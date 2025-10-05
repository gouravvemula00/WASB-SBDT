import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

log = logging.getLogger(__name__)

class BounceLoss(nn.Module):
    """
    Loss function for bounce detection
    """
    def __init__(self, loss_type='bce', auto_weight=False, scales=[0], neg_factor=3):
        super().__init__()
        self.loss_type = loss_type
        self.auto_weight = auto_weight
        self.scales = scales
        self.neg_factor = neg_factor
        
        if loss_type == 'bce':
            self.loss_fn = nn.BCEWithLogitsLoss()
        elif loss_type == 'ce':
            self.loss_fn = nn.CrossEntropyLoss()
        elif loss_type == 'focal':
            self.loss_fn = FocalLoss(alpha=1, gamma=2)
        else:
            raise ValueError(f'Unsupported loss type: {loss_type}')
            
        # Initialize learnable weights for multi-scale loss
        if self.auto_weight:
            self._ws = nn.ParameterDict()
            for scale in self.scales:
                self._ws[f'loss_w_s{scale}'] = nn.Parameter(torch.tensor(0.0))

    def forward(self, predicts, targets):
        """
        Args:
            predicts: dict with 'bounce' key containing bounce predictions
            targets: dict with 'bounce' key containing bounce targets
        """
        if 'bounce' not in predicts:
            return torch.tensor(0.0, device=next(iter(predicts.values())).device)
            
        loss_acc = 0
        for scale in self.scales:
            if scale in predicts['bounce']:
                pred = predicts['bounce'][scale]
                target = targets['bounce'][scale]
                
                # Convert target to appropriate format
                if self.loss_type == 'bce':
                    # For BCE, target should be binary (0 or 1)
                    if target.dim() == 4 and target.size(1) == 1:
                        target = target.squeeze(1)
                    target = target.float()
                elif self.loss_type == 'ce':
                    # For CE, target should be class indices
                    if target.dim() == 4 and target.size(1) == 1:
                        target = target.squeeze(1)
                    target = target.long()
                
                # Reshape for loss computation
                if pred.dim() == 4:
                    b, c, h, w = pred.shape
                    pred = pred.view(b, c, -1)
                    target = target.view(b, -1)
                
                loss = self.loss_fn(pred, target)
                
                if self.auto_weight:
                    loss_acc += loss * torch.exp(-self._ws[f'loss_w_s{scale}']) + self._ws[f'loss_w_s{scale}']
                else:
                    loss_acc += loss
                    
        return loss_acc


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    """
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()
