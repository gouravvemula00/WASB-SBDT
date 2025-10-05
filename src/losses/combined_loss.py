import torch
import torch.nn as nn
import logging

from .heatmap import HeatmapLoss
from .bounce_loss import BounceLoss

log = logging.getLogger(__name__)

class CombinedLoss(nn.Module):
    """
    Combined loss for ball detection and bounce classification
    """
    def __init__(self, cfg):
        super().__init__()
        
        # Ball detection loss
        self.ball_loss = HeatmapLoss(cfg)
        
        # Bounce detection loss
        bounce_cfg = cfg.get('bounce_loss', {})
        self.bounce_loss = BounceLoss(
            loss_type=bounce_cfg.get('loss_type', 'bce'),
            auto_weight=bounce_cfg.get('auto_weight', False),
            scales=bounce_cfg.get('scales', [0]),
            neg_factor=bounce_cfg.get('neg_factor', 3)
        )
        
        # Loss weights
        self.ball_weight = cfg.get('ball_weight', 1.0)
        self.bounce_weight = cfg.get('bounce_weight', 1.0)
        
        # Enable bounce loss only if bounce detection is enabled
        self.enable_bounce = cfg.get('enable_bounce_detection', False)

    def forward(self, predicts, targets):
        """
        Args:
            predicts: dict with '0' (ball detection) and optionally 'bounce' keys
            targets: dict with '0' (ball detection) and optionally 'bounce' keys
        """
        # Extract ball detection parts for HeatmapLoss
        ball_predicts = {0: predicts[0]} if 0 in predicts else {}
        ball_targets = {0: targets[0]} if 0 in targets else {}
        
        # Ball detection loss
        ball_loss = self.ball_loss(ball_predicts, ball_targets)
        
        # Bounce detection loss (only if enabled and predictions available)
        bounce_loss = torch.tensor(0.0, device=ball_loss.device)
        if self.enable_bounce and 'bounce' in predicts:
            bounce_loss = self.bounce_loss(predicts, targets)
        
        # Combined loss
        total_loss = self.ball_weight * ball_loss + self.bounce_weight * bounce_loss
        
        return total_loss
