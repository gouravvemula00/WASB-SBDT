from torch import nn

from .heatmap import HeatmapLoss
from .segmentation import SegmentationLoss
from .bounce_loss import BounceLoss
from .combined_loss import CombinedLoss

__factory = {
        'heatmap': HeatmapLoss,
        'segmentation': SegmentationLoss,
        'bounce': BounceLoss,
        'combined': CombinedLoss
        }

def build_loss_criteria(cfg):
    #print(cfg)
    #print( cfg['loss']['name'] )
    loss_name = cfg['loss']['name']

    if not loss_name in __factory.keys():
        raise KeyError('invalid loss: {}'.format(loss_name ))

    loss = __factory[loss_name](cfg)

    return loss

