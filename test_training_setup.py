#!/usr/bin/env python3
"""
Test script to verify the training setup for tennis bounce detection
"""

import os
import sys
import torch
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.models import build_model
from src.losses import build_loss_criteria
from src.utils.dataclasses import Center

# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def test_wasb_model():
    """Test WASB model with bounce detection"""
    print("Testing WASB model with bounce detection...")
    
    # Create mock config
    cfg = {
        'model': {
            'name': 'hrnet',
            'frames_in': 3,
            'frames_out': 3,
            'inp_height': 288,
            'inp_width': 512,
            'out_height': 288,
            'out_width': 512,
            'rgb_diff': False,
            'out_scales': [0],
            'enable_bounce_detection': True,
            'MODEL': {
                'EXTRA': {
                    'FINAL_CONV_KERNEL': 1,
                    'PRETRAINED_LAYERS': ['*'],
                    'STEM': {
                        'INPLANES': 64,
                        'STRIDES': [1, 1]
                    },
                    'STAGE1': {
                        'NUM_MODULES': 1,
                        'NUM_BRANCHES': 1,
                        'BLOCK': 'BOTTLENECK',
                        'NUM_BLOCKS': [1],
                        'NUM_CHANNELS': [32],
                        'FUSE_METHOD': 'SUM'
                    },
                    'STAGE2': {
                        'NUM_MODULES': 1,
                        'NUM_BRANCHES': 2,
                        'BLOCK': 'BASIC',
                        'NUM_BLOCKS': [2, 2],
                        'NUM_CHANNELS': [16, 32],
                        'FUSE_METHOD': 'SUM'
                    },
                    'STAGE3': {
                        'NUM_MODULES': 1,
                        'NUM_BRANCHES': 3,
                        'BLOCK': 'BASIC',
                        'NUM_BLOCKS': [2, 2, 2],
                        'NUM_CHANNELS': [16, 32, 64],
                        'FUSE_METHOD': 'SUM'
                    },
                    'STAGE4': {
                        'NUM_MODULES': 1,
                        'NUM_BRANCHES': 4,
                        'BLOCK': 'BASIC',
                        'NUM_BLOCKS': [2, 2, 2, 2],
                        'NUM_CHANNELS': [16, 32, 64, 128],
                        'FUSE_METHOD': 'SUM'
                    },
                    'DECONV': {
                        'NUM_DECONVS': 0,
                        'KERNEL_SIZE': [],
                        'NUM_BASIC_BLOCKS': 2
                    }
                },
                'INIT_WEIGHTS': True
            }
        }
    }
    
    # Build model
    model = build_model(cfg)
    
    # Test forward pass
    x = torch.randn(1, 9, 288, 512)  # 3 frames * 3 channels
    outputs = model(x)
    
    print(f"Model outputs keys: {list(outputs.keys())}")
    print(f"Ball detection output shape: {outputs[0].shape}")
    print(f"Bounce detection output shape: {outputs['bounce'][0].shape}")
    
    assert 0 in outputs, "Ball detection output missing"
    assert 'bounce' in outputs, "Bounce detection output missing"
    assert outputs[0].shape == (1, 3, 288, 512), f"Unexpected ball detection shape: {outputs[0].shape}"
    assert outputs['bounce'][0].shape == (1, 2, 288, 512), f"Unexpected bounce detection shape: {outputs['bounce'][0].shape}"
    
    print("✓ WASB model test passed!")

def test_combined_loss():
    """Test combined loss function"""
    print("Testing combined loss...")
    
    # Create mock config
    cfg = {
        'ball_weight': 1.0,
        'bounce_weight': 0.5,
        'enable_bounce_detection': True,
        'bounce_loss': {
            'loss_type': 'bce',
            'auto_weight': False,
            'scales': [0],
            'neg_factor': 3
        }
    }
    
    # Create loss function directly
    from src.losses.combined_loss import CombinedLoss
    loss_fn = CombinedLoss(cfg)
    
    # Create mock predictions and targets
    predicts = {
        0: torch.randn(1, 3, 288, 512),  # Ball detection
        'bounce': {0: torch.randn(1, 2, 288, 512)}  # Bounce detection (multi-scale format)
    }
    
    targets = {
        0: torch.randn(1, 3, 288, 512),  # Ball detection targets
        'bounce': {0: torch.randn(1, 2, 288, 512)}  # Bounce detection targets (multi-scale format)
    }
    
    # Test loss computation
    loss = loss_fn(predicts, targets)
    
    assert isinstance(loss, torch.Tensor), "Loss should be a tensor"
    assert loss.requires_grad, "Loss should require gradients"
    
    print("✓ Combined loss test passed!")

def test_center_with_bounce():
    """Test Center dataclass with bounce information"""
    print("Testing Center dataclass with bounce...")
    
    # Test with bounce
    center_with_bounce = Center(x=100.0, y=200.0, is_visible=True, is_bounce=True)
    assert center_with_bounce.is_bounce == True
    assert center_with_bounce.xy == (100.0, 200.0)
    
    # Test without bounce
    center_without_bounce = Center(x=150.0, y=250.0, is_visible=True, is_bounce=False)
    assert center_without_bounce.is_bounce == False
    
    print("✓ Center dataclass test passed!")

def main():
    """Run all tests"""
    print("Running training setup tests...\n")
    
    try:
        test_center_with_bounce()
        test_wasb_model()
        test_combined_loss()
        
        print("\n🎉 All tests passed! Training setup is ready for tennis bounce detection.")
        print("\nTo start training, run:")
        print("python train_tennis_bounce.py")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
