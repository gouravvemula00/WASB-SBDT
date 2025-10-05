#!/usr/bin/env python3
"""
Test script for bounce detection implementation
"""

import torch
import numpy as np
from src.models.deepball import DeepBall
from src.utils.dataclasses import Center
from src.utils.heatmap import gen_bounce_heatmap
from src.losses.combined_loss import CombinedLoss

def test_model_architecture():
    """Test that the model can be created with bounce detection enabled"""
    print("Testing model architecture...")
    
    # Test model creation
    model = DeepBall(
        n_channels=3,
        n_classes=2,
        enable_bounce_detection=True
    )
    
    # Test forward pass
    x = torch.randn(1, 3, 180, 320)
    outputs = model(x)
    
    print(f"Model outputs keys: {list(outputs.keys())}")
    print(f"Ball detection output shape: {outputs[0].shape}")
    print(f"Bounce detection output shape: {outputs['bounce'].shape}")
    
    assert 0 in outputs, "Ball detection output missing"
    assert 'bounce' in outputs, "Bounce detection output missing"
    assert outputs[0].shape == (1, 2, 180, 320), f"Unexpected ball detection shape: {outputs[0].shape}"
    assert outputs['bounce'].shape == (1, 2, 180, 320), f"Unexpected bounce detection shape: {outputs['bounce'].shape}"
    
    print("✓ Model architecture test passed!")

def test_center_dataclass():
    """Test the updated Center dataclass with bounce information"""
    print("Testing Center dataclass...")
    
    # Test with bounce
    center_with_bounce = Center(x=100.0, y=200.0, is_visible=True, is_bounce=True)
    assert center_with_bounce.is_bounce == True
    assert center_with_bounce.xy == (100.0, 200.0)
    
    # Test without bounce
    center_without_bounce = Center(x=150.0, y=250.0, is_visible=True, is_bounce=False)
    assert center_without_bounce.is_bounce == False
    
    print("✓ Center dataclass test passed!")

def test_bounce_heatmap_generation():
    """Test bounce heatmap generation"""
    print("Testing bounce heatmap generation...")
    
    # Test bounce case
    bounce_hm = gen_bounce_heatmap(
        wh=(320, 180),
        cxy=(160, 90),
        r=5.0,
        is_bounce=True
    )
    
    assert bounce_hm.shape == (2, 180, 320), f"Unexpected bounce heatmap shape: {bounce_hm.shape}"
    assert np.max(bounce_hm[1]) > 0, "Bounce channel should have non-zero values"
    assert np.max(bounce_hm[0]) > 0, "No-bounce channel should have non-zero values"
    
    # Test no-bounce case
    no_bounce_hm = gen_bounce_heatmap(
        wh=(320, 180),
        cxy=(160, 90),
        r=5.0,
        is_bounce=False
    )
    
    assert no_bounce_hm.shape == (2, 180, 320), f"Unexpected no-bounce heatmap shape: {no_bounce_hm.shape}"
    assert np.max(no_bounce_hm[1]) == 0, "Bounce channel should be zero for no-bounce case"
    assert np.max(no_bounce_hm[0]) == 1.0, "No-bounce channel should be 1.0 for no-bounce case"
    
    print("✓ Bounce heatmap generation test passed!")

def test_combined_loss():
    """Test the combined loss function"""
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
    
    # Create loss function
    loss_fn = CombinedLoss(cfg)
    
    # Create mock predictions and targets
    predicts = {
        0: torch.randn(1, 2, 180, 320),  # Ball detection
        'bounce': torch.randn(1, 2, 180, 320)  # Bounce detection
    }
    
    targets = {
        0: torch.randn(1, 2, 180, 320),  # Ball detection targets
        'bounce': torch.randn(1, 2, 180, 320)  # Bounce detection targets
    }
    
    # Test loss computation
    loss = loss_fn(predicts, targets)
    
    assert isinstance(loss, torch.Tensor), "Loss should be a tensor"
    assert loss.requires_grad, "Loss should require gradients"
    
    print("✓ Combined loss test passed!")

def main():
    """Run all tests"""
    print("Running bounce detection tests...\n")
    
    try:
        test_center_dataclass()
        test_bounce_heatmap_generation()
        test_model_architecture()
        test_combined_loss()
        
        print("\n🎉 All tests passed! Bounce detection implementation is working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise

if __name__ == "__main__":
    main()
