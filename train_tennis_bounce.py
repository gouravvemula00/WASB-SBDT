#!/usr/bin/env python3
"""
Training script for tennis ball detection with bounce classification using WASB
"""

import os
import sys
import logging
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.hydra_config import HydraConfig

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.runners import select_runner

# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="src/configs", config_name="train_tennis_bounce")
def main(cfg: DictConfig) -> None:
    """
    Main training function
    """
    log.info("Starting tennis ball detection training with bounce classification")
    log.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    
    # Create output directory
    output_dir = HydraConfig.get().run.dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Save configuration
    with open(os.path.join(output_dir, "config.yaml"), "w") as f:
        OmegaConf.save(cfg, f)
    
    # Select and run trainer
    runner = select_runner(cfg)
    runner.run()
    
    log.info("Training completed!")

if __name__ == "__main__":
    main()
