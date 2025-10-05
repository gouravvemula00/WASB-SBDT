# Tennis Ball Bounce Detection

This implementation adds bounce detection capability to the WASB (Wide Area Sports Ball) detection system for tennis. The system can now detect both ball positions and classify whether the ball is bouncing or not.

## Features Added

### 1. Enhanced Data Structure
- Extended `Center` dataclass to include `is_bounce` field
- Updated tennis dataset loader to handle bounce annotations from CSV files
- Added bounce column support in tennis configuration

### 2. Model Architecture
- **DeepBall Model**: Added bounce classification head alongside ball detection
- **WASB/HRNet Model**: Extended with bounce detection heads
- Both models now output both ball detection and bounce classification predictions

### 3. Loss Functions
- **BounceLoss**: Dedicated loss function for bounce classification
- **CombinedLoss**: Combines ball detection and bounce classification losses
- Support for different loss types (BCE, CE, Focal Loss)

### 4. Data Processing
- **Bounce Heatmap Generation**: Creates 2-channel heatmaps for bounce/no-bounce classification
- **Enhanced Data Loading**: Processes bounce annotations during training
- **Multi-scale Support**: Works with different output scales

## Usage

### Training with Bounce Detection

1. **Prepare your data**: Ensure your tennis CSV files include a 'bounce' column with boolean values indicating whether each frame contains a ball bounce.

2. **Run training**:
   ```bash
   python train_tennis_bounce.py
   ```

3. **Test the setup**:
   ```bash
   python test_training_setup.py
   ```

### Configuration

The training uses the configuration in `src/configs/train_tennis_bounce.yaml`. Key settings:

- **Model**: Uses WASB (HRNet) architecture with bounce detection enabled
- **Loss**: Combined loss with configurable weights for ball detection and bounce classification
- **Dataset**: Tennis dataset with bounce column support
- **Training**: 100 epochs with validation every 5 epochs

### Model Outputs

The trained model outputs:
- **Ball Detection**: `outputs[0]` - Shape: `(batch, frames_out, height, width)`
- **Bounce Classification**: `outputs['bounce']` - Shape: `(batch, 2, height, width)`
  - Channel 0: No-bounce probability
  - Channel 1: Bounce probability

## File Structure

### New Files
- `src/losses/bounce_loss.py` - Bounce classification loss
- `src/losses/combined_loss.py` - Combined ball detection + bounce loss
- `src/configs/loss/combined.yaml` - Combined loss configuration
- `src/configs/train_tennis_bounce.yaml` - Complete training configuration
- `train_tennis_bounce.py` - Training script
- `test_training_setup.py` - Setup verification script

### Modified Files
- `src/utils/dataclasses.py` - Added `is_bounce` to Center class
- `src/utils/file.py` - Updated CSV loading for bounce annotations
- `src/datasets/tennis.py` - Added bounce column support
- `src/models/deepball.py` - Added bounce detection head
- `src/models/hrnet.py` - Added bounce detection heads for WASB
- `src/models/__init__.py` - Updated model building for bounce detection
- `src/dataloaders/heatmaps/heatmaps.py` - Added bounce heatmap generation
- `src/dataloaders/dataset_loader.py` - Updated to handle bounce annotations
- `src/detectors/deepball_detector.py` - Updated to handle bounce predictions
- `src/detectors/deepball_postprocessor.py` - Added bounce prediction processing

## Data Format

### CSV File Format
Your tennis CSV files should include a 'bounce' column:

```csv
file name,visibility,x-coordinate,y-coordinate,bounce
000001.jpg,1,100.5,200.3,True
000002.jpg,1,105.2,198.7,False
000003.jpg,1,110.1,195.2,True
...
```

- `bounce`: Boolean indicating whether the ball is bouncing in this frame
- `True`: Ball is bouncing
- `False`: Ball is not bouncing

## Training Tips

1. **Data Quality**: Ensure accurate bounce annotations for better performance
2. **Loss Weights**: Adjust `ball_weight` and `bounce_weight` in the config based on your priorities
3. **Batch Size**: Start with smaller batch sizes if you encounter memory issues
4. **Learning Rate**: The default learning rate (0.001) works well, but you can adjust based on your data

## Evaluation

The model outputs both ball detection and bounce classification results. You can evaluate:
- Ball detection accuracy (position and visibility)
- Bounce classification accuracy
- Combined performance metrics

## Troubleshooting

1. **Memory Issues**: Reduce batch size or input resolution
2. **Training Instability**: Check bounce annotation quality and loss weights
3. **Poor Bounce Performance**: Ensure bounce annotations are accurate and consistent

## Future Enhancements

- Support for other sports (basketball, volleyball, etc.)
- Temporal consistency for bounce detection
- Confidence scoring for bounce predictions
- Real-time inference optimization
