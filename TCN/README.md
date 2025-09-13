# Pose Validation TCN

A simple and modular Temporal Convolutional Network (TCN) for validating human pose movements using 3D keypoint data.

## Features

- **Simple Architecture**: Clean TCN implementation with temporal blocks
- **Modular Design**: Separated concerns across multiple files
- **Easy to Use**: Simple training and prediction scripts
- **Configurable**: Centralized configuration management
- **Visualization**: Training plots and evaluation metrics

## Project Structure

```
TCN/
├── model.py           # TCN model definition
├── data_utils.py      # Data loading and preprocessing
├── trainer.py         # Training logic
├── utils.py          # Utility functions and visualization
├── config.py         # Configuration management
├── train.py          # Main training script
├── predict.py        # Prediction script
└── README.md         # This file
```

## Installation

Install required dependencies:

```bash
pip install torch torchvision numpy matplotlib seaborn scikit-learn
```

## Data Format

Your data should be in JSON format with the following structure:

```json
[
  {
    "label": 0,  // 0 = correct, 1, 2, 3, 4, 5, 6 = different incorrect types
    "data": [
      {
        "ts": 1757717714.1701035,  // timestamp
        "keypoints_3d": [
          [0.24, 0.008, -2.14],    // [x, y, z] for joint 1
          [0.24, 0.19, -2.17],     // [x, y, z] for joint 2
          // ... more joints
        ]
      },
      // ... more frames
    ]
  },
  // ... more samples
]
```

## Quick Start

### 1. Training

```bash
python train.py
```

This will:
- Load data from `merged_data.json`
- Train the TCN model
- Show training plots
- Evaluate on test set
- Save the trained model

### 2. Prediction

```bash
# Predict on new data file
python predict.py pose_validation_model.pth test_data.json

# Interactive mode (with sample data)
python predict.py pose_validation_model.pth
```

## Configuration

Edit `config.py` to adjust settings:

```python
# Data settings
DATA_PATH = 'your_data.json'
SEQUENCE_LENGTH = 30  # Number of frames per sequence

# Model settings
HIDDEN_CHANNELS = [64, 128]  # TCN channel sizes
DROPOUT = 0.2

# Training settings
BATCH_SIZE = 32
NUM_EPOCHS = 100
LEARNING_RATE = 0.001
```

## Model Architecture

The TCN consists of:
- **Temporal Blocks**: Dilated convolutions with residual connections
- **Increasing Dilation**: Captures different temporal scales
- **Global Pooling**: Aggregates temporal information
- **Classification Head**: Binary classification (correct/incorrect)

## Advanced Usage

### Custom Training

```python
from model import PoseValidationTCN
from trainer import Trainer
from data_utils import create_data_loaders

# Create model
model = PoseValidationTCN(input_size=103, hidden_channels=[64, 128])

# Create trainer
trainer = Trainer(model)

# Create data loaders
train_loader, val_loader, test_loader = create_data_loaders('data.json')

# Train
history = trainer.train(train_loader, val_loader, num_epochs=50)
```

### Custom Prediction

```python
from model import PoseValidationTCN
from utils import PosePredictor
import torch

# Load model
model = PoseValidationTCN(input_size=103, hidden_channels=[64, 128])
checkpoint = torch.load('model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Create predictor
predictor = PosePredictor(model)

# Predict
keypoints_3d = [...]  # Your 3D keypoints
timestamps = [...]    # Your timestamps
prediction, confidence = predictor.predict_sequence(keypoints_3d, timestamps)
```

## Model Parameters

- **Input Size**: Depends on number of joints (num_joints × 3 + 1 for timestamp)
- **Hidden Channels**: List of channel sizes for each TCN layer
- **Kernel Size**: Convolution kernel size (default: 3)
- **Dropout**: Dropout rate for regularization

## Tips

1. **Sequence Length**: Longer sequences capture more temporal context but require more memory
2. **Hidden Channels**: More channels increase model capacity but may overfit
3. **Data Quality**: Ensure consistent keypoint ordering and coordinate systems
4. **Preprocessing**: Consider normalizing coordinates and timestamps

## Troubleshooting

**Memory Issues**: Reduce batch size or sequence length

**Poor Performance**: 
- Increase model capacity (more channels)
- Collect more training data
- Check data quality and labeling

**Slow Training**: 
- Use GPU if available
- Reduce sequence length
- Use fewer workers in data loading

## License

MIT License - feel free to use and modify as needed.