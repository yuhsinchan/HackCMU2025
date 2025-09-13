"""
Configuration file for pose validation TCN
Centralized configuration management
"""

import os


class Config:
    """Main configuration class"""
    
    # Data settings
    DATA_PATH = 'pose_data.json'
    SEQUENCE_LENGTH = 9  # Number of consecutive frames per sequence
    TRAIN_SPLIT = 0.7
    VAL_SPLIT = 0.15
    # TEST_SPLIT = 0.15 (calculated automatically)
    
    # Model settings - will be auto-detected from data
    # Expected: 34 keypoints × 3 coordinates + 1 timestamp = 103 features per frame
    HIDDEN_CHANNELS = [64, 128]
    KERNEL_SIZE = 3
    DROPOUT = 0.2
    NUM_CLASSES = 7
    
    # Training settings
    BATCH_SIZE = 16
    NUM_EPOCHS = 15
    LEARNING_RATE = 0.001
    RANDOM_SEED = 42
    
    # System settings
    NUM_WORKERS = 1
    DEVICE = 'auto'  # 'auto', 'cpu', or 'cuda'
    
    # Output settings
    MODEL_SAVE_PATH = 'pose_validation_model.pth'
    VERBOSE = True
    PLOT_RESULTS = True
    
    @classmethod
    def from_dict(cls, config_dict):
        """Create config from dictionary"""
        config = cls()
        for key, value in config_dict.items():
            if hasattr(config, key.upper()):
                setattr(config, key.upper(), value)
        return config
    
    @classmethod
    def from_file(cls, config_path):
        """Load config from JSON file"""
        import json
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def to_dict(self):
        """Convert config to dictionary"""
        return {
            key.lower(): value 
            for key, value in self.__class__.__dict__.items() 
            if not key.startswith('_') and not callable(value)
        }
    
    def save(self, config_path):
        """Save config to JSON file"""
        import json
        with open(config_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def validate(self):
        """Validate configuration values"""
        assert os.path.exists(self.DATA_PATH), f"Data file not found: {self.DATA_PATH}"
        assert self.SEQUENCE_LENGTH > 0, "Sequence length must be positive"
        assert 0 < self.TRAIN_SPLIT < 1, "Train split must be between 0 and 1"
        assert 0 < self.VAL_SPLIT < 1, "Validation split must be between 0 and 1"
        assert self.TRAIN_SPLIT + self.VAL_SPLIT < 1, "Train + Val splits must be less than 1"
        assert self.BATCH_SIZE > 0, "Batch size must be positive"
        assert self.NUM_EPOCHS > 0, "Number of epochs must be positive"
        assert self.LEARNING_RATE > 0, "Learning rate must be positive"
        
        return True


# Default configuration instance
default_config = Config()