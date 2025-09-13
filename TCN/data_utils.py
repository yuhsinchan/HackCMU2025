"""
Data utilities for pose validation
Simple data loading and processing
"""

import json
import numpy as np
import torch
from torch.utils.data import Dataset


class PoseDataset(Dataset):
    """Dataset for pose validation with 3D keypoints"""
    
    def __init__(self, data_path, sequence_length=30):
        self.sequence_length = sequence_length
        self.sequences, self.labels = self._load_data(data_path)
        
    def _load_data(self, data_path):
        """Load and process pose data from JSON file"""
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        sequences = []
        labels = []
        
        for sample in data:
            label = sample['label']
            frames = sample['data']

            # Extract keypoints and create sequences
            keypoints_sequence = []
            timestamps = []
            
            for frame in frames:
                # Flatten 3D keypoints: [[x1,y1,z1], [x2,y2,z2], ...] -> [x1,y1,z1,x2,y2,z2,...]
                keypoints_flat = np.array(frame['keypoints_3d']).flatten()
                keypoints_sequence.append(keypoints_flat)
                timestamps.append(frame['ts'])

            # Create sliding windows if sequence is long enough
            if len(keypoints_sequence) >= self.sequence_length:
                for i in range(len(keypoints_sequence) - self.sequence_length + 1):
                    seq = keypoints_sequence[i:i + self.sequence_length]
                    seq_timestamps = timestamps[i:i + self.sequence_length]
                    
                    # Add time differences as features
                    seq_with_time = self._add_temporal_features(seq, seq_timestamps)
                    
                    sequences.append(seq_with_time)
                    labels.append(label)
        
        return np.array(sequences), np.array(labels)
    
    def _add_temporal_features(self, keypoints_seq, timestamps):
        """Add temporal information to keypoints"""
        seq_with_time = []
        start_time = timestamps[0]
        
        for i, keypoints in enumerate(keypoints_seq):
            # Add time difference from start
            time_diff = timestamps[i] - start_time
            features = np.append(keypoints, time_diff)
            seq_with_time.append(features)
            
        return np.array(seq_with_time)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        # Convert to tensor and transpose for TCN: (features, time_steps)
        sequence = torch.FloatTensor(self.sequences[idx]).transpose(0, 1)
        label = torch.LongTensor([self.labels[idx]])[0]
        return sequence, label


def create_data_loaders(data_path, sequence_length=30, batch_size=32, train_split=0.7, val_split=0.15):
    """Create train, validation, and test data loaders"""
    
    # Create dataset
    dataset = PoseDataset(data_path, sequence_length)
    # print("Dataset shape:", dataset.sequences.shape)
    # Calculate split sizes
    total_size = len(dataset)

    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    
    return train_loader, val_loader, test_loader


def get_input_size(data_path, sequence_length=30):
    """Calculate input size based on data structure"""
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    if len(data) > 0 and len(data[0]['data']) > 0:
        # Get number of keypoints from first frame
        num_keypoints = len(data[0]['data'][0]['keypoints_3d'])
        # Each keypoint has x, y, z coordinates + 1 timestamp feature
        input_size = num_keypoints * 3 + 1
        
        print(f"Detected {num_keypoints} keypoints per frame")
        print(f"Input size: {num_keypoints} keypoints × 3 coordinates + 1 timestamp = {input_size} features per frame")
        
        return input_size
    
    print("Error: Could not determine input size from data")
    return None