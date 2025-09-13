"""
Simple prediction service for web applications
Load model once, predict on demand
"""

import json
import torch
import numpy as np
from model import PoseValidationTCN


class PredictionService:
    """Simple prediction service that loads model once"""
    
    def __init__(self, model_path):
        """Initialize and load model"""
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_model(model_path)
        print(f"Model loaded and ready for predictions on {self.device}")
    
    def load_model(self, model_path):
        """Load trained model"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Get model config or use defaults
            if 'model_config' in checkpoint:
                config = checkpoint['model_config']
            else:
                config = {
                    'input_size': 103,
                    'hidden_channels': [64, 128], 
                    'kernel_size': 3,
                    'dropout': 0.2,
                    'num_classes': 7
                }
            
            # Create and load model
            self.model = PoseValidationTCN(**config)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
        except Exception as e:
            raise Exception(f"Failed to load model: {e}")
    
    def predict_json_file(self, json_file_path):
        """Predict on a JSON file and return results"""
        try:
            # Load JSON data
            with open(json_file_path, 'r') as f:
                data = json.load(f)
            
            # Handle single sample or list format
            if isinstance(data, dict) and 'data' in data:
                data = [data]  # Convert single sample to list
            
            results = []
            
            for i, sample in enumerate(data):
                try:
                    # Extract keypoints and timestamps
                    keypoints_3d = [frame['keypoints_3d'] for frame in sample['data']]
                    timestamps = [frame['ts'] for frame in sample['data']]
                    
                    # Make prediction
                    prediction, confidence = self.predict_sequence(keypoints_3d, timestamps)
                    
                    results.append({
                        'sample_id': i,
                        'prediction': int(prediction),
                        'confidence': float(confidence),
                        'is_correct': prediction == 0
                    })
                    
                except Exception as e:
                    results.append({
                        'sample_id': i,
                        'error': str(e),
                        'prediction': None,
                        'confidence': None,
                        'is_correct': None
                    })
            
            return results
            
        except Exception as e:
            raise Exception(f"Failed to process JSON file: {e}")
    
    def predict_sequence(self, keypoints_3d, timestamps, sequence_length=10):
        """Predict on a single sequence"""
        try:
            if len(keypoints_3d) < sequence_length:
                raise ValueError(f"Need at least {sequence_length} frames, got {len(keypoints_3d)}")
            
            # Use the last sequence_length frames
            recent_keypoints = keypoints_3d[-sequence_length:]
            recent_timestamps = timestamps[-sequence_length:]
            
            # Process data same as training
            sequence = []
            start_time = recent_timestamps[0]
            
            for i, keypoints in enumerate(recent_keypoints):
                # Flatten keypoints and add time difference
                keypoints_flat = np.array(keypoints).flatten()
                time_diff = recent_timestamps[i] - start_time
                features = np.append(keypoints_flat, time_diff)
                sequence.append(features)
            
            # Convert to tensor and predict
            sequence_tensor = torch.FloatTensor(sequence).transpose(0, 1).unsqueeze(0)
            sequence_tensor = sequence_tensor.to(self.device)
            
            with torch.no_grad():
                logits = self.model(sequence_tensor)
                probs = torch.softmax(logits, dim=1)
                prediction = torch.argmax(logits, dim=1).item()
                confidence = torch.max(probs, dim=1)[0].item()
            
            return prediction, confidence
            
        except Exception as e:
            raise Exception(f"Prediction failed: {e}")


# Global prediction service instance
_prediction_service = None


def initialize_service(model_path):
    """Initialize the prediction service (call once at startup)"""
    global _prediction_service
    _prediction_service = PredictionService(model_path)
    return _prediction_service


def predict_file(json_file_path):
    """Predict on a JSON file (call this from web endpoints)"""
    if _prediction_service is None:
        raise Exception("Service not initialized. Call initialize_service() first.")
    
    return _prediction_service.predict_json_file(json_file_path)