"""
Prediction script for pose validation
Simple inference on new data
"""

import torch
import json
from model import PoseValidationTCN
from trainer import Trainer
from utils import PosePredictor


def load_trained_model(model_path):
    """Load a trained model from file"""
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        
        if 'model_config' not in checkpoint:
            print("Warning: Model config not found. Using default parameters.")
            # Default config - adjust if needed
            config = {
                'input_size': 103,  # Adjust based on your data
                'hidden_channels': [64, 128],
                'kernel_size': 3,
                'dropout': 0.2
            }
        else:
            config = checkpoint['model_config']
        
        # Create model
        model = PoseValidationTCN(**config)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"Model loaded successfully from {model_path}")
        print(f"Model config: {config}")
        
        return model
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def predict_single_sample(model_path, sample_data):
    """
    Predict on a single sample
    
    Args:
        model_path: Path to trained model
        sample_data: Dict with 'keypoints_3d' and 'ts' keys
    """
    # Load model
    model = load_trained_model(model_path)
    if model is None:
        return None, None
    
    # Create predictor
    predictor = PosePredictor(model)
    
    # Make prediction
    try:
        predictions, confidences = predictor.predict_from_dict(sample_data)
        
        # Get active classes (where prediction > 0.5)
        active_classes = [i for i, pred in enumerate(predictions) if pred > 0.5]
        
        if not active_classes:
            print("No classes detected")
        else:
            print(f"Classes detected: {active_classes}")
        
        # Print confidence for each class
        for i, conf in enumerate(confidences):
            print(f"Class {i} confidence: {conf:.3f}")
        
        return predictions, confidences
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None, None


def predict_from_file(model_path, data_path):
    """
    Predict on data from a JSON file
    
    Args:
        model_path: Path to trained model
        data_path: Path to JSON file with pose data
    """
    # Load model
    model = load_trained_model(model_path)
    if model is None:
        return
    
    # Load data
    try:
        with open(data_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Create predictor
    predictor = PosePredictor(model)
    
    # Predict on each sample
    print(f"\nPredicting on {len(data)} samples...")
    print("-" * 50)
    
    correct_predictions = 0
    total_predictions = 0
    
    for i, sample in enumerate(data):
        try:
            # Extract data
            keypoints_3d = [frame['keypoints_3d'] for frame in sample['data']]
            timestamps = [frame['ts'] for frame in sample['data']]
            true_label = sample.get('label', None)
            
            # Make prediction
            predictions, confidences = predictor.predict_sequence(keypoints_3d, timestamps)
            
            # Get active classes (where prediction > 0.5)
            active_classes = [i for i, pred in enumerate(predictions) if pred > 0.5]
            
            if not active_classes:
                result_text = "No classes detected"
            else:
                result_text = f"Classes detected: {active_classes}"
            
            # Format confidence scores
            confidence_text = " ".join([f"Class {i}: {conf:.3f}" for i, conf in enumerate(confidences)])
            print(f"Sample {i+1}: {result_text}")
            print(f"Confidences: {confidence_text}")
            
            # Compare with true labels if available
            if true_label is not None:
                true_classes = true_label if isinstance(true_label, list) else [true_label]
                true_classes = set(true_classes)
                pred_classes = set(active_classes)
                
                # Calculate metrics
                correct_preds = len(true_classes.intersection(pred_classes))
                total_true = len(true_classes)
                total_pred = len(pred_classes)
                
                print(f"  True labels: {sorted(true_classes)}")
                print(f"  Precision: {correct_preds/total_pred if total_pred > 0 else 0:.3f}")
                print(f"  Recall: {correct_preds/total_true if total_true > 0 else 0:.3f}")
                
                if true_classes == pred_classes:
                    correct_predictions += 1
                total_predictions += 1
            
            print()
            
        except Exception as e:
            print(f"Error predicting sample {i+1}: {e}")
    
    # Print overall accuracy if labels were available
    if total_predictions > 0:
        accuracy = correct_predictions / total_predictions
        print(f"Overall accuracy: {accuracy:.3f} ({correct_predictions}/{total_predictions})")


def main():
    """Main prediction function"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python predict.py <model_path> [data_path]")
        print("\nExamples:")
        print("  python predict.py pose_validation_model.pth")
        print("  python predict.py pose_validation_model.pth test_data.json")
        return
    
    model_path = sys.argv[1]
    
    if len(sys.argv) >= 3:
        # Predict from file
        data_path = sys.argv[2]
        predict_from_file(model_path, data_path)
    else:
        # Interactive mode - create sample data for demonstration
        print("Interactive mode - demonstrating with sample data")
        print("In practice, replace this with your actual pose data")
        
        # Example sample data (replace with real data)
        # Note: This creates 30 frames, each with 34 keypoints (x,y,z coordinates)
        sample_keypoints = []
        sample_timestamps = []
        
        # Create 30 frames of data
        for frame_idx in range(30):
            # Create 34 keypoints for this frame
            frame_keypoints = []
            for joint_idx in range(34):
                # Example keypoint coordinates (replace with real data)
                x = 0.24 + joint_idx * 0.01 + frame_idx * 0.001
                y = 0.008 + joint_idx * 0.02 + frame_idx * 0.002  
                z = -2.14 + joint_idx * 0.01 + frame_idx * 0.001
                frame_keypoints.append([x, y, z])
            
            sample_keypoints.append(frame_keypoints)
            sample_timestamps.append(1757717714.17 + frame_idx * 0.033)  # 30 FPS
        
        sample_data = {
            'keypoints_3d': sample_keypoints,  # 30 frames × 34 keypoints × 3 coordinates
            'ts': sample_timestamps           # 30 timestamps
        }
        
        predict_single_sample(model_path, sample_data)


if __name__ == "__main__":
    main()