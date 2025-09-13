"""
Utility functions for pose validation
Simple helper functions for visualization and inference
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


def plot_training_history(history):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    ax1.plot(history['train_loss'], label='Train Loss', color='blue')
    ax1.plot(history['val_loss'], label='Val Loss', color='red')
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(history['train_acc'], label='Train Acc', color='blue')
    ax2.plot(history['val_acc'], label='Val Acc', color='red')
    ax2.set_title('Training Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names=['Correct', 'Incorrect']):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()


def print_evaluation_report(accuracy, report):
    """Print evaluation metrics in a clean format"""
    print(f"\n{'='*50}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*50}")
    print(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"\nPer-Class Metrics:")
    print(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
    print(f"{'-'*55}")
    
    # Get all class labels from the report
    class_keys = [key for key in report.keys() if key.isdigit()]
    
    for class_key in sorted(class_keys):
        metrics = report[class_key]
        class_name = f"Class {class_key}"
        if class_key == '0':
            class_name = "Correct (0)"
        else:
            class_name = f"Incorrect ({class_key})"
            
        print(f"{class_name:<15} {metrics['precision']:<10.3f} "
              f"{metrics['recall']:<10.3f} {metrics['f1-score']:<10.3f}")
    
    # Print macro/weighted averages if available
    if 'macro avg' in report:
        print(f"{'-'*55}")
        macro = report['macro avg']
        print(f"{'Macro Avg':<15} {macro['precision']:<10.3f} "
              f"{macro['recall']:<10.3f} {macro['f1-score']:<10.3f}")


class PosePredictor:
    """Simple predictor for single sequences"""
    
    def __init__(self, model, device=None):
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
    
    def predict_sequence(self, keypoints_3d, timestamps, sequence_length=10):
        """
        Predict on a single sequence of keypoints
        
        Args:
            keypoints_3d: List of 3D keypoints for each frame [[x1,y1,z1], [x2,y2,z2], ...]
            timestamps: List of timestamps for each frame
            sequence_length: Length of sequence to use for prediction
            
        Returns:
            prediction: 0 for correct, 1 for incorrect
            confidence: Confidence score (0-1)
        """
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
    
    def predict_from_dict(self, data_dict):
        """
        Predict from data in the same format as training data
        
        Args:
            data_dict: Dictionary with 'keypoints_3d' and 'ts' lists
            
        Returns:
            prediction: 0 for correct, 1 for incorrect
            confidence: Confidence score (0-1)
        """
        keypoints_3d = data_dict['keypoints_3d']
        timestamps = data_dict['ts']
        
        return self.predict_sequence(keypoints_3d, timestamps)


def count_parameters(model):
    """Count trainable parameters in model"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    return total_params, trainable_params


def set_random_seeds(seed=42):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Make CuDNN deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False