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


def plot_multilabel_metrics(y_true, y_pred, y_prob=None):
    """Plot multi-label classification metrics including the 7th (implicit) class"""
    # num_classes should be 7 (6 explicit + 1 implicit)
    num_classes = y_true.shape[1]  # Should be 7
    class_names = [f"Class {i}" if i < 6 else "Class 6 (All Zero)" for i in range(num_classes)]
    
    # Create subplots for different visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Per-class accuracy
    accuracies = np.mean(y_true == y_pred, axis=0)
    axes[0, 0].bar(range(num_classes), accuracies)
    axes[0, 0].set_title('Per-Class Accuracy')
    axes[0, 0].set_xlabel('Class')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_xticks(range(num_classes))
    axes[0, 0].set_xticklabels(class_names, rotation=45)
    
    # 2. Label frequency
    true_freq = np.mean(y_true, axis=0)
    pred_freq = np.mean(y_pred, axis=0)
    x = np.arange(num_classes)
    width = 0.35
    axes[0, 1].bar(x - width/2, true_freq, width, label='True')
    axes[0, 1].bar(x + width/2, pred_freq, width, label='Predicted')
    axes[0, 1].set_title('Label Frequency')
    axes[0, 1].set_xlabel('Class')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    axes[0, 1].set_xticks(range(num_classes))
    axes[0, 1].set_xticklabels(class_names, rotation=45)
    
    # 3. Co-occurrence matrix for true labels
    cooc_true = y_true.T @ y_true
    sns.heatmap(cooc_true, annot=True, fmt='.2f', cmap='Blues', ax=axes[1, 0])
    axes[1, 0].set_title('True Label Co-occurrence')
    
    # 4. Co-occurrence matrix for predicted labels
    cooc_pred = y_pred.T @ y_pred
    sns.heatmap(cooc_pred, annot=True, fmt='.2f', cmap='Blues', ax=axes[1, 1])
    axes[1, 1].set_title('Predicted Label Co-occurrence')
    
    plt.tight_layout()
    plt.show()
    
    # If probabilities are provided, plot ROC curves
    if y_prob is not None:
        from sklearn.metrics import roc_curve, auc
        plt.figure(figsize=(10, 6))
        for i in range(num_classes):
            if np.any(y_true[:, i] > 0):  # Only plot if there are positive samples
                fpr, tpr, _ = roc_curve(y_true[:, i], y_prob[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend(loc="lower right")
        plt.show()


def print_evaluation_report(metrics):
    """Print multi-label evaluation metrics in a clean format"""
    print(f"\n{'='*60}")
    print(f"MULTI-LABEL EVALUATION RESULTS")
    print(f"{'='*60}")
    
    # Overall metrics
    print(f"Exact Match Accuracy: {metrics['exact_match_accuracy']:.4f} ({metrics['exact_match_accuracy']*100:.2f}%)")
    print(f"Average Precision: {metrics['average_precision']:.4f}")
    print(f"\nPer-Class Metrics:")
    print(f"{'Class':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
    print(f"{'-'*50}")
    
    for i in range(len(metrics['per_class_precision'])):
        print(f"{i:<10} {metrics['per_class_precision'][i]:<10.3f} "
              f"{metrics['per_class_recall'][i]:<10.3f} "
              f"{metrics['per_class_f1'][i]:<10.3f}")
    
    print(f"{'-'*50}")
    
    # Calculate and print averages
    avg_precision = np.mean(metrics['per_class_precision'])
    avg_recall = np.mean(metrics['per_class_recall'])
    avg_f1 = np.mean(metrics['per_class_f1'])
    
    print(f"{'Average':<10} {avg_precision:<10.3f} "
          f"{avg_recall:<10.3f} {avg_f1:<10.3f}")


class PosePredictor:
    """Simple predictor for single sequences"""
    
    def __init__(self, model, device=None):
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
    
    def predict_sequence(self, keypoints_3d, timestamps, sequence_length=30):
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
            probs = torch.sigmoid(logits)
            predictions = (probs > 0.5).float()
            
            # Convert to numpy for easier handling
            predictions = predictions.cpu().numpy()[0]
            probabilities = probs.cpu().numpy()[0]
        
        return predictions, probabilities
    
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