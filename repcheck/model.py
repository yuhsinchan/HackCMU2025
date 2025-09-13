"""
TCN Model for Pose Validation
Simple and focused model definition
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalBlock(nn.Module):
    """Single temporal block with dilated convolutions"""
    
    def __init__(self, n_inputs, n_outputs, kernel_size, dilation, dropout=0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, padding=padding, dilation=dilation)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
        # Residual connection
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        
        # Remove future information (causal)
        self.chomp_size = padding
        
    def forward(self, x):
        # First convolution
        out = self.conv1(x)
        if self.chomp_size > 0:
            out = out[:, :, :-self.chomp_size]
        out = self.relu(out)
        out = self.dropout(out)
        
        # Second convolution
        out = self.conv2(out)
        if self.chomp_size > 0:
            out = out[:, :, :-self.chomp_size]
        out = self.relu(out)
        out = self.dropout(out)
        
        # Residual connection
        res = x if self.downsample is None else self.downsample(x)
        if res.size(2) != out.size(2):
            res = res[:, :, :out.size(2)]
            
        return self.relu(out + res)


class PoseValidationTCN(nn.Module):
    """Complete TCN model for pose validation"""
    
    def __init__(self, input_size, hidden_channels=[64, 128], kernel_size=3, dropout=0.2, num_classes=6):
        super().__init__()
        
        layers = []
        num_channels = [input_size] + hidden_channels
        
        for i in range(len(hidden_channels)):
            dilation = 2 ** i
            layers.append(TemporalBlock(
                num_channels[i], 
                num_channels[i+1], 
                kernel_size, 
                dilation, 
                dropout
            ))
        
        self.tcn = nn.Sequential(*layers)
        self.classifier = nn.Linear(hidden_channels[-1], num_classes)  # Multi-class classification
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: (batch_size, input_size, sequence_length)
        tcn_out = self.tcn(x)  # (batch_size, hidden_channels[-1], sequence_length)
        
        # Global average pooling over time
        pooled = F.adaptive_avg_pool1d(tcn_out, 1).squeeze(2)  # (batch_size, hidden_channels[-1])
        
        # Classification
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)  # (batch_size, 2)
        
        return logits
    
    def predict(self, x):
        """Get predictions and confidence scores for multi-label classification
        
        Returns:
            predictions: tensor of shape (batch_size, 7) where the last dimension
                       represents the 7th class (when all others are 0)
            confidence: tensor of shape (batch_size, 7) with confidence scores
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits)
            
            # Get initial predictions for first N-1 classes (where N is total number of classes)
            num_base_classes = probs.shape[1]  # This will be 7 now
            predictions_base = (probs > 0.5).float()
            
            # Create tensors for predictions and confidence
            predictions = predictions_base.clone()
            confidence = probs.clone()
            
            # For the last class: if all other probabilities are < 0.5
            all_zeros = (predictions_base[:, :-1].sum(dim=1) == 0)
            predictions[all_zeros, -1] = 1.0
            
            # Confidence for the last class: 1 - max probability of other classes
            confidence[:, -1] = 1 - probs[:, :-1].max(dim=1)[0]
            
        return predictions, confidence