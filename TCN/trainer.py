"""
Training utilities for pose validation TCN
Simple and focused training logic
"""

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report


class Trainer:
    """Simple trainer for pose validation model"""
    
    def __init__(self, model, device=None):
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = None
        self.scheduler = None
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
    
    def setup_optimizer(self, lr=0.001):
        """Setup optimizer and scheduler"""
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=10, factor=0.5
        )
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(self.device), targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def validate_epoch(self, val_loader):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def train(self, train_loader, val_loader, num_epochs=50, lr=0.001, verbose=True):
        """Full training loop"""
        self.setup_optimizer(lr)
        
        best_val_acc = 0
        best_model_state = None
        
        for epoch in range(num_epochs):
            # Train and validate
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate_epoch(val_loader)
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = self.model.state_dict().copy()
            
            # Print progress
            if verbose and (epoch % 10 == 0 or epoch == num_epochs - 1):
                print(f'Epoch {epoch+1}/{num_epochs}:')
                print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
                print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
                print(f'  Best Val Acc: {best_val_acc:.2f}%')
                print('-' * 50)
        
        # Load best model
        if best_model_state:
            self.model.load_state_dict(best_model_state)
        
        return self.history
    
    def evaluate(self, test_loader):
        """Evaluate model on test set"""
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)
                _, predicted = outputs.max(1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_targets, all_predictions)
        report = classification_report(
            all_targets, all_predictions, 
            target_names=['Correct', 'Incorrect'],
            output_dict=True
        )
        
        return accuracy, report, all_predictions, all_targets
    
    def save_model(self, filepath, include_config=True):
        """Save trained model"""
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'history': self.history
        }
        
        if include_config:
            # Try to extract model config
            try:
                save_dict['model_config'] = {
                    'input_size': self.model.tcn[0].conv1.in_channels,
                    'hidden_channels': [layer.conv1.out_channels for layer in self.model.tcn],
                    'kernel_size': self.model.tcn[0].conv1.kernel_size[0],
                }
            except:
                pass
        
        torch.save(save_dict, filepath)
        print(f"Model saved to {filepath}")
    
    @staticmethod
    def load_model(filepath, model_class):
        """Load a saved model"""
        checkpoint = torch.load(filepath, map_location='cpu')
        
        if 'model_config' in checkpoint:
            config = checkpoint['model_config']
            model = model_class(**config)
        else:
            # Need to provide config manually
            raise ValueError("Model config not found. Please provide model parameters.")
        
        model.load_state_dict(checkpoint['model_state_dict'])
        history = checkpoint.get('history', None)
        
        return model, history