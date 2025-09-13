"""
Main training script for pose validation TCN
Simple and clean training pipeline
"""

import torch
from model import PoseValidationTCN
from data_utils import create_data_loaders, get_input_size
from trainer import Trainer
from utils import plot_training_history, plot_confusion_matrix, print_evaluation_report, count_parameters, set_random_seeds


def main():
    """Main training function"""
    
    # Configuration
    CONFIG = {
        'data_path': '../Dataset/merged_data.json',
        'sequence_length': 9,
        'batch_size': 16,
        'num_epochs': 15,
        'learning_rate': 0.001,
        'hidden_channels': [64, 128],
        'kernel_size': 3,
        'dropout': 0.2,
        'random_seed': 42
    }
    
    # Set random seeds for reproducibility
    set_random_seeds(CONFIG['random_seed'])
    
    print("Loading data...")
    # Get input size from data
    input_size = get_input_size(CONFIG['data_path'], CONFIG['sequence_length'])
    if input_size is None:
        print("Error: Could not determine input size from data")
        return
    
    print(f"Input size: {input_size}")
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        CONFIG['data_path'],
        sequence_length=CONFIG['sequence_length'],
        batch_size=CONFIG['batch_size']
    )
    
    print(f"Dataset sizes:")
    print(f"  Train: {len(train_loader.dataset)}")
    print(f"  Validation: {len(val_loader.dataset)}")
    print(f"  Test: {len(test_loader.dataset)}")
    
    # Create model
    model = PoseValidationTCN(
        input_size=input_size,
        hidden_channels=CONFIG['hidden_channels'],
        kernel_size=CONFIG['kernel_size'],
        dropout=CONFIG['dropout'],
        num_classes=7  # 0=correct, 1-6=incorrect types
    )
    
    print(f"\nModel created:")
    count_parameters(model)
    
    # Create trainer
    trainer = Trainer(model)
    
    # Train model
    print(f"\nStarting training for {CONFIG['num_epochs']} epochs...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=CONFIG['num_epochs'],
        lr=CONFIG['learning_rate']
    )
    
    # Plot training history
    print("\nPlotting training history...")
    plot_training_history(history)
    
    # Evaluate on test set
    print("Evaluating on test set...")
    accuracy, report, predictions, targets = trainer.evaluate(test_loader)
    
    # Print results
    print_evaluation_report(accuracy, report)
    
    # Plot confusion matrix
    plot_confusion_matrix(targets, predictions)
    
    # Save model
    model_path = 'pose_validation_model.pth'
    trainer.save_model(model_path)
    
    print(f"\nTraining completed!")
    print(f"Final test accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    main()