#!/usr/bin/env python3
"""
Model Training Script for Defect Detection System
Handles the training and evaluation of autoencoder models.
"""

import os
import sys
import time
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import our modules
from config import config
from models.autoencoder import ConvAutoencoder, train_autoencoder, test_autoencoder
from utils.logger import setup_logger, log_exception

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Autoencoder Model for Defect Detection")
    parser.add_argument("--train-dir", type=str, help="Directory with normal training images")
    parser.add_argument("--val-dir", type=str, help="Directory with normal validation images")
    parser.add_argument("--test-normal-dir", type=str, help="Directory with normal test images")
    parser.add_argument("--test-anomaly-dir", type=str, help="Directory with anomaly test images")
    parser.add_argument("--output-model", type=str, help="Path to save trained model")
    parser.add_argument("--latent-dim", type=int, default=32, help="Dimension of latent space")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--evaluate-only", action="store_true", help="Only evaluate existing model")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser.parse_args()

def plot_loss_curves(history, output_path=None):
    """
    Plot training and validation loss curves.
    
    Args:
        history: Dictionary with training history
        output_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Training Loss')
    
    if 'val_loss' in history and history['val_loss']:
        plt.plot(history['val_loss'], label='Validation Loss')
    
    plt.title('Autoencoder Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    if output_path:
        plt.savefig(output_path)
        print(f"Loss curves saved to {output_path}")
    else:
        plt.show()

def plot_error_distributions(results, output_path=None):
    """
    Plot error distributions for normal and anomaly images.
    
    Args:
        results: Dictionary with test results
        output_path: Path to save the plot
    """
    plt.figure(figsize=(12, 6))
    
    # Create histograms
    plt.hist(results['normal_errors'], bins=50, alpha=0.5, label='Normal', color='green')
    plt.hist(results['anomaly_errors'], bins=50, alpha=0.5, label='Anomaly', color='red')
    
    # Add vertical line for threshold
    plt.axvline(x=results['optimal_threshold'], color='black', linestyle='--', 
               label=f"Threshold: {results['optimal_threshold']:.6f}")
    
    # Add text with metrics
    metrics_text = (
        f"Precision: {results['precision']:.4f}\n"
        f"Recall: {results['recall']:.4f}\n"
        f"F1 Score: {results['f1_score']:.4f}\n"
        f"Normal μ: {results['normal_mean']:.6f}, σ: {results['normal_std']:.6f}\n"
        f"Anomaly μ: {results['anomaly_mean']:.6f}, σ: {results['anomaly_std']:.6f}"
    )
    
    plt.annotate(metrics_text, xy=(0.02, 0.95), xycoords='axes fraction', 
                fontsize=10, bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8))
    
    plt.title('Reconstruction Error Distributions')
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    
    if output_path:
        plt.savefig(output_path)
        print(f"Error distributions saved to {output_path}")
    else:
        plt.show()

def main():
    """Main function for model training and evaluation."""
    args = parse_arguments()
    
    # Set up logging
    log_level = "DEBUG" if args.debug else "INFO"
    logger = setup_logger("train_model", level=log_level)
    logger.info("Starting model training script")
    
    # Resolve directories and files
    train_dir = args.train_dir or config.PATHS["NORMAL_DIR"]
    val_dir = args.val_dir
    test_normal_dir = args.test_normal_dir or config.PATHS["NORMAL_DIR"]
    test_anomaly_dir = args.test_anomaly_dir or config.PATHS["ANOMALY_DIR"]
    output_model = args.output_model or config.PATHS["MODEL_FILE"]
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_model), exist_ok=True)
    
    # Create visualization directory
    viz_dir = os.path.join(config.BASE_DIR, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    # Check for CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create model
    latent_dim = args.latent_dim or config.MODEL.get("LATENT_DIM", 32)
    logger.info(f"Creating autoencoder with latent dimension {latent_dim}")
    model = ConvAutoencoder(latent_dim=latent_dim)
    
    # Check directories and files
    if not args.evaluate_only:
        if not os.path.exists(train_dir):
            logger.error(f"Training directory not found: {train_dir}")
            return 1
        
        if val_dir and not os.path.exists(val_dir):
            logger.error(f"Validation directory not found: {val_dir}")
            return 1
    
    if not os.path.exists(test_normal_dir):
        logger.error(f"Test normal directory not found: {test_normal_dir}")
        return 1
    
    if not os.path.exists(test_anomaly_dir):
        logger.error(f"Test anomaly directory not found: {test_anomaly_dir}")
        return 1
    
    # Training or loading the model
    if not args.evaluate_only:
        try:
            logger.info(f"Starting training with {args.epochs} epochs")
            logger.info(f"Training directory: {train_dir}")
            if val_dir:
                logger.info(f"Validation directory: {val_dir}")
            
            # Train the model
            start_time = time.time()
            model, history = train_autoencoder(
                model=model,
                train_dir=train_dir,
                val_dir=val_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                device=device,
                save_path=output_model
            )
            
            training_time = time.time() - start_time
            logger.info(f"Training completed in {training_time:.2f} seconds")
            
            # Plot and save loss curves
            loss_plot_path = os.path.join(viz_dir, "training_loss.png")
            plot_loss_curves(history, loss_plot_path)
        
        except Exception as e:
            log_exception(logger, f"Error during training: {e}")
            return 1
    else:
        # Load existing model
        if not os.path.exists(output_model):
            logger.error(f"Model file not found: {output_model}")
            return 1
        
        try:
            logger.info(f"Loading model from {output_model}")
            model.load_state_dict(torch.load(output_model, map_location=device))
        except Exception as e:
            log_exception(logger, f"Error loading model: {e}")
            return 1
    
    # Evaluate the model
    try:
        logger.info("Starting model evaluation")
        logger.info(f"Test normal directory: {test_normal_dir}")
        logger.info(f"Test anomaly directory: {test_anomaly_dir}")
        
        # Test the model
        results = test_autoencoder(
            model=model,
            normal_dir=test_normal_dir,
            anomaly_dir=test_anomaly_dir,
            device=device,
            batch_size=args.batch_size
        )
        
        # Print results
        logger.info("Evaluation Results:")
        logger.info(f"Normal images mean error: {results['normal_mean']:.6f}")
        logger.info(f"Normal images std dev: {results['normal_std']:.6f}")
        logger.info(f"Anomaly images mean error: {results['anomaly_mean']:.6f}")
        logger.info(f"Anomaly images std dev: {results['anomaly_std']:.6f}")
        logger.info(f"Optimal threshold: {results['optimal_threshold']:.6f}")
        logger.info(f"Precision: {results['precision']:.4f}")
        logger.info(f"Recall: {results['recall']:.4f}")
        logger.info(f"F1 Score: {results['f1_score']:.4f}")
        
        # Plot and save error distributions
        error_plot_path = os.path.join(viz_dir, "error_distributions.png")
        plot_error_distributions(results, error_plot_path)
        
        # Save optimal threshold to a file
        threshold_file = os.path.join(os.path.dirname(output_model), "optimal_threshold.txt")
        with open(threshold_file, "w") as f:
            f.write(f"{results['optimal_threshold']:.6f}")
        logger.info(f"Saved optimal threshold to {threshold_file}")
        
        # Suggest threshold to use in settings
        logger.info("\nRecommended settings:")
        logger.info(f"Global threshold: {results['optimal_threshold']:.4f}")
        logger.info(f"Patch threshold: {results['optimal_threshold'] * 0.9:.4f}")
        logger.info(f"Patch defect ratio: 0.45")
        
    except Exception as e:
        log_exception(logger, f"Error during evaluation: {e}")
        return 1
    
    logger.info("Model training and evaluation completed successfully")
    return 0

if __name__ == "__main__":
    sys.exit(main())
