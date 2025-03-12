#!/usr/bin/env python3
"""
Autoencoder Model for Anomaly Detection
Implements a convolutional autoencoder for unsupervised defect detection.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from pathlib import Path

from config import config

class ConvAutoencoder(nn.Module):
    """
    Convolutional Autoencoder for anomaly detection.
    Uses a symmetric encoder-decoder architecture with skip connections.
    """
    
    def __init__(self, latent_dim=32):
        """
        Initialize the autoencoder.
        
        Args:
            latent_dim: Dimension of the latent space
        """
        super(ConvAutoencoder, self).__init__()
        
        # Encoder layers
        self.encoder = nn.Sequential(
            # Layer 1: 3 -> 32 channels, 256x256 -> 128x128
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
            # Layer 2: 32 -> 64 channels, 128x128 -> 64x64
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            # Layer 3: 64 -> 128 channels, 64x64 -> 32x32
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            # Layer 4: 128 -> 256 channels, 32x32 -> 16x16
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            # Layer 5: 256 -> 512 channels, 16x16 -> 8x8
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 8 * 8, latent_dim),
            nn.ReLU(True),
            nn.Linear(latent_dim, 512 * 8 * 8),
            nn.ReLU(True)
        )
        
        # Decoder layers
        self.decoder = nn.Sequential(
            # Layer 1: 512 channels, 8x8 -> 16x16
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            # Layer 2: 256 channels, 16x16 -> 32x32
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            # Layer 3: 128 channels, 32x32 -> 64x64
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            # Layer 4: 64 channels, 64x64 -> 128x128
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
            # Layer 5: 32 channels, 128x128 -> 256x256
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  # Output in range [0, 1]
        )
        
        # Store latent dimension for reference
        self.latent_dim = latent_dim
    
    def forward(self, x):
        """
        Forward pass through the autoencoder.
        
        Args:
            x: Input tensor (batch_size, 3, 256, 256)
            
        Returns:
            Tuple of (reconstructed, latent)
        """
        # Encode
        encoded = self.encoder(x)
        
        # Bottleneck
        flattened = encoded.view(encoded.size(0), -1)
        bottleneck = self.bottleneck(flattened)
        latent = bottleneck[:, :self.latent_dim]  # Extract latent representation
        bottleneck_reshaped = bottleneck.view(encoded.size())
        
        # Decode
        reconstructed = self.decoder(bottleneck_reshaped)
        
        return reconstructed, latent

class NormalImageDataset(Dataset):
    """Dataset of normal (non-defective) images for training."""
    
    def __init__(self, root_dir, transform=None):
        """
        Initialize the dataset.
        
        Args:
            root_dir: Directory containing normal images
            transform: PyTorch transform pipeline
        """
        self.root_dir = root_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize(config.MODEL["IMAGE_SIZE"]),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=config.MODEL["NORMALIZE_MEAN"],
                std=config.MODEL["NORMALIZE_STD"]
            )
        ])
        
        # Get list of image files
        self.image_files = []
        for ext in ['.jpg', '.jpeg', '.png']:
            self.image_files.extend(list(Path(root_dir).glob(f'*{ext}')))
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image

def train_autoencoder(model, train_dir, val_dir=None, epochs=50, batch_size=16, 
                      learning_rate=0.001, device=None, save_path=None):
    """
    Train the autoencoder model.
    
    Args:
        model: Autoencoder model
        train_dir: Directory with training images
        val_dir: Directory with validation images (optional)
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        device: Device to use for training ('cuda', 'mps', or 'cpu')
        save_path: Path to save the trained model
        
    Returns:
        Trained model and dictionary of training history
    """
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else 
                             "mps" if torch.backends.mps.is_available() else 
                             "cpu")
    
    model = model.to(device)
    print(f"Training on device: {device}")
    
    # Create datasets and data loaders
    train_dataset = NormalImageDataset(train_dir)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    if val_dir:
        val_dataset = NormalImageDataset(val_dir)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    else:
        val_loader = None
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': []
    }
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch_idx, data in enumerate(train_loader):
            # Move data to device
            data = data.to(device)
            
            # Forward pass
            reconstructed, _ = model(data)
            loss = criterion(reconstructed, data)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Print progress
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(train_loader)}], "
                      f"Loss: {loss.item():.6f}")
        
        # Calculate average training loss
        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        
        # Validation
        if val_loader:
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for data in val_loader:
                    data = data.to(device)
                    reconstructed, _ = model(data)
                    loss = criterion(reconstructed, data)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            history['val_loss'].append(avg_val_loss)
            
            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.6f}, "
                  f"Val Loss: {avg_val_loss:.6f}")
        else:
            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.6f}")
    
    # Save the trained model
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")
    
    return model, history

def test_autoencoder(model, normal_dir, anomaly_dir, device=None, batch_size=16):
    """
    Test the autoencoder on normal and anomaly images.
    
    Args:
        model: Trained autoencoder model
        normal_dir: Directory with normal images
        anomaly_dir: Directory with anomaly images
        device: Device to use for inference
        batch_size: Batch size for testing
        
    Returns:
        Dictionary with test results
    """
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else 
                             "mps" if torch.backends.mps.is_available() else 
                             "cpu")
    
    model = model.to(device)
    model.eval()
    
    # Create datasets and data loaders
    normal_dataset = NormalImageDataset(normal_dir)
    normal_loader = DataLoader(normal_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    anomaly_dataset = NormalImageDataset(anomaly_dir)
    anomaly_loader = DataLoader(anomaly_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Define loss function
    criterion = nn.MSELoss(reduction='none')
    
    # Collect reconstruction errors
    normal_errors = []
    anomaly_errors = []
    
    # Process normal images
    with torch.no_grad():
        for data in normal_loader:
            data = data.to(device)
            reconstructed, _ = model(data)
            
            # Calculate per-sample MSE
            error = criterion(reconstructed, data).mean(dim=[1, 2, 3])
            normal_errors.extend(error.cpu().numpy())
    
    # Process anomaly images
    with torch.no_grad():
        for data in anomaly_loader:
            data = data.to(device)
            reconstructed, _ = model(data)
            
            # Calculate per-sample MSE
            error = criterion(reconstructed, data).mean(dim=[1, 2, 3])
            anomaly_errors.extend(error.cpu().numpy())
    
    # Calculate statistics
    normal_mean = np.mean(normal_errors)
    normal_std = np.std(normal_errors)
    anomaly_mean = np.mean(anomaly_errors)
    anomaly_std = np.std(anomaly_errors)
    
    # Calculate optimal threshold (midpoint between normal and anomaly means)
    optimal_threshold = (normal_mean + anomaly_mean) / 2
    
    # Calculate metrics at optimal threshold
    true_positives = sum(error > optimal_threshold for error in anomaly_errors)
    false_negatives = len(anomaly_errors) - true_positives
    true_negatives = sum(error <= optimal_threshold for error in normal_errors)
    false_positives = len(normal_errors) - true_negatives
    
    # Calculate precision, recall, and F1 score
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Return results
    results = {
        'normal_mean': normal_mean,
        'normal_std': normal_std,
        'anomaly_mean': anomaly_mean,
        'anomaly_std': anomaly_std,
        'optimal_threshold': optimal_threshold,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'true_negatives': true_negatives,
        'false_negatives': false_negatives,
        'normal_errors': normal_errors,
        'anomaly_errors': anomaly_errors
    }
    
    return results

def evaluate_on_image(model, image_path, threshold=None, device=None):
    """
    Evaluate the model on a single image.
    
    Args:
        model: Trained autoencoder model
        image_path: Path to the image to evaluate
        threshold: Reconstruction error threshold for anomaly detection
        device: Device to use for inference
        
    Returns:
        Dictionary with evaluation results
    """
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else 
                             "mps" if torch.backends.mps.is_available() else 
                             "cpu")
    
    model = model.to(device)
    model.eval()
    
    # Create transform
    transform = transforms.Compose([
        transforms.Resize(config.MODEL["IMAGE_SIZE"]),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=config.MODEL["NORMALIZE_MEAN"],
            std=config.MODEL["NORMALIZE_STD"]
        )
    ])
    
    # Load and transform image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Perform inference
    with torch.no_grad():
        reconstructed, latent = model(image_tensor)
        
        # Calculate reconstruction error
        criterion = nn.MSELoss()
        reconstruction_error = criterion(reconstructed, image_tensor).item()
        
        # Determine if anomaly
        is_anomaly = reconstruction_error > threshold if threshold is not None else None
        
        # Convert tensors for visualization
        input_np = image_tensor.cpu().squeeze(0).permute(1, 2, 0).numpy()
        input_np = np.clip(input_np * np.array(config.MODEL["NORMALIZE_STD"]) + 
                         np.array(config.MODEL["NORMALIZE_MEAN"]), 0, 1)
        
        recon_np = reconstructed.cpu().squeeze(0).permute(1, 2, 0).numpy()
        
        # Calculate residual (error) map
        residual = torch.abs(image_tensor - reconstructed)
        residual_np = residual.cpu().squeeze(0).permute(1, 2, 0).mean(axis=2).numpy()
        
        # Normalize residual map for visualization
        residual_np = (residual_np - residual_np.min()) / max(1e-8, residual_np.max() - residual_np.min())
    
    # Return results
    results = {
        'reconstruction_error': reconstruction_error,
        'is_anomaly': is_anomaly,
        'threshold': threshold,
        'input_image': input_np,
        'reconstructed_image': recon_np,
        'residual_map': residual_np,
        'latent_vector': latent.cpu().numpy()
    }
    
    return results

if __name__ == "__main__":
    # Example usage
    import matplotlib.pyplot as plt
    
    # Create model
    model = ConvAutoencoder(latent_dim=config.MODEL["LATENT_DIM"])
    
    # Check if model file exists
    model_path = config.PATHS["MODEL_FILE"]
    if os.path.exists(model_path):
        # Load pre-trained model
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        print(f"Loaded model from {model_path}")
    else:
        print(f"Model file not found at {model_path}")
        
        # Train model
        print("Would you like to train a new model? (y/n)")
        choice = input().lower()
        if choice == 'y' or choice == 'yes':
            normal_dir = config.PATHS["NORMAL_DIR"]
            if os.path.exists(normal_dir) and os.listdir(normal_dir):
                train_autoencoder(model, normal_dir, save_path=model_path)
            else:
                print(f"No training images found in {normal_dir}")
