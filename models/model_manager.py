#!/usr/bin/env python3
"""
AI Model Management System for Defect Detection
Handles model loading, versioning, evaluation, and inference.
"""

import os
import json
import time
import datetime
import shutil
import hashlib
import threading
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union

# Third-party libraries with error handling
try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as transforms
    import numpy as np
    from PIL import Image
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("WARNING: PyTorch is required for model management functionality.")

# Import centralized configuration
from config import config

# Set up logger
logger = config.setup_logger("model_manager")

class ModelError(Exception):
    """Custom exception for model-related errors"""
    pass

class AutoencoderModel(nn.Module):
    """
    Autoencoder model for detecting anomalies in images.
    Uses reconstruction error to identify potential defects.
    """
    
    def __init__(self, latent_dim):
        """
        Initialize the autoencoder with encoder and decoder components.
        
        Args:
            latent_dim: Dimension of the latent space representation
        """
        super(AutoencoderModel, self).__init__()
        
        # Encoder with dropout to encourage general features
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 32 * 32, latent_dim)
        )
        
        # Decoder reconstructs the image
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128 * 32 * 32),
            nn.Unflatten(1, (128, 32, 32)),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Forward pass through the autoencoder.
        
        Args:
            x: Input tensor (image)
            
        Returns:
            reconstructed: Reconstructed image
            latent: Latent space representation
        """
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent

class ModelManager:
    """
    Manages AI models for defect detection, including loading, versioning,
    and evaluation.
    """
    
    def __init__(self, model_dir=None, cache_models=True):
        """
        Initialize the model manager.
        
        Args:
            model_dir: Directory for model storage (defaults to config)
            cache_models: Whether to cache loaded models in memory
        """
        self.model_dir = model_dir or os.path.join(config.BASE_DIR, "models")
        self.cache_models = cache_models
        self.model_cache = {}  # Cache for loaded models
        self.locks = {}  # Locks for thread-safe model loading
        self.default_model_path = config.PATHS.get("MODEL_FILE", "")
        
        # Device for computations
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Create model directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Track the active model
        self.active_model_path = None
        
        # Load model info
        self._load_model_info()
    
    def _load_model_info(self):
        """Load model information from the model directory"""
        self.models_info = {}
        info_file = os.path.join(self.model_dir, "models_info.json")
        
        if os.path.exists(info_file):
            try:
                with open(info_file, "r") as f:
                    self.models_info = json.load(f)
            except json.JSONDecodeError:
                logger.error(f"Error parsing models_info.json. Creating new file.")
                self._save_model_info()
        else:
            # Create empty model info file
            self._save_model_info()
    
    def _save_model_info(self):
        """Save model information to the model directory"""
        info_file = os.path.join(self.model_dir, "models_info.json")
        
        try:
            with open(info_file, "w") as f:
                json.dump(self.models_info, f, indent=4)
        except Exception as e:
            logger.error(f"Error saving model info: {e}")
    
    def _calculate_model_hash(self, model_path):
        """
        Calculate a hash for the model file.
        
        Args:
            model_path: Path to the model file
            
        Returns:
            Hash string
        """
        hash_md5 = hashlib.md5()
        
        with open(model_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
                
        return hash_md5.hexdigest()
    
    def load_model(self, model_path=None, force_reload=False):
        """
        Load a model from the specified path or the default model.
        
        Args:
            model_path: Path to the model file (defaults to active model)
            force_reload: Whether to reload the model even if it's cached
            
        Returns:
            Loaded model
        """
        if not HAS_TORCH:
            raise ModelError("PyTorch is required for model loading")
        
        # Use default model path if not specified
        if model_path is None:
            if self.active_model_path:
                model_path = self.active_model_path
            else:
                model_path = self.default_model_path
        
        # Check if model file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Create a lock for this model path if it doesn't exist
        if model_path not in self.locks:
            self.locks[model_path] = threading.RLock()
        
        # Use the lock for thread safety
        with self.locks[model_path]:
            # Check cache
            if not force_reload and self.cache_models and model_path in self.model_cache:
                logger.debug(f"Using cached model from {model_path}")
                return self.model_cache[model_path]
            
            try:
                # Load model architecture
                latent_dim = config.MODEL.get("LATENT_DIM", 32)
                model = AutoencoderModel(latent_dim).to(self.device)
                
                # Load weights
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                model.eval()  # Set model to evaluation mode
                
                # Update active model path
                self.active_model_path = model_path
                
                # Cache the model if enabled
                if self.cache_models:
                    self.model_cache[model_path] = model
                
                logger.info(f"Successfully loaded model from {model_path}")
                return model
            except Exception as e:
                logger.error(f"Error loading model from {model_path}: {e}")
                raise ModelError(f"Failed to load model: {e}")
    
    def register_model(self, model_path, name=None, description=None, metrics=None):
        """
        Register a model in the model directory with metadata.
        
        Args:
            model_path: Path to the model file
            name: Name for the model (defaults to filename)
            description: Optional description
            metrics: Optional dictionary of evaluation metrics
            
        Returns:
            Model ID (hash)
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Generate a name if not provided
        if name is None:
            name = os.path.basename(model_path)
        
        # Calculate model hash
        model_hash = self._calculate_model_hash(model_path)
        
        # Create destination path in model directory
        dest_dir = os.path.join(self.model_dir, model_hash)
        os.makedirs(dest_dir, exist_ok=True)
        
        # Copy model file
        dest_path = os.path.join(dest_dir, os.path.basename(model_path))
        if os.path.abspath(model_path) != os.path.abspath(dest_path):
            shutil.copy2(model_path, dest_path)
        
        # Create model info
        timestamp = datetime.datetime.now().isoformat()
        model_info = {
            "id": model_hash,
            "name": name,
            "description": description or "",
            "path": dest_path,
            "original_path": model_path,
            "registered_time": timestamp,
            "last_used": timestamp,
            "metrics": metrics or {}
        }
        
        # Save model info
        self.models_info[model_hash] = model_info
        self._save_model_info()
        
        logger.info(f"Registered model '{name}' with ID {model_hash}")
        return model_hash
    
    def set_active_model(self, model_id=None, model_path=None):
        """
        Set the active model for inference.
        
        Args:
            model_id: ID of the model in the registry
            model_path: Path to the model file (alternative to model_id)
            
        Returns:
            Boolean indicating success
        """
        # If model_id is provided, look up in registry
        if model_id is not None:
            if model_id not in self.models_info:
                logger.error(f"Model ID not found in registry: {model_id}")
                return False
                
            model_path = self.models_info[model_id]["path"]
        
        # If model_path is provided, use it directly
        if model_path is not None:
            if not os.path.exists(model_path):
                logger.error(f"Model file not found: {model_path}")
                return False
            
            # Try to load the model to verify it works
            try:
                self.load_model(model_path, force_reload=True)
                
                # Update last_used timestamp if it's a registered model
                for model_id, info in self.models_info.items():
                    if info["path"] == model_path:
                        info["last_used"] = datetime.datetime.now().isoformat()
                        self._save_model_info()
                        break
                
                return True
            except Exception as e:
                logger.error(f"Error setting active model: {e}")
                return False
        
        logger.error("Either model_id or model_path must be provided")
        return False
    
    def get_model_list(self):
        """
        Get a list of all registered models with their metadata.
        
        Returns:
            List of model info dictionaries
        """
        return list(self.models_info.values())
    
    def get_model_info(self, model_id):
        """
        Get metadata for a specific model.
        
        Args:
            model_id: ID of the model
            
        Returns:
            Model info dictionary or None if not found
        """
        return self.models_info.get(model_id)
    
    def evaluate_model(self, model_id=None, model_path=None, test_data_dir=None):
        """
        Evaluate a model on a test dataset and record metrics.
        
        Args:
            model_id: ID of the model in the registry
            model_path: Path to the model file (alternative to model_id)
            test_data_dir: Directory containing test images
            
        Returns:
            Dictionary of evaluation metrics
        """
        if not HAS_TORCH:
            raise ModelError("PyTorch is required for model evaluation")
        
        # Determine model path
        if model_id is not None:
            if model_id not in self.models_info:
                raise ModelError(f"Model ID not found in registry: {model_id}")
            model_path = self.models_info[model_id]["path"]
        elif model_path is None:
            raise ModelError("Either model_id or model_path must be provided")
        
        # Default test data directory
        if test_data_dir is None:
            test_data_dir = os.path.join(config.BASE_DIR, "test_data")
        
        if not os.path.exists(test_data_dir):
            raise FileNotFoundError(f"Test data directory not found: {test_data_dir}")
        
        # Load the model
        model = self.load_model(model_path)
        
        # Create transforms for preprocessing
        transform = transforms.Compose([
            transforms.Resize(config.MODEL["IMAGE_SIZE"]),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=config.MODEL["NORMALIZE_MEAN"],
                std=config.MODEL["NORMALIZE_STD"]
            )
        ])
        
        # Setup for evaluation
        criterion = nn.MSELoss()
        total_error = 0.0
        normal_error = 0.0
        anomaly_error = 0.0
        normal_count = 0
        anomaly_count = 0
        
        # Process normal images
        normal_dir = os.path.join(test_data_dir, "normal")
        if os.path.exists(normal_dir):
            normal_files = [os.path.join(normal_dir, f) for f in os.listdir(normal_dir)
                           if f.endswith(('.jpg', '.jpeg', '.png'))]
            
            for file_path in normal_files:
                try:
                    # Preprocess image
                    img = Image.open(file_path).convert("RGB")
                    img_tensor = transform(img).unsqueeze(0).to(self.device)
                    
                    # Get reconstruction
                    with torch.no_grad():
                        reconstructed, _ = model(img_tensor)
                        error = criterion(reconstructed, img_tensor).item()
                    
                    total_error += error
                    normal_error += error
                    normal_count += 1
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")
        
        # Process anomaly images
        anomaly_dir = os.path.join(test_data_dir, "anomaly")
        if os.path.exists(anomaly_dir):
            anomaly_files = [os.path.join(anomaly_dir, f) for f in os.listdir(anomaly_dir)
                            if f.endswith(('.jpg', '.jpeg', '.png'))]
            
            for file_path in anomaly_files:
                try:
                    # Preprocess image
                    img = Image.open(file_path).convert("RGB")
                    img_tensor = transform(img).unsqueeze(0).to(self.device)
                    
                    # Get reconstruction
                    with torch.no_grad():
                        reconstructed, _ = model(img_tensor)
                        error = criterion(reconstructed, img_tensor).item()
                    
                    total_error += error
                    anomaly_error += error
                    anomaly_count += 1
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")
        
        # Calculate metrics
        total_count = normal_count + anomaly_count
        avg_error = total_error / total_count if total_count > 0 else 0
        avg_normal_error = normal_error / normal_count if normal_count > 0 else 0
        avg_anomaly_error = anomaly_error / anomaly_count if anomaly_count > 0 else 0
        
        # Calculate separation (difference between normal and anomaly errors)
        error_separation = avg_anomaly_error - avg_normal_error
        
        # Calculate optimal threshold (midpoint between normal and anomaly errors)
        optimal_threshold = (avg_normal_error + avg_anomaly_error) / 2 if normal_count > 0 and anomaly_count > 0 else 0
        
        # Prepare metrics dictionary
        metrics = {
            "avg_error": avg_error,
            "avg_normal_error": avg_normal_error,
            "avg_anomaly_error": avg_anomaly_error,
            "error_separation": error_separation,
            "optimal_threshold": optimal_threshold,
            "normal_count": normal_count,
            "anomaly_count": anomaly_count,
            "total_count": total_count,
            "evaluation_time": datetime.datetime.now().isoformat()
        }
        
        # Update model info if it's a registered model
        if model_id is not None:
            self.models_info[model_id]["metrics"] = metrics
            self.models_info[model_id]["last_evaluated"] = datetime.datetime.now().isoformat()
            self._save_model_info()
        
        logger.info(f"Model evaluation complete. Metrics: {metrics}")
        return metrics
    
    def compare_models(self, model_ids=None):
        """
        Compare multiple models based on their evaluation metrics.
        
        Args:
            model_ids: List of model IDs to compare (defaults to all models with metrics)
            
        Returns:
            Dictionary with comparison results
        """
        # If no model IDs are provided, use all models with metrics
        if model_ids is None:
            model_ids = [model_id for model_id, info in self.models_info.items() 
                        if "metrics" in info and info["metrics"]]
        
        # Check if we have models to compare
        if not model_ids:
            logger.warning("No models with metrics available for comparison")
            return {"models": [], "best_model": None, "comparison_metrics": {}}
        
        # Collect models with their metrics
        models_with_metrics = []
        for model_id in model_ids:
            if model_id in self.models_info and "metrics" in self.models_info[model_id]:
                model_info = self.models_info[model_id].copy()
                models_with_metrics.append(model_info)
        
        # Sort models by different metrics
        by_error_separation = sorted(models_with_metrics, 
                                   key=lambda x: x["metrics"].get("error_separation", 0), 
                                   reverse=True)
        
        by_normal_error = sorted(models_with_metrics, 
                                key=lambda x: x["metrics"].get("avg_normal_error", float('inf')))
        
        by_anomaly_error = sorted(models_with_metrics, 
                                 key=lambda x: x["metrics"].get("avg_anomaly_error", 0), 
                                 reverse=True)
        
        # Determine best overall model (highest error separation is best)
        best_model = by_error_separation[0] if by_error_separation else None
        
        # Create comparison result
        comparison = {
            "models": models_with_metrics,
            "best_model": best_model,
            "comparison_metrics": {
                "by_error_separation": [m["id"] for m in by_error_separation],
                "by_normal_error": [m["id"] for m in by_normal_error],
                "by_anomaly_error": [m["id"] for m in by_anomaly_error]
            }
        }
        
        return comparison
    
    def get_recommended_threshold(self, model_id=None):
        """
        Get the recommended threshold value for a model based on its evaluation metrics.
        
        Args:
            model_id: ID of the model (defaults to active model)
            
        Returns:
            Recommended threshold value or None if not available
        """
        # If no model_id is provided, use the active model if available
        if model_id is None:
            # Find the model_id corresponding to the active model path
            for mid, info in self.models_info.items():
                if info["path"] == self.active_model_path:
                    model_id = mid
                    break
        
        # Check if model exists and has metrics
        if model_id and model_id in self.models_info and "metrics" in self.models_info[model_id]:
            metrics = self.models_info[model_id]["metrics"]
            
            # Use optimal_threshold if available
            if "optimal_threshold" in metrics:
                return metrics["optimal_threshold"]
            
            # Otherwise calculate from normal and anomaly errors
            if "avg_normal_error" in metrics and "avg_anomaly_error" in metrics:
                return (metrics["avg_normal_error"] + metrics["avg_anomaly_error"]) / 2
        
        return None
    
    def backup_model(self, model_id, backup_dir=None):
        """
        Create a backup of a registered model.
        
        Args:
            model_id: ID of the model to backup
            backup_dir: Directory to store the backup (defaults to BACKUP_DIR in config)
            
        Returns:
            Path to the backup file
        """
        if model_id not in self.models_info:
            raise ModelError(f"Model ID not found in registry: {model_id}")
        
        model_info = self.models_info[model_id]
        model_path = model_info["path"]
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Default backup directory
        if backup_dir is None:
            backup_dir = config.PATHS.get("BACKUP_DIR", os.path.join(config.BASE_DIR, "backups"))
        
        # Create backup directory if it doesn't exist
        os.makedirs(backup_dir, exist_ok=True)
        
        # Create a timestamped backup filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = os.path.basename(model_path)
        backup_filename = f"{timestamp}_{model_id[:8]}_{model_name}"
        backup_path = os.path.join(backup_dir, backup_filename)
        
        # Copy the model file
        shutil.copy2(model_path, backup_path)
        
        # Save model info with backup
        backup_info_path = backup_path + ".json"
        with open(backup_info_path, "w") as f:
            json.dump(model_info, f, indent=4)
        
        logger.info(f"Created backup of model {model_id} at {backup_path}")
        return backup_path
    
    def delete_model(self, model_id, backup=True):
        """
        Delete a model from the registry.
        
        Args:
            model_id: ID of the model to delete
            backup: Whether to create a backup before deleting
            
        Returns:
            Boolean indicating success
        """
        if model_id not in self.models_info:
            logger.error(f"Model ID not found in registry: {model_id}")
            return False
        
        model_info = self.models_info[model_id]
        model_path = model_info["path"]
        
        # Create backup if requested
        if backup:
            try:
                self.backup_model(model_id)
            except Exception as e:
                logger.error(f"Error creating backup before delete: {e}")
        
        # Remove from cache if it exists
        if model_path in self.model_cache:
            del self.model_cache[model_path]
        
        # Remove from registry
        del self.models_info[model_id]
        self._save_model_info()
        
        # Delete model directory
        model_dir = os.path.dirname(model_path)
        try:
            shutil.rmtree(model_dir)
            logger.info(f"Deleted model {model_id} from registry")
            return True
        except Exception as e:
            logger.error(f"Error deleting model directory: {e}")
            return False
    
    def infer(self, image_path, model_id=None, model_path=None, threshold=None):
        """
        Perform inference on an image using the specified model.
        
        Args:
            image_path: Path to the image to analyze
            model_id: ID of the model to use (defaults to active model)
            model_path: Path to the model file (alternative to model_id)
            threshold: Threshold for defect detection (defaults to model's recommended threshold)
            
        Returns:
            Dictionary with inference results
        """
        if not HAS_TORCH:
            raise ModelError("PyTorch is required for model inference")
        
        # Determine model to use
        if model_id is not None:
            if model_id not in self.models_info:
                raise ModelError(f"Model ID not found in registry: {model_id}")
            model_path = self.models_info[model_id]["path"]
        elif model_path is None:
            # Use active model
            if self.active_model_path is None:
                raise ModelError("No active model set and no model specified")
            model_path = self.active_model_path
        
        # Load the model
        model = self.load_model(model_path)
        
        # Set threshold
        if threshold is None:
            # Use model's recommended threshold if available
            if model_id is not None:
                threshold = self.get_recommended_threshold(model_id)
            
            # Fall back to default threshold from config
            if threshold is None:
                threshold = config.DEFAULT_SETTINGS.get("threshold", 0.85)
        
        # Create transform for preprocessing
        transform = transforms.Compose([
            transforms.Resize(config.MODEL["IMAGE_SIZE"]),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=config.MODEL["NORMALIZE_MEAN"],
                std=config.MODEL["NORMALIZE_STD"]
            )
        ])
        
        try:
            # Load and preprocess the image
            img = Image.open(image_path).convert("RGB")
            img_tensor = transform(img).unsqueeze(0).to(self.device)
            
            # Get reconstruction
            with torch.no_grad():
                start_time = time.time()
                reconstructed, latent = model(img_tensor)
                inference_time = time.time() - start_time
                
                # Calculate global reconstruction error
                criterion = nn.MSELoss()
                global_error = criterion(reconstructed, img_tensor).item()
                
                # Determine if it's a defect
                is_defect = global_error > threshold
                
                # Create result dictionary
                result = {
                    "global_error": global_error,
                    "threshold": threshold,
                    "is_defect": is_defect,
                    "inference_time": inference_time,
                    "latent_features": latent.cpu().numpy().tolist() if latent.numel() < 100 else None,
                    "model_id": model_id,
                    "model_path": model_path
                }
                
                return result
        except Exception as e:
            logger.error(f"Error performing inference on {image_path}: {e}")
            raise ModelError(f"Inference failed: {e}")
    
    def batch_infer(self, image_paths, model_id=None, model_path=None, threshold=None, batch_size=8):
        """
        Perform batch inference on multiple images.
        
        Args:
            image_paths: List of image paths to analyze
            model_id: ID of the model to use
            model_path: Path to the model file (alternative to model_id)
            threshold: Threshold for defect detection
            batch_size: Number of images to process in each batch
            
        Returns:
            List of inference result dictionaries
        """
        if not HAS_TORCH:
            raise ModelError("PyTorch is required for model inference")
        
        # Determine model to use
        if model_id is not None:
            if model_id not in self.models_info:
                raise ModelError(f"Model ID not found in registry: {model_id}")
            model_path = self.models_info[model_id]["path"]
        elif model_path is None:
            # Use active model
            if self.active_model_path is None:
                raise ModelError("No active model set and no model specified")
            model_path = self.active_model_path
        
        # Load the model
        model = self.load_model(model_path)
        
        # Set threshold
        if threshold is None:
            # Use model's recommended threshold if available
            if model_id is not None:
                threshold = self.get_recommended_threshold(model_id)
            
            # Fall back to default threshold from config
            if threshold is None:
                threshold = config.DEFAULT_SETTINGS.get("threshold", 0.85)
        
        # Create transform for preprocessing
        transform = transforms.Compose([
            transforms.Resize(config.MODEL["IMAGE_SIZE"]),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=config.MODEL["NORMALIZE_MEAN"],
                std=config.MODEL["NORMALIZE_STD"]
            )
        ])
        
        # Process images in batches
        results = []
        criterion = nn.MSELoss(reduction='none')
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_tensors = []
            valid_indices = []
            
            # Load and preprocess images
            for j, path in enumerate(batch_paths):
                try:
                    img = Image.open(path).convert("RGB")
                    img_tensor = transform(img)
                    batch_tensors.append(img_tensor)
                    valid_indices.append(j)
                except Exception as e:
                    logger.error(f"Error loading image {path}: {e}")
                    results.append({
                        "error": str(e),
                        "is_defect": False,
                        "image_path": path
                    })
            
            if not batch_tensors:
                continue
            
            # Convert to batch tensor
            batch_tensor = torch.stack(batch_tensors).to(self.device)
            
            # Get reconstruction
            with torch.no_grad():
                start_time = time.time()
                reconstructed, latent = model(batch_tensor)
                inference_time = time.time() - start_time
                
                # Calculate reconstruction errors
                errors = criterion(reconstructed, batch_tensor).mean(dim=[1, 2, 3])
                
                # Process results
                for j, idx in enumerate(valid_indices):
                    error = errors[j].item()
                    is_defect = error > threshold
                    
                    result = {
                        "global_error": error,
                        "threshold": threshold,
                        "is_defect": is_defect,
                        "inference_time": inference_time / len(batch_tensors),
                        "image_path": batch_paths[idx],
                        "model_id": model_id,
                        "model_path": model_path
                    }
                    
                    results.append(result)
        
        return results
    
    def generate_report(self, model_id=None):
        """
        Generate a report about a model's performance and characteristics.
        
        Args:
            model_id: ID of the model (defaults to all models if None)
            
        Returns:
            Report as a formatted string
        """
        if model_id is not None:
            # Report for a specific model
            if model_id not in self.models_info:
                return f"Model ID {model_id} not found in registry"
            
            model_info = self.models_info[model_id]
            
            # Build report
            report = []
            report.append(f"Model Report: {model_info['name']}")
            report.append("=" * 50)
            report.append(f"ID: {model_id}")
            report.append(f"Description: {model_info.get('description', 'N/A')}")
            report.append(f"Path: {model_info['path']}")
            report.append(f"Registered: {model_info.get('registered_time', 'N/A')}")
            report.append(f"Last Used: {model_info.get('last_used', 'N/A')}")
            
            # Add metrics if available
            if "metrics" in model_info and model_info["metrics"]:
                report.append("\nEvaluation Metrics:")
                report.append("-" * 50)
                metrics = model_info["metrics"]
                report.append(f"Average Error: {metrics.get('avg_error', 'N/A'):.6f}")
                report.append(f"Normal Error: {metrics.get('avg_normal_error', 'N/A'):.6f}")
                report.append(f"Anomaly Error: {metrics.get('avg_anomaly_error', 'N/A'):.6f}")
                report.append(f"Error Separation: {metrics.get('error_separation', 'N/A'):.6f}")
                report.append(f"Optimal Threshold: {metrics.get('optimal_threshold', 'N/A'):.6f}")
                report.append(f"Test Images: {metrics.get('total_count', 'N/A')} (Normal: {metrics.get('normal_count', 'N/A')}, Anomaly: {metrics.get('anomaly_count', 'N/A')})")
                report.append(f"Evaluation Time: {metrics.get('evaluation_time', 'N/A')}")
            else:
                report.append("\nNo evaluation metrics available")
            
            return "\n".join(report)
        else:
            # Report for all models
            if not self.models_info:
                return "No models registered in the system"
            
            report = []
            report.append("Model Registry Summary")
            report.append("=" * 50)
            report.append(f"Total Models: {len(self.models_info)}")
            
            # Sort models by last used time
            sorted_models = sorted(
                self.models_info.values(),
                key=lambda x: x.get("last_used", ""),
                reverse=True
            )
            
            # List models
            report.append("\nRegistered Models:")
            report.append("-" * 50)
            for model in sorted_models:
                model_id = model["id"]
                name = model["name"]
                last_used = model.get("last_used", "Never").split("T")[0]  # Just date part
                has_metrics = "Yes" if "metrics" in model and model["metrics"] else "No"
                
                report.append(f"ID: {model_id[:8]}... | Name: {name} | Last Used: {last_used} | Metrics: {has_metrics}")
            
            # Add active model info
            if self.active_model_path:
                active_id = "Unknown"
                for model_id, info in self.models_info.items():
                    if info["path"] == self.active_model_path:
                        active_id = model_id
                        break
                
                report.append("\nActive Model:")
                report.append("-" * 50)
                report.append(f"ID: {active_id}")
                report.append(f"Path: {self.active_model_path}")
            else:
                report.append("\nNo active model set")
            
            return "\n".join(report)

# For testing
if __name__ == "__main__":
    print("Testing Model Manager")
    
    # Check if PyTorch is available
    if not HAS_TORCH:
        print("WARNING: PyTorch not available. Skipping tests.")
        sys.exit(0)
    
    # Create model manager
    model_manager = ModelManager()
    
    # Register the default model if it exists
    if os.path.exists(config.PATHS.get("MODEL_FILE", "")):
        model_path = config.PATHS["MODEL_FILE"]
        print(f"\nRegistering default model from {model_path}")
        model_id = model_manager.register_model(
            model_path=model_path,
            name="Default Autoencoder",
            description="Default model for defect detection"
        )
        print(f"Registered model with ID: {model_id}")
        
        # Set as active model
        model_manager.set_active_model(model_id=model_id)
        print(f"Set active model to: {model_id}")
    
    # List registered models
    print("\nRegistered models:")
    for model in model_manager.get_model_list():
        print(f"  - {model['name']} (ID: {model['id'][:8]}...)")
    
    # Generate report
    print("\nModel report:")
    report = model_manager.generate_report()
    print(report)
    
    print("\nModel Manager tests completed.")