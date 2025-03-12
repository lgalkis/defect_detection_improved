#!/usr/bin/env python3
"""
Image Processing Utilities for Defect Detection System
Provides optimized functions for image manipulation, processing, and analysis.
"""

import os
import io
import shutil
import datetime
import multiprocessing
from pathlib import Path

# Third-party libraries with error handling
try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as transforms
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("WARNING: Some image processing features require PyTorch, torchvision, and matplotlib.")

# Import centralized configuration
from config import config

# Set up logger
logger = config.setup_logger("image_utils")

class ImageUtilsError(Exception):
    """Custom exception for image utilities errors"""
    pass

def optimize_image_processing():
    """
    Configure and optimize image processing based on available hardware.
    Returns a dict with optimization info.
    """
    optimization_info = {
        "device": "cpu",
        "num_workers": 1,
        "torch_available": HAS_TORCH,
        "gpu_available": False,
        "mps_available": False  # Apple Metal Performance Shaders
    }
    
    # Skip optimizations if PyTorch is not available
    if not HAS_TORCH:
        logger.warning("PyTorch not available. Using basic image processing only.")
        return optimization_info
    
    # Determine device for computations
    if torch.cuda.is_available():
        optimization_info["device"] = "cuda"
        optimization_info["gpu_available"] = True
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
        optimization_info["device"] = "mps"
        optimization_info["mps_available"] = True
        logger.info("Using Apple MPS (Metal Performance Shaders)")
    else:
        logger.info("Using CPU for image processing")
    
    # Set optimal number of workers for data loading
    optimization_info["num_workers"] = min(4, multiprocessing.cpu_count())
    
    # Set PyTorch to use the selected device
    device = torch.device(optimization_info["device"])
    torch.set_default_device(device) if hasattr(torch, 'set_default_device') else None
    
    return optimization_info

def create_image_transforms(size=None):
    """
    Create standard image transforms for preprocessing.
    
    Args:
        size: Optional tuple of (width, height) for resizing
        
    Returns:
        Dictionary of transform objects
    """
    if not HAS_TORCH:
        logger.error("Cannot create transforms: PyTorch not available")
        return None
        
    # Use model settings from config
    size = size or config.MODEL["IMAGE_SIZE"]
    mean = config.MODEL["NORMALIZE_MEAN"]
    std = config.MODEL["NORMALIZE_STD"]
    
    # Create transform pipelines
    transforms_dict = {
        # Standard preprocessing for model input
        "model_input": transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ]),
        
        # Transform for visualization (no normalization)
        "visualization": transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor()
        ]),
        
        # Simple resize for storage
        "resize_only": transforms.Compose([
            transforms.Resize(size)
        ])
    }
    
    return transforms_dict

def get_optimal_image_format(image_type="good"):
    """
    Get optimal image format and quality settings based on image type.
    
    Args:
        image_type: String indicator of image type ("good" or "bad")
        
    Returns:
        Dict with format settings
    """
    if image_type.lower() == "good":
        # Good images: smaller size, lower quality
        return {
            "format": "JPEG",
            "quality": 85,
            "optimize": True
        }
    else:
        # Bad/defect images: higher quality for analysis
        return {
            "format": "JPEG",
            "quality": 92,
            "optimize": True
        }

def resize_image(image_path, output_path=None, size=None, image_type="good"):
    """
    Resize an image with optimal settings based on type.
    
    Args:
        image_path: Path to the input image
        output_path: Path to save the resized image (if None, overwrite input)
        size: Tuple of (width, height) for the target size
        image_type: String indicator of image type ("good" or "bad")
        
    Returns:
        Path to the resized image
    """
    # Set target size based on image type if not specified
    if size is None:
        size = (
            config.INFERENCE["GOOD_IMAGE_RESIZE"] if image_type.lower() == "good" 
            else config.INFERENCE["BAD_IMAGE_RESIZE"]
        )
    
    # Default output path to input path if not specified
    output_path = output_path or image_path
    
    try:
        # Get format settings
        format_settings = get_optimal_image_format(image_type)
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Open and resize image
        with Image.open(image_path) as img:
            # Use LANCZOS for high-quality downsampling
            img = img.resize(size, Image.LANCZOS)
            
            # Save with optimal settings
            img.save(
                output_path,
                format=format_settings["format"],
                quality=format_settings["quality"],
                optimize=format_settings["optimize"]
            )
        
        logger.debug(f"Resized image from {image_path} to {output_path} ({size[0]}x{size[1]})")
        return output_path
    except Exception as e:
        logger.error(f"Error resizing image {image_path}: {e}")
        if output_path != image_path:
            # If we're not overwriting, attempt a simple copy as fallback
            try:
                shutil.copy2(image_path, output_path)
                logger.warning(f"Resize failed, copied original image to {output_path}")
                return output_path
            except Exception as copy_error:
                logger.error(f"Error copying image: {copy_error}")
        
        # Return original path if all operations failed
        return image_path

def annotate_image(image_path, output_path=None, text=None, metrics=None, add_timestamp=True):
    """
    Add annotations to an image (timestamp, error values, etc.).
    
    Args:
        image_path: Path to the input image
        output_path: Path to save the annotated image
        text: Custom text to add to the image
        metrics: Dictionary of metrics to display
        add_timestamp: Whether to add a timestamp
        
    Returns:
        Path to the annotated image
    """
    if output_path is None:
        # Create output path with _annotated suffix
        base_path, ext = os.path.splitext(image_path)
        output_path = f"{base_path}_annotated{ext}"
    
    try:
        # Open the image
        with Image.open(image_path) as img:
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Create a drawing context
            draw = ImageDraw.Draw(img)
            
            # Try to load a font, use default if not available
            try:
                font = ImageFont.truetype("DejaVuSans.ttf", 14)
            except IOError:
                font = ImageFont.load_default()
            
            # Prepare annotation text
            lines = []
            
            if add_timestamp:
                timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                lines.append(f"Time: {timestamp}")
            
            if text:
                lines.append(text)
            
            if metrics:
                for key, value in metrics.items():
                    if isinstance(value, float):
                        lines.append(f"{key}: {value:.4f}")
                    else:
                        lines.append(f"{key}: {value}")
            
            # Draw text with shadow for better visibility
            y_position = 10
            for line in lines:
                # Draw shadow
                draw.text((11, y_position + 1), line, fill=(0, 0, 0), font=font)
                # Draw text
                draw.text((10, y_position), line, fill=(255, 255, 0), font=font)
                y_position += 20
            
            # Save the annotated image
            img.save(output_path)
            
            logger.debug(f"Created annotated image: {output_path}")
            return output_path
    except Exception as e:
        logger.error(f"Error annotating image {image_path}: {e}")
        
        # If annotation fails, try to at least copy the original
        if output_path != image_path:
            try:
                shutil.copy2(image_path, output_path)
                logger.warning(f"Annotation failed, copied original image to {output_path}")
                return output_path
            except Exception as copy_error:
                logger.error(f"Error copying image: {copy_error}")
        
        return image_path

def generate_heatmap(image_path, model, error_map=None, output_path=None):
    """
    Generate a heatmap overlay for a defective image.
    
    Args:
        image_path: Path to the image
        model: Trained model for reconstruction
        error_map: Optional precomputed error map
        output_path: Path to save the heatmap
        
    Returns:
        Path to the generated heatmap
    """
    if not HAS_TORCH:
        logger.error("Cannot generate heatmap: PyTorch not available")
        return None
        
    # Default output path
    if output_path is None:
        base_path, _ = os.path.splitext(image_path)
        output_path = f"{base_path}_heatmap.png"  # Always use PNG for heatmaps
    
    try:
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Create transforms
        transforms_dict = create_image_transforms()
        if transforms_dict is None:
            raise ImageUtilsError("Failed to create image transforms")
        
        # Get device for computation
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load and preprocess the image
        image = Image.open(image_path).convert("RGB")
        input_tensor = transforms_dict["model_input"](image).unsqueeze(0).to(device)
        
        # Generate error map if not provided
        if error_map is None:
            error_map = compute_error_map(
                input_tensor[0], 
                model, 
                patch_size=config.INFERENCE["PATCH_SIZE"],
                stride=config.INFERENCE["STRIDE"]
            )
        
        # Create visualization
        plt.figure(figsize=(10, 8))
        
        # Convert input tensor to a displayable image
        img_tensor = input_tensor[0].clone().detach().cpu()
        for c in range(3):  # RGB channels
            img_tensor[c] = img_tensor[c] * config.MODEL["NORMALIZE_STD"][c] + config.MODEL["NORMALIZE_MEAN"][c]
        img_np = img_tensor.permute(1, 2, 0).numpy()
        img_np = np.clip(img_np, 0, 1)  # Ensure values are in [0,1]
        
        # Display the original image
        plt.imshow(img_np)
        
        # Create custom colormap: transparent for low values, red for high values
        colors = [(0, 0, 0, 0), (1, 0, 0, 0.7)]
        cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=256)
        
        # Overlay heatmap
        plt.imshow(error_map, cmap=cmap, alpha=0.5)
        plt.colorbar(label='Reconstruction Error')
        plt.title('Defect Heatmap')
        plt.axis('off')
        
        # Save the visualization
        plt.savefig(output_path, format='png', bbox_inches='tight', pad_inches=0)
        plt.close()
        
        logger.info(f"Heatmap generated: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Error generating heatmap: {e}")
        return None

def compute_error_map(image_tensor, model, patch_size=64, stride=32, reduction='none'):
    """
    Compute error map for an image using sliding window analysis.
    
    Args:
        image_tensor: Input image tensor (C, H, W)
        model: Trained model for reconstruction
        patch_size: Size of each patch
        stride: Step size between patches
        reduction: How to reduce error values ('none', 'mean', 'sum')
        
    Returns:
        NumPy array containing the error map
    """
    if not HAS_TORCH:
        logger.error("Cannot compute error map: PyTorch not available")
        return None
        
    try:
        # Get tensor dimensions
        C, H, W = image_tensor.shape
        
        # Initialize error map and count map
        error_map = np.zeros((H, W))
        count_map = np.zeros((H, W))
        
        # Create MSE criterion
        criterion = nn.MSELoss(reduction=reduction)
        
        # Extract patches for sliding window analysis
        patches, (rows, cols) = extract_patches(image_tensor, patch_size, stride)
        
        # Make sure model is in evaluation mode
        model.eval()
        
        # Process each patch
        with torch.no_grad():
            for idx, patch in enumerate(patches):
                # Calculate position in the grid
                row_idx = idx // cols
                col_idx = idx % cols
                
                # Calculate pixel coordinates in the original image
                row_start = row_idx * stride
                col_start = col_idx * stride
                
                # Ensure we don't go out of bounds
                row_end = min(row_start + patch_size, H)
                col_end = min(col_start + patch_size, W)
                
                # Upsample patch to model input size
                patch_upsampled = torch.nn.functional.interpolate(
                    patch.unsqueeze(0), 
                    size=config.MODEL["IMAGE_SIZE"], 
                    mode='bilinear', 
                    align_corners=False
                )
                
                # Get reconstruction
                reconstructed_patch, _ = model(patch_upsampled)
                
                # Calculate patch-wise reconstruction error
                if reduction == 'none':
                    # For element-wise error, compute mean across channels
                    error = criterion(reconstructed_patch, patch_upsampled).mean(dim=1).squeeze().cpu().numpy()
                    
                    # Resize error to patch size if needed
                    if error.shape != (patch_size, patch_size):
                        from skimage.transform import resize
                        error = resize(error, (patch_size, patch_size), preserve_range=True)
                    
                    # Add error to error map at the patch position
                    error_map[row_start:row_end, col_start:col_end] += error[:row_end-row_start, :col_end-col_start]
                else:
                    # For scalar error, add to entire patch region
                    error = criterion(reconstructed_patch, patch_upsampled).item()
                    error_map[row_start:row_end, col_start:col_end] += error
                
                # Increment count map
                count_map[row_start:row_end, col_start:col_end] += 1
        
        # Normalize error map by count to get average error per pixel
        count_map[count_map == 0] = 1  # Avoid division by zero
        error_map = error_map / count_map
        
        # Normalize to [0, 1] range for visualization
        if np.max(error_map) > np.min(error_map):
            error_map = (error_map - np.min(error_map)) / (np.max(error_map) - np.min(error_map))
        
        return error_map
    except Exception as e:
        logger.error(f"Error computing error map: {e}")
        return None

def extract_patches(tensor, patch_size, stride):
    """
    Extract patches from an image tensor using a sliding window.
    
    Args:
        tensor: Image tensor (C, H, W)
        patch_size: Size of each patch
        stride: Step size between patches
        
    Returns:
        Tuple of (patches, grid_dimensions)
    """
    if not HAS_TORCH:
        logger.error("Cannot extract patches: PyTorch not available")
        return None, (0, 0)
        
    try:
        C, H, W = tensor.shape
        num_patches_h = (H - patch_size) // stride + 1
        num_patches_w = (W - patch_size) // stride + 1
        
        # Use unfold to extract patches
        patches = tensor.unfold(1, patch_size, stride).unfold(2, patch_size, stride)
        patches = patches.permute(1, 2, 0, 3, 4)
        patches = patches.contiguous().view(-1, C, patch_size, patch_size)
        
        return patches, (num_patches_h, num_patches_w)
    except Exception as e:
        logger.error(f"Error extracting patches: {e}")
        return None, (0, 0)

def backup_images(source_dir, max_images=1000, delete_after_backup=True):
    """
    Create a backup of images when a directory reaches its limit.
    
    Args:
        source_dir: Directory to monitor and backup
        max_images: Maximum number of images before backup
        delete_after_backup: Whether to delete images after backup
        
    Returns:
        Tuple of (backup_created, backup_path)
    """
    try:
        # Count images in directory
        if not os.path.exists(source_dir):
            logger.warning(f"Source directory does not exist: {source_dir}")
            return False, None
            
        # Count only files, not directories
        image_count = sum(1 for item in os.listdir(source_dir) 
                        if os.path.isfile(os.path.join(source_dir, item)))
        
        # Check if backup is needed
        if image_count < max_images:
            logger.debug(f"No backup needed yet. Image count: {image_count}/{max_images}")
            return False, None
            
        # Create backup directory with timestamp
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_dir = os.path.join(config.PATHS["BACKUP_DIR"], timestamp)
        os.makedirs(backup_dir, exist_ok=True)
        
        # Copy structure from source dir
        source_name = os.path.basename(source_dir)
        backup_subdir = os.path.join(backup_dir, source_name)
        os.makedirs(backup_subdir, exist_ok=True)
        
        # Copy files
        copied_count = 0
        for item in os.listdir(source_dir):
            source_path = os.path.join(source_dir, item)
            if os.path.isfile(source_path):
                dest_path = os.path.join(backup_subdir, item)
                shutil.copy2(source_path, dest_path)
                copied_count += 1
        
        logger.info(f"Backed up {copied_count} images to {backup_dir}")
        
        # Delete originals if requested
        if delete_after_backup:
            deleted_count = 0
            for item in os.listdir(source_dir):
                source_path = os.path.join(source_dir, item)
                if os.path.isfile(source_path):
                    os.unlink(source_path)
                    deleted_count += 1
            logger.info(f"Deleted {deleted_count} images from {source_dir} after backup")
        
        return True, backup_dir
    except Exception as e:
        logger.error(f"Error during image backup: {e}")
        return False, None

# For testing
if __name__ == "__main__":
    print("Testing Image Utilities")
    
    # Test optimization detection
    optimization_info = optimize_image_processing()
    print("\nOptimization Info:")
    for key, value in optimization_info.items():
        print(f"  {key}: {value}")
    
    # Test image resize
    print("\nTesting image resize...")
    test_image_path = "/tmp/test_image.jpg"
    
    # Create a test image if it doesn't exist
    if not os.path.exists(test_image_path):
        try:
            img = Image.new('RGB', (640, 480), color=(73, 109, 137))
            d = ImageDraw.Draw(img)
            d.text((10, 10), "Test Image", fill=(255, 255, 0))
            img.save(test_image_path)
            print(f"Created test image: {test_image_path}")
        except Exception as e:
            print(f"Error creating test image: {e}")
    
    # Test resize
    if os.path.exists(test_image_path):
        resized_path = resize_image(
            test_image_path,
            output_path="/tmp/resized_image.jpg",
            size=(320, 240),
            image_type="good"
        )
        print(f"Resized image saved to: {resized_path}")
    
    # Test annotation
    if os.path.exists(test_image_path):
        metrics = {
            "Error": 0.123,
            "Threshold": 0.5,
            "Status": "OK"
        }
        annotated_path = annotate_image(
            test_image_path,
            output_path="/tmp/annotated_image.jpg",
            text="Test Annotation",
            metrics=metrics
        )
        print(f"Annotated image saved to: {annotated_path}")
    
    print("\nImage Utilities tests completed.")