"""
Module for generating variations of glyph images using data augmentation techniques.
This module provides functionality to create multiple variants of input images
to enhance training datasets for machine learning models.
"""

import os
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path


class GlyphVariantGenerator:
    """
    Class for generating variations of glyph images using data augmentation.
    """
    def __init__(self, transform=None):
        """
        Initialize the GlyphVariantGenerator.
        
        Args:
            transform (albumentations.Compose, optional): Custom transformation pipeline.
                                                         If None, a default pipeline will be used.
        """
        # Use provided transform or create default one
        self.transform = transform if transform is not None else self._create_default_transform()
    
    def _create_default_transform(self):
        """
        Create a default transformation pipeline for data augmentation.
        
        Returns:
            albumentations.Compose: Default transformation pipeline.
        """
        return A.Compose([
            A.Rotate(limit=30, p=1.0, border_mode=cv2.BORDER_CONSTANT),  # Rotation with constant border
            A.RandomScale(scale_limit=(0.8, 1.2), p=1.0),  # Scale between 0.8 and 1.2
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),  # Brightness and contrast
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.0),  # Color shift
            A.GaussianBlur(blur_limit=3, p=0.3),  # Gaussian blur
            A.RandomGamma(p=0.5),  # Random gamma adjustment
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),  # Contrast Limited Adaptive Histogram Equalization
            ToTensorV2()
        ])
    
    def generate_variations(self, image_path, output_dir, num_variants=10, prefix=None):
        """
        Generate variations of a single image.
        
        Args:
            image_path (str): Path to the input image.
            output_dir (str): Directory to save generated variants.
            num_variants (int, optional): Number of variants to generate. Defaults to 10.
            prefix (str, optional): Prefix for output filenames. Defaults to None.
            
        Returns:
            list: Paths to the generated images.
        """
        # Read the original image
        img = cv2.imread(image_path)
        
        # Verify that the image was loaded correctly
        if img is None:
            print(f"Error: Could not load image: {image_path}")
            return []
        
        # Get the folder name from the image path
        folder_name = os.path.basename(os.path.dirname(image_path))
        output_folder = os.path.join(output_dir, folder_name)
        
        # Create the output directory if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        
        # Get the base filename
        filename = os.path.basename(image_path)
        base_name = os.path.splitext(filename)[0]
        
        # Use provided prefix or base name
        prefix = prefix or base_name
        
        generated_paths = []
        
        # Generate variants
        for i in range(num_variants):
            try:
                # Apply transformations
                augmented = self.transform(image=img)
                augmented_image = augmented['image']
                
                # Convert the tensor to a NumPy array (HWC)
                augmented_image = augmented_image.permute(1, 2, 0).cpu().numpy()
                
                # Ensure the image is in the range [0, 255] and type uint8
                augmented_image = np.clip(augmented_image * 255, 0, 255).astype(np.uint8)
                
                # Create output path
                output_path = os.path.join(output_folder, f"{prefix}_variant_{i+1}.png")
                
                # Replace backslashes with forward slashes
                output_path = output_path.replace("\\", "/")
                
                # Save the generated image
                cv2.imwrite(output_path, augmented_image)
                
                generated_paths.append(output_path)
                print(f"Generated image: {output_path}")
            except Exception as e:
                print(f"Error generating variant {i+1}: {str(e)}")
        
        return generated_paths
    
    def process_directory(self, input_dir, output_dir, num_variants=10):
        """
        Process all images in a directory to generate variants.
        
        Args:
            input_dir (str): Directory containing input images.
            output_dir (str): Directory to save generated variants.
            num_variants (int, optional): Number of variants to generate per image. Defaults to 10.
            
        Returns:
            dict: Dictionary mapping input images to lists of generated variant paths.
        """
        # Ensure the input directory exists
        if not os.path.exists(input_dir):
            print(f"Error: Input directory not found: {input_dir}")
            return {}
        
        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        results = {}
        
        # Iterate over each image in the input directory
        for filename in os.listdir(input_dir):
            # Ensure it's an image
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(input_dir, filename)
                
                try:
                    # Generate variations for this image
                    variant_paths = self.generate_variations(
                        image_path, 
                        output_dir, 
                        num_variants
                    )
                    
                    results[image_path] = variant_paths
                except Exception as e:
                    print(f"Error processing image {filename}: {str(e)}")
        
        # Print summary
        total_variants = sum(len(paths) for paths in results.values())
        print(f"\nProcessing completed:")
        print(f"  - Input images processed: {len(results)}")
        print(f"  - Total variants generated: {total_variants}")
        
        return results
    
    def create_custom_transform(self, config):
        """
        Create a custom transformation pipeline based on configuration.
        
        Args:
            config (dict): Configuration dictionary with transformation parameters.
            
        Returns:
            albumentations.Compose: Custom transformation pipeline.
        """
        transforms = []
        
        # Add transformations based on configuration
        if config.get('rotate', True):
            rotate_limit = config.get('rotate_limit', 30)
            transforms.append(A.Rotate(limit=rotate_limit, p=1.0, border_mode=cv2.BORDER_CONSTANT))
        
        if config.get('scale', True):
            scale_min = config.get('scale_min', 0.8)
            scale_max = config.get('scale_max', 1.2)
            transforms.append(A.RandomScale(scale_limit=(scale_min, scale_max), p=1.0))
        
        if config.get('brightness_contrast', True):
            brightness_limit = config.get('brightness_limit', 0.2)
            contrast_limit = config.get('contrast_limit', 0.2)
            transforms.append(A.RandomBrightnessContrast(
                brightness_limit=brightness_limit, 
                contrast_limit=contrast_limit, 
                p=1.0
            ))
        
        if config.get('color_shift', True):
            hue_limit = config.get('hue_limit', 20)
            sat_limit = config.get('sat_limit', 30)
            val_limit = config.get('val_limit', 20)
            transforms.append(A.HueSaturationValue(
                hue_shift_limit=hue_limit,
                sat_shift_limit=sat_limit,
                val_shift_limit=val_limit,
                p=1.0
            ))
        
        if config.get('blur', True):
            blur_limit = config.get('blur_limit', 3)
            blur_prob = config.get('blur_prob', 0.3)
            transforms.append(A.GaussianBlur(blur_limit=blur_limit, p=blur_prob))
        
        if config.get('gamma', True):
            gamma_prob = config.get('gamma_prob', 0.5)
            transforms.append(A.RandomGamma(p=gamma_prob))
        
        if config.get('clahe', True):
            clip_limit = config.get('clip_limit', 2.0)
            grid_size = config.get('grid_size', 8)
            clahe_prob = config.get('clahe_prob', 0.5)
            transforms.append(A.CLAHE(
                clip_limit=clip_limit,
                tile_grid_size=(grid_size, grid_size),
                p=clahe_prob
            ))
        
        # Always add ToTensorV2 at the end
        transforms.append(ToTensorV2())
        
        return A.Compose(transforms)


# Module-level functions for backward compatibility
def generate_variations(image_path, output_dir, num_variants=10):
    """
    Generate variations of a single image.
    
    Args:
        image_path (str): Path to the input image.
        output_dir (str): Directory to save generated variants.
        num_variants (int, optional): Number of variants to generate. Defaults to 10.
        
    Returns:
        list: Paths to the generated images.
    """
    generator = GlyphVariantGenerator()
    return generator.generate_variations(image_path, output_dir, num_variants)

def process_directory(input_dir, output_dir, num_variants=10):
    """
    Process all images in a directory to generate variants.
    
    Args:
        input_dir (str): Directory containing input images.
        output_dir (str): Directory to save generated variants.
        num_variants (int, optional): Number of variants to generate per image. Defaults to 10.
        
    Returns:
        dict: Dictionary mapping input images to lists of generated variant paths.
    """
    generator = GlyphVariantGenerator()
    return generator.process_directory(input_dir, output_dir, num_variants)


if __name__ == "__main__":
    # Example usage
    generator = GlyphVariantGenerator()
    
    # Process a directory of images
    input_dir = 'assets/dataset_glifos/a'
    output_dir = 'assets/dataset_glifos/generated_images'
    generator.process_directory(input_dir, output_dir, num_variants=5)
