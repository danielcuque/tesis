"""
Module for vessel recognition and analysis in archaeological images.
This module provides functionality to detect and analyze vessel fragments
in images using color-based segmentation and contour analysis.
"""

import os
import cv2
import numpy as np
from utils import utils
from pathlib import Path


class VesselAnalyzer:
    """
    Class for analyzing and detecting vessel fragments in archaeological images.
    """
    def __init__(self, target_color=None):
        """
        Initialize the VesselAnalyzer.
        
        Args:
            target_color (str, optional): Target color in hex format (e.g., '#A98876').
                                         Defaults to None.
        """
        self.target_color = target_color or '#A98876'  # Default terracotta color
        self.image = None
        self.hsv_image = None
        self.mask = None
        self.contours = []
        self.filtered_contours = []
    
    def load_image(self, image_path):
        """
        Load an image for analysis.
        
        Args:
            image_path (str): Path to the image file.
            
        Returns:
            bool: True if image was loaded successfully, False otherwise.
        """
        try:
            # Read image
            self.image = cv2.imread(image_path)
            
            # Verify that the image was loaded correctly
            if self.image is None:
                print(f"Error: Could not load image: {image_path}")
                return False
            
            # Convert to HSV
            self.hsv_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
            
            return True
        except Exception as e:
            print(f"Error loading image: {str(e)}")
            return False
    
    def set_target_color(self, color_hex):
        """
        Set the target color for vessel detection.
        
        Args:
            color_hex (str): Target color in hex format (e.g., '#A98876').
        """
        self.target_color = color_hex
    
    def create_color_mask(self, hue_offset=10, saturation_offset=50, value_offset=50):
        """
        Create a mask for the specified target color.
        
        Args:
            hue_offset (int, optional): Hue range offset. Defaults to 10.
            saturation_offset (int, optional): Saturation range offset. Defaults to 50.
            value_offset (int, optional): Value range offset. Defaults to 50.
            
        Returns:
            numpy.ndarray: Binary mask highlighting areas matching the target color.
        """
        if self.hsv_image is None:
            print("Error: No image loaded")
            return None
        
        try:
            # Get color range
            lower_bound, upper_bound = utils.generate_color_range(
                self.target_color, hue_offset, saturation_offset, value_offset
            )
            
            # Create mask
            self.mask = cv2.inRange(self.hsv_image, lower_bound, upper_bound)
            
            return self.mask
        except Exception as e:
            print(f"Error creating color mask: {str(e)}")
            return None
    
    def process_mask(self, dilate_iterations=1, erode_iterations=1, blur_size=5):
        """
        Process the mask with morphological operations to improve detection.
        
        Args:
            dilate_iterations (int, optional): Number of dilation iterations. Defaults to 1.
            erode_iterations (int, optional): Number of erosion iterations. Defaults to 1.
            blur_size (int, optional): Size of Gaussian blur kernel. Defaults to 5.
            
        Returns:
            numpy.ndarray: Processed mask.
        """
        if self.mask is None:
            print("Error: No mask available. Create a color mask first.")
            return None
        
        try:
            # Create kernel for morphological operations
            kernel = np.ones((5, 5), np.uint8)
            
            # Apply dilation
            dilated = cv2.dilate(self.mask, kernel, iterations=dilate_iterations)
            
            # Apply erosion
            eroded = cv2.erode(dilated, kernel, iterations=erode_iterations)
            
            # Apply Gaussian blur
            smoothed = cv2.GaussianBlur(eroded, (blur_size, blur_size), 0)
            
            # Update the mask
            self.mask = smoothed
            
            return self.mask
        except Exception as e:
            print(f"Error processing mask: {str(e)}")
            return None
    
    def find_contours(self):
        """
        Find contours in the processed mask.
        
        Returns:
            list: List of contours found in the mask.
        """
        if self.mask is None:
            print("Error: No mask available. Create and process a color mask first.")
            return []
        
        try:
            # Find contours
            contours, _ = cv2.findContours(self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            self.contours = contours
            
            return contours
        except Exception as e:
            print(f"Error finding contours: {str(e)}")
            return []
    
    def filter_contours(self, min_ratio=0.2, max_ratio=4.0, min_area=10, max_area=1000):
        """
        Filter contours by aspect ratio and area.
        
        Args:
            min_ratio (float, optional): Minimum aspect ratio. Defaults to 0.2.
            max_ratio (float, optional): Maximum aspect ratio. Defaults to 4.0.
            min_area (int, optional): Minimum contour area. Defaults to 10.
            max_area (int, optional): Maximum contour area. Defaults to 1000.
            
        Returns:
            list: Filtered contours.
        """
        if not self.contours:
            print("Error: No contours available. Find contours first.")
            return []
        
        try:
            filtered = []
            
            for contour in self.contours:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Calculate aspect ratio
                aspect_ratio = float(w) / h if h != 0 else 0
                
                # Calculate area
                area = cv2.contourArea(contour)
                
                # Filter by aspect ratio and area
                if min_ratio <= aspect_ratio <= max_ratio and min_area <= area <= max_area:
                    filtered.append(contour)
            
            self.filtered_contours = filtered
            return filtered
        except Exception as e:
            print(f"Error filtering contours: {str(e)}")
            return []
    
    def draw_contours(self, image=None, color=(0, 255, 0), thickness=1, label=True):
        """
        Draw contours on an image.
        
        Args:
            image (numpy.ndarray, optional): Image to draw on. If None, uses the original image.
            color (tuple, optional): Color for contours (BGR). Defaults to (0, 255, 0) (green).
            thickness (int, optional): Line thickness. Defaults to 1.
            label (bool, optional): Whether to add a label with the count. Defaults to True.
            
        Returns:
            numpy.ndarray: Image with contours drawn.
        """
        if self.image is None:
            print("Error: No image loaded")
            return None
        
        if not self.filtered_contours:
            print("Warning: No filtered contours available. Using all contours.")
            contours_to_draw = self.contours
        else:
            contours_to_draw = self.filtered_contours
        
        if not contours_to_draw:
            print("Error: No contours to draw")
            return None
        
        try:
            # Use provided image or copy the original
            result = image.copy() if image is not None else self.image.copy()
            
            # Draw contours
            cv2.drawContours(result, contours_to_draw, -1, color, thickness)
            
            # Add label with count if requested
            if label:
                count = len(contours_to_draw)
                text = f'Cantidad de restos encontrados: {count}'
                cv2.putText(
                    result, text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA
                )
            
            return result
        except Exception as e:
            print(f"Error drawing contours: {str(e)}")
            return None
    
    def save_image(self, image, output_path):
        """
        Save an image to disk.
        
        Args:
            image (numpy.ndarray): Image to save.
            output_path (str): Path to save the image.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save image
            cv2.imwrite(output_path, image)
            
            return True
        except Exception as e:
            print(f"Error saving image: {str(e)}")
            return False
    
    def analyze_image(self, image_path, output_folder=None, save_intermediates=False):
        """
        Perform complete analysis on an image.
        
        Args:
            image_path (str): Path to the image file.
            output_folder (str, optional): Folder to save results. If None, results are not saved.
            save_intermediates (bool, optional): Whether to save intermediate results. Defaults to False.
            
        Returns:
            dict: Analysis results including count, contours, and result image.
        """
        # Load image
        if not self.load_image(image_path):
            return {'success': False, 'error': 'Failed to load image'}
        
        # Create output folder if specified
        if output_folder:
            # Get image name without extension
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            image_output_folder = os.path.join(output_folder, image_name)
            os.makedirs(image_output_folder, exist_ok=True)
        else:
            image_output_folder = None
        
        # Create color mask
        mask = self.create_color_mask()
        
        # Save HSV image and mask if requested
        if save_intermediates and image_output_folder:
            # Resize for better visualization
            hsv_resized = utils.resize_image(self.hsv_image, 780, 540)
            mask_resized = utils.resize_image(mask, 780, 540)
            
            self.save_image(hsv_resized, os.path.join(image_output_folder, 'hsv.png'))
            self.save_image(mask_resized, os.path.join(image_output_folder, 'mask.png'))
        
        # Process mask
        processed_mask = self.process_mask()
        
        # Save intermediate processing results if requested
        if save_intermediates and image_output_folder:
            # Resize for better visualization
            mask_resized = utils.resize_image(processed_mask, 780, 540)
            
            # Save different processing stages
            dilated = cv2.dilate(mask_resized, np.ones((5, 5), np.uint8), iterations=1)
            eroded = cv2.erode(dilated, np.ones((5, 5), np.uint8), iterations=1)
            smoothed = cv2.GaussianBlur(eroded, (5, 5), 0)
            
            self.save_image(dilated, os.path.join(image_output_folder, 'dilated.png'))
            self.save_image(eroded, os.path.join(image_output_folder, 'eroded.png'))
            self.save_image(smoothed, os.path.join(image_output_folder, 'smooth.png'))
        
        # Find contours
        self.find_contours()
        
        # Filter contours
        self.filter_contours()
        
        # Resize original image for visualization
        original_resized = utils.resize_image(self.image, 780, 540)
        
        # Draw contours
        result_image = self.draw_contours(original_resized)
        
        # Save result if output folder is specified
        if image_output_folder:
            # Save contour image
            contour_image = self.draw_contours(original_resized, label=False)
            self.save_image(contour_image, os.path.join(image_output_folder, 'contours.png'))
            
            # Save final result
            self.save_image(result_image, os.path.join(image_output_folder, 'final_result.png'))
        
        # Return results
        return {
            'success': True,
            'count': len(self.filtered_contours),
            'contours': self.filtered_contours,
            'result_image': result_image,
            'output_folder': image_output_folder
        }
    
    def batch_process(self, input_folder, output_folder, file_extensions=None):
        """
        Process all images in a folder.
        
        Args:
            input_folder (str): Folder containing images to process.
            output_folder (str): Folder to save results.
            file_extensions (list, optional): List of file extensions to process.
                                             Defaults to ['.png', '.jpg', '.jpeg'].
            
        Returns:
            dict: Results for each processed image.
        """
        if file_extensions is None:
            file_extensions = ['.png', '.jpg', '.jpeg']
        
        # Ensure input folder exists
        if not os.path.exists(input_folder):
            print(f"Error: Input folder not found: {input_folder}")
            return {}
        
        # Create output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        
        results = {}
        
        # Get list of image files
        image_files = [
            f for f in os.listdir(input_folder) 
            if any(f.lower().endswith(ext) for ext in file_extensions)
        ]
        
        # Process each image
        for image_file in image_files:
            image_path = os.path.join(input_folder, image_file)
            
            print(f"Processing image: {image_file}")
            
            # Analyze image
            result = self.analyze_image(image_path, output_folder, save_intermediates=True)
            
            if result['success']:
                print(f"Image: {image_file} - Vessel fragments detected: {result['count']}")
                results[image_file] = result
            else:
                print(f"Failed to process image: {image_file}")
        
        # Print summary
        print("\nProcessing completed:")
        print(f"  - Images processed: {len(results)}/{len(image_files)}")
        print(f"  - Results saved to: {output_folder}")
        
        return results


# Module-level function for backward compatibility
def filter_contours_by_aspect_ratio(contours, min_ratio=0.2, max_ratio=4.0, min_area=10, max_area=1000):
    """
    Filter contours by aspect ratio and area.
    
    Args:
        contours (list): List of contours to filter.
        min_ratio (float, optional): Minimum aspect ratio. Defaults to 0.2.
        max_ratio (float, optional): Maximum aspect ratio. Defaults to 4.0.
        min_area (int, optional): Minimum contour area. Defaults to 10.
        max_area (int, optional): Maximum contour area. Defaults to 1000.
        
    Returns:
        list: Filtered contours.
    """
    filtered_contours = []
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h if h != 0 else 0
        area = cv2.contourArea(contour)
        
        if min_ratio <= aspect_ratio <= max_ratio and min_area <= area <= max_area:
            filtered_contours.append(contour)
    
    return filtered_contours

def save_image(output_folder, filename, image):
    """
    Save an image to a directory.
    
    Args:
        output_folder (str): Directory to save the image.
        filename (str): Filename for the saved image.
        image (numpy.ndarray): Image to save.
        
    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        cv2.imwrite(os.path.join(output_folder, filename), image)
        return True
    except Exception as e:
        print(f"Error saving image: {str(e)}")
        return False

def main():
    """
    Main function to demonstrate the VesselAnalyzer class.
    """
    # Create the output folder if it doesn't exist
    output_folder = './assets/results/vasijas/'
    os.makedirs(output_folder, exist_ok=True)
    
    # Directory of images to process
    input_folder = './assets/dataset_vasijas/'
    
    # Create a VesselAnalyzer instance
    analyzer = VesselAnalyzer(target_color='#A98876')
    
    # Process all images in the input folder
    analyzer.batch_process(input_folder, output_folder)


if __name__ == "__main__":
    main()
