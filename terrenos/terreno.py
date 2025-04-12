import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils.utils import resize_image, generate_color_range, hex_to_bgr

class Terreno:
    """
    Class for processing and analyzing satellite images to map structures.
    """
    def __init__(self, name=None):
        """
        Initialize a Terreno object.
        
        Args:
            name (str, optional): Name identifier for the terrain. Defaults to None.
        """
        self.name = name
        self.image = None
        self.processed_image = None
        self.structures = []
        self.contours = []
        self.structure_mask = None
        
    def load_image(self, image_path):
        """
        Load a satellite image from the specified path.
        
        Args:
            image_path (str): Path to the satellite image file.
            
        Returns:
            bool: True if image was loaded successfully, False otherwise.
        """
        if not os.path.exists(image_path):
            print(f"Error: Image file not found at {image_path}")
            return False
        
        try:
            self.image = cv2.imread(image_path)
            if self.image is None:
                print(f"Error: Could not read image at {image_path}")
                return False
            
            # Create a copy for processing
            self.processed_image = self.image.copy()
            return True
        except Exception as e:
            print(f"Error loading image: {str(e)}")
            return False
    
    def load_image_from_array(self, image_array):
        """
        Load a satellite image from a numpy array.
        
        Args:
            image_array (numpy.ndarray): Image as a numpy array.
            
        Returns:
            bool: True if image was loaded successfully, False otherwise.
        """
        try:
            if image_array is None or not isinstance(image_array, np.ndarray):
                print("Error: Invalid image array")
                return False
            
            self.image = image_array.copy()
            self.processed_image = self.image.copy()
            return True
        except Exception as e:
            print(f"Error loading image from array: {str(e)}")
            return False
    
    def resize(self, width, height):
        """
        Resize the loaded image.
        
        Args:
            width (int): Target width.
            height (int): Target height.
            
        Returns:
            bool: True if resizing was successful, False otherwise.
        """
        if self.image is None:
            print("Error: No image loaded")
            return False
        
        try:
            self.image = resize_image(self.image, width, height)
            self.processed_image = resize_image(self.processed_image, width, height)
            return True
        except Exception as e:
            print(f"Error resizing image: {str(e)}")
            return False
    
    def preprocess(self, denoise=True, contrast_enhance=True):
        """
        Preprocess the image to enhance features for structure detection.
        
        Args:
            denoise (bool, optional): Apply denoising. Defaults to True.
            contrast_enhance (bool, optional): Enhance contrast. Defaults to True.
            
        Returns:
            numpy.ndarray: Preprocessed image.
        """
        if self.image is None:
            print("Error: No image loaded")
            return None
        
        try:
            # Create a copy of the original image
            processed = self.image.copy()
            
            # Convert to grayscale
            if len(processed.shape) > 2 and processed.shape[2] == 3:
                gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
            else:
                gray = processed.copy()
            
            # Apply denoising if requested
            if denoise:
                gray = cv2.GaussianBlur(gray, (5, 5), 0)
                
            # Enhance contrast if requested
            if contrast_enhance:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                gray = clahe.apply(gray)
            
            self.processed_image = gray
            return gray
        except Exception as e:
            print(f"Error preprocessing image: {str(e)}")
            return None
    
    def detect_structures(self, threshold_method='adaptive', structure_min_area=100, structure_max_area=None):
        """
        Detect structures in the satellite image.
        
        Args:
            threshold_method (str, optional): Method for thresholding ('adaptive', 'otsu', or 'binary'). 
                                             Defaults to 'adaptive'.
            structure_min_area (int, optional): Minimum area for a structure to be considered valid. 
                                               Defaults to 100.
            structure_max_area (int, optional): Maximum area for a structure to be considered valid. 
                                               Defaults to None.
            
        Returns:
            list: List of detected structure contours.
        """
        if self.processed_image is None:
            print("Error: No processed image available")
            return []
        
        try:
            # Ensure we're working with grayscale
            if len(self.processed_image.shape) > 2:
                gray = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = self.processed_image.copy()
            
            # Apply thresholding based on the selected method
            if threshold_method == 'adaptive':
                binary = cv2.adaptiveThreshold(
                    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                    cv2.THRESH_BINARY_INV, 11, 2
                )
            elif threshold_method == 'otsu':
                _, binary = cv2.threshold(
                    gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
                )
            else:  # binary
                _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
            
            # Apply morphological operations to clean up the binary image
            kernel = np.ones((3, 3), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours by area
            valid_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area >= structure_min_area and (structure_max_area is None or area <= structure_max_area):
                    valid_contours.append(contour)
            
            self.contours = valid_contours
            
            # Create a mask for the structures
            self.structure_mask = np.zeros_like(binary)
            cv2.drawContours(self.structure_mask, valid_contours, -1, 255, -1)
            
            # Extract structure information
            self.structures = []
            for i, contour in enumerate(valid_contours):
                # Calculate bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Calculate centroid
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    cx, cy = x + w // 2, y + h // 2
                
                # Calculate area
                area = cv2.contourArea(contour)
                
                # Store structure information
                structure = {
                    'id': i,
                    'contour': contour,
                    'bbox': (x, y, w, h),
                    'centroid': (cx, cy),
                    'area': area
                }
                self.structures.append(structure)
            
            return valid_contours
        except Exception as e:
            print(f"Error detecting structures: {str(e)}")
            return []
    
    def detect_structures_by_color(self, target_color_hex, hue_offset=10, 
                                  saturation_offset=50, value_offset=50,
                                  structure_min_area=100, structure_max_area=None):
        """
        Detect structures in the satellite image based on color.
        
        Args:
            target_color_hex (str): Target color in hex format (e.g., '#FF0000' for red).
            hue_offset (int, optional): Hue range offset. Defaults to 10.
            saturation_offset (int, optional): Saturation range offset. Defaults to 50.
            value_offset (int, optional): Value range offset. Defaults to 50.
            structure_min_area (int, optional): Minimum area for a structure. Defaults to 100.
            structure_max_area (int, optional): Maximum area for a structure. Defaults to None.
            
        Returns:
            list: List of detected structure contours.
        """
        if self.image is None:
            print("Error: No image loaded")
            return []
        
        try:
            # Convert to HSV color space
            hsv_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
            
            # Generate color range
            lower_bound, upper_bound = generate_color_range(
                target_color_hex, hue_offset, saturation_offset, value_offset
            )
            
            # Create a mask for the specified color
            color_mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
            
            # Apply morphological operations to clean up the mask
            kernel = np.ones((5, 5), np.uint8)
            color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel, iterations=1)
            color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            
            # Find contours
            contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours by area
            valid_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area >= structure_min_area and (structure_max_area is None or area <= structure_max_area):
                    valid_contours.append(contour)
            
            self.contours = valid_contours
            
            # Create a mask for the structures
            self.structure_mask = np.zeros_like(color_mask)
            cv2.drawContours(self.structure_mask, valid_contours, -1, 255, -1)
            
            # Extract structure information
            self.structures = []
            for i, contour in enumerate(valid_contours):
                # Calculate bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Calculate centroid
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    cx, cy = x + w // 2, y + h // 2
                
                # Calculate area
                area = cv2.contourArea(contour)
                
                # Store structure information
                structure = {
                    'id': i,
                    'contour': contour,
                    'bbox': (x, y, w, h),
                    'centroid': (cx, cy),
                    'area': area
                }
                self.structures.append(structure)
            
            return valid_contours
        except Exception as e:
            print(f"Error detecting structures by color: {str(e)}")
            return []
    
    def segment_image(self, num_segments=5):
        """
        Segment the image using K-means clustering.
        
        Args:
            num_segments (int, optional): Number of segments to create. Defaults to 5.
            
        Returns:
            numpy.ndarray: Segmented image.
        """
        if self.image is None:
            print("Error: No image loaded")
            return None
        
        try:
            # Reshape the image
            pixel_values = self.image.reshape((-1, 3))
            pixel_values = np.float32(pixel_values)
            
            # Define criteria and apply kmeans
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
            _, labels, centers = cv2.kmeans(
                pixel_values, num_segments, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
            )
            
            # Convert back to uint8
            centers = np.uint8(centers)
            segmented_image = centers[labels.flatten()]
            segmented_image = segmented_image.reshape(self.image.shape)
            
            return segmented_image
        except Exception as e:
            print(f"Error segmenting image: {str(e)}")
            return None
    
    def visualize_structures(self, output_path=None, show=True):
        """
        Visualize detected structures on the original image.
        
        Args:
            output_path (str, optional): Path to save the visualization. Defaults to None.
            show (bool, optional): Whether to display the visualization. Defaults to True.
            
        Returns:
            numpy.ndarray: Visualization image.
        """
        if self.image is None or not self.contours:
            print("Error: No image loaded or no structures detected")
            return None
        
        try:
            # Create a copy of the original image
            visualization = self.image.copy()
            
            # Draw contours
            cv2.drawContours(visualization, self.contours, -1, (0, 255, 0), 2)
            
            # Draw bounding boxes and IDs
            for structure in self.structures:
                x, y, w, h = structure['bbox']
                cv2.rectangle(visualization, (x, y), (x + w, y + h), (255, 0, 0), 2)
                
                # Add structure ID
                cv2.putText(
                    visualization, f"#{structure['id']}", (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1
                )
            
            # Save the visualization if output path is provided
            if output_path:
                cv2.imwrite(output_path, visualization)
            
            # Show the visualization if requested
            if show:
                plt.figure(figsize=(10, 8))
                plt.imshow(cv2.cvtColor(visualization, cv2.COLOR_BGR2RGB))
                plt.title("Detected Structures")
                plt.axis('off')
                plt.tight_layout()
                plt.show()
            
            return visualization
        except Exception as e:
            print(f"Error visualizing structures: {str(e)}")
            return None
    
    def calculate_structure_density(self):
        """
        Calculate the density of structures in the image.
        
        Returns:
            float: Structure density (percentage of image area covered by structures).
        """
        if self.image is None or self.structure_mask is None:
            print("Error: No image loaded or no structures detected")
            return 0.0
        
        try:
            # Calculate total image area
            total_area = self.image.shape[0] * self.image.shape[1]
            
            # Calculate area covered by structures
            structure_area = np.sum(self.structure_mask > 0)
            
            # Calculate density
            density = (structure_area / total_area) * 100.0
            
            return density
        except Exception as e:
            print(f"Error calculating structure density: {str(e)}")
            return 0.0
    
    def get_structure_statistics(self):
        """
        Get statistics about the detected structures.
        
        Returns:
            dict: Dictionary containing structure statistics.
        """
        if not self.structures:
            print("Error: No structures detected")
            return {}
        
        try:
            # Calculate areas
            areas = [structure['area'] for structure in self.structures]
            
            # Calculate statistics
            stats = {
                'count': len(self.structures),
                'total_area': sum(areas),
                'min_area': min(areas),
                'max_area': max(areas),
                'mean_area': sum(areas) / len(areas),
                'density': self.calculate_structure_density()
            }
            
            return stats
        except Exception as e:
            print(f"Error calculating structure statistics: {str(e)}")
            return {}
    
    def export_structure_data(self, output_path):
        """
        Export structure data to a CSV file.
        
        Args:
            output_path (str): Path to save the CSV file.
            
        Returns:
            bool: True if export was successful, False otherwise.
        """
        if not self.structures:
            print("Error: No structures detected")
            return False
        
        try:
            import csv
            
            with open(output_path, 'w', newline='') as csvfile:
                fieldnames = ['id', 'x', 'y', 'width', 'height', 'centroid_x', 'centroid_y', 'area']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                writer.writeheader()
                for structure in self.structures:
                    x, y, w, h = structure['bbox']
                    cx, cy = structure['centroid']
                    
                    writer.writerow({
                        'id': structure['id'],
                        'x': x,
                        'y': y,
                        'width': w,
                        'height': h,
                        'centroid_x': cx,
                        'centroid_y': cy,
                        'area': structure['area']
                    })
            
            return True
        except Exception as e:
            print(f"Error exporting structure data: {str(e)}")
            return False
    
    def create_structure_mask(self, output_path=None):
        """
        Create a binary mask of the detected structures.
        
        Args:
            output_path (str, optional): Path to save the mask. Defaults to None.
            
        Returns:
            numpy.ndarray: Binary mask.
        """
        if self.image is None or not self.contours:
            print("Error: No image loaded or no structures detected")
            return None
        
        try:
            # Create an empty mask
            mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
            
            # Draw contours on the mask
            cv2.drawContours(mask, self.contours, -1, 255, -1)
            
            # Save the mask if output path is provided
            if output_path:
                cv2.imwrite(output_path, mask)
            
            return mask
        except Exception as e:
            print(f"Error creating structure mask: {str(e)}")
            return None


def main():
    """
    Main function to demonstrate the Terreno class functionality.
    """
    # Example usage
    print("Terreno - Satellite Image Structure Mapping")
    print("-------------------------------------------")
    
    # Create a Terreno object
    terreno = Terreno(name="Example Terrain")
    
    # Here you would load a satellite image
    # For example:
    # terreno.load_image("path/to/satellite_image.jpg")
    
    # Then preprocess the image
    # terreno.preprocess()
    
    # Detect structures
    # terreno.detect_structures()
    
    # Visualize the results
    # terreno.visualize_structures()
    
    # Get statistics
    # stats = terreno.get_structure_statistics()
    # print("Structure Statistics:")
    # for key, value in stats.items():
    #     print(f"  {key}: {value}")
    
    print("\nTo use this module, import the Terreno class and follow the example in the main function.")


if __name__ == "__main__":
    main()
