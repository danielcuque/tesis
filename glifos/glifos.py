"""
Module for glyph analysis using YOLOv8.
This module provides functionality for training and predicting glyphs in images.
"""

from ultralytics import YOLO
import os
import cv2
import numpy as np
import yaml
from pathlib import Path

class GlifosAnalyzer:
    """
    Class for analyzing glyphs using YOLOv8.
    """
    def __init__(self, config_path='glifos/config.yaml'):
        """
        Initialize the GlifosAnalyzer.
        
        Args:
            config_path (str): Path to the YAML configuration file.
        """
        self.config_path = config_path
        self.model = None
        self.results = None
        
        # Load configuration
        self._load_config()
    
    def _load_config(self):
        """
        Load configuration from YAML file.
        """
        try:
            with open(self.config_path, 'r') as file:
                self.config = yaml.safe_load(file)
        except Exception as e:
            print(f"Error loading configuration: {str(e)}")
            self.config = {}
    
    def calculate_average_img_size(self, input_dir):
        """
        Calculate the average image size in a directory.
        Returns the larger value (width or height) to use as imgsz.
        
        Args:
            input_dir (str): Directory containing images.
            
        Returns:
            int: Recommended image size for training.
            
        Raises:
            ValueError: If no valid images are found in the directory.
        """
        widths, heights = [], []
        
        # Ensure the directory exists
        if not os.path.exists(input_dir):
            raise ValueError(f"Directory not found: {input_dir}")
        
        # Process each image in the directory
        for filename in os.listdir(input_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(input_dir, filename)
                try:
                    img = cv2.imread(img_path)
                    if img is not None:
                        h, w = img.shape[:2]
                        heights.append(h)
                        widths.append(w)
                except Exception as e:
                    print(f"Error processing image {filename}: {str(e)}")
        
        # Calculate average size
        if widths and heights:
            avg_width = int(np.mean(widths))
            avg_height = int(np.mean(heights))
            return max(avg_width, avg_height)
        else:
            raise ValueError("No valid images found in the directory.")
    
    def train(self, model_yaml='yolov8n.yaml', epochs=50, imgsz=640, input_dir=None):
        """
        Train a YOLOv8 model for glyph detection.
        
        Args:
            model_yaml (str): Path to the YOLO model YAML file.
            epochs (int): Number of training epochs.
            imgsz (int): Image size for training.
            input_dir (str, optional): Directory containing training images.
                                      If provided, will calculate average image size.
            
        Returns:
            object: Training results.
        """
        print(f"Current working directory: {os.getcwd()}")
        
        # Calculate average image size if input directory is provided
        if input_dir:
            try:
                calculated_imgsz = self.calculate_average_img_size(input_dir)
                imgsz = calculated_imgsz
                print(f"Using calculated average image size: {imgsz}")
            except ValueError as e:
                print(f"Warning: {str(e)} Using default image size: {imgsz}")
        else:
            print(f"Using default image size: {imgsz}")
        
        # Initialize the model
        try:
            self.model = YOLO(model_yaml)
            
            # Train the model
            self.results = self.model.train(
                data=self.config_path,
                epochs=epochs,
                imgsz=imgsz,
                verbose=True
            )
            
            print("Training completed successfully.")
            return self.results
        except Exception as e:
            print(f"Error during training: {str(e)}")
            return None
    
    def predict(self, model_path='runs/detect/train/weights/best.pt', 
                source_dir=None, output_dir=None, conf=0.01):
        """
        Perform prediction using a trained model.
        
        Args:
            model_path (str): Path to the trained model weights.
            source_dir (str, optional): Directory containing images for prediction.
            output_dir (str, optional): Directory to save prediction results.
            conf (float): Confidence threshold for detections.
            
        Returns:
            object: Prediction results.
        """
        try:
            # Set default directories if not provided
            if source_dir is None:
                source_dir = 'assets/dataset_glifos/a'
            
            if output_dir is None:
                output_dir = 'assets/results/glifos'
            
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Load the model
            self.model = YOLO(model_path)
            
            # Perform prediction
            results = self.model.predict(
                source=source_dir,
                save=True,
                project=output_dir,
                conf=conf,
                verbose=True
            )
            
            print(f"Prediction completed. Results saved to {output_dir}")
            return results
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            return None
    
    def export_model(self, format='onnx', output_path=None):
        """
        Export the trained model to different formats.
        
        Args:
            format (str): Export format ('onnx', 'torchscript', etc.)
            output_path (str, optional): Path to save the exported model.
            
        Returns:
            str: Path to the exported model.
        """
        if self.model is None:
            print("Error: No model loaded. Train or load a model first.")
            return None
        
        try:
            # Export the model
            path = self.model.export(format=format, output=output_path)
            print(f"Model exported to {path}")
            return path
        except Exception as e:
            print(f"Error exporting model: {str(e)}")
            return None


# Module-level functions for backward compatibility
def calculate_average_img_size(input_dir):
    """
    Calculate the average image size in a directory.
    
    Args:
        input_dir (str): Directory containing images.
        
    Returns:
        int: Recommended image size for training.
    """
    analyzer = GlifosAnalyzer()
    return analyzer.calculate_average_img_size(input_dir)

def train():
    """
    Train a YOLOv8 model for glyph detection.
    
    Returns:
        object: Training results.
    """
    analyzer = GlifosAnalyzer()
    return analyzer.train()

def predict():
    """
    Perform prediction using a trained model.
    
    Returns:
        object: Prediction results.
    """
    analyzer = GlifosAnalyzer()
    return analyzer.predict()


if __name__ == "__main__":
    # Example usage
    analyzer = GlifosAnalyzer()
    
    # Train a model
    # analyzer.train()
    
    # Predict using the trained model
    analyzer.predict()
