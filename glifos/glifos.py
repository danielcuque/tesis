from ultralytics import YOLO
import os
import cv2
import numpy as np

def calculate_average_img_size(input_dir):
    """
    Calcula el tamaño promedio de las imágenes en un directorio.
    Retorna el mayor valor (ancho o alto) para usar como imgsz.
    """
    widths, heights = [], []
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img = cv2.imread(os.path.join(input_dir, filename))
            if img is not None:
                h, w = img.shape[:2]
                heights.append(h)
                widths.append(w)
    if widths and heights:
        avg_width = int(np.mean(widths))
        avg_height = int(np.mean(heights))
        return max(avg_width, avg_height)
    else:
        raise ValueError("No se encontraron imágenes válidas en el directorio.")

def train():
    print(f"Current working directory: {os.getcwd()}")
    # Calcular el tamaño promedio de las imágenes del dataset
    # input_dir = 'assets/dataset_glifos/a'
    imgsz = 640
    print(f"Usando imgsz promedio: {imgsz}")

    # Entrenamiento de un modelo YOLOv8
    model = YOLO('yolov8n.yaml')  # Usa un modelo YOLOv8 pequeño como base
    results = model.train(data='glifos/config.yaml', epochs=50, imgsz=imgsz)  # Entrena con el tamaño calculado
    return results

def predict():
    # Cargar el modelo previamente entrenado
    model = YOLO('runs/detect/train/weights/best.pt')

    # Directorio de entrada y salida
    source_dir = 'assets/dataset_glifos/a'
    output_dir = 'assets/dataset_glifos/images'

    # Crear el directorio de salida si no existe
    os.makedirs(output_dir, exist_ok=True)

    # Realizar predicciones
    model.predict(
        source=source_dir,  # Directorio o archivo de entrada
        save=True,          # Guarda automáticamente las imágenes procesadas
        project=output_dir, # Especifica el directorio base para guardar resultados
        conf=0.01,         # Umbral de confianza para las detecciones
    )
