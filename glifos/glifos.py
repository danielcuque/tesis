from ultralytics import YOLO
import os

def train():
    print(f"Current working directory: {os.getcwd()}")
    # Entrenamiento de un modelo YOLOv8
    model = YOLO('yolov8n.yaml')  # Usa un modelo YOLOv8 pequeño como base
    results = model.train(data='glifos/config.yaml', epochs=1, imgsz=640)  # Entrena con datos de glifos
    return results

def predict():
    # Cargar el modelo previamente entrenado
    model = YOLO('runs/detect/train/weights/best.pt')

    # Directorio de entrada y salida
    source_dir = 'assets/dataset_glifos/images/val'
    output_dir = 'assets/dataset_glifos/images/val/predictions'

    # Crear el directorio de salida si no existe
    os.makedirs(output_dir, exist_ok=True)

    # Realizar predicciones
    model.predict(
        source=source_dir,  # Directorio o archivo de entrada
        save=True,          # Guarda automáticamente las imágenes procesadas
        project=output_dir, # Especifica el directorio base para guardar resultados
        conf=0.01,           # Umbral de confianza para las detecciones
        
    )
