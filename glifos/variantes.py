import os
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Definir el pipeline de aumento de datos
transform = A.Compose([
    A.Rotate(limit=30, p=1.0),  # Rotación entre -30 y 30 grados
    A.RandomScale(scale_limit=(0.8, 1.2), p=1.0),  # Escalado entre 0.8 y 1.2
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),  # Brillo y contraste
    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.0),  # Cambio de color
    A.GaussianBlur(blur_limit=3, p=0.3),  # Desenfoque Gaussiano
    A.RandomGamma(p=0.5),  # Ajuste gamma aleatorio
    A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),  # Mejora de contraste adaptativa CLAHE
    ToTensorV2()
])

def generate_variations(image_path, output_dir, num_variants=300):
    # Leer la imagen original
    img = cv2.imread(image_path)
    
    # Verificar que la imagen fue cargada correctamente
    if img is None:
        print(f"No se pudo cargar la imagen: {image_path}")
        return
    
    # Generar variantes
    for i in range(num_variants):
        augmented = transform(image=img)
        augmented_image = augmented['image']
        
        # Convertir el tensor a un array de NumPy (HWC)
        augmented_image = augmented_image.permute(1, 2, 0).cpu().numpy()
        
        # Asegurarse de que la imagen esté en el rango [0, 255] y tipo uint8
        augmented_image = np.clip(augmented_image * 255, 0, 255).astype(np.uint8)
        
        filename = os.path.basename(image_path)
        output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_variant_{i+1}.png")

        # Reemplazar las diagonales invertidas por diagonales hacia adelante
        output_path = output_path.replace("\\", "/")
        
        # Guardar la imagen generada
        cv2.imwrite(output_path, augmented_image)
        print(f"Imagen generada: {output_path}")

def process_directory(input_dir, output_dir, num_variants=300):
    # Crear directorio de salida si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    # Iterar sobre cada imagen en el directorio de entrada
    for filename in os.listdir(input_dir):
        # Asegurarse de que es una imagen
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_dir, filename)
            generate_variations(image_path, output_dir, num_variants)
