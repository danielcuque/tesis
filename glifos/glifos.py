import os
import cv2
import numpy as np

def main():
    # Crear el output folder si no existe
    output_folder = './assets/results/glifos/'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Directorio de las imágenes a procesar
    input_folder = './assets/dataset_glifos/'

    # Obtener lista de archivos en el directorio de entrada
    image_files = [f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.jpeg', 'webp'))]

    # Recorrer todos los archivos de imagen
    for image_file in image_files:
        # Path completo de la imagen
        image_path = os.path.join(input_folder, image_file)

        # Leer imagen
        image = cv2.imread(image_path)

        # Convertir la imagen a escala de grises
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Convertir a escala de grises
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Aplicar suavizado Gaussiano
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Detección de bordes con Canny
        edges = cv2.Canny(blurred, 100, 150)
        
        # Mejorar bordes con operaciones morfológicas
        kernel = np.ones((3,3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        edges = cv2.erode(edges, kernel, iterations=1)
        
        # Umbralización adaptativa
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY, 11, 2)
        
        # Combinar bordes y umbralización
        result = cv2.bitwise_and(edges, thresh)

        # invert colors
        result = 255 - result

        # Guardar la imagen procesada en el directorio de salida
        output_path = os.path.join(output_folder, f'contours_{image_file}')
        cv2.imwrite(output_path, result)

        print(f'Procesada la imagen: {image_file}')

    # Terminar el procesamiento
    print('Procesamiento completado.')

if __name__ == "__main__":
    main()
