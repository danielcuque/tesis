import os
import cv2
import numpy as np
from utils import utils

def filter_contours_by_aspect_ratio(contours, min_ratio=0.2, max_ratio=4.0, min_area=10, max_area=1000):
    filtered_contours = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h if h != 0 else 0
        area = cv2.contourArea(contour)
        
        if min_ratio <= aspect_ratio <= max_ratio and min_area <= area <= max_area:
            filtered_contours.append(contour)
    
    return filtered_contours

def save_image(output_folder, filename, image):
    """Función auxiliar para guardar la imagen en el directorio."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    cv2.imwrite(os.path.join(output_folder, filename), image)

def main():
    # Crear el output folder general si no existe
    output_folder = './assets/results/vasijas/'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Directorio de las imágenes a procesar
    input_folder = './assets/dataset_vasijas/'

    # Colores a detectar
    color_to_detect = '#A98876'

    # Obtener lista de archivos en el directorio de entrada
    image_files = [f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

    # Recorrer todos los archivos de imagen
    for image_file in image_files:
        # Crear carpeta específica para cada imagen
        image_name = os.path.splitext(image_file)[0]  # Quitar extensión del nombre de la imagen
        image_folder = os.path.join(output_folder, image_name)
        if not os.path.exists(image_folder):
            os.makedirs(image_folder)

        # Path completo de la imagen
        image_path = os.path.join(input_folder, image_file)

        # Leer imagen
        image = cv2.imread(image_path)

        # Verificar si la imagen se pudo leer correctamente
        if image is None:
            print(f'Error al cargar la imagen: {image_path}')
            continue

        # Convertir a HSV
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        save_image(image_folder, 'hsv.png', hsv_image)

        # Obtener límites de color
        lower_bound, upper_bound = utils.generate_color_range(color_to_detect, 10, 50, 50)

        # Crear máscara
        mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

        # Redimensionar imágenes
        original_resized = utils.resize_image(image, 780, 540)
        
        mask_resized = utils.resize_image(mask, 780, 540)
        save_image(image_folder, 'mask.png', mask_resized)

        """ Aplicar operaciones morfológicas para suavizar los bordes """
        kernel = np.ones((5, 5), np.uint8)

        """ Dilatación """
        mask_dilated = cv2.dilate(mask_resized, kernel, iterations=1)
        save_image(image_folder, 'dilated.png', mask_dilated)

        """ Erosionada """
        mask_eroded = cv2.erode(mask_dilated, kernel, iterations=1)
        save_image(image_folder, 'eroded.png', mask_eroded)

        """ Suavizado gaussiano """
        mask_smoothed = cv2.GaussianBlur(mask_eroded, (5, 5), 0)
        save_image(image_folder, 'smooth.png', mask_smoothed)

        # Encontrar contornos en la máscara suavizada
        contours, _ = cv2.findContours(mask_smoothed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filtrar contornos por aspect ratio
        filtered_contours = filter_contours_by_aspect_ratio(contours)

        # Contar la cantidad de figuras después del filtrado
        num_figures = len(filtered_contours)

        # Dibujar los contornos filtrados en la imagen original
        contour_image = original_resized.copy()
        cv2.drawContours(contour_image, filtered_contours, -1, (0, 255, 0), 1)  # Dibujar contornos en verde

        # Guardar la imagen con contornos
        save_image(image_folder, 'contours.png', contour_image)

        # Agregar un label con la cantidad de figuras detectadas
        text = f'Cantidad de restos encontrados: {num_figures}'
        cv2.putText(contour_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Guardar la imagen final con el texto
        save_image(image_folder, 'final_result.png', contour_image)

        # Mostrar la cantidad de figuras detectadas en consola
        print(f'Imagen: {image_file} - Cantidad de figuras detectadas: {num_figures}')

    # Terminar el procesamiento
    print('Procesamiento completado.')

if __name__ == "__main__":
    main()
