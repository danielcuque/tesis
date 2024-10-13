import os
import cv2
import numpy as np
from utils import utils

def main():
    # Crear el output folder si no existe
    output_folder = './assets/results/vasijas/'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Directorio de las imágenes a procesar
    input_folder = './assets/dataset_vasijas/'

    # Colores a detectar
    color_to_detect = '#A98875'
    # color_to_detect = '#B58C7E'

    # Obtener lista de archivos en el directorio de entrada
    image_files = [f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

    # Recorrer todos los archivos de imagen
    for image_file in image_files:
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

        # Obtener límites de color
        lower_bound, upper_bound = utils.generate_color_range(color_to_detect, 10, 51, 51)

        # Crear máscara
        mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

        # Redimensionar imágenes
        original_resized = utils.resize_image(image, 780, 540)
        mask_resized = utils.resize_image(mask, 780, 540)

        # Encontrar contornos en la máscara
        contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Contar la cantidad de figuras
        num_figures = len(contours)

        # Dibujar los contornos en la imagen original
        contour_image = original_resized.copy()
        cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)  # Dibujar contornos en verde

        # Agregar un label con la cantidad de figuras detectadas
        text = f'Cantidad de restos encontrados: {num_figures}'
        cv2.putText(contour_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Guardar la imagen procesada en el directorio de salida con el mismo nombre que la original
        output_path = os.path.join(output_folder, image_file)
        cv2.imwrite(output_path, contour_image)

        # Mostrar la cantidad de figuras detectadas en consola
        print(f'Imagen: {image_file} - Cantidad de figuras detectadas: {num_figures}')

    # Terminar el procesamiento
    print('Procesamiento completado.')
