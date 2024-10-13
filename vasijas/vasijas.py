import os
import cv2
import numpy as np

from utils import utils

def main():
    # crear el output folder
    if not os.path.exists('./assets/results/vasijas/'):
        os.makedirs('./assets/results/vasijas/')
        
    # Path de la imagen
    image_path = 'assets/dataset_vasijas/1.png'

    color_to_detect = '#A98875'

    # Leer imagen
    image = cv2.imread(image_path)

    # Convertir a HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Obtener límites de color
    lower_bound, upper_bound = utils.generate_color_range(color_to_detect, 7, 50, 50)

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
    # color text: 
    cv2.putText(contour_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Mostrar las imágenes
    # cv2.imshow('original', original_resized)
    # cv2.imshow('mask', mask_resized)
    # cv2.imshow('contours', contour_image)

    # Mostrar la cantidad de figuras detectadas en consola
    print(f'Cantidad de figuras detectadas: {num_figures}')


    # Guardar imagen
    cv2.imwrite('./assets/results/vasijas/1.png', contour_image)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return
