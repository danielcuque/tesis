import cv2
import numpy as np

from utils import utils

def main():
    # path de la imagen
    image_path = 'assets/dataset_vasijas/1.png'

    color_to_detect = '#A98875'

    # leer imagen
    image = cv2.imread(image_path)

    # convertir a hsv
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_bound, upper_bound = utils.generate_color_range(color_to_detect)

    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

    original_resized = utils.resize_image(image, 780, 540)
    mask_resized = utils.resize_image(mask, 780, 540)

    cv2.imshow('original', original_resized)
    cv2.imshow('mask', mask_resized)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return
