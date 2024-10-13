import cv2
import numpy as np

def resize_image(image, width, height):
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)

# Función para convertir HEX a BGR
def hex_to_bgr(hex_color):
    hex_color = hex_color.lstrip('#')
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return (rgb[2], rgb[1], rgb[0])  # Cambiar el orden a BGR

# Función para convertir BGR a HSV
def bgr_to_hsv(bgr_color):
    bgr_array = np.uint8([[bgr_color]])  # Crear un arreglo 2D
    hsv_color = cv2.cvtColor(bgr_array, cv2.COLOR_BGR2HSV)[0][0]  # Convertir a HSV
    return hsv_color

# Función para convertir HEX a HSV
def hex_to_hsv(hex_color):
    bgr_color = hex_to_bgr(hex_color)  # Convertir HEX a BGR
    hsv_color = bgr_to_hsv(bgr_color)  # Convertir BGR a HSV
    return hsv_color

# Función para generar un rango de colores
def generate_color_range(hex_color, hue_offset=10, saturation_offset=50, value_offset=50):
    base_hsv = hex_to_hsv(hex_color)  # Obtener el color base en HSV
    lower_bound = np.array([
        max(0, base_hsv[0] - hue_offset),   # Limite inferior para el matiz (H)
        max(0, base_hsv[1] - saturation_offset),  # Limite inferior para la saturación (S)
        max(0, base_hsv[2] - value_offset)  # Limite inferior para el valor (V)
    ])
    upper_bound = np.array([
        min(179, base_hsv[0] + hue_offset),  # Limite superior para el matiz (H)
        min(255, base_hsv[1] + saturation_offset),  # Limite superior para la saturación (S)
        min(255, base_hsv[2] + value_offset)  # Limite superior para el valor (V)
    ])
    return lower_bound, upper_bound

# Ejemplo de uso
hex_color = '#A98875'
lower_bound, upper_bound = generate_color_range(hex_color)

print(f"Lower Bound HSV: {lower_bound}")
print(f"Upper Bound HSV: {upper_bound}")
