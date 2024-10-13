import cv2
import numpy as np

def main():

    # Cargar la red YOLO preentrenada
    net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # Cargar las etiquetas (nombres de los objetos que YOLO puede detectar)
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    # Cargar la imagen donde se quiere detectar el perro
    image_path = 'assets/dataset_vasijas/1.png'
    image = cv2.imread(image_path)
    height, width, channels = image.shape

    # Preprocesar la imagen para YOLO
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Listas para guardar los detalles de los objetos detectados
    class_ids = []
    confidences = []
    boxes = []

    # Procesar los resultados de YOLO
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > 0.5:  # Confianza mínima de detección
                # Obtenemos las coordenadas del objeto
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                
                # Definir el cuadro delimitador
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Usamos Non-maxima suppression para eliminar múltiples cajas delimitadoras del mismo objeto
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Dibujar las cajas delimitadoras y etiquetas
    for i in range(len(boxes)):
        if i in indices:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            
            if label == "dog":  # Verificar si el objeto detectado es un perro
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(image, f'{label} {confidence:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Mostrar la imagen con las detecciones
    cv2.imshow('Imagen con Detección', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Guardar la imagen con las detecciones
    cv2.imwrite('resultado_deteccion_perro.jpg', image)
