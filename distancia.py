import cv2
import numpy as np
from helpers import relative



def calculate_distance(frame, points, resultsd):
    """
    Calcula la distancia entre la cámara y los ojos usando solo los puntos del iris de ambos ojos.
    """
    # Puntos del iris izquierdo y derecho
    left_iris_indices = [468, 469, 470, 471]  # Puntos del iris del ojo izquierdo
    right_iris_indices = [473, 474, 475, 476]  # Puntos del iris del ojo derecho

    # Parámetros de la cámara
    focal_length = frame.shape[1]  # La longitud focal es el ancho de la imagen
    real_iris_width_cm = 1.2  # Ancho promedio del iris humano en cm son 11.7mm +- 0.5mm

    def calculate_iris_distance(iris_indices):
        """
        Calcula la distancia promedio entre los puntos del iris de un ojo.
        """
        iris_points = np.array([relative(points.landmark[idx], frame.shape) for idx in iris_indices])

        # Calcular la distancia promedio entre los puntos del iris
        avg_pixel_distance = np.mean([np.linalg.norm(iris_points[i] - iris_points[j])
                                      for i in range(len(iris_points)) 
                                      for j in range(i + 1, len(iris_points))])
        return avg_pixel_distance

    # Calcular la distancia promedio para ambos ojos
    left_iris_pixel_distance = calculate_iris_distance(left_iris_indices)
    right_iris_pixel_distance = calculate_iris_distance(right_iris_indices)

    # Estimar la distancia entre la cámara y cada ojo (en cm)
    left_iris_distance_cm = (real_iris_width_cm * focal_length) / left_iris_pixel_distance
    right_iris_distance_cm = (real_iris_width_cm * focal_length) / right_iris_pixel_distance

    # Mostrar en la imagen las distancias
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_left = f'Distancia Ojo Izq: {left_iris_distance_cm:.2f} cm'
    text_right = f'Distancia Ojo Der: {right_iris_distance_cm:.2f} cm'
    
    cv2.putText(frame, text_left, (50, 200), font, 1, (26, 127, 239), 2, cv2.LINE_AA)
    cv2.putText(frame, text_right, (50, 250), font, 1, (26, 127, 239), 2, cv2.LINE_AA)



