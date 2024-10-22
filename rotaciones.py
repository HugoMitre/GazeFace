import cv2
import numpy as np
from helpers import relative, relativeT



def gaze(frame, points):
    """
    La función de mirada obtiene una imagen y puntos de referencia del rostro del marco de MediaPipe.
    La función dibuja la dirección de la mirada en el marco.
    """

    '''
    Puntos 2D de la imagen.
    relative toma los puntos de mediapipe que son normalizados [-1, 1] y regresa los puntos
    de la imagen con un formato (x,y)
    '''
    image_points = np.array([
        relative(points.landmark[4], frame.shape),  # Punta de Nariz
        relative(points.landmark[152], frame.shape),  # Barbilla
        relative(points.landmark[263], frame.shape),  # Esquina del ojo izquierdo 
        relative(points.landmark[33], frame.shape),  # Esquina del ojo derecho 
        relative(points.landmark[287], frame.shape),  # Esquina Izquierda de la boca
        relative(points.landmark[57], frame.shape)  # Esquina derecha de la boca 
    ], dtype="double")

    '''
    Puntos 2D de la imagen.
    relativeT toma los puntos de Mediapipe que son normalizados a [-1, 1] y regresa los puntos de la imagen en 
    el formato (x,y,0).
    '''
    image_points1 = np.array([
        relativeT(points.landmark[4], frame.shape),  # Punta de Nariz
        relativeT(points.landmark[152], frame.shape),  # Barbilla
        relativeT(points.landmark[263], frame.shape),  # Esquina del ojo izquierdo
        relativeT(points.landmark[33], frame.shape),  # Esquina del ojo derecho
        relativeT(points.landmark[287], frame.shape),  # Esquina Izquierda de la boca
        relativeT(points.landmark[57], frame.shape)  # Esquina derecha de la boca
    ], dtype="double")

    # puntos con modelo 3D .
    model_points = np.array([
        (0.0, 0.0, 0.0),  # Nose tip
        (0, -63.6, -12.5),  # Chin
        (-43.3, 32.7, -26),  # Left eye, left corner
        (43.3, 32.7, -26),  # Right eye, right corner
        (-28.9, -28.9, -24.1),  # Left Mouth corner
        (28.9, -28.9, -24.1)  # Right mouth corner
    ])

    '''
    Puntos del modelo 3D 
    El centro de la bola del ojo y centro de la nariz
    '''
    Eye_ball_center_right = np.array([[-29.05], [32.7], [-39.5]]) # El centro de la bola del ojo der como un vector.
    Eye_ball_center_left = np.array([[29.05], [32.7], [-39.5]])  # El centro de la bola del ojo izq como un vector.
    Nose_center = np.array([[0.0], [0.0], [-29.05]])  # El centro de la nariz como un vector (X, Y, Z) Z Es la profundidad.
    '''
    Estimacion de la matriz camara
    '''
    focal_length = frame.shape[1]
    center = (frame.shape[1] / 2, frame.shape[0] / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )

    dist_coeffs = np.zeros((4, 1))  # Suponiendo que no hay distorsión de la lente
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                  dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE) # devuelve un vector de rotación y traslación y por lo tanto una transformación que nos ayudará a proyectar un punto desde el punto del mundo 3D al plano 2D

    # Localizacion 2D de la pupila
    left_pupil = relative(points.landmark[468], frame.shape)
    right_pupil = relative(points.landmark[473], frame.shape)
    
    # Localizacion 2D de la Nariz
    Nose = relative(points.landmark[4], frame.shape)

    # Transformación del punto de imagen al punto del mundo
    _, transformation, _ = cv2.estimateAffine3D(image_points1, model_points)  # De la imagen a la transformación del mundo

    if transformation is not None:  # Si estimateAffine3D fue exitoso
        ################ MIRADA ###################
        ################ OJO IZQUIERDO ##################
        # Proyecta el punto de la pupila de la imagen dentro de un punto en 3D
        pupil_world_cord = transformation @ np.array([[left_pupil[0], left_pupil[1], 0, 1]]).T

        # Punto de la mirada en 3D (7 es un valor arbitrario que indica la distancia de la mirada)
        S = Eye_ball_center_left + (pupil_world_cord - Eye_ball_center_left) * 7

        # Proyecta una dirección de mirada 3D sobre una imagen plana.
        (eye_pupil2D_left, _) = cv2.projectPoints((int(S[0]), int(S[1]), int(S[2])), rotation_vector,
                                              translation_vector, camera_matrix, dist_coeffs)
        
        
        # Coordenadas del centro del ojo (punto de referencia) y coordenadas 3D de la pupila
        eye_center = Eye_ball_center_left  # O Eye_ball_center_right para el ojo derecho
        pupil_direction = pupil_world_cord - eye_center

        # Calcular los ángulos de rotación a lo largo de los ejes X, Y y Z
        def calculateEulerFromDirection(direction_vector):
            x = np.arctan2(direction_vector[1], direction_vector[2])  # Rotación en el eje X
            y = np.arctan2(direction_vector[0], np.sqrt(direction_vector[1]**2 + direction_vector[2]**2))  # Rotación en el eje Y
            z = np.arctan2(direction_vector[0], direction_vector[1])  # Rotación en el eje Z
            return np.degrees(np.array([x, y, z]))

        # Obtener los ángulos de rotación
        eulerAngles = calculateEulerFromDirection(pupil_direction)

        REL = 'Rotacion de Ojo Izquierdo: X={:.2f}, Y={:.2f}, Z={:.2f}'.format(float(eulerAngles[0]), float(eulerAngles[1]), float(eulerAngles[2])
)

        # REL = 'Rotacion de Ojo Izquierdo: X={:.2f}, Y={:.2f}, Z={:.2f}'.format(rotation_vector[0][0], rotation_vector[1][0], rotation_vector[2][0])
    

        # Proyecta una direccion de la posicion de la cabeza sobre una imagen plana
        (head_pose, _) = cv2.projectPoints((int(pupil_world_cord[0]), int(pupil_world_cord[1]), int(40)),
                                           rotation_vector,
                                           translation_vector, camera_matrix, dist_coeffs)
        
        # Corrige la mirada para rotacion de la cabeza
        gaze = left_pupil + (eye_pupil2D_left[0][0] - left_pupil) - (head_pose[0][0] - left_pupil)

        # Draw gaze line into screen
        p1_left_eye = (int(left_pupil[0]), int(left_pupil[1]))
        p2_left_eye = (int(gaze[0]), int(gaze[1]))
        cv2.line(frame, p1_left_eye, p2_left_eye, (0, 255, 0), 2)


        #REL = 'Rotacion de Ojo Izquierdo: X={:.2f}, Y={:.2f}, Z={:.2f}'.format(left_eye_angle_x_deg, left_eye_angle_y_deg, left_eye_angle_z_deg)

         ################ OJO DERECHO ##################
        # Proyecta el punto de la pupila de la imagen dentro de un punto en 3D
        pupil_world_cord_r = transformation @ np.array([[right_pupil[0], right_pupil[1], 0, 1]]).T

        # Punto de la mirada en 3D (7 es un valor arbitrario que indica la distancia de la mirada)
        Sr = Eye_ball_center_right + (pupil_world_cord_r - Eye_ball_center_right) * 7

        # Proyecta una dirección de mirada 3D sobre una imagen plana.
        (eye_pupil2D_right, _) = cv2.projectPoints((int(Sr[0]), int(Sr[1]), int(Sr[2])), rotation_vector,
                                               translation_vector, camera_matrix, dist_coeffs)
        
        # Proyecta una direccion de la posicion de la cabeza sobre una imagen plana
        (head_pose, _) = cv2.projectPoints((int(pupil_world_cord_r[0]), int(pupil_world_cord_r[1]), int(40)),
                                           rotation_vector,
                                           translation_vector, camera_matrix, dist_coeffs)
        

        # Coordenadas del centro del ojo (punto de referencia) y coordenadas 3D de la pupila
        eye_center = Eye_ball_center_right  # O Eye_ball_center_right para el ojo derecho
        pupil_direction_r = pupil_world_cord_r - eye_center

        # Calcular los ángulos de rotación a lo largo de los ejes X, Y y Z
        def calculateEulerFromDirection(direction_vector):
            x = np.arctan2(direction_vector[1], direction_vector[2])  # Rotación en el eje X
            y = np.arctan2(direction_vector[0], np.sqrt(direction_vector[1]**2 + direction_vector[2]**2))  # Rotación en el eje Y
            z = np.arctan2(direction_vector[0], direction_vector[1])  # Rotación en el eje Z
            return np.degrees(np.array([x, y, z]))

        # Obtener los ángulos de rotación
        eulerAngles_r = calculateEulerFromDirection(pupil_direction_r)

        RER = 'Rotacion de Ojo Derecho: X={:.2f}, Y={:.2f}, Z={:.2f}'.format(float(eulerAngles_r[0]), float(eulerAngles_r[1]), float(eulerAngles_r[2]))
                                                                               
        # Corrige la mirada para rotacion de la cabeza
        gaze = right_pupil + (eye_pupil2D_right[0][0] - right_pupil) - (head_pose[0][0] - right_pupil)

        # Draw gaze line into screen
        p1_right_eye = (int(right_pupil[0]), int(right_pupil[1]))
        p2_right_eye = (int(gaze[0]), int(gaze[1]))
        cv2.line(frame, p1_right_eye, p2_right_eye, (255, 0, 0), 2)

        ################ HEAD ################
        # Proyecta el punto de la nariz de la imagen dentro de un punto en 3D
        pupil_world_cord_h = transformation @ np.array([[Nose[0], Nose[1], 0, 1]]).T

        # Punto de la mirada en 3D (7 es un valor arbitrario que indica la distancia de la mirada)
        H = Nose_center + (pupil_world_cord_h - Nose_center) * 7

        # Proyecta una dirección de la cabeza 3D sobre una imagen plana.
        (head_pose, _) = cv2.projectPoints(Nose_center, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
        

        (head_pose_h, _) = cv2.projectPoints((int(pupil_world_cord_h[0]), int(pupil_world_cord_h[1]), int(40)),
                                           rotation_vector,
                                           translation_vector, camera_matrix, dist_coeffs)
        
         # Coordenadas del centro del ojo (punto de referencia) y coordenadas 3D de la pupila
        eye_center = Nose_center  # O Eye_ball_center_right para el ojo derecho
        pupil_direction_h = pupil_world_cord_h - Nose_center

        # Calcular los ángulos de rotación a lo largo de los ejes X, Y y Z
        def calculateEulerFromDirection(direction_vector):
            x = np.arctan2(direction_vector[1], direction_vector[2])  # Rotación en el eje X
            y = np.arctan2(direction_vector[0], np.sqrt(direction_vector[1]**2 + direction_vector[2]**2))  # Rotación en el eje Y
            z = np.arctan2(direction_vector[0], direction_vector[1])  # Rotación en el eje Z
            return np.degrees(np.array([x, y, z]))

        # Obtener los ángulos de rotación
        eulerAngles_h = calculateEulerFromDirection(pupil_direction_h)

        RH = 'Rotacion de Cabeza: X={:.2f}, Y={:.2f}, Z={:.2f}'.format(float(eulerAngles_h[0]), float(eulerAngles_h[1]), float(eulerAngles_h[2]))

        

        # Corrige la direccion para rotacion de la cabeza
        HeadPoint = Nose + (head_pose_h[0][0] - Nose) - (head_pose[0][0] - Nose)

        # Dibuja la linea en la imagen
        p1_head = (int(Nose[0]), int(Nose[1]))
        p2_head = (int(HeadPoint[0]), int(HeadPoint[1]))
        cv2.line(frame, p1_head, p2_head, (0, 0, 150), 2)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        # Imprime en pantalla los valores de rotacion de: mirada del ojo izquierdo, mirada del ojo derecho y del movimiento de cabeza.
        cv2.putText(frame, RH, (50, 50), font, 1, (0, 0, 150), 2, cv2.LINE_4)
        cv2.putText(frame, RER, (50, 100), font, 1, (255, 0, 0), 2, cv2.LINE_4)
        cv2.putText(frame, REL, (50, 150), font, 1, (0, 255, 0), 2, cv2.LINE_4) 

