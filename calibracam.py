import cv2
import numpy as np


def calibra():
    # Tamaño del tablero de ajedrez: 9x6 (se puede ajustar según el tablero)
    chessboard_size = (9, 6)

    # Tamaño real de cada casilla en el tablero (en cm o cualquier unidad que estés usando)
    square_size = 2.2  # Tamaño de cada casilla (por ejemplo, 2.5 cm). Toma la distancia en cm de un lado de los cuadros

    # Crea los puntos 3D en el espacio real del tablero de ajedrez
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp *= square_size  # Escala según el tamaño de las casillas

    # Listas para almacenar los puntos del mundo real (3D) y de la imagen (2D)
    obj_points = []  # Puntos 3D del mundo real
    img_points = []  # Puntos 2D de la imagen

    # Captura de video desde la cámara
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detecta las esquinas del tablero de ajedrez
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

        if ret:
            obj_points.append(objp)
            img_points.append(corners)

            # Dibuja el tablero de ajedrez en la imagen
            frame = cv2.drawChessboardCorners(frame, chessboard_size, corners, ret)

        cv2.imshow('Tablero de Ajedrez', frame)

        # Rompe el bucle con la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)
            return
            #break

    #cap.release()
    #cv2.destroyAllWindows()

    # Calibración de la cámara
    