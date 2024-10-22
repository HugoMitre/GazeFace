import mediapipe as mp
import cv2
import rotaciones, distancia

# INSTALAR: 
# mediapipe 0.10.14
# opencv-python 4.10.0.84
# numpy 2.1.0


mp_face_mesh = mp.solutions.face_mesh  # inicializa el modelo face mesh 
# streaming de la camara:
cap = cv2.VideoCapture(0)  # elige el indice de la camara (prueba 1, 2, 3)
with mp_face_mesh.FaceMesh(
        max_num_faces=1,  # numero de caras a trastrear en cada frame 
        refine_landmarks=True,  # incluye los landmarks del iris en el modelo face mesh  
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6) as face_mesh:
    while cap.isOpened():
        success, image = cap.read()
        if not success:  # no hay entrada de frames 
            print("Ignoring empty camera frame.")
            continue
        # Para mejorar el rendimiento, opcionalmente marque la imagen como no grabable para pasarla como referencia.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # frame a RGB para el modelo face-mesh 
        results = face_mesh.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # regresa el frame a BGR para OpenCV 
        if results.multi_face_landmarks:
            rotaciones.gaze(image, results.multi_face_landmarks[0])
            distancia.calculate_distance(image,results.multi_face_landmarks[0],results)

        cv2.imshow('output window', image)
        if cv2.waitKey(2) & 0xFF == 27:
            break
cap.release()