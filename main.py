"""
PROYECTO: Air Drawing MNIST - Reconocimiento de Números en el Aire
PROPÓSITO:
Este script permite al usuario dibujar números en el aire utilizando la punta del dedo índice
frente a una webcam. El dibujo se procesa en tiempo real mediante una red neuronal profunda.

TECNOLOGÍAS UTILIZADAS:
- MediaPipe: Para el seguimiento preciso de los puntos de referencia de la mano.
- TensorFlow/Keras: Para la ejecución de la red neuronal convolucional (CNN) entrenada.
- OpenCV: Para la captura de vídeo, procesamiento de imagen y visualización en pantalla.
- NumPy: Para la gestión de matrices de imagen y normalización de datos.

FLUJO DE EJECUCIÓN:
1. Inicialización: Carga del modelo de IA y configuración de los detectores de manos.
2. Captura: Lectura de frames de la webcam y conversión a formato adecuado (RGB).
3. Seguimiento: Detección de la punta del dedo índice y determinación de estados (dibujar/borrar).
4. Procesamiento: Identificación de contornos en el lienzo y extracción del área de dibujo.
5. Inferencia: Clasificación del número dibujado mediante la red neuronal.
6. Visualización: Superposición del dibujo e información de la predicción en el vídeo.
"""

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

# Definición del mapa de etiquetas para los números del 0 al 9
label_map = "0123456789"

try:
    # Intento de carga del modelo de red neuronal previamente entrenado
    model = tf.keras.models.load_model('modelo_caracteres.h5')
    print("Modelo MNIST cargado con éxito.")
except:
    # Gestión de error en caso de que el archivo del modelo no esté disponible
    print("Error: No se encontró 'modelo_caracteres.h5'.")
    model = None

# Configuración de las utilidades de MediaPipe para el seguimiento de manos
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8, min_tracking_confidence=0.8)
mp_draw = mp.solutions.drawing_utils

# Apertura de la cámara y creación del lienzo negro virtual donde se guardará el trazo
cap = cv2.VideoCapture(0)
canvas = np.zeros((480, 640, 3), dtype=np.uint8)

# Variables de control para mantener la continuidad del trazo y colores
prev_x, prev_y = 0, 0
draw_color = (0, 255, 0) # Color verde para el dibujo
prediction_text = ""

# Inicio del ciclo de captura y procesamiento continuo de vídeo
while True:
    ret, frame = cap.read()
    if not ret: break
    # Inversión de la imagen para que el usuario se vea como frente a un espejo
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    
    # Preparación de la imagen para que el sistema de detección de manos pueda leerla
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    # Verificación de si se ha detectado alguna mano en la escena
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            lm = hand_landmarks.landmark
            # Localización de las coordenadas en píxeles de la punta del dedo índice
            cx, cy = int(lm[8].x * w), int(lm[8].y * h)
            
            # Cálculo de qué dedos están levantados comparando la altura de las puntas
            tips = [8, 12, 16, 20]
            fingers = [lm[tip].y < lm[tip - 2].y for tip in tips]
            finger_count = sum(fingers)

            # Lógica para decidir si dibujar, borrar o mantenerse en reposo
            if finger_count == 1: # Si solo el dedo índice está arriba, empezamos a pintar
                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = cx, cy
                # Unión de la posición anterior con la actual mediante una línea continua
                cv2.line(canvas, (prev_x, prev_y), (cx, cy), draw_color, 15)
                prev_x, prev_y = cx, cy
            
            elif finger_count >= 3: # Si hay tres o más dedos, limpiamos todo el lienzo
                canvas = np.zeros_like(canvas)
                prediction_text = ""
                prev_x, prev_y = 0, 0
            
            else: # Reinicio de posiciones cuando la mano no está en modo dibujo
                prev_x, prev_y = 0, 0

            # Dibujo visual de los puntos de la mano detectados sobre la imagen real
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Identificación del número dibujado en el lienzo virtual
    gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    # Búsqueda de formas cerradas o trazos dentro de la imagen gris
    contours, _ = cv2.findContours(gray_canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Procedimiento para pasar el dibujo a la inteligencia artificial
    if contours and model:
        largest_contour = max(contours, key=cv2.contourArea)
        # Filtro para ignorar dibujos demasiado pequeños o accidentales
        if cv2.contourArea(largest_contour) > 500:
            x, y, cw, ch = cv2.boundingRect(largest_contour)
            
            # Recorte del área donde está el número añadiendo un margen de seguridad
            margin = 30
            roi = gray_canvas[max(0,y-margin):min(h,y+ch+margin), max(0,x-margin):min(w,x+cw+margin)]
            
            if roi.size > 0:
                # Ajuste del dibujo al tamaño que espera la red neuronal (28x28 píxeles)
                roi_resized = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
                # Normalización de los colores para que varíen entre 0 y 1
                roi_normalized = roi_resized / 255.0
                # Reorganización de los datos para que coincidan con la entrada del modelo
                roi_input = np.reshape(roi_normalized, (1, 28, 28, 1))
                
                # Ejecución de la IA para obtener la probabilidad de cada número
                prediction = model.predict(roi_input, verbose=0)
                idx = np.argmax(prediction)
                conf = np.max(prediction)
                
                # Solo informamos el resultado si la inteligencia artificial está segura
                if conf > 0.8:
                    prediction_text = f"Numero: {label_map[idx]} ({int(conf*100)}%)"
                    cv2.rectangle(frame, (x, y), (x + cw, y + ch), (255, 0, 0), 2)

    # Creación de una imagen mixta sumando la realidad y el lienzo de dibujo
    combined = cv2.addWeighted(frame, 0.7, canvas, 0.3, 0)
    
    # Escritura del nombre del número detectado en la parte superior de la pantalla
    cv2.putText(combined, prediction_text, (20, 50), 
                cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2)

    # Apertura de la ventana principal para mostrar todo el sistema funcionando
    cv2.imshow("Air Drawing MNIST - Detector de Numeros", combined)

    # Control de salida del programa al pulsar la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberación de recursos de cámara y cierre de ventanas al terminar
cap.release()
cv2.destroyAllWindows()