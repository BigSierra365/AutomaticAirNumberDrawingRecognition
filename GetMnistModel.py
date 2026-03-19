"""
PROYECTO: Generador de Modelo de IA para Reconocimiento de Números
PROPÓSITO:
Este script se encarga de descargar el dataset MNIST, construir una red neuronal
convolucional (CNN) y entrenarla para que sea capaz de identificar dígitos del 0 al 9.

TECNOLOGÍAS UTILIZADAS:
- TensorFlow: Framework principal para creación y entrenamiento de redes neuronales.
- Keras: Interfaz de alto nivel utilizada para definir la arquitectura del modelo.
- NumPy: Utilizado internamente para el manejo de los arrays de datos del dataset.

FLUJO DE EJECUCIÓN:
1. Carga: Descarga automática de miles de imágenes de números escritos a mano.
2. Procesamiento: Ajuste de escala, dimensiones y normalización de los píxeles.
3. Construcción: Diseño de las capas ocultas de la red neuronal (Convoluciones y Filtrado).
4. Entrenamiento: Aplicación de los datos a la red para que aprenda patrones.
5. Exportación: Guardado del conocimiento del modelo en un archivo reutilizable (.h5).
"""

import tensorflow as tf
from tensorflow.keras import layers, models

# Descarga y carga de las imágenes de entrenamiento y prueba desde el repositorio oficial
print("Cargando dataset MNIST (Números 0-9)...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Preparación matemática de los datos para que la red neuronal los procese mejor
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255

# Definición de la estructura de capas que forman el "cerebro" del modelo de IA
print("Configurando modelo para 10 clases...")
model = models.Sequential([
    # Especificamos el tamaño de entrada de la imagen (28x28 píxeles en blanco y negro)
    layers.Input(shape=(28, 28, 1)),
    # Capas de convolución para detectar bordes y formas simples
    layers.Conv2D(32, (3, 3), activation='relu'),
    # Capa de reducción para optimizar la velocidad y evitar errores
    layers.MaxPooling2D((2, 2)),
    # Capa adicional para detectar formas más complejas
    layers.Conv2D(64, (3, 3), activation='relu'),
    # Aplanado de la imagen para convertirla en una lista de valores lineal
    layers.Flatten(),
    # Capas neuronales densas para la decisión final
    layers.Dense(128, activation='relu'),
    # Capa de salida con 10 opciones correspondientes a los números del 0 al 9
    layers.Dense(10, activation='softmax')
])

# Configuración técnica del método que usará la red para corregir sus propios errores
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# Inicio del proceso de aprendizaje mediante la repetición de ciclos sobre el dataset
print("Entrenando (3 épocas son suficientes para MNIST)...")
model.fit(x_train, y_train, epochs=3, batch_size=128)

# Almacenamiento del modelo completo en el disco duro para su uso posterior
model.save('modelo_caracteres.h5')
print("\n¡ÉXITO! El archivo 'modelo_caracteres.h5' (solo números) está listo.")
