# ✍️ Air MNIST: Reconocimiento Automático de Números en el Aire



![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)

![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-FF6F00.svg?logo=tensorflow)

![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10-00C5FF.svg)

![OpenCV](https://img.shields.io/badge/OpenCV-4.13-5C3EE8.svg?logo=opencv)

![License](https://img.shields.io/badge/license-MIT-green.svg)



## 🚀 Descripción / Propósito

**Air MNIST** es un sistema de visión artificial interactivo que permite a los usuarios dibujar números (del 0 al 9) en el aire utilizando únicamente su dedo índice apuntando a una cámara web.

El propósito principal es demostrar la integración fluida entre algoritmos de seguimiento de puntos de referencia de la mano (*Hand Tracking*) en tiempo real y redes neuronales convolucionales (CNN) especializadas en clasificación de imágenes. Es la evolución natural del clásico dataset MNIST llevado a un entorno dinámico y *contactless*.



## ⚙️ Stack Tecnológico



| Dominio | Tecnología | Propósito |

|---------|:-----------|:----------|

| **Frontend / Visión** | **OpenCV** & **MediaPipe** | Captura de vídeo, seguimiento de la mano (21 landmarks) y renderizado HUD superpuesto. |

| **IA / Deep Learning** | **TensorFlow** & **Keras** | Inferencia del modelo de red neuronal (clasificador de dígitos de 10 clases). |

| **Data / Numérico** | **NumPy** | Manipulación de matrices, normalización del ROI (Region of Interest) y preprocesamiento matemático. |



## ✨ Características Principales

- **Interacción Sin Contacto (Air Drawing):** Dibuja números frente a la cámara usando la punta de tu dedo índice.

- **Borrado Inteligente:** Levanta 3 o más dedos simultáneamente para limpiar el lienzo de forma instantánea.

- **Procesamiento ROI Dinámico:** Detección de contornos automáticos para extraer y redimensionar matemáticamente el dibujo a 28x28 píxeles.

- **Inferencia en Tiempo Real:** Realiza predicciones por *frame* con un umbral estricto de confianza (>80%) para evitar falsos positivos.



## 🧠 Arquitectura y Lógica

El flujo de datos del sistema está optimizado para baja latencia:

1. **Captura y Tracking:** `cv2.VideoCapture` obtiene el frame. MediaPipe extrae la coordenada del *landmark 8* (punta del índice).

2. **Máquina de Estados:** Se cuentan los dedos levantados.

   - *1 dedo:* Se interpolan las posiciones anteriores y actuales dibujando sobre un lienzo virtual (matriz genérica).

   - *≥3 dedos:* Se limpia la matriz por completo.

3. **Computer Vision (Pipeline ROI):** Se aplica `cv2.findContours` al lienzo. El trazado más grande se encapsula en una Bounding Box (con un margen de relleno de seguridad) para aislar el número dibujado.

4. **Predictive Layer:** La ROI extraída se redimensiona a (28, 28) y se normaliza dividiendo entre 255.0. El modelo CNN procesa el tensor y devuelve la etiqueta asociada a la mayor probabilidad mediante `argmax`.



## 🎥 Demostración


https://github.com/user-attachments/assets/3d9a3e69-4b61-46d0-b21a-f6c802ca7f7b




## ⚠️ Prerrequisitos de Hardware y Modelos

- **Hardware (GPU/CPU):** El modelo CNN preentrenado (`modelo_caracteres.h5`) es extremadamente ligero. **No requiere GPU** (CUDA) para funcionar a +30 FPS de forma nativa. Solo se necesita una **cámara web** activa.

- **Modelo de Pesos (.h5): El modelo entrenado(`modelo_caracteres.h5`) está incluido en el repositorio. Aunque se puede generar localmente ejecutando el script de entrenamiento integrado antes de lanzar la aplicación principal.



## 💻 Instalación (Plug & Play)



1. **Clonar el repositorio:**

   ```bash

   git clone https://github.com/BigSierra365/AutomaticAirNumberDrawingRecognition.git

   cd AutomaticAirNumberDrawingRecognition

   ```



2. **Crear y activar entorno virtual (Recomendado):**

   ```bash

   python -m venv .venv

   

   # En Windows:

   .venv\Scripts\activate

   

   # En Linux/Mac:

   source .venv/bin/activate

   ```



3. **Instalar dependencias:**

   ```bash

   pip install -r requirements.txt

   ```



## 🎮 Uso del Sistema



Los archivos de entrenamiento e inferencia se encuentran en la raíz del proyecto.



**Paso 1: Generar el modelo de IA**

Dado que el archivo de pesos no fue subido al control de versiones, primero debes instanciar y entrenar la red neuronal localmente.

```bash

python GetMnistModel.py

```

*(Este script descargará el dataset MNIST oficial y tras 3 épocas de entrenamiento, generará el archivo `modelo_caracteres.h5` en tu entorno local).*



**Paso 2: Lanzar el Detector Interactivo**

Una vez localizado el modelo `.h5` en la raíz, puedes ejecutar el reconocedor visual.

```bash

python main.py

```



**Comandos de la aplicación en vivo:**

- **Index Up (1 dedo):** Dibujar / Pintar.

- **Palma abierta (≥3 dedos):** Borrar lienzo inmediatamente.

- **Tecla `q`:** Finalizar el entorno interactivo de OpenCV y cerrar sesión."
