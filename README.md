# Proyecto: Detección de Atención en Video y Tiempo Real

Este proyecto utiliza **Deep Learning** y **Visión Computacional** para detectar el estado de atención ("atento" o "desatento") de una persona usando webcam o video pregrabado.

# INTEGRANTES:
 - Donayre Alvarez, Jose 
 - Fernandez Gutierrez, Valentin
 - Leon Rojas, Franco 
 - Moreno Quevedo, Camila
 - Valera Flores, Lesly
## Estructura de Archivos

- `app.py` — Detección en tiempo real usando webcam. Muestra en pantalla el estado y FPS.
- `video_main.py` — Procesa videos MP4 y muestra el estado de atención frame a frame.
- `ENTRENAMIENTO_DE_MODELO.ipynb` — Notebook para preparar datos, entrenar y evaluar el modelo de atención.
- `requirements.txt` — Dependencias necesarias (OpenCV, TensorFlow, Numpy, etc).
- `modelos/` — Carpeta con modelos entrenados y archivos de detección de rostro:
    - `atencion_mnv2_final_mejorado.keras` — Modelo final de atención.
    - `face_detection_yunet_2023mar.onnx` — Detector de rostros YuNet.
- Videos de ejemplo: `aten.mp4`, `Atento.mp4`, `desa.mp4`, `desatento.mp4`.

## ¿Cómo funciona?

1. **Detección de rostro:** Se usa YuNet (ONNX) para localizar la cara en cada frame.
2. **Clasificación de atención:** Un modelo MobileNetV2 entrenado distingue entre "atento" y "desatento".
3. **Visualización:** Se muestra el estado en pantalla, junto con la probabilidad y FPS.

## Uso rápido

1. Instala dependencias:
   ```bash
   pip install -r requirements.txt
   ```
2. Ejecuta en tiempo real:
   ```bash
   python app.py
   ```
3. Procesa un video:
   ```bash
   python video_main.py
   ```

## Entrenamiento de modelo

- Usa el notebook `ENTRENAMIENTO_DE_MODELO.ipynb` para:
    - Extraer frames de videos etiquetados.
    - Preparar dataset balanceado.
    - Entrenar y ajustar el modelo.
    - Guardar el modelo final en `modelos/`.

## Requisitos

- Python 3.8+
- OpenCV 4.7+
- TensorFlow 2.x
- Numpy

## Créditos y referencias

- Detector de rostros: [YuNet - OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)
- Backbone: MobileNetV2 (preentrenado en ImageNet)

---


