# app_fast.py
import cv2, numpy as np, tensorflow as tf, time
from pathlib import Path
from urllib.request import urlretrieve

# ========= Config =========
MODEL_PATH = Path("modelos/atencion_mnv2_final_mejorado.keras")
IMG_SIZE = 224
UMBRAL = 0.615

# Cámara
CAM_INDEX   = 0
CAM_BACKEND = cv2.CAP_DSHOW  # prueba CAP_MSMF si fuera necesario
FRAME_W, FRAME_H = 640, 360  # baja resolución = más FPS
FOURCC = 'MJPG'              # ayuda mucho en Windows

# Detector YuNet (detección barata en tamaño reducido)
YUNET_ONNX = Path("modelos/face_detection_yunet_2023mar.onnx")
YUNET_URL  = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
DETECT_W, DETECT_H = 320, 180   # resolución para la detección
SCORE_TH = 0.6
NMS_TH   = 0.3
TOP_K    = 1

# Frecuencias
DETECT_EVERY   = 4   # detecta 1 vez cada 4 frames (resto = tracking)
CLASSIFY_EVERY = 2   # clasifica 1 vez cada 2 frames
MISS_TOLERANCE = 8   # histéresis si detection falla
SMOOTH_ALPHA_BBOX = 0.6
SMOOTH_ALPHA_PROB = 0.6  # EMA para estabilizar probabilidad

def ensure_yunet():
    if not YUNET_ONNX.exists():
        YUNET_ONNX.parent.mkdir(parents=True, exist_ok=True)
        print("[INFO] Descargando YuNet ONNX...")
        urlretrieve(YUNET_URL, YUNET_ONNX.as_posix())
        print("[OK] YuNet descargado:", YUNET_ONNX)

def smooth_bbox(prev, curr, alpha=SMOOTH_ALPHA_BBOX):
    if prev is None: return curr
    return [
        int(alpha*prev[0] + (1-alpha)*curr[0]),
        int(alpha*prev[1] + (1-alpha)*curr[1]),
        int(alpha*prev[2] + (1-alpha)*curr[2]),
        int(alpha*prev[3] + (1-alpha)*curr[3]),
    ]

def clip_bbox(b, W, H):
    x,y,w,h = b
    x = max(0, min(x, W-1)); y = max(0, min(y, H-1))
    w = max(1, min(w, W-x)); h = max(1, min(h, H-y))
    return [x,y,w,h]

def classify_face(frame_bgr, bbox_xywh, model):
    x,y,w,h = bbox_xywh
    face_roi = frame_bgr[y:y+h, x:x+w]
    face_roi = cv2.resize(face_roi, (IMG_SIZE, IMG_SIZE))
    arr = np.expand_dims(face_roi, axis=0).astype(np.float32)
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    prob = float(model.predict(arr, verbose=0)[0][0])
    return prob

# ========= Carga modelo =========
model = tf.keras.models.load_model(MODEL_PATH)

# ========= YuNet =========
ensure_yunet()
detector = cv2.FaceDetectorYN.create(
    model=YUNET_ONNX.as_posix(),
    config="",
    input_size=(DETECT_W, DETECT_H),  # detectaremos sobre un frame reducido
    score_threshold=SCORE_TH,
    nms_threshold=NMS_TH,
    top_k=TOP_K
)

# ========= Cámara =========
cap = cv2.VideoCapture(CAM_INDEX, CAM_BACKEND)
if not cap.isOpened():
    cap.release()
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_ANY)
if not cap.isOpened():
    raise RuntimeError("No se pudo abrir la cámara.")

cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*FOURCC))
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
cap.set(cv2.CAP_PROP_FPS, 30)

# ========= Tracking (KCF = más veloz) =========
tracker = None
prev_bbox = None
last_ok_bbox = None
miss_counter = 0

# Probabilidad suavizada
prob_ema = None
frame_id = 0

print("[INFO] Presiona 'q' para salir.")
t0, n, fps_val = time.time(), 0, 0.0

while True:
    ok, frame = cap.read()
    if not ok:
        break
    H, W = frame.shape[:2]

    # --- Detección cada N frames (sobre frame reducido) ---
    bbox = None
    if frame_id % DETECT_EVERY == 0:
        small = cv2.resize(frame, (DETECT_W, DETECT_H), interpolation=cv2.INTER_LINEAR)
        # actualizar input_size al tamaño reducido
        detector.setInputSize((DETECT_W, DETECT_H))
        res = detector.detect(small)
        faces = res[1] if isinstance(res, tuple) else res

        if isinstance(faces, np.ndarray) and faces.size > 0:
            x, y, w, h = faces[0][:4].astype(int)
            # re-escalar bbox al tamaño original del frame
            scale_x = W / DETECT_W
            scale_y = H / DETECT_H
            x = int(x * scale_x); y = int(y * scale_y)
            w = int(w * scale_x); h = int(h * scale_y)
            bbox = [x, y, w, h]

            # (re)iniciar tracker con bbox nueva
            tracker = cv2.legacy.TrackerKCF_create()
            tracker.init(frame, tuple(bbox))
            prev_bbox = bbox
            last_ok_bbox = bbox
            miss_counter = 0

    # --- Si no detectamos ahora, intentamos tracker ---
    if bbox is None and tracker is not None:
        ok_tr, tb = tracker.update(frame)
        if ok_tr:
            miss_counter += 1
            if miss_counter <= MISS_TOLERANCE:
                tb = [int(tb[0]), int(tb[1]), int(tb[2]), int(tb[3])]
                tb = clip_bbox(tb, W, H)
                tb = smooth_bbox(prev_bbox, tb, alpha=0.7)
                prev_bbox = tb
                bbox = tb
            else:
                tracker = None

    # --- Último recurso: histéresis con último bbox bueno ---
    if bbox is None and last_ok_bbox is not None and miss_counter < MISS_TOLERANCE:
        miss_counter += 1
        bbox = last_ok_bbox

    # --- Clasificación (cada M frames) + EMA ---
    label_text = "Sin rostro"
    color = (200, 200, 200)

    do_classify = (frame_id % CLASSIFY_EVERY == 0)
    if bbox is not None:
        bbox = clip_bbox(bbox, W, H)
        bbox = smooth_bbox(prev_bbox, bbox)
        prev_bbox = bbox
        x,y,w,h = bbox
        cv2.rectangle(frame, (x, y), (x+w, y+h), (180, 255, 180), 2)

        if do_classify:
            prob = classify_face(frame, bbox, model)
            prob_ema = prob if prob_ema is None else (SMOOTH_ALPHA_PROB*prob_ema + (1-SMOOTH_ALPHA_PROB)*prob)
        elif prob_ema is None:
            # garantía: si recién inicia, clasifica al menos una vez
            prob = classify_face(frame, bbox, model)
            prob_ema = prob

        if prob_ema is not None:
            if prob_ema >= UMBRAL:
                label_text = f"Desatento ({prob_ema:.2f})"
                color = (0, 0, 255)
            else:
                label_text = f"Atento ({prob_ema:.2f})"
                color = (0, 200, 0)

    # --- UI + FPS ---
    cv2.putText(frame, label_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

    n += 1
    if time.time() - t0 >= 0.5:
        fps_val = n / (time.time() - t0)
        t0 = time.time(); n = 0
    cv2.putText(frame, f"FPS: {fps_val:.1f}", (10, H-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    cv2.imshow("Atencion - MNV2 (FAST)", frame)
    frame_id += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
