# test_face_landmarks_npu.py
import cv2
import numpy as np
import onnxruntime as ort
import time

MODEL_PATH = "mediapipe_face-facelandmarkdetector-float.onnx/model.onnx/model.onnx"   # put your downloaded model here

# --- Create ONNX session (QNN/NPU first, CPU fallback) ---
providers = [
    ("QNNExecutionProvider", {"backend_path": "QnnHtp.dll"}),  # Snapdragon NPU
    "CPUExecutionProvider"
]
sess = ort.InferenceSession(MODEL_PATH, providers=providers)

print("Available providers:", ort.get_available_providers())
print("Session providers:", sess.get_providers())

inp = sess.get_inputs()[0]
inp_name = inp.name
# Common shapes: [1,3,H,W] or dynamic dims. We’ll read from the model and default to 192x192 if unknown.
inp_shape = list(inp.shape)
# Resolve any dynamic dims to concrete values
if None in inp_shape or "None" in str(inp_shape) or -1 in inp_shape:
    # Try a common MediaPipe-friendly size
    N, C, H, W = 1, 3, 192, 192
else:
    # Expected [1,3,H,W]
    N, C, H, W = inp_shape

print(f"Model input: name={inp_name}, shape={N}x{C}x{H}x{W}")

# --- Simple drawing helper for landmarks ---
def draw_landmarks(img, lms, color=(0,255,0)):
    # lms expected as (num_points, 2) normalized [0,1] coords OR pixel coords depending on model
    h, w = img.shape[:2]
    for (x, y) in lms:
        # If looks normalized, scale to pixels
        if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
            px, py = int(x * w), int(y * h)
        else:
            px, py = int(x), int(y)
        cv2.circle(img, (px, py), 2, color, -1)

# --- Preprocess frame to model tensor ---
def preprocess_bgr(frame):
    # Center-crop to square then resize to (H, W)
    h, w = frame.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    crop = frame[y0:y0+side, x0:x0+side]

    # BGR -> RGB, resize, to float32 [0,1]
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA)
    tensor = resized.astype(np.float32) / 255.0
    # NCHW
    tensor = np.transpose(tensor, (2, 0, 1))[np.newaxis, ...]  # (1,3,H,W)
    return crop, tensor

# --- Decode model outputs into 2D landmarks ---
def decode_landmarks(outputs, frame):
    """
    This part depends on the exact model head.
    Many face-landmark models output either:
      - (1, num_points*2) normalized [0,1] x/y pairs, or
      - (1, num_points, 2), sometimes with z.
    We’ll try to infer a reasonable shape.
    """
    out = outputs[0]
    arr = np.array(out)

    # Squeeze batch if present
    arr = np.squeeze(arr)

    # If flat vector length is even, reshape to (num_points, 2)
    if arr.ndim == 1 and arr.size % 2 == 0:
        pts = arr.reshape(-1, 2)
    elif arr.ndim == 2 and arr.shape[1] >= 2:
        pts = arr[:, :2]
    else:
        # Fallback: nothing we can parse
        return None

    # Some models output pixel coords, others normalized. We’ll let draw_landmarks scale if normalized.
    return pts

# --- Webcam loop ---
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)  # fallback

print("Press 'q' to quit.")
fps_avg = None
last_t = time.time()

while True:
    ok, frame = cap.read()
    if not ok:
        print("Camera read failed")
        break

    # Preprocess (square crop + resize + NCHW float32)
    vis_crop, tensor = preprocess_bgr(frame)

    # Inference
    t0 = time.time()
    outputs = sess.run(None, {inp_name: tensor})
    t1 = time.time()
    dt = (t1 - t0)
    fps = 1.0 / dt if dt > 0 else 0.0
    fps_avg = fps if fps_avg is None else (0.9 * fps_avg + 0.1 * fps)

    # Decode & draw
    pts = decode_landmarks(outputs, vis_crop)
    if pts is not None:
        draw_landmarks(vis_crop, pts, color=(0, 255, 0))
    else:
        cv2.putText(vis_crop, "No landmarks decoded", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # HUD with provider + speed
    prov = sess.get_providers()[0] if sess.get_providers() else "unknown"
    cv2.putText(vis_crop, f"Provider: {prov}", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
    cv2.putText(vis_crop, f"FPS: {fps_avg:.1f}", (10, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

    cv2.imshow("Face Landmarks (NPU test)", vis_crop)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
