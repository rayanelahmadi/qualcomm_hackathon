import time, cv2, numpy as np
import onnxruntime as ort

MODEL_PATH = "emotion-ferplus-8.onnx"   # expects 64x64 or 224x224 depending on variant
IMG_SIZE = 64

# --- Create ONNX session (QNN/NPU first, CPU fallback) ---
providers = [
    ("QNNExecutionProvider", {"backend_path": "QnnHtp.dll"})  # Snapdragon NPU
]
sess = ort.InferenceSession(MODEL_PATH, providers=providers)
inp = sess.get_inputs()[0].name
out = sess.get_outputs()[0].name
labels = ["neutral","happiness","surprise","sadness","anger","disgust","fear","contempt"][:8]

det = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0); assert cap.isOpened()
t0 = 0
while True:
    ok, frame = cap.read()
    if not ok: break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = det.detectMultiScale(gray, 1.2, 5)
    if len(faces):
        (x,y,w,h) = faces[0]
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
        x_in = face[None, None, ...]  # [1,1,H,W]
        logits = sess.run([out], {inp: x_in})[0].squeeze()
        probs = np.exp(logits) / np.exp(logits).sum()
        print(probs)
        idx = int(np.argmax(probs))
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(frame, f"{labels[idx]} ({probs[idx]:.2f})", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.imshow("FER+", frame)
    if cv2.waitKey(1) & 0xFF == 27: break
cap.release(); cv2.destroyAllWindows()
