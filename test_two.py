import cv2, numpy as np
import onnxruntime as ort

MODEL_PATH = "mediapipe_face-facelandmarkdetector-float.onnx/model.onnx/model.onnx"   # put your downloaded model here

# --- Create ONNX session (QNN/NPU first, CPU fallback) ---
providers = [
    ("QNNExecutionProvider", {"backend_path": "QnnHtp.dll"})  # Snapdragon NPU
]
sess = ort.InferenceSession(MODEL_PATH, providers=providers)

print("Available providers:", ort.get_available_providers())
print("Session providers:", sess.get_providers())

inp = sess.get_inputs()[0]
inp_name, inp_shape = inp.name, [d if isinstance(d, int) else 192 for d in inp.shape]
# default to 192x192 if dynamic
_, c, H, W = (1, 3, 192, 192) if len(inp_shape) != 4 else inp_shape
H = 192 if H in (0, -1, None) else H
W = 192 if W in (0, -1, None) else W

out0 = sess.get_outputs()[0].name

cap = cv2.VideoCapture(0)
assert cap.isOpened()

def preprocess(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    im = cv2.resize(rgb, (W, H)).astype(np.float32) / 255.0
    x = np.transpose(im, (2,0,1))[None, ...]  # [1,3,H,W]
    return x

print(f"Input shape -> [1,3,{H},{W}]  | Output: {out0}")
while True:
    ok, frame = cap.read()
    if not ok: break
    x = preprocess(frame)
    y = sess.run(None, {inp_name: x})  # list of outputs
    print(y)
    #print shape of y
    print("Output shapes:", [np.array(o).shape for o in y])
    out = y[1]
    h, w = frame.shape[:2]

    # Try to interpret as [1, N*2] or [1, N, 2] or [1, N, 3]
    pts = None
    arr = out.squeeze()
    if arr.ndim == 1 and arr.size % 2 == 0:
        N = arr.size // 2; pts = arr.reshape(N,2)
    elif arr.ndim == 2 and arr.shape[1] in (2,3):
        pts = arr[:, :2]
    elif arr.ndim == 3 and arr.shape[-1] in (2,3):
        pts = arr[0,:, :2]

    if pts is not None:
        # assume normalized in [0,1]; if looks tiny/huge, switch to pixel scaling
        if pts.max() <= 1.2:
            pts_px = np.stack([pts[:,0]*w, pts[:,1]*h], axis=1).astype(int)
        else:
            pts_px = pts.astype(int)
        for (px,py) in pts_px:
            cv2.circle(frame, (int(px),int(py)), 1, (0,255,255), -1)
        cv2.putText(frame, f"landmarks: {len(pts_px)}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
    else:
        cv2.putText(frame, f"Output shape: {out.shape}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    cv2.imshow("Face Landmarks (best-effort)", frame)
    if cv2.waitKey(1) & 0xFF == 27: break
cap.release(); cv2.destroyAllWindows()
