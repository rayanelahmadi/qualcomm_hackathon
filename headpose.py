import cv2, numpy as np
from common.ort_utils import make_session

MODEL = "models/6drepnet.onnx"   # expects 224x224 RGB, normalized
sess = make_session(MODEL)
inp = sess.get_inputs()[0].name
out = sess.get_outputs()[0].name

# Use OpenCV’s built-in face detector for the crop (keeps dependencies light)
face_det = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def preprocess(img_rgb, box, size=224):
    x,y,w,h = box
    cx, cy = x+w//2, y+h//2
    s = int(1.2 * max(w, h))
    x1, y1 = max(0, cx - s//2), max(0, cy - s//2)
    x2, y2 = min(img_rgb.shape[1], x1 + s), min(img_rgb.shape[0], y1 + s)
    crop = img_rgb[y1:y2, x1:x2]
    crop = cv2.resize(crop, (size, size))
    x = crop.astype(np.float32) / 255.0
    x = (x - 0.5) / 0.5  # normalize to [-1,1]
    x = np.transpose(x, (2,0,1))[None, ...]  # [1,3,224,224]
    return x, (x1,y1,x2,y2)

cap = cv2.VideoCapture(0)
assert cap.isOpened()
while True:
    ret, frame_bgr = cap.read()
    if not ret: break
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    faces = face_det.detectMultiScale(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY), 1.2, 5)

    for (x,y,w,h) in faces[:1]:  # first face
        x_in, rect = preprocess(frame_rgb, (x,y,w,h))
        yaw_pitch_roll = sess.run([out], {inp: x_in})[0].flatten()  # [yaw, pitch, roll] in degrees
        yaw, pitch, roll = map(float, yaw_pitch_roll)

        # Draw box + text
        x1,y1,x2,y2 = rect
        cv2.rectangle(frame_bgr, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(frame_bgr, f"yaw:{yaw:+.1f}  pitch:{pitch:+.1f}  roll:{roll:+.1f}",
                    (x1, max(0,y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.imshow("Head Pose", frame_bgr)
    if cv2.waitKey(1) & 0xFF == 27: break  # ESC to quit
cap.release(); cv2.destroyAllWindows()
