import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import os
from collections import deque

# download hand detection model if not present
model_path = 'hand_landmarker.task'
if not os.path.exists(model_path):
    print("downloading hand detection model...")
    url = 'https://storage.googleapis.com/mediapipe-assets/hand_landmarker.task'
    urllib.request.urlretrieve(url, model_path)
    print("done")

# initialize hand landmarker
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(base_options=base_options)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

strokes = []
current_stroke = []
smooth_buf = deque(maxlen=5)
print("controls:  q=quit  c=clear  z=undo  s=save")
count = -1

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, c = frame.shape
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    results = detector.detect(mp_image)
    drawing_this_frame = False
    if results.hand_landmarks:
        for hand_landmarks in results.hand_landmarks:
            tip = hand_landmarks[8]
            pip = hand_landmarks[6]
            tip_x = int(tip.x * w)
            tip_y = int(tip.y * h)
            pip_y = int(pip.y * h)
            if tip_y < pip_y:
                smooth_buf.append((tip_x, tip_y))
                sx = int(sum(p[0] for p in smooth_buf) / len(smooth_buf))
                sy = int(sum(p[1] for p in smooth_buf) / len(smooth_buf))
                current_stroke.append((sx, sy))
                drawing_this_frame = True

    if not drawing_this_frame and current_stroke:
        strokes.append(current_stroke)
        current_stroke = []
        smooth_buf.clear()

    for stroke in strokes + [current_stroke]:
        for i in range(1, len(stroke)):
            cv2.line(frame, stroke[i-1], stroke[i], (0, 255, 0), 4)

    cv2.imshow('hand detection', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('z'):
        if strokes:
            strokes.pop()
    elif key == ord('c'):
        strokes = []
        current_stroke = []
    elif key == ord('s'):
        count += 1
        cv2.imwrite(f"drawing_{count}.png", frame)
        print(f"drawing saved as drawing_{count}.png")

cap.release()
cv2.destroyAllWindows()