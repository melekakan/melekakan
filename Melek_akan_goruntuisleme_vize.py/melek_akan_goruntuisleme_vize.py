import cv2
import numpy as np
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

# MediaPipe modeli yükleniyor
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

# Her top için bilgiler
NUM_BALLS = 3
BALL_RADIUS = 30  # Resmin boyutunu buna göre ayarlayacağız
BALL_COLOR = (0, 0, 255)  # Bu renk artık kullanılmayacak çünkü top görseli olacak

# Top listesi
balls = []
for i in range(NUM_BALLS):
    ball = {
        "pos": [100 + i * 100, 100 + i * 60],  # Başlangıç pozisyonları
        "vel": [5 - i*2, 3 + i],               # Farklı hızlar
        "visible": True,
        "hide_time": 0
    }
    balls.append(ball)

# Top resmi yükleniyor (top.png dosyasını yükle)
ball_image = cv2.imread('ball.png', cv2.IMREAD_UNCHANGED)  # Şeffaf PNG desteği için

# Kamera başlat
cam = cv2.VideoCapture(0)

# Top resmini ekranda çizdirmek için fonksiyon
def draw_ball_with_image(frame, ball_pos, ball_img, radius):
    # Görseli yeniden boyutlandır
    ball_img = cv2.resize(ball_img, (radius*2, radius*2))
    x, y = int(ball_pos[0]), int(ball_pos[1])
    h, w, _ = ball_img.shape

    # Topun sol üst köşesi
    x1, y1 = x - radius, y - radius
    x2, y2 = x1 + w, y1 + h

    # Görselin taşmaması için sınır kontrolü
    if x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
        return  # Ekran dışına taşarsa çizme

    roi = frame[y1:y2, x1:x2]
    
    # Eğer top şeffafsa (PNG), alfa kanalını kullan
    if ball_img.shape[2] == 4:
        alpha = ball_img[:, :, 3] / 255.0
        for c in range(3):
            roi[:, :, c] = (1 - alpha) * roi[:, :, c] + alpha * ball_img[:, :, c]
    else:
        roi[:] = ball_img[:, :, :3]

    frame[y1:y2, x1:x2] = roi


while cam.isOpened():
    success, frame = cam.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    detection_result = detector.detect(mp_image)

    h, w, _ = frame.shape
    current_time = time.time()

    # Topları güncelle
    for ball in balls:
        if not ball["visible"]:
            if current_time - ball["hide_time"] > 3:
                ball["visible"] = True
        else:
            for i in range(2):
                ball["pos"][i] += ball["vel"][i]

            if ball["pos"][0] <= BALL_RADIUS or ball["pos"][0] >= w - BALL_RADIUS:
                ball["vel"][0] *= -1
            if ball["pos"][1] <= BALL_RADIUS or ball["pos"][1] >= h - BALL_RADIUS:
                ball["vel"][1] *= -1

    # Sadece işaret parmağı ucu kontrol edilir
    for hand_landmarks in detection_result.hand_landmarks:
        for idx in [8]:  # 👈 işaret parmağı
            landmark = hand_landmarks[idx]
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            cv2.circle(frame, (x, y), 5, (255, 255, 255), -1)

            for ball in balls:
                if ball["visible"]:
                    dist = np.linalg.norm(np.array([x, y]) - np.array(ball["pos"]))
                    if dist < BALL_RADIUS:
                        ball["visible"] = False
                        ball["hide_time"] = current_time

    # Topları resimle çiz
    for ball in balls:
        if ball["visible"]:
            draw_ball_with_image(frame, ball["pos"], ball_image, BALL_RADIUS)

    cv2.imshow("Touch the Balls!", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
