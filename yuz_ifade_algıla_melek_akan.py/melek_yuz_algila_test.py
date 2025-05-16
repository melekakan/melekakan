from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import matplotlib.pyplot as plt
import pickle
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Modeli yükle
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Tahmin edilen sonucu emoji'ye dönüştür
etiket_emoji = {
    "gulen": "happy",     # Gülen
    "uzgun": "sad",     # Üzgün
    "sokta": "in shock",     # Şokta
    "kizgin": "angry"     # Kızgın
}

# Görüntü üzerine landmarkları çizme ve sonucu yazma
def draw_landmarks_on_image(rgb_image, detection_result):
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(rgb_image)

    for face_landmarks in face_landmarks_list:
        # Koordinatları çıkar
        koordinatlar = []
        for landmark in face_landmarks:
            koordinatlar.append(round(landmark.x, 4))
            koordinatlar.append(round(landmark.y, 4))

        # Tahmin yap
        sonuc = model.predict([koordinatlar])[0]
        emoji = etiket_emoji.get(sonuc, "❓")

        # Sonucu görüntüye yaz (Not: OpenCV Türkçe karakter veya emoji tam desteklemez ama bazı sistemlerde çalışır)
        annotated_image = cv2.putText(annotated_image,
                                      emoji,
                                      (60, 60),
                                      cv2.FONT_HERSHEY_COMPLEX,
                                      3,
                                      (0, 255, 255),
                                      6,
                                      cv2.LINE_AA)
    return annotated_image

# MediaPipe yüz tanıma ayarları
base_options = python.BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task')
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       output_face_blendshapes=True,
                                       output_facial_transformation_matrixes=True,
                                       num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)

# Kamera başlat
cam = cv2.VideoCapture(0)
while cam.isOpened():
    basari, frame = cam.read()
    if not basari:
        break

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

    detection_result = detector.detect(mp_image)

    annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), detection_result)
    cv2.imshow("Yüz İfade Tanıma (Emoji)", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

    key = cv2.waitKey(1)
    if key == ord('q') or key == ord('Q'):
        break

cam.release()
cv2.destroyAllWindows()
