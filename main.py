import cv2
import numpy as np
import mediapipe as mp
from PIL import Image, ImageDraw, ImageFont

# Инициализация MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Зеркальный эффект
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    gesture = "No gesture detected"

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = hand_landmarks.landmark

            # Координаты кончиков пальцев (по оси Y)
            thumb_tip = landmarks[4].y
            index_tip = landmarks[8].y
            middle_tip = landmarks[12].y
            ring_tip = landmarks[16].y
            pinky_tip = landmarks[20].y

            fingers = [
                index_tip < landmarks[6].y,  # Указательный
                middle_tip < landmarks[10].y,  # Средний
                ring_tip < landmarks[14].y,  # Безымянный
                pinky_tip < landmarks[18].y  # Мизинец
            ]

            if fingers == [False, True, False, False]:
                gesture = "🖕 Fuck you"
            elif fingers == [True, True, True, True]:
                gesture = "🖐 Open Palm"
            elif fingers == [True, False, False, False]:
                gesture = "☝️ Pointing Up"
            elif fingers == [True, True, False, False]:
                gesture = "✌️ Peace Sign"
            elif fingers == [False, False, False, False] and thumb_tip < index_tip:
                gesture = "👍 Thumbs Up"

    # Конвертация OpenCV в PIL
    image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image_pil)

    # Загрузка шрифта (замени путь на свой)
    font_path = "arial.ttf"  # Убедись, что у тебя есть этот шрифт
    try:
        font = ImageFont.truetype(font_path, 40)
    except IOError:
        font = ImageFont.load_default()

    # Отображение текста
    draw.text((50, 100), gesture, font=font, fill=(0, 255, 0))

    # Конвертация PIL обратно в OpenCV
    frame = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

    # Показываем изображение
    cv2.imshow("Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
