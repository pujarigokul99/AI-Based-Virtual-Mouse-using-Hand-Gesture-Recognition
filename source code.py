import cv2
import mediapipe as mp
import pyautogui
import math

# Start camera
cap = cv2.VideoCapture(0)

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Screen size
screen_w, screen_h = pyautogui.size()

# Smoothing variables
prev_x, prev_y = 0, 0
smoothening = 7

# Click delay
click_delay = 0

while True:
    success, img = cap.read()
    if not success:
        continue

    h, w, c = img.shape

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:

            # 👉 Index finger (for cursor)
            index = handLms.landmark[8]
            cx, cy = int(index.x * w), int(index.y * h)

            # 👉 Convert to screen coordinates (sensitivity 2.0)
            screen_x = int(index.x * screen_w * 2.0)
            screen_y = int(index.y * screen_h * 2.0)

            # Keep inside screen
            screen_x = min(screen_w, max(0, screen_x))
            screen_y = min(screen_h, max(0, screen_y))

            # 👉 Ultra stability (ignore small movements)
            if abs(screen_x - prev_x) >= 5 or abs(screen_y - prev_y) >= 5:

                # 👉 Smooth movement
                curr_x = prev_x + (screen_x - prev_x) / smoothening
                curr_y = prev_y + (screen_y - prev_y) / smoothening

                pyautogui.moveTo(curr_x, curr_y)
                prev_x, prev_y = curr_x, curr_y

            # 👉 CLICK GESTURE (thumb + index)
            thumb = handLms.landmark[4]

            tx, ty = int(thumb.x * w), int(thumb.y * h)
            ix, iy = int(index.x * w), int(index.y * h)

            distance = math.hypot(ix - tx, iy - ty)

            # Draw line between fingers
            cv2.line(img, (tx, ty), (ix, iy), (255, 255, 0), 2)

            # 👉 Balanced click
            if distance < 40 and click_delay == 0:
                pyautogui.click()
                click_delay = 15

                # Show click text
                cv2.putText(img, "CLICK", (ix, iy - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 0), 3)

            # Reduce delay
            if click_delay > 0:
                click_delay -= 1

            # 👉 Draw visuals
            cv2.circle(img, (cx, cy), 25, (0, 255, 255), 5)
            cv2.putText(img, "INDEX", (cx, cy - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 255), 3)

            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

    else:
        cv2.putText(img, "No Hand Detected",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2)

    cv2.imshow("Virtual Mouse", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()