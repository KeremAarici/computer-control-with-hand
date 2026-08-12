import cv2
import mediapipe as mp
import pyautogui
import math

# Taking the screen size
screen_width, screen_height = pyautogui.size()
pyautogui.FAILSAFE = True

# MediaPipe setups
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# Soft mouse
prev_x, prev_y = 0, 0

# ==========================================
# Instant rection
# ==========================================
smoothening = 7         
margin_x = 70           # For slow and sensitive movement
margin_y = 50           # For slow and sensitive movement
DEAD_ZONE = 3           # Shake filter
CLICK_THRESHOLD = 35    # Pointer and middle fingers distance to click action
# ==========================================

mouse_down = False 

print("Hand mode active! Press 'q' to exit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    

    cv2.rectangle(frame, (margin_x, margin_y), (w - margin_x, h - margin_y), (255, 0, 0), 2)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Coordinates of pointer and midle finger tips
            index_finger_tip = hand_landmarks.landmark[8]
            middle_finger_tip = hand_landmarks.landmark[12]

            index_cx, index_cy = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
            middle_cx, middle_cy = int(middle_finger_tip.x * w), int(middle_finger_tip.y * h)

            # Calculating the distance between finger tips
            distance = math.hypot(index_cx - middle_cx, index_cy - middle_cy)

            # Restriction areas in a box and screen ratio
            clamped_x = max(margin_x, min(index_cx, w - margin_x))
            clamped_y = max(margin_y, min(index_cy, h - margin_y))

            active_width = w - (2 * margin_x)
            active_height = h - (2 * margin_y)
            
            target_x = int(((clamped_x - margin_x) / active_width) * screen_width)
            target_y = int(((clamped_y - margin_y) / active_height) * screen_height)

            # Location calculation
            curr_x = prev_x + (target_x - prev_x) / smoothening
            curr_y = prev_y + (target_y - prev_y) / smoothening

            # Shake Filter
            movement = math.hypot(curr_x - prev_x, curr_y - prev_y)
            if movement > DEAD_ZONE:
                pyautogui.moveTo(int(curr_x), int(curr_y))
                prev_x, prev_y = curr_x, curr_y
            else:
                curr_x, curr_y = prev_x, prev_y

            # Clicking and dragging trigger
            if distance < CLICK_THRESHOLD:
                if not mouse_down:
                    pyautogui.mouseDown()
                    mouse_down = True
                    print("Dragging started")
                cv2.line(frame, (index_cx, index_cy), (middle_cx, middle_cy), (0, 255, 0), 3)
            else:
                if mouse_down:
                    pyautogui.mouseUp()
                    mouse_down = False
                    

    cv2.imshow("Mouse Control", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if mouse_down:
    pyautogui.mouseUp()

cap.release()
cv2.destroyAllWindows()