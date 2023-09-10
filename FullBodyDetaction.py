import mediapipe as mp
import cv2
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import pyautogui

# Initialize MediaPipe and Holistic
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# Get the screen resolution
screen_width, screen_height = 1920, 1080  # Adjust to your screen resolution

# Initialize the camera feed
cap = cv2.VideoCapture(0)
cap.set(3, screen_width)  # Set the width of the camera feed
cap.set(4, screen_height)  # Set the height of the camera feed

# Get the default audio playback device
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

# Mouse movement scaling factors (adjust as needed)
mouse_scale_x = screen_width
mouse_scale_y = screen_height

# Minimum distance for pinch gesture (adjust as needed)
pinch_distance_threshold = 0.03

# Initialize the pinch state
pinch_gesture = False

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

    while cap.isOpened():
        ret, frame = cap.read()

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = holistic.process(image)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw facial landmarks, hand landmarks, and pose landmarks
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                                  mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                                  mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
                                  )
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                                  mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                                  mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
                                  )
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
                                  )
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                                  )
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(224, 117, 66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                  )

        # Detect smile expression
        if results.face_landmarks:
            upper_lip_center = results.face_landmarks.landmark[61]
            lower_lip_center = results.face_landmarks.landmark[91]

            # Calculate the distance between upper and lower lip centers
            lip_distance = np.sqrt((upper_lip_center.x - lower_lip_center.x) ** 2 +
                                   (upper_lip_center.y - lower_lip_center.y) ** 2)

            # Detect smile
            if lip_distance > 0.03:
                cv2.putText(image, 'Smile', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Detect sad expression (example)
            left_eye = results.face_landmarks.landmark[33]
            right_eye = results.face_landmarks.landmark[263]

            # Calculate the vertical distance between left and right eyes
            eye_distance = abs(left_eye.y - right_eye.y)

            # Detect sad expression
            if eye_distance > 0.02:
                cv2.putText(image, 'Sad', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Detect surprised expression (example)
            left_eyebrow = results.face_landmarks.landmark[164]
            right_eyebrow = results.face_landmarks.landmark[46]

            # Calculate the horizontal distance between left and right eyebrows
            eyebrow_distance = abs(left_eyebrow.x - right_eyebrow.x)

            # Detect surprised expression
            if eyebrow_distance > 0.02:
                cv2.putText(image, 'Surprised', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Detect hand tracking volume gestures
        if results.right_hand_landmarks:
            hand_landmarks = results.right_hand_landmarks.landmark
            thumb_tip = hand_landmarks[4]
            index_finger_tip = hand_landmarks[8]
            middle_finger_tip = hand_landmarks[12]
            ring_finger_tip = hand_landmarks[16]
            pinky_tip = hand_landmarks[20]

            # Calculate the distance between thumb and other finger tips
            thumb_to_index_distance = np.sqrt(
                (thumb_tip.x - index_finger_tip.x) ** 2 + (thumb_tip.y - index_finger_tip.y) ** 2)
            thumb_to_middle_distance = np.sqrt(
                (thumb_tip.x - middle_finger_tip.x) ** 2 + (thumb_tip.y - middle_finger_tip.y) ** 2)
            thumb_to_ring_distance = np.sqrt(
                (thumb_tip.x - ring_finger_tip.x) ** 2 + (thumb_tip.y - ring_finger_tip.y) ** 2)
            thumb_to_pinky_distance = np.sqrt(
                (thumb_tip.x - pinky_tip.x) ** 2 + (thumb_tip.y - pinky_tip.y) ** 2)

            # Determine the gesture based on distances
            if thumb_to_index_distance < 0.03:
                cv2.putText(image, 'Fist', (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                # Decrease volume (adjustment value increased)
                current_volume = volume.GetMasterVolumeLevelScalar()
                new_volume = max(current_volume - 0.2, 0.0)
                volume.SetMasterVolumeLevelScalar(new_volume, None)
            elif thumb_to_middle_distance < 0.03:
                cv2.putText(image, 'Thumbs Up', (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                # Increase volume (adjustment value increased)
                current_volume = volume.GetMasterVolumeLevelScalar()
                new_volume = min(current_volume + 0.2, 1.0)
                volume.SetMasterVolumeLevelScalar(new_volume, None)

            # Check for pinch gesture (thumb and index finger close)
            if thumb_to_index_distance < pinch_distance_threshold:
                if not pinch_gesture:
                    cv2.putText(image, 'Pinch (Click)', (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    # Simulate a left mouse click
                    pyautogui.click()
                    pinch_gesture = True
            else:
                pinch_gesture = False

            # Move the virtual mouse based on hand position
            hand_x = int(thumb_tip.x * mouse_scale_x)
            hand_y = int(thumb_tip.y * mouse_scale_y)
            pyautogui.moveTo(hand_x, hand_y)

        cv2.imshow('Holistic Model Detections', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
