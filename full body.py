import cv2
import mediapipe as mp
import threading
import tkinter as tk
from tkinter import messagebox
import time

# Initialize MediaPipe Pose and Hands
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Global variables for camera control
camera_running = False
camera_source = 0  # Default to local webcam
fps_limit = 30  # Max FPS for processing
frame_time = 1 / fps_limit

# Function to dynamically switch camera source
def set_camera_source(new_source):
    global camera_source
    camera_source = new_source
    messagebox.showinfo("Camera Source", f"Switched to {new_source}.")

# Function to classify poses
def classify_pose(landmarks):
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]

    if abs(left_shoulder.y - right_shoulder.y) < 0.1:  # Shoulders horizontal
        if abs(left_wrist.y - left_shoulder.y) < 0.1 and abs(right_wrist.y - right_shoulder.y) < 0.1:
            return "T-pose"
        elif left_wrist.x < left_shoulder.x and right_wrist.x > right_shoulder.x:
            return "Arms Extended"
        else:
            return "Standing Upright"
    return "Unknown"

# Function to classify hand gestures
def classify_hand_gesture(hand_landmarks):
    FINGER_TIPS = [4, 8, 12, 16, 20]
    finger_states = []
    for tip in FINGER_TIPS:
        tip_y = hand_landmarks.landmark[tip].y
        pip_y = hand_landmarks.landmark[tip - 2].y
        finger_states.append(1 if tip_y < pip_y else 0)

    if finger_states == [1, 0, 0, 0, 0]:
        return "Thumb Up"
    elif finger_states == [1, 1, 1, 1, 1]:
        return "All Fingers Extended"
    elif finger_states == [0, 1, 1, 0, 0]:
        return "Victory"
    return "Unknown"

# Camera thread to process frames
def run_camera():
    global camera_running, camera_source

    cap = cv2.VideoCapture(camera_source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    last_frame_time = time.time()

    while camera_running:
        current_time = time.time()
        if current_time - last_frame_time < frame_time:
            continue  # Skip frame if processing too fast
        last_frame_time = current_time

        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect pose
        pose_results = pose.process(rgb_frame)
        if pose_results.pose_landmarks:
            landmarks = pose_results.pose_landmarks.landmark
            current_pose = classify_pose(landmarks)
            mp_draw.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        else:
            current_pose = "Unknown"

        # Detect hand gestures
        hand_results = hands.process(rgb_frame)
        current_gestures = []
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                gesture = classify_hand_gesture(hand_landmarks)
                current_gestures.append(gesture)
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Display results
        cv2.putText(frame, f"Pose: {current_pose}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        for idx, gesture in enumerate(current_gestures):
            cv2.putText(frame, f"Hand {idx + 1}: {gesture}", (10, 60 + idx * 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        # Show the frame
        cv2.imshow("Camera Feed", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# GUI Control
def start_camera():
    global camera_running
    camera_running = True
    threading.Thread(target=run_camera).start()

def stop_camera():
    global camera_running
    camera_running = False
    messagebox.showinfo("Camera", "Camera stopped successfully.")

# Main GUI
def open_ip_camera_prompt():
    ip = tk.simpledialog.askstring("IP Camera", "Enter the IP Camera URL:")
    if ip:
        set_camera_source(ip)

root = tk.Tk()
root.title("Camera Switch and FPS Optimization")

start_button = tk.Button(root, text="Start Camera", command=start_camera, width=20, height=2, bg="green", fg="white")
start_button.pack(pady=10)

stop_button = tk.Button(root, text="Stop Camera", command=stop_camera, width=20, height=2, bg="red", fg="white")
stop_button.pack(pady=10)

ip_camera_button = tk.Button(root, text="Set IP Camera", command=open_ip_camera_prompt, width=20, height=2, bg="orange", fg="black")
ip_camera_button.pack(pady=10)

exit_button = tk.Button(root, text="Exit", command=root.quit, width=20, height=2, bg="blue", fg="white")
exit_button.pack(pady=10)

root.mainloop()
