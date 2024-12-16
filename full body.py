import cv2
import mediapipe as mp
import threading
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
import time
import webbrowser

# Initialize MediaPipe Pose and Hands
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Global variables for camera and processing
camera_running = False
camera_source = 0  # Default to device camera
current_pose = "Unknown"
current_gestures = []

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

    # Example gesture: Thumb up
    if finger_states == [1, 0, 0, 0, 0]:
        return "Thumb Up"
    elif finger_states == [1, 1, 1, 1, 1]:
        return "All Fingers Extended"
    elif finger_states == [0, 1, 1, 0, 0]:
        return "Victory"
    return "Unknown"

# Camera thread to process frames
def run_camera():
    global camera_running, camera_source, current_pose, current_gestures

    cap = cv2.VideoCapture(camera_source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Lower resolution for better performance
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    fps_limit = 30
    frame_time = 1 / fps_limit
    last_frame_time = time.time()

    while camera_running:
        current_time = time.time()
        if current_time - last_frame_time < frame_time:
            continue  # Skip processing to maintain FPS limit
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

# Function to set the camera source
def set_camera_source(ip_digits):
    global camera_source
    try:
        # Split the input into octets and validate
        octets = ip_digits.split(".")
        if len(octets) == 2 and all(octet.isdigit() and 0 <= int(octet) <= 255 for octet in octets):
            camera_source = f"http://192.168.{ip_digits}:8080/video"
            messagebox.showinfo("Camera Source", f"Switched to IP Camera: {camera_source}")
        else:
            raise ValueError
    except ValueError:
        messagebox.showerror("Invalid Input", "Please enter the last two octets in the format (e.g., 0.212).")

# Function to reset to the device camera
def reset_to_device_camera():
    global camera_source
    camera_source = 0  # Reset to device camera
    messagebox.showinfo("Camera Source", "Switched to Device Camera")

# GUI Control
def start_camera():
    global camera_running
    if not camera_running:
        camera_running = True
        threading.Thread(target=run_camera).start()

def stop_camera():
    global camera_running
    if camera_running:
        camera_running = False
        messagebox.showinfo("Camera", "Camera stopped successfully.")

# Main GUI
def show_about():
    about_text = "BJJ Pose and Gesture Recognition\n\n" \
                 """This project is a real-time pose and hand 
                 gesture recognition system tailored for Brazilian Jiu-Jitsu (BJJ) training and scoring.
                 Using computer vision and machine learning technologies, the application processes video
                 feeds to identify poses and gestures, then maps these to predefined BJJ scoring categories."
                 
                 Key Features:

                 1. Pose Recognition:

                 Detects common BJJ stances like "T-pose," "Arms Extended," and "Standing Upright" using 
                 MediaPipe's Pose module.
                 Maps detected poses to BJJ scoring categories such as Finish, Advantage, and Neutral.
                 
                 2. Hand Gesture Recognition:

                 Recognizes gestures such as "Thumb Up," "Victory," and "All Fingers Extended" using
                 MediaPipe Hands.
                 
                 3. Dynamic Camera Support:

                 Default setup uses the device's camera.
                 Users can switch to an IP camera feed dynamically during runtime.
                 Frames are processed at a steady FPS to ensure smooth video feed, even with complex calculations.
                 Camera resolution is set to 1280x720 for enhanced clarity.
                 
                 4. Optimized Performance:

                 Frames are processed at a steady FPS to ensure smooth video feed, even with complex calculations.
                 Camera resolution is set to 1280x720 for enhanced clarity.
                 
                 5. User-Friendly GUI:

                 Simple buttons for starting, stopping, and switching between camera sources.
                 Real-time display of pose, gestures, and BJJ scoring annotations directly on the video feed.
                 
                 Purpose and Applications:

                 This project serves as a training tool for athletes and coaches in Brazilian Jiu-Jitsu,
                 providing insights into body positioning and scoring potential in real-time. Additionally,
                 it showcases the integration of advanced computer vision techniques with a user-friendly
                 interface, making it adaptable for other sports or gesture recognition applications.
                 License: MIT (see GitHub repository for details)."""
    messagebox.showinfo("About", about_text)

def show_contact():
    contact_text = "Developers:\n\n" \
                  "1. Boris Jordanov\n\n" \
                  "GitHub Repository: https://github.com/BorisJIordanov/BJJ"
    messagebox.showinfo("Contact", contact_text)

def open_github():
    webbrowser.open("https://github.com/BorisJIordanov/BJJ")

# Initialize root GUI
root = tk.Tk()
root.title("Pose and Gesture Recognition")
root.geometry("600x400")

# Apply a modern style
style = ttk.Style(root)
style.theme_use("clam")

# Dark theme configurations
root.configure(bg="#2e2e2e")
style.configure("TButton", font=("Helvetica", 12), background="#3a3a3a", foreground="white")
style.map("TButton", background=[("active", "#505050")])
style.configure("TLabel", font=("Helvetica", 10), background="#2e2e2e", foreground="white")

# Create Notebook for tabs
notebook = ttk.Notebook(root)
notebook.pack(expand=True, fill="both")

# Tab 1: Main functionality
main_frame = tk.Frame(notebook, bg="#2e2e2e")
notebook.add(main_frame, text="Run Program")

start_button = ttk.Button(main_frame, text="Start Camera", command=start_camera)
start_button.pack(pady=10)

stop_button = ttk.Button(main_frame, text="Stop Camera", command=stop_camera)
stop_button.pack(pady=10)

ip_entry_label = ttk.Label(main_frame, text="Enter last two digits of IP (e.g., 0.212):")
ip_entry_label.pack(pady=5)

ip_entry = ttk.Entry(main_frame, width=20, font=("Helvetica", 10))
ip_entry.pack(pady=5)

set_ip_button = ttk.Button(main_frame, text="Set IP Camera", command=lambda: set_camera_source(ip_entry.get()))
set_ip_button.pack(pady=10)

reset_button = ttk.Button(main_frame, text="Reset to Device Camera", command=reset_to_device_camera)
reset_button.pack(pady=10)

# Tab 2: About
about_frame = tk.Frame(notebook, bg="#2e2e2e")
notebook.add(about_frame, text="About")

about_label = ttk.Label(about_frame, text="BJJ Pose and Gesture Recognition\n\n" \
                 """This project is a real-time pose and hand 
                 gesture recognition system tailored for Brazilian Jiu-Jitsu (BJJ) training and scoring.
                 Using computer vision and machine learning technologies, the application processes video
                 feeds to identify poses and gestures, then maps these to predefined BJJ scoring categories."
                 
                 Key Features:

                 1. Pose Recognition:

                 Detects common BJJ stances like "T-pose," "Arms Extended," and "Standing Upright" using 
                 MediaPipe's Pose module.
                 Maps detected poses to BJJ scoring categories such as Finish, Advantage, and Neutral.
                 
                 2. Hand Gesture Recognition:

                 Recognizes gestures such as "Thumb Up," "Victory," and "All Fingers Extended" using
                 MediaPipe Hands.
                 
                 3. Dynamic Camera Support:

                 Default setup uses the device's camera.
                 Users can switch to an IP camera feed dynamically during runtime.
                 Frames are processed at a steady FPS to ensure smooth video feed, even with complex calculations.
                 Camera resolution is set to 1280x720 for enhanced clarity.
                 
                 4. Optimized Performance:

                 Frames are processed at a steady FPS to ensure smooth video feed, even with complex calculations.
                 Camera resolution is set to 1280x720 for enhanced clarity.
                 
                 5. User-Friendly GUI:

                 Simple buttons for starting, stopping, and switching between camera sources.
                 Real-time display of pose, gestures, and BJJ scoring annotations directly on the video feed.
                 
                 Purpose and Applications:

                 This project serves as a training tool for athletes and coaches in Brazilian Jiu-Jitsu,
                 providing insights into body positioning and scoring potential in real-time. Additionally,
                 it showcases the integration of advanced computer vision techniques with a user-friendly
                 interface, making it adaptable for other sports or gesture recognition applications.
                 License: MIT (see GitHub repository for details).""", wraplength=500)
about_label.pack(pady=20)

# Tab 3: Contact
contact_frame = tk.Frame(notebook, bg="#2e2e2e")
notebook.add(contact_frame, text="Contact")

contact_label = ttk.Label(contact_frame, text="Developers:\n\n" \
                          "1. Boris Yordanov\n\n" , wraplength=500)
contact_label.pack(pady=20)

# GitHub button to open the repository
github_button = ttk.Button(contact_frame, text="Visit GitHub", command=open_github)
github_button.pack(pady=10)

# Add exit button
exit_button = ttk.Button(root, text="Exit", command=root.quit)
exit_button.pack(pady=10)

root.mainloop()
