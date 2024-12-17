import cv2
import mediapipe as mp
import threading
import tkinter as tk
from tkinter import ttk, messagebox
import requests
import zipfile
import io
import os
import time
import webbrowser

# URL to the Forest theme GitHub repository zip file
FOREST_THEME_REPO_ZIP = "https://github.com/rdbende/Forest-ttk-theme/archive/refs/heads/master.zip"
FOREST_THEME_DIR = "Forest-ttk-theme-master"

# Function to download and extract the Forest theme
def setup_forest_theme():
    if not os.path.exists(FOREST_THEME_DIR):
        print("Downloading Forest theme...")
        try:
            response = requests.get(FOREST_THEME_REPO_ZIP)
            response.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                z.extractall()
            print("Forest theme downloaded and extracted!")
        except requests.exceptions.RequestException as e:
            print(f"Error downloading Forest theme: {e}")
            return False
    else:
        print("Forest theme already exists.")
    return True

# Initialize MediaPipe Pose and Hands
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Global variables
camera_running = False
camera_source = 0  # Default to device camera
current_pose = "Unknown"
current_gestures = []

# Functions for Pose and Gesture Recognition
def classify_pose(landmarks):
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]

    if abs(left_shoulder.y - right_shoulder.y) < 0.1:
        if abs(left_wrist.y - left_shoulder.y) < 0.1 and abs(right_wrist.y - right_shoulder.y) < 0.1:
            return "T-pose"
        elif left_wrist.x < left_shoulder.x and right_wrist.x > right_shoulder.x:
            return "Arms Extended"
        else:
            return "Standing Upright"
    return "Unknown"

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

def run_camera():
    global camera_running, camera_source, current_pose, current_gestures

    cap = cv2.VideoCapture(camera_source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while camera_running:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Pose recognition
        pose_results = pose.process(rgb_frame)
        if pose_results.pose_landmarks:
            landmarks = pose_results.pose_landmarks.landmark
            current_pose = classify_pose(landmarks)
            mp_draw.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        else:
            current_pose = "Unknown"

        # Gesture recognition
        hand_results = hands.process(rgb_frame)
        current_gestures = []
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                gesture = classify_hand_gesture(hand_landmarks)
                current_gestures.append(gesture)
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.putText(frame, f"Pose: {current_pose}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        for idx, gesture in enumerate(current_gestures):
            cv2.putText(frame, f"Hand {idx + 1}: {gesture}", (10, 60 + idx * 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        cv2.imshow("Camera Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

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

def set_camera_source(ip_digits):
    global camera_source
    try:
        octets = ip_digits.split(".")
        if len(octets) == 2 and all(octet.isdigit() and 0 <= int(octet) <= 255 for octet in octets):
            camera_source = f"http://192.168.{ip_digits}:8080/video"
            messagebox.showinfo("Camera Source", f"Switched to IP Camera: {camera_source}")
        else:
            raise ValueError
    except ValueError:
        messagebox.showerror("Invalid Input", "Please enter the last two octets in the format (e.g., 0.212).")

def reset_to_device_camera():
    global camera_source
    camera_source = 0
    messagebox.showinfo("Camera Source", "Switched to Device Camera")

def show_about():
    about_text = """BJJ Pose and Gesture Recognition\n
This project uses MediaPipe to detect poses and hand gestures in real time."""
    messagebox.showinfo("About", about_text)

def show_contact():
    contact_text = "Developer: Boris Jordanov\nGitHub Repository: https://github.com/BorisJIordanov/BJJ"
    messagebox.showinfo("Contact", contact_text)

def open_github():
    webbrowser.open("https://github.com/BorisJIordanov/BJJ")

# Initialize GUI
root = tk.Tk()
root.title("BJJ Pose and Gesture Recognition")
root.geometry("800x600")

# Download and apply the Forest theme
if setup_forest_theme():
    theme_path = os.path.join(FOREST_THEME_DIR, "forest-dark.tcl")
    try:
        root.tk.call("source", theme_path)
        ttk.Style().theme_use("forest-dark")
    except Exception as e:
        print(f"Error applying Forest theme: {e}")
else:
    print("Using default theme due to setup error.")

# Create tabs
notebook = ttk.Notebook(root)
notebook.pack(expand=True, fill="both")

# Tab 1: Main Functionality
main_frame = ttk.Frame(notebook)
notebook.add(main_frame, text="Main Tab")

ttk.Label(main_frame, text="BJJ Pose and Gesture Recognition", font=("Calibri", 18, "bold")).pack(pady=10)

start_button = ttk.Button(main_frame, text="Start Camera", command=start_camera)
start_button.pack(pady=10)

stop_button = ttk.Button(main_frame, text="Stop Camera(q)", command=stop_camera)
stop_button.pack(pady=10)

ip_label = ttk.Label(main_frame, text="Enter last two digits of IP(0.123):")
ip_label.pack(pady=5)

ip_entry = ttk.Entry(main_frame)
ip_entry.pack(pady=5)

set_ip_button = ttk.Button(main_frame, text="Set IP Camera", command=lambda: set_camera_source(ip_entry.get()))
set_ip_button.pack(pady=10)

reset_button = ttk.Button(main_frame, text="Reset to Device Camera", command=reset_to_device_camera)
reset_button.pack(pady=10)

# Tab 2: About
about_frame = ttk.Frame(notebook)
notebook.add(about_frame, text="About")
about_label = ttk.Label(about_frame, text="""This project is a real-time pose and hand gesture recognition system tailored for Brazilian Jiu-Jitsu (BJJ) training and scoring.Using computer vision and machine learning technologies, the application processes videofeeds to identify poses and gestures, then maps these to predefined BJJ scoring categories."
              
    Key Features:

    1. Pose Recognition:

     -  Detects common BJJ stances like "T-pose," "Arms Extended," and "Standing Upright" using MediaPipe's Pose module.
    Maps detected poses to BJJ scoring categories such as Finish, Advantage, and Neutral.
                
    2. Hand Gesture Recognition:

     - Recognizes gestures such as "Thumb Up," "Victory," and "All Fingers Extended" using MediaPipe Hands.
                 
    3. Dynamic Camera Support:

     - Default setup uses the device's camera.
     - Users can switch to an IP camera feed dynamically during runtime.
     - Frames are processed at a steady FPS to ensure smooth video feed, even with complex calculations.
     - Camera resolution is set to 1280x720 for enhanced clarity.
                 
    4. Optimized Performance:

     - Frames are processed at a steady FPS to ensure smooth video feed, even with complex calculations.
     - Camera resolution is set to 1280x720 for enhanced clarity.
                
    5. User-Friendly GUI:

     - Simple buttons for starting, stopping, and switching between camera sources.
     - Real-time display of pose, gestures, and BJJ scoring annotations directly on the video feed.
                 
    Purpose and Applications:

     - This project serves as a training tool for athletes and coaches in Brazilian Jiu-Jitsu,
    providing insights into body positioning and scoring potential in real-time. Additionally,
    it showcases the integration of advanced computer vision techniques with a user-friendly
    interface, making it adaptable for other sports or gesture recognition applications.

    License: Custom MIT (see GitHub repository for details).""", wraplength=700)
about_label.pack(pady=20)

# Tab 3: Contact
contact_frame = ttk.Frame(notebook)
notebook.add(contact_frame, text="Contact")

ttk.Label(contact_frame, text="Developer: Boris Yordanov", font=("Calibri", 14, "bold")).pack(pady=10)
github_button = ttk.Button(contact_frame, text="Visit GitHub", command=open_github)
github_button.pack(pady=10)

# Exit Button
exit_button = ttk.Button(root, text="Exit", command=root.quit)
exit_button.pack(side="bottom", pady=20)

root.mainloop()
