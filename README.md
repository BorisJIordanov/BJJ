**BJJ Pose and Gesture Recognition**

This project is a real-time pose and hand gesture recognition system tailored for Brazilian Jiu-Jitsu (BJJ) training and scoring. Using computer vision and machine learning technologies, the application processes video feeds to identify poses and gestures, then maps these to predefined BJJ scoring categories.

**Key Features:**

**1. Pose Recognition:**

- Detects common BJJ stances like "T-pose," "Arms Extended," and "Standing Upright" using MediaPipe's Pose module.
- Maps detected poses to BJJ scoring categories such as Finish, Advantage, and Neutral.
  
**2. Hand Gesture Recognition:**

- Recognizes gestures such as "Thumb Up," "Victory," and "All Fingers Extended" using MediaPipe Hands.
  
**3. Dynamic Camera Support:**

- Default setup uses the device's camera.
- Users can switch to an IP camera feed dynamically during runtime.
- Frames are processed at a steady FPS to ensure smooth video feed, even with complex calculations. Camera resolution is set to 1280x720 for enhanced clarity.

**4. Optimized Performance:**

- Frames are processed at a steady FPS to ensure smooth video feed, even with complex calculations.
- Camera resolution is set to 1280x720 for enhanced clarity.

**5. User-Friendly GUI:**

- Simple buttons for starting, stopping, and switching between camera sources.
- Real-time display of pose, gestures, and BJJ scoring annotations directly on the video feed.

**Purpose and Applications:**

This project serves as a training tool for athletes and coaches in Brazilian Jiu-Jitsu, providing insights into body positioning and scoring potential in real-time. Additionally, it showcases the integration of advanced computer vision techniques with a user-friendly interface, making it adaptable for other sports or gesture recognition applications.
