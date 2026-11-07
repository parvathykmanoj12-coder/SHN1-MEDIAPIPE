🚀 FreshPick: 360° Assistive Freshness Tracker
💡 What's the idea?
FreshPick is an assistive non-visual kitchen tool designed for visually impaired individuals to manage their produce. Because users cannot see if an item is properly framed in the camera, FreshPick uses MediaPipe to track their hand and guides them into the center frame using spatial audio "sonar" beeps. It tracks the 3D rotation of their wrist to ensure a full 360° scan of the fruit (front and back), analyzes the ripeness using a Vision-Language Model, and speaks a priority list of which fruits to eat first and which to save for later.

🌐 Check it out!
Live Website: N/A (Desktop Python Application)

GitHub Repo: https://github.com/parvathykmanoj12-coder/SHN1-MEDIAPIPE.git
             https://github.com/annrose2277-glitch/SHN1-MEDIAPIPE.git

🛠️ What did you use?
Core Language: Python 3.10+

Spatial Tracking: Google MediaPipe (HandLandmarker API)

Computer Vision: OpenCV (cv2) & HSV Color Masking

AI / VLM: Google Gemini 3.8 Flash API (google-genai)

Accessibility: pyttsx3 (Offline Text-to-Speech) & winsound (Spatial Audio Cues)

👥 The Humans Behind It
Ann Rose Mathew – [@annrose2277-glitch]

Parvathy K Manoj – [@parvathykmanoj12-coder]


⚙️ Hardware Corner
Standard Laptop Webcam (RGB)

📸 Show Us What You Built!
https://drive.google.com/file/d/1Ws1r2bnMx5jifdZZR2zvMCnhVmQaRIuS/view?usp=drivesdk

✨ Cool Things It Does
Zero-Vision Audio Homing: Uses low/high pitch beeps to actively guide the user's hand to the center of the camera.

360° Palm Normal Tracking: Calculates the 3D vector of the palm to detect when the user has physically rotated their wrist 180 degrees to capture the back of the fruit.

Smart Priority Sorting: Remembers every fruit scanned during the session and ranks them from "Eat Today" (Use First) to "Store Longer" (Use Last).

Gesture Controls: Detects an "Open Palm" gesture to read aloud the pantry priority list, or a "Pinch" to force an instant scan.

Built with curiosity at Saturday Hack Night 💜
