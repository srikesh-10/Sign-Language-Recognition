# Sign Language Recognition - Project Setup

A Python-based foundation for building a real-time Sign Language Recognition system. This repository contains the configuration, environment verification tools, and test scripts necessary to establish a stable workspace using **Python 3.11.9**, **OpenCV**, and **MediaPipe**.

---

## 📁 Directory Structure

```text
SignLanguageRecognition/
├── venv/                   # Virtual environment (ignored by git)
├── .gitignore              # Python gitignore configuration
├── requirements.txt        # Project dependencies list
├── README.md               # Setup and verification documentation
├── setup_report.txt        # Generated environment verification report
├── test_environment.py     # Checks dependencies, Python version, & webcam
├── webcam_test.py          # Verifies raw camera capture & live FPS display
└── mediapipe_hand_test.py  # Runs real-time hand detection and landmark tracking
```

---

## 🛠️ Environment Setup

### 1. Prerequisites
Ensure you have **Python 3.11.9** installed on your system.

### 2. Activate Virtual Environment
Before installing packages or running scripts, activate the virtual environment in your terminal:

*   **PowerShell (Windows):**
    ```powershell
    .\venv\Scripts\Activate.ps1
    ```
*   **Command Prompt (Windows):**
    ```cmd
    .\venv\Scripts\activate.bat
    ```
*   **Bash/zsh (Linux/macOS):**
    ```bash
    source venv/bin/activate
    ```

### 3. Install Dependencies
Install all required packages from `requirements.txt`:
```bash
pip install -r requirements.txt
```

---

## 🚀 Usage Guide

### 1. Verify Environment
Run the environment verification script to ensure Python compatibility, check installed library versions, test the webcam, and produce a diagnostic report.
```bash
python test_environment.py
```
This generates a detailed `setup_report.txt` in the root directory.

### 2. Test Raw Webcam Feed
Run the basic webcam testing script to confirm OpenCV is reading video frames properly and displaying them with live FPS calculation.
```bash
python webcam_test.py
```
*   **Control**: Press `ESC` inside the video window to close the application.

### 3. Test Hand Tracking (MediaPipe)
Run the hand tracking script to perform real-time landmark detection. This script draws connections, coordinates, and shows the count of detected hands.
```bash
python mediapipe_hand_test.py
```
*   **Control**: Press `ESC` inside the video window to exit.

---

## 📦 Dependencies Reference

*   `opencv-python` (v4.13.0.92): Handles webcam capture, image processing, and window rendering.
*   `mediapipe` (v0.10.35): Provides low-latency ML solutions for hand and landmark detection.
*   `numpy` (v2.4.6): Performs fast numerical array manipulations on frame data.

---

## ❓ Troubleshooting

*   **Webcam fails to open:**
    *   Verify the camera is plugged in.
    *   Ensure no other application (like Zoom, Teams, or the native Camera app) is using the webcam.
    *   On Windows, check **Settings > Privacy & security > Camera** to ensure apps are allowed to access your camera.
*   **MediaPipe issues on Python 3.12+:**
    *   This project is strictly configured and tested for **Python 3.11.9** to prevent version compatibility issues with MediaPipe.
 
## Author

*  P.Srikesh
