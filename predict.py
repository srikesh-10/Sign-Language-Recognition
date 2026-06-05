"""
=============================================================================
predict.py - Real-Time Sign Language Communication System
=============================================================================
Project  : Sign Language Recognition
Python   : 3.11.9
MediaPipe: 0.10.14

Description:
    Opens the webcam and performs real-time sign language recognition using
    MediaPipe Hands for feature extraction and a trained Random Forest
    model for classification.

    Optimizations:
    - HD Camera Feed (1280x720) @ 30 FPS with low latency (BUFFERSIZE=1).
    - Real-time Image Enhancement (Brightness/Contrast/Noise Reduction).
    - Auto Lighting Assist to detect and warn about low-light environments.
    - MediaPipe tuned with high tracking & detection confidence (0.8).
    - Advanced Performance Monitoring (Current FPS, Avg FPS, Frame Time).
    - Professional semi-transparent HUD layout with improved readability.
=============================================================================
"""

import cv2
import mediapipe as mp
import numpy as np
import joblib
import os
import time
import sys
import threading
from collections import deque, Counter
from datetime import datetime
import traceback
import pyttsx3

# ---------------------------------------------------------------------------
# Constants & Settings
# ---------------------------------------------------------------------------
MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, "sign_model.pkl")
ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.pkl")
SAVED_DIR = "saved_sentences"

CONFIDENCE_THRESHOLD = 0.50  # Only accept predictions with >= 50% confidence
SMOOTHING_FRAMES = 10        # Buffer size for majority voting

# Camera Quality Settings
TARGET_WIDTH = 1280
TARGET_HEIGHT = 720
TARGET_FPS = 30

# Image Enhancement & Lighting Settings
APPLY_ENHANCEMENT = True
LIGHTING_THRESHOLD = 60      # If average pixel intensity < 60, show low light warning

# ---------------------------------------------------------------------------
# TTS Function (Threaded)
# ---------------------------------------------------------------------------
def speak_text(text: str):
    """
    Runs Text-to-Speech in a separate thread to prevent UI freezing.
    """
    if not text.strip():
        return
    try:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"[-] TTS Error: {e}")

# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------
def extract_landmarks(hand_landmarks) -> list:
    """
    Flatten MediaPipe hand landmarks into a plain list of floats.
    """
    flat = []
    for lm in hand_landmarks.landmark:
        flat.extend([lm.x, lm.y, lm.z])
    return flat

def apply_image_enhancement(frame):
    """
    Applies real-time image enhancement to improve detection quality
    while maintaining high FPS.
    - Contrast/Brightness mapping
    - Mild Gaussian Blur for noise reduction
    """
    if not APPLY_ENHANCEMENT:
        return frame

    # 1. Mild Contrast (alpha) and Brightness (beta) correction
    # alpha > 1 increases contrast, beta > 0 increases brightness
    enhanced = cv2.convertScaleAbs(frame, alpha=1.1, beta=10)

    # 2. Mild noise reduction (small kernel to avoid heavy blurring)
    enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)

    return enhanced

def check_lighting(frame) -> bool:
    """
    Calculates average brightness of the frame.
    Returns True if low light is detected.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    avg_brightness = np.mean(gray)
    return avg_brightness < LIGHTING_THRESHOLD

def draw_text_with_shadow(frame, text: str, pos: tuple,
                          font_scale: float = 0.8,
                          fg_color: tuple = (255, 255, 255),
                          thickness: int = 2) -> None:
    """
    Draw text with a dark drop-shadow for improved readability on any background.
    """
    x, y = pos
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Shadow
    cv2.putText(frame, text, (x + 2, y + 2), font, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    # Foreground
    cv2.putText(frame, text, (x, y), font, font_scale, fg_color, thickness, cv2.LINE_AA)

def draw_hud(frame, metrics: dict, state: dict) -> None:
    """
    Render a professional multi-section HUD displaying performance metrics,
    predictions, tracking quality, and session statistics.
    """
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # --- Semi-Transparent Panels ---
    # Top Panel (Prediction, Sentence, Warnings)
    cv2.rectangle(overlay, (0, 0), (w, 160), (20, 20, 20), -1)
    
    # Bottom Panel (Stats, Controls)
    cv2.rectangle(overlay, (0, h - 130), (w, h), (20, 20, 20), -1)
    
    # Left Sidebar (Performance Monitoring)
    cv2.rectangle(overlay, (0, 160), (330, 320), (25, 25, 25), -1)
    
    # Apply alpha blending for HUD transparency
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # --- Top Panel Info ---
    # Hand Tracking Status
    tracking_color = (0, 255, 0) if state['tracking_status'] == "GOOD" else (0, 165, 255) if state['tracking_status'] == "MODERATE" else (0, 0, 255)
    tracking_text = f"Tracking: {state['tracking_status']}"
    draw_text_with_shadow(frame, tracking_text, (w - 300, 35), font_scale=0.7, fg_color=tracking_color)

    # Auto Lighting Assist Warning
    if metrics['lighting_warning']:
        draw_text_with_shadow(frame, "LOW LIGHT DETECTED", (w - 300, 75), font_scale=0.6, fg_color=(0, 0, 255))
        draw_text_with_shadow(frame, "Improve room lighting for better accuracy", (w - 380, 100), font_scale=0.5, fg_color=(0, 165, 255))
    
    # Low FPS Warning
    if metrics['fps'] > 0 and metrics['fps'] < 15:
         draw_text_with_shadow(frame, "LOW FPS DETECTED", (w - 300, 130), font_scale=0.6, fg_color=(0, 0, 255))

    # Main Prediction
    pred_color = (0, 255, 255) if state['prediction'] != "UNKNOWN" else (150, 150, 150)
    display_pred = state['prediction'] if state['hand_detected'] else "Waiting..."
    draw_text_with_shadow(frame, f"Prediction: {display_pred}", (20, 45), font_scale=1.3, fg_color=pred_color, thickness=3)

    # Confidence Score
    conf_color = (0, 255, 0) if state['confidence'] >= 0.8 else (0, 200, 255) if state['confidence'] >= 0.5 else (0, 0, 255)
    conf_str = f"Confidence: {state['confidence'] * 100:.1f}%" if state['hand_detected'] and state['prediction'] != "UNKNOWN" else "Confidence: N/A"
    draw_text_with_shadow(frame, conf_str, (20, 90), font_scale=0.8, fg_color=conf_color, thickness=2)

    # Word Builder / Sentence Box
    sentence_disp = state['sentence'] if state['sentence'] else "_"
    draw_text_with_shadow(frame, f"Sentence: {sentence_disp}", (20, 135), font_scale=1.1, fg_color=(255, 255, 255), thickness=3)

    # Dynamic Notification / Status Message (Centered horizontally below top panel)
    if state['status_message']:
        (sw, sh), _ = cv2.getTextSize(state['status_message'], cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        draw_text_with_shadow(frame, state['status_message'], ((w - sw) // 2, 195), font_scale=0.8, fg_color=(0, 255, 0), thickness=2)

    # --- Left Sidebar (Performance Monitoring) ---
    draw_text_with_shadow(frame, "Performance Monitoring:", (15, 190), font_scale=0.65, fg_color=(200, 200, 255))
    draw_text_with_shadow(frame, f"Resolution : {metrics['resolution']}", (15, 225), font_scale=0.55, fg_color=(220, 220, 220))
    draw_text_with_shadow(frame, f"Curr FPS   : {metrics['fps']:.1f}", (15, 255), font_scale=0.55, fg_color=(220, 220, 220))
    draw_text_with_shadow(frame, f"Avg FPS    : {metrics['avg_fps']:.1f}", (15, 285), font_scale=0.55, fg_color=(220, 220, 220))
    draw_text_with_shadow(frame, f"Frame Time : {metrics['proc_time']:.1f} ms", (15, 315), font_scale=0.55, fg_color=(220, 220, 220))

    # --- Bottom Panel (Stats & Controls) ---
    start_str = state['stats']['start_time'].strftime("%H:%M:%S")
    stats_str1 = f"Session Start: {start_str}   |   Total Accepted: {state['stats']['accepted']}   |   Current Length: {len(state['sentence'])}"
    draw_text_with_shadow(frame, stats_str1, (20, h - 90), font_scale=0.65, fg_color=(200, 200, 200))

    controls1 = "[SPACE]: Append  |  [BACKSPACE]: Remove  |  [C]: Clear"
    controls2 = "[ENTER]: Speak   |  [S]: Save            |  [ESC]: Exit"
    draw_text_with_shadow(frame, controls1, (20, h - 45), font_scale=0.6, fg_color=(150, 255, 150))
    draw_text_with_shadow(frame, controls2, (20, h - 15), font_scale=0.6, fg_color=(150, 255, 150))

# ---------------------------------------------------------------------------
# Main Application Pipeline
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("   Sign Language Communication System - High Quality Build")
    print("=" * 60)

    # 1. Project Structure - Ensure save directory exists
    os.makedirs(SAVED_DIR, exist_ok=True)

    # 2. Robustness - Validate models
    if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODER_PATH):
        print("[-] Critical Error: Classification model or Label encoder not found.")
        print("[*] Please ensure you have trained the model and generated the .pkl files.")
        sys.exit(1)

    print("[*] Loading model and label encoder...")
    try:
        model = joblib.load(MODEL_PATH)
        encoder = joblib.load(ENCODER_PATH)
        print("[+] Model and Encoder loaded successfully.")
    except Exception as e:
        print(f"[-] Failed to load model/encoder: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 3. MediaPipe Improvements
    # Increased tracking and detection confidence to 0.8 for maximum reliability
    print("[*] Initializing MediaPipe Hands (v0.10+)...")
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.8,  # Increased for reliability
        min_tracking_confidence=0.8    # Increased for stability
    )

    print("[*] Accessing Primary Webcam...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("[-] Critical Error: Cannot open webcam. Please check your camera permissions and connection.")
        sys.exit(1)

    # 4. Camera Quality Improvements
    # Force HD resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_HEIGHT)
    # Set webcam FPS
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
    # Reduce webcam latency by setting minimal buffer size
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    resolution_str = f"{actual_width}x{actual_height}"
    print(f"[+] Camera initialized successfully at resolution: {resolution_str}")

    if actual_width < TARGET_WIDTH or actual_height < TARGET_HEIGHT:
        print("[-] Warning: Camera does not support full 1280x720 HD. Falling back gracefully.")

    # State Variables & Tracking
    prediction_history = deque(maxlen=SMOOTHING_FRAMES)
    
    # Word Builder variables
    sentence = ""
    status_message = ""
    status_msg_time = 0.0
    last_action_time = time.time()
    
    # Performance Monitoring
    prev_frame_time = time.time()
    session_start_time = time.time()
    total_frames = 0
    
    stats = {
        'start_time': datetime.now(),
        'accepted': 0
    }
    
    window_name = "Sign Language Communication System (HD)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    print("[+] System fully initialized and ready.")
    print("[*] Focus the video window to use keyboard controls.")
    print("[*] Press ESC inside the video window to quit gracefully.")
    
    try:
        while True:
            loop_start = time.time()
            
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[-] Warning: Failed to capture frame from webcam. Retrying...")
                time.sleep(0.5)
                continue

            # Mirror the frame naturally for the user
            frame = cv2.flip(frame, 1)
            
            # Apply Image Enhancements (Brightness, Contrast, Noise Reduction)
            frame = apply_image_enhancement(frame)
            
            # Auto Lighting Assist
            is_low_light = check_lighting(frame)

            # MediaPipe operates on RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            hand_detected = False
            raw_prediction = "UNKNOWN"
            confidence = 0.0
            tracking_status = "NONE"

            if results.multi_hand_landmarks:
                hand_detected = True
                hand_lms = results.multi_hand_landmarks[0]
                
                # Draw skeleton landmarks on top of the hand
                mp_drawing.draw_landmarks(
                    frame, hand_lms, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                # Extract and format features for the model
                landmarks = extract_landmarks(hand_lms)
                X_input = np.array([landmarks])

                try:
                    # Model Inference
                    probabilities = model.predict_proba(X_input)[0]
                    max_idx = np.argmax(probabilities)
                    confidence = probabilities[max_idx]

                    if confidence >= CONFIDENCE_THRESHOLD:
                        encoded_class = model.classes_[max_idx]
                        raw_prediction = encoder.inverse_transform([encoded_class])[0]
                    else:
                        raw_prediction = "UNKNOWN"
                except Exception as e:
                    print(f"[-] Inference computation error: {e}")
                    raw_prediction = "UNKNOWN"

                # Buffer the prediction for smoothing
                prediction_history.append(raw_prediction)
                
                # Calculate Tracking Status based on confidence
                if confidence >= 0.80:
                    tracking_status = "GOOD"
                elif confidence >= 0.50:
                    tracking_status = "MODERATE"
                else:
                    tracking_status = "POOR"
            else:
                # If no hand, immediately insert UNKNOWN to rapidly decay old predictions
                prediction_history.append("UNKNOWN")

            # Phase 1: Majority Voting for Anti-Flickering
            if len(prediction_history) > 0:
                smoothed_prediction = Counter(prediction_history).most_common(1)[0][0]
            else:
                smoothed_prediction = "UNKNOWN"

            # Performance Monitoring (FPS & Loop Time)
            loop_end = time.time()
            proc_time_ms = (loop_end - loop_start) * 1000.0
            
            # Current FPS
            curr_fps = 1.0 / (loop_end - prev_frame_time) if (loop_end - prev_frame_time) > 0 else 0.0
            prev_frame_time = loop_end
            
            # Average FPS
            total_frames += 1
            session_duration = loop_end - session_start_time
            avg_fps = total_frames / session_duration if session_duration > 0 else 0.0

            # Manage temporary status messages (Fade out after 2 seconds)
            if status_message and (loop_end - status_msg_time > 2.0):
                status_message = ""

            # Prepare metrics and state dictionaries
            metrics = {
                'fps': curr_fps,
                'avg_fps': avg_fps,
                'proc_time': proc_time_ms,
                'resolution': resolution_str,
                'lighting_warning': is_low_light
            }
            state = {
                'hand_detected': hand_detected,
                'prediction': smoothed_prediction,
                'confidence': confidence,
                'tracking_status': tracking_status,
                'sentence': sentence,
                'stats': stats,
                'status_message': status_message
            }

            # Render the updated HUD
            draw_hud(frame, metrics, state)
            cv2.imshow(window_name, frame)

            # Phase 2 & 3 & 4: Keyboard Input Processing
            key = cv2.waitKey(1) & 0xFF
            now = time.time()
            
            if key == 27: # [ESC] key
                print("[*] ESC pressed. Exiting...")
                break
                
            elif key == 32: # [SPACE] key
                # Enforce a 0.5s cooldown to prevent accidental duplicates
                if now - last_action_time > 0.5:
                    if smoothed_prediction != "UNKNOWN":
                        sentence += smoothed_prediction
                        stats['accepted'] += 1
                        last_action_time = now
                    else:
                        status_message = "Prediction Unstable!"
                        status_msg_time = now
                        
            elif key == 8: # [BACKSPACE] key
                if now - last_action_time > 0.2: # Faster repeat for deletions
                    if len(sentence) > 0:
                        sentence = sentence[:-1]
                    last_action_time = now
                    
            elif key == ord('c') or key == ord('C'): # [C] key
                if len(sentence) > 0:
                    sentence = ""
                    status_message = "Sentence Cleared"
                    status_msg_time = now
                
            elif key == 13: # [ENTER] key
                if now - last_action_time > 1.0: # 1s cooldown to prevent overlapping speech
                    if sentence:
                        status_message = "Speaking..."
                        status_msg_time = now
                        threading.Thread(target=speak_text, args=(sentence,), daemon=True).start()
                    last_action_time = now
                    
            elif key == ord('s') or key == ord('S'): # [S] key
                if now - last_action_time > 1.0:
                    if sentence:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"sentence_{timestamp}.txt"
                        filepath = os.path.join(SAVED_DIR, filename)
                        try:
                            with open(filepath, "w", encoding="utf-8") as f:
                                f.write(sentence)
                            status_message = "Sentence Saved Successfully"
                            status_msg_time = now
                            print(f"[+] Saved conversation to: {filepath}")
                        except Exception as e:
                            status_message = "Save Failed!"
                            status_msg_time = now
                            print(f"[-] Failed to save sentence: {e}")
                    last_action_time = now

    except KeyboardInterrupt:
        print("\n[*] Manual interruption (Ctrl+C) received. Shutting down gracefully...")
    except Exception as e:
        print(f"[-] FATAL Unexpected application error: {e}")
        traceback.print_exc()
    finally:
        print("[*] Releasing resources and tearing down...")
        if cap is not None:
            cap.release()
        if hands is not None:
            hands.close()
        cv2.destroyAllWindows()
        print("[+] System terminated successfully.")
        print("==================================================")

if __name__ == "__main__":
    main()
