"""
=============================================================================
app.py - Sign Language Recognition Flask Web Application
=============================================================================
Project  : SignLang AI
Python   : 3.11.9
MediaPipe: 0.10.14

Description:
    Flask web application that provides real-time sign language recognition
    via webcam. Uses MediaPipe Hands for landmark extraction, a trained
    Random Forest classifier for prediction, and pyttsx3 for text-to-speech.

Routes:
    /              -> Main dashboard
    /video_feed    -> MJPEG video stream
    /api/state     -> JSON endpoint for real-time prediction state
    /api/append    -> POST: append predicted character to sentence
    /api/delete    -> POST: delete last character from sentence
    /api/clear     -> POST: clear sentence
    /api/speak     -> POST: speak current sentence via TTS
    /api/save      -> POST: save sentence to file
=============================================================================
"""

import cv2
import mediapipe as mp
import numpy as np
import joblib
import os
import sys
import time
import threading
import pyttsx3
import json
import traceback
from flask import Flask, render_template, Response, jsonify, request
from collections import deque, Counter
from datetime import datetime

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, "sign_model.pkl")
ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.pkl")
SAVED_DIR = "saved_sentences"

CONFIDENCE_THRESHOLD = 0.50
SMOOTHING_FRAMES = 10

# ---------------------------------------------------------------------------
# Flask App
# ---------------------------------------------------------------------------
app = Flask(__name__)

# ---------------------------------------------------------------------------
# Global State (thread-safe via GIL for simple reads/writes)
# ---------------------------------------------------------------------------
class AppState:
    """Holds the shared application state between the video thread and Flask."""
    def __init__(self):
        self.prediction = "—"
        self.confidence = 0.0
        self.tracking_status = "NONE"
        self.hand_detected = False
        self.fps = 0.0
        self.sentence = ""
        self.accepted_count = 0
        self.session_start = datetime.now()
        self.is_speaking = False
        self.status_message = ""
        self.status_time = 0.0
        self.camera_active = False
        self.avg_confidence = 0.0
        self.confidence_history = deque(maxlen=100)
        self.lock = threading.Lock()

state = AppState()

# ---------------------------------------------------------------------------
# Load ML Model
# ---------------------------------------------------------------------------
print("[*] Loading model and label encoder...")
if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODER_PATH):
    print("[-] FATAL: Model files not found. Run train_model.py first.")
    sys.exit(1)

try:
    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
    print("[+] Model and encoder loaded successfully.")
except Exception as e:
    print(f"[-] Failed to load model: {e}")
    traceback.print_exc()
    sys.exit(1)

# Ensure save directory exists
os.makedirs(SAVED_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# MediaPipe Setup
# ---------------------------------------------------------------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)

# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------
def extract_landmarks(hand_landmarks) -> list:
    """Flatten MediaPipe hand landmarks into a list of 63 floats."""
    flat = []
    for lm in hand_landmarks.landmark:
        flat.extend([lm.x, lm.y, lm.z])
    return flat


def speak_text_sync(text: str):
    """Run TTS in a blocking manner (called from a thread)."""
    if not text.strip():
        return
    state.is_speaking = True
    try:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"[-] TTS Error: {e}")
    finally:
        state.is_speaking = False


# ---------------------------------------------------------------------------
# Video Streaming Generator
# ---------------------------------------------------------------------------
prediction_history = deque(maxlen=SMOOTHING_FRAMES)

def generate_frames():
    """
    Generator that captures webcam frames, runs MediaPipe + ML inference,
    annotates the frame with hand landmarks, and yields MJPEG frames.
    """
    global prediction_history

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[-] Cannot open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    state.camera_active = True
    prev_time = time.time()

    print("[+] Video stream started.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            frame = cv2.flip(frame, 1)

            # Mild enhancement
            frame = cv2.convertScaleAbs(frame, alpha=1.05, beta=5)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            hand_detected = False
            raw_prediction = "—"
            confidence = 0.0
            tracking_status = "NONE"

            if results.multi_hand_landmarks:
                hand_detected = True
                hand_lms = results.multi_hand_landmarks[0]

                # Draw skeleton on the frame
                mp_drawing.draw_landmarks(
                    frame, hand_lms, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                landmarks = extract_landmarks(hand_lms)
                X_input = np.array([landmarks])

                try:
                    probabilities = model.predict_proba(X_input)[0]
                    max_idx = np.argmax(probabilities)
                    confidence = float(probabilities[max_idx])

                    if confidence >= CONFIDENCE_THRESHOLD:
                        encoded_class = model.classes_[max_idx]
                        raw_prediction = encoder.inverse_transform([encoded_class])[0]
                    else:
                        raw_prediction = "—"
                except Exception:
                    raw_prediction = "—"

                prediction_history.append(raw_prediction)

                if confidence >= 0.80:
                    tracking_status = "GOOD"
                elif confidence >= 0.50:
                    tracking_status = "MODERATE"
                else:
                    tracking_status = "POOR"

                state.confidence_history.append(confidence)
            else:
                prediction_history.append("—")

            # Majority voting
            if len(prediction_history) > 0:
                smoothed = Counter(prediction_history).most_common(1)[0][0]
            else:
                smoothed = "—"

            # FPS
            now = time.time()
            fps = 1.0 / (now - prev_time) if (now - prev_time) > 0 else 0.0
            prev_time = now

            # Avg confidence
            avg_conf = (sum(state.confidence_history) / len(state.confidence_history)
                        if state.confidence_history else 0.0)

            # Update shared state
            state.prediction = smoothed
            state.confidence = confidence
            state.tracking_status = tracking_status
            state.hand_detected = hand_detected
            state.fps = fps
            state.avg_confidence = avg_conf

            # Encode to JPEG
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    except GeneratorExit:
        pass
    finally:
        cap.release()
        state.camera_active = False
        print("[*] Video stream stopped.")


# ---------------------------------------------------------------------------
# Flask Routes
# ---------------------------------------------------------------------------
@app.route('/')
def index():
    """Serve the main dashboard."""
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """MJPEG video stream endpoint."""
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/api/state')
def get_state():
    """Return the current prediction state as JSON."""
    elapsed = (datetime.now() - state.session_start).total_seconds()
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)

    return jsonify({
        'prediction': state.prediction,
        'confidence': round(state.confidence * 100, 1),
        'tracking': state.tracking_status,
        'hand_detected': state.hand_detected,
        'fps': round(state.fps, 1),
        'sentence': state.sentence,
        'accepted': state.accepted_count,
        'duration': f"{minutes:02d}:{seconds:02d}",
        'avg_confidence': round(state.avg_confidence * 100, 1),
        'is_speaking': state.is_speaking,
        'camera_active': state.camera_active,
        'status_message': state.status_message,
    })


@app.route('/api/append', methods=['POST'])
def append_char():
    """Append the current stable prediction to the sentence."""
    pred = state.prediction
    if pred and pred != "—" and state.hand_detected:
        state.sentence += pred
        state.accepted_count += 1
        return jsonify({'ok': True, 'sentence': state.sentence, 'char': pred})
    return jsonify({'ok': False, 'error': 'No stable prediction available'})


@app.route('/api/space', methods=['POST'])
def add_space():
    """Add a space to the sentence."""
    state.sentence += " "
    return jsonify({'ok': True, 'sentence': state.sentence})


@app.route('/api/delete', methods=['POST'])
def delete_char():
    """Remove the last character from the sentence."""
    if state.sentence:
        state.sentence = state.sentence[:-1]
        return jsonify({'ok': True, 'sentence': state.sentence})
    return jsonify({'ok': False, 'error': 'Sentence is empty'})


@app.route('/api/clear', methods=['POST'])
def clear_sentence():
    """Clear the entire sentence."""
    state.sentence = ""
    return jsonify({'ok': True, 'sentence': ''})


@app.route('/api/speak', methods=['POST'])
def speak():
    """Speak the current sentence via TTS in a background thread."""
    if state.is_speaking:
        return jsonify({'ok': False, 'error': 'Already speaking'})
    if not state.sentence.strip():
        return jsonify({'ok': False, 'error': 'Nothing to speak'})

    threading.Thread(target=speak_text_sync, args=(state.sentence,), daemon=True).start()
    return jsonify({'ok': True})


@app.route('/api/save', methods=['POST'])
def save_sentence():
    """Save the current sentence to a timestamped text file."""
    if not state.sentence.strip():
        return jsonify({'ok': False, 'error': 'Nothing to save'})

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"sentence_{timestamp}.txt"
    filepath = os.path.join(SAVED_DIR, filename)

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(state.sentence)
        print(f"[+] Saved: {filepath}")
        return jsonify({'ok': True, 'filename': filename})
    except Exception as e:
        print(f"[-] Save failed: {e}")
        return jsonify({'ok': False, 'error': str(e)})


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print("=" * 60)
    print("   SignLang AI - Flask Web Application")
    print("=" * 60)
    print(f"   Model  : {os.path.abspath(MODEL_PATH)}")
    print(f"   Encoder: {os.path.abspath(ENCODER_PATH)}")
    print(f"   Saves  : {os.path.abspath(SAVED_DIR)}")
    print("=" * 60)
    print("\n   Open http://127.0.0.1:5000 in your browser.\n")

    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
