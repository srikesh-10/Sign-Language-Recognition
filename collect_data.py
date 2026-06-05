"""
=============================================================================
collect_data.py - Sign Language Dataset Collection Module
=============================================================================
Project  : Sign Language Recognition
Author   : (Your Name)
Python   : 3.11.9
MediaPipe: 0.10.14

Description:
    Captures live webcam frames, detects one hand using MediaPipe Hands,
    extracts all 21 hand landmarks (x, y, z per point = 63 features) and
    writes them as rows to a CSV file.

Controls:
    A-E -> Select the active letter label
    0-9 -> Select the active number label
    I  -> Toggle Alphabet Improvement Mode
    T  -> Toggle Auto-Save (saves every 0.3s)
    S  -> Manual Save (only when hand visible)
    Q  -> Quit and close the application

Output CSV format (64 columns total):
    label, x1, y1, z1, x2, y2, z2, ..., x21, y21, z21

Goal: Collect 300 samples per sign (A, B, C, D, E) and 200 per number.
=============================================================================
"""

import cv2
import mediapipe as mp
import csv
import os
import time
import sys

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CSV_DIR  = os.path.join("dataset", "raw")
CSV_FILE = os.path.join(CSV_DIR, "sign_landmarks.csv")

NUM_LANDMARKS = 21

TARGET_SAMPLES_ALPHA = 300
TARGET_SAMPLES_NUM = 300

def get_target_samples(label: str) -> int:
    if label and label.isalpha():
        return TARGET_SAMPLES_ALPHA
    return TARGET_SAMPLES_NUM

SIGN_KEYS = {}
for char in "abcde":
    SIGN_KEYS[ord(char)] = char.upper()
for num in range(10):
    SIGN_KEYS[ord(str(num))] = str(num)

CSV_HEADER = ["label"]
for i in range(1, NUM_LANDMARKS + 1):
    CSV_HEADER += [f"x{i}", f"y{i}", f"z{i}"]

REMINDERS = [
    "Move hand slightly",
    "Change angle",
    "Change distance",
    "Avoid identical poses"
]

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def ensure_csv_exists(filepath: str) -> None:
    directory = os.path.dirname(filepath)
    if directory and not os.path.isdir(directory):
        os.makedirs(directory, exist_ok=True)
        print(f"[+] Created directory: {directory}")

    if not os.path.isfile(filepath):
        try:
            with open(filepath, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(CSV_HEADER)
            print(f"[+] Created CSV with header: {filepath}")
        except OSError as e:
            print(f"[-] Failed to create CSV file: {e}")
            sys.exit(1)
    else:
        print(f"[*] Appending to existing CSV: {filepath}")

def count_samples_per_label(filepath: str) -> dict:
    counts = {label: 0 for label in SIGN_KEYS.values()}
    if not os.path.isfile(filepath):
        return counts

    try:
        with open(filepath, mode='r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                lbl = row.get("label", "").strip().upper()
                if lbl in counts:
                    counts[lbl] += 1
    except Exception as e:
        print(f"[!] Warning: Could not read existing sample counts: {e}")

    return counts

def extract_landmarks(hand_landmarks) -> list:
    flat = []
    for lm in hand_landmarks.landmark:
        flat.extend([lm.x, lm.y, lm.z])
    return flat

def append_sample_to_csv(filepath: str, label: str, landmark_values: list) -> bool:
    try:
        with open(filepath, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([label] + [f"{v:.6f}" for v in landmark_values])
        return True
    except OSError as e:
        print(f"[-] Failed to write sample: {e}")
        return False

# ---------------------------------------------------------------------------
# HUD (Heads-Up Display) drawing helpers
# ---------------------------------------------------------------------------

def draw_text_with_shadow(frame, text: str, pos: tuple,
                          font_scale: float = 0.8,
                          fg_color: tuple = (255, 255, 255),
                          thickness: int = 2) -> None:
    x, y = pos
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, text, (x + 2, y + 2),
                font, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y),
                font, font_scale, fg_color, thickness, cv2.LINE_AA)

def draw_hud(frame, current_label: str, sample_counts: dict,
             fps: float, hand_detected: bool, auto_save: bool,
             alphabet_mode: bool, current_reminder: str) -> None:
    h, w = frame.shape[:2]

    # Calculate dynamic banner height based on active modes
    y_offset = 155
    if alphabet_mode:
        y_offset += 30
    if current_reminder:
        y_offset += 30
    banner_height = y_offset + 70

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_height), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    # FPS
    fps_color = (0, 255, 100)
    draw_text_with_shadow(frame, f"FPS: {fps:.1f}", (15, 30), font_scale=0.7, fg_color=fps_color)

    # Label & Count Info
    label_color = (0, 200, 255) if current_label != "None" else (100, 100, 100)
    target = get_target_samples(current_label) if current_label != "None" else 0
    current_count = sample_counts.get(current_label, 0) if current_label != "None" else 0
    remaining = max(0, target - current_count) if current_label != "None" else 0

    draw_text_with_shadow(frame, f"Label: {current_label}", (15, 65), font_scale=0.8, fg_color=label_color, thickness=2)
    draw_text_with_shadow(frame, f"Count: {current_count} / {target}", (15, 95), font_scale=0.7, fg_color=(255, 255, 255), thickness=1)
    draw_text_with_shadow(frame, f"Remaining: {remaining}", (15, 125), font_scale=0.7, fg_color=(255, 255, 255), thickness=1)

    # Auto Save Status
    auto_text = "AUTO SAVE: ON" if auto_save else "AUTO SAVE: OFF"
    auto_color = (0, 255, 0) if auto_save else (0, 0, 255)
    draw_text_with_shadow(frame, auto_text, (15, 155), font_scale=0.7, fg_color=auto_color, thickness=2)

    # Alphabet Improvement Mode
    draw_y = 155
    if alphabet_mode:
        draw_y += 30
        draw_text_with_shadow(frame, "ALPHABET IMPROVEMENT MODE", (15, draw_y), font_scale=0.7, fg_color=(255, 100, 100), thickness=2)
    
    # Quality Collection Mode Reminder
    if current_reminder:
        draw_y += 30
        draw_text_with_shadow(frame, f"Reminder: {current_reminder}", (15, draw_y), font_scale=0.6, fg_color=(0, 255, 255), thickness=1)

    # Sample counts for all labels
    letters = [lbl for lbl in sorted(SIGN_KEYS.values()) if lbl.isalpha()]
    numbers = [lbl for lbl in sorted(SIGN_KEYS.values()) if lbl.isdigit()]
    
    letters_str = "  ".join([f"{lbl}:{sample_counts.get(lbl, 0)}/{get_target_samples(lbl)}" for lbl in letters])
    numbers_str = "  ".join([f"{lbl}:{sample_counts.get(lbl, 0)}/{get_target_samples(lbl)}" for lbl in numbers])
    
    draw_y += 30
    draw_text_with_shadow(frame, letters_str, (15, draw_y), font_scale=0.55, fg_color=(200, 200, 200), thickness=1)
    draw_text_with_shadow(frame, numbers_str, (15, draw_y + 30), font_scale=0.55, fg_color=(200, 200, 200), thickness=1)

    # Hand detection status
    if hand_detected:
        badge_text  = "HAND DETECTED"
        badge_color = (0, 220, 80)
    else:
        badge_text  = "NO HAND"
        badge_color = (50, 50, 200)

    (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    badge_x = w - tw - 25
    cv2.putText(frame, badge_text, (badge_x, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, badge_color, 2, cv2.LINE_AA)

    # Target Reached / Complete Status
    if current_label != "None":
        if current_count >= target:
            draw_text_with_shadow(frame, f"{current_label} COMPLETE ({target}/{target})", (w // 2 - 200, h // 2 - 50), font_scale=1.0, fg_color=(0, 255, 0), thickness=3)
            draw_text_with_shadow(frame, "TARGET REACHED", (w // 2 - 120, h // 2), font_scale=0.8, fg_color=(0, 255, 255), thickness=2)

    # Bottom instruction bar
    instr_y = h - 12
    instr = "[A-E] [0-9] Label  [I] Improve Mode  [T] Auto Save  [S] Save  [Q] Quit"
    draw_text_with_shadow(frame, instr, (10, instr_y), font_scale=0.5, fg_color=(180, 180, 180), thickness=1)

def draw_save_flash(frame) -> None:
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 255, 80), 6)
    draw_text_with_shadow(frame, "SAVED!", (w // 2 - 45, h // 2), font_scale=1.5, fg_color=(0, 255, 80), thickness=3)

# ---------------------------------------------------------------------------
# Main application loop
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("   Sign Language Dataset Collection - collect_data.py")
    print("=" * 60)
    print(f"  CSV output : {os.path.abspath(CSV_FILE)}")
    print(f"  Controls   : [A-E, 0-9] Label | [I] Improve Mode | [T] Auto Save | [S] Save | [Q] Quit")
    print("=" * 60)

    ensure_csv_exists(CSV_FILE)

    sample_counts = count_samples_per_label(CSV_FILE)
    print(f"[*] Existing sample counts: {sample_counts}")

    mp_hands         = mp.solutions.hands
    mp_drawing       = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[-] Error: Cannot open webcam. Check connection and permissions.")
        sys.exit(1)

    current_label  = "None"
    hand_detected  = False
    current_landmarks = []
    
    auto_save = False
    alphabet_mode = False
    last_auto_save_time = 0.0
    AUTO_SAVE_INTERVAL = 0.3

    flash_frames   = 0
    FLASH_DURATION = 8

    prev_time = time.time()
    fps        = 0.0

    reminder_start_time = time.time()
    reminder_index = 0
    current_reminder = REMINDERS[0]

    window_name = "Sign Language Data Collection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("[*] Webcam started. Select a label (A-E, 0-9) then press S to save samples.")

    new_samples_collected = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[-] Failed to capture frame. Retrying...")
                continue

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = hands.process(rgb_frame)

            hand_detected     = False
            current_landmarks = []

            if results.multi_hand_landmarks:
                hand_lms = results.multi_hand_landmarks[0]
                hand_detected     = True
                current_landmarks = extract_landmarks(hand_lms)

                mp_drawing.draw_landmarks(
                    frame,
                    hand_lms,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )

            now      = time.time()
            fps      = 1.0 / (now - prev_time) if (now - prev_time) > 0 else 0.0
            prev_time = now

            if now - reminder_start_time >= 10.0:
                reminder_index = (reminder_index + 1) % len(REMINDERS)
                current_reminder = REMINDERS[reminder_index]
                reminder_start_time = now

            if current_label != "None":
                target = get_target_samples(current_label)
                if sample_counts[current_label] >= target:
                    if auto_save:
                        auto_save = False
                        print("[*] Auto Save Disabled")
                        print(f"[✓] {current_label} reached {target} samples. Auto-Save stopped.")

            if auto_save and current_label != "None" and hand_detected and current_landmarks:
                target = get_target_samples(current_label)
                if sample_counts[current_label] < target:
                    if now - last_auto_save_time >= AUTO_SAVE_INTERVAL:
                        success = append_sample_to_csv(CSV_FILE, current_label, current_landmarks)
                        if success:
                            sample_counts[current_label] += 1
                            new_samples_collected += 1
                            flash_frames = FLASH_DURATION
                            last_auto_save_time = now
                            print(f"[+] Auto Saved {current_label} : {sample_counts[current_label]}/{target}")
                        else:
                            print("[-] Auto Save failed due to a write error.")

            if flash_frames > 0:
                draw_save_flash(frame)
                flash_frames -= 1

            draw_hud(frame, current_label, sample_counts, fps, hand_detected, auto_save, alphabet_mode, current_reminder)

            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF

            if key in (ord('q'), 27):
                print("[*] Quit requested. Exiting...")
                break

            elif key in SIGN_KEYS:
                lbl = SIGN_KEYS[key]
                if alphabet_mode and not lbl.isalpha():
                    print("[!] Alphabet Improvement Mode is ON. Only labels A-E are allowed.")
                else:
                    current_label = lbl
                    target = get_target_samples(current_label)
                    print(f"[*] Label set to '{current_label}'  "
                          f"(current count: {sample_counts[current_label]}/{target})")

            elif key == ord('i'):
                alphabet_mode = not alphabet_mode
                if alphabet_mode:
                    print("[*] ALPHABET IMPROVEMENT MODE Enabled. Only A-E allowed.")
                    if current_label != "None" and not current_label.isalpha():
                        current_label = "None"
                        auto_save = False
                        print("[!] Deselected number label due to Alphabet Improvement Mode.")
                else:
                    print("[*] ALPHABET IMPROVEMENT MODE Disabled.")

            elif key == ord('t'):
                auto_save = not auto_save
                if auto_save:
                    print("[*] Auto Save Enabled")
                else:
                    print("[*] Auto Save Disabled")

            elif key == ord('s'):
                if current_label == "None":
                    print("[!] No label selected.")
                elif not hand_detected:
                    print("[!] No hand visible in frame. Show your hand before saving.")
                elif not current_landmarks:
                    print("[!] Landmark data unavailable. Try again.")
                else:
                    target = get_target_samples(current_label)
                    if sample_counts[current_label] >= target:
                        print(f"[!] Target reached for {current_label}. Cannot save more samples.")
                    else:
                        success = append_sample_to_csv(CSV_FILE, current_label, current_landmarks)
                        if success:
                            sample_counts[current_label] += 1
                            new_samples_collected += 1
                            flash_frames = FLASH_DURATION
                            print(f"[+] Saved sample for '{current_label}'  "
                                  f"→ total: {sample_counts[current_label]}/{target}")
                        else:
                            print("[-] Sample was NOT saved due to a write error.")

    except KeyboardInterrupt:
        print("\n[*] Keyboard interrupt received. Shutting down...")
    except Exception as e:
        print(f"[-] Unexpected error in main loop: {e}")
    finally:
        cap.release()
        hands.close()
        cv2.destroyAllWindows()

        print("\n[+] Resources released successfully.")
        print("\n--- Collection Session Summary ---")
        
        for lbl in sorted(SIGN_KEYS.values()):
            if lbl.isalpha():
                count = sample_counts[lbl]
                target = get_target_samples(lbl)
                print(f"  {lbl}: {count:>3}/{target}")

        print(f"\n  Total new samples collected in this session: {new_samples_collected}")
        print(f"  Dataset saved to: {os.path.abspath(CSV_FILE)}")
        print("----------------------------------\n")

if __name__ == "__main__":
    main()
