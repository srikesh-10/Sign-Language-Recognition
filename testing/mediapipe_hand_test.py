import cv2
import mediapipe as mp
import time
import sys

def main():
    print("[*] Initializing MediaPipe Hands and Webcam... Press 'ESC' to exit.")
    
    # Initialize MediaPipe Hands solution
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    # Configure the hands detection module
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("[-] Error: Could not open webcam. Make sure it is connected and not in use.")
        sys.exit(1)
        
    # Variables for FPS calculation
    prev_frame_time = 0
    new_frame_time = 0
    
    # Create a resizable window
    window_name = "MediaPipe Hands Test - Sign Language Recognition"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[-] Error: Failed to capture frame from webcam. Exiting...")
                break
                
            # Flip the frame horizontally for a natural mirror-like viewing experience
            frame = cv2.flip(frame, 1)
            
            # MediaPipe processes images in RGB format, but OpenCV loads them in BGR
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the image and find hands
            results = hands.process(rgb_frame)
            
            # Count detected hands
            hand_count = 0
            
            # Draw the hand annotations on the image
            if results.multi_hand_landmarks:
                hand_count = len(results.multi_hand_landmarks)
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw landmarks with default styled connections (colorful landmarks)
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )
            
            # Calculate FPS
            new_frame_time = time.time()
            time_difference = new_frame_time - prev_frame_time
            fps = 1.0 / time_difference if time_difference > 0 else 0
            prev_frame_time = new_frame_time
            
            # Setup visual feedback overlays (with drop shadows)
            fps_text = f"FPS: {fps:.1f}"
            hand_text = f"Hands Detected: {hand_count}"
            instruction_text = "Press 'ESC' to exit"
            
            # Draw FPS
            cv2.putText(frame, fps_text, (22, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, fps_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
            
            # Draw Hand Count
            cv2.putText(frame, hand_text, (22, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, hand_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 100, 0), 2, cv2.LINE_AA)
            
            # Draw Instruction
            cv2.putText(frame, instruction_text, (22, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, instruction_text, (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
            
            # Show output
            cv2.imshow(window_name, frame)
            
            # Exit on ESC key
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                print("[*] ESC pressed. Exiting...")
                break
                
    except KeyboardInterrupt:
        print("[*] Keyboard interrupt received. Exiting...")
    except Exception as e:
        print(f"[-] An unexpected error occurred: {e}")
    finally:
        # Release all resources
        cap.release()
        hands.close()
        cv2.destroyAllWindows()
        print("[+] Webcam and MediaPipe resources released. Windows closed.")

if __name__ == "__main__":
    main()
