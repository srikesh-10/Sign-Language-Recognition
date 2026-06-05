import cv2
import time
import sys

def main():
    print("[*] Initializing webcam... Press 'ESC' in the video window to exit.")
    
    # Open default webcam (index 0)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("[-] Error: Could not open webcam. Please verify it is connected and not used by another application.")
        sys.exit(1)
        
    # Variables for calculating FPS
    prev_frame_time = 0
    new_frame_time = 0
    
    # Create window and make it resizable
    window_name = "Webcam Test - Sign Language Recognition"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    try:
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            
            if not ret or frame is None:
                print("[-] Error: Failed to capture frame from webcam. Exiting...")
                break
                
            # Calculate FPS
            new_frame_time = time.time()
            # Avoid division by zero
            time_difference = new_frame_time - prev_frame_time
            fps = 1.0 / time_difference if time_difference > 0 else 0
            prev_frame_time = new_frame_time
            
            # Format FPS to 1 decimal place
            fps_text = f"FPS: {fps:.1f}"
            
            # Draw FPS overlay (text with shadow for readability)
            # Coordinates: (x, y) = (20, 40)
            # cv2.putText parameters: image, text, org, fontFace, fontScale, color, thickness, lineType
            cv2.putText(frame, fps_text, (22, 42), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, fps_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            # Add helper exit instruction in the window
            instruction_text = "Press 'ESC' to exit"
            cv2.putText(frame, instruction_text, (22, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, instruction_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
            
            # Display the resulting frame
            cv2.imshow(window_name, frame)
            
            # Check for the key press. 27 is the ASCII code for ESC
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                print("[*] ESC pressed. Exiting...")
                break
                
    except KeyboardInterrupt:
        print("[*] Keyboard interrupt received. Exiting...")
    except Exception as e:
        print(f"[-] An unexpected error occurred: {e}")
    finally:
        # Release the webcam capture and close all windows
        cap.release()
        cv2.destroyAllWindows()
        print("[+] Webcam released and windows closed successfully.")

if __name__ == "__main__":
    main()
