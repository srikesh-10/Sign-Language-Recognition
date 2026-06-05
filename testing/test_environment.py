import sys
import os
import time

def test_python_version():
    """Verifies that the Python version is compatible with 3.11.9."""
    current_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    target_version = "3.11.9"
    
    print(f"[*] Checking Python version...")
    print(f"    - Current version: {current_version}")
    print(f"    - Target version:  {target_version}")
    
    if current_version == target_version:
        print("[+] Python version matches target version exactly.")
        return True, current_version, "Success: Exact match."
    elif sys.version_info.major == 3 and sys.version_info.minor == 11:
        print("[!] Python version is 3.11.x but not exactly 3.11.9. It should still be compatible.")
        return True, current_version, "Warning: Version is 3.11.x (not exactly 3.11.9), but compatible."
    else:
        print("[-] Python version is not 3.11.x. Compatibility issues may arise.")
        return False, current_version, f"Error: Python {current_version} is not in the 3.11 branch."

def test_dependencies():
    """Verifies that opencv, mediapipe, and numpy can be imported and prints their versions."""
    dependencies = ["cv2", "mediapipe", "numpy"]
    status = {}
    
    print("\n[*] Verifying dependencies...")
    for lib in dependencies:
        try:
            module = __import__(lib)
            # Find version
            version = getattr(module, "__version__", "Unknown Version")
            print(f"[+] Successfully imported '{lib}' (v{version})")
            status[lib] = (True, version, "Imported successfully")
        except ImportError as e:
            print(f"[-] Failed to import '{lib}'. Error: {e}")
            status[lib] = (False, "N/A", str(e))
            
    return status

def test_webcam():
    """Tests if a webcam is available and can capture a frame."""
    import cv2
    
    print("\n[*] Testing webcam availability...")
    # Attempt to open default camera (index 0)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("[-] Error: Could not open webcam at index 0.")
        return False, "Could not open webcam at index 0. Make sure the webcam is plugged in and not in use by another program."
    
    # Try to read a frame
    ret, frame = cap.read()
    
    # Release camera resource immediately
    cap.release()
    
    if ret and frame is not None:
        height, width, channels = frame.shape
        resolution = f"{width}x{height}"
        print(f"[+] Webcam successfully initialized and captured a frame.")
        print(f"    - Resolution: {resolution}")
        return True, f"Webcam active (Resolution: {resolution})"
    else:
        print("[-] Error: Webcam opened but failed to capture a frame.")
        return False, "Webcam opened but failed to retrieve frame. Access might be blocked or driver issues."

def generate_report(py_status, dep_status, webcam_status_info):
    """Generates a text report summarizing the environment setup."""
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("           SIGN LANGUAGE RECOGNITION SETUP REPORT")
    report_lines.append("=" * 60)
    report_lines.append(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Operating System: {sys.platform}")
    report_lines.append("-" * 60)
    
    # Python Section
    py_ok, py_ver, py_msg = py_status
    report_lines.append(f"1. Python Environment:")
    report_lines.append(f"   - Installed Version: {py_ver}")
    report_lines.append(f"   - Status: {'[PASS]' if py_ok else '[FAIL]'} - {py_msg}")
    report_lines.append("-" * 60)
    
    # Dependencies Section
    report_lines.append(f"2. Required Libraries:")
    all_deps_pass = True
    for lib, (ok, ver, msg) in dep_status.items():
        pass_str = "[PASS]" if ok else "[FAIL]"
        report_lines.append(f"   - {lib.ljust(12)}: {ver.ljust(12)} {pass_str} - {msg}")
        if not ok:
            all_deps_pass = False
    report_lines.append("-" * 60)
    
    # Webcam Section
    webcam_ok, webcam_msg = webcam_status_info
    report_lines.append(f"3. Hardware / Webcam Check:")
    report_lines.append(f"   - Webcam Status: {'[PASS]' if webcam_ok else '[FAIL]'}")
    report_lines.append(f"   - Detail: {webcam_msg}")
    report_lines.append("-" * 60)
    
    # Final Verdict
    overall_pass = py_ok and all_deps_pass and webcam_ok
    report_lines.append("OVERALL SYSTEM STATUS:")
    if overall_pass:
        report_lines.append(">>> READY! Environment is fully configured for Sign Language Recognition.")
    else:
        report_lines.append(">>> ATTENTION REQUIRED: Fix the failures listed above before proceeding.")
    report_lines.append("=" * 60)
    
    report_content = "\n".join(report_lines)
    
    # Write to file
    report_path = "setup_report.txt"
    try:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)
        print(f"\n[+] Setup report successfully saved to '{report_path}'.")
    except Exception as e:
        print(f"\n[-] Failed to save report to file. Error: {e}")
        
    # Print the report to console
    print("\n" + report_content + "\n")
    return overall_pass

def main():
    print("==================================================")
    print("      Environment Verification & Setup Test")
    print("==================================================")
    
    py_ok, py_ver, py_msg = test_python_version()
    
    # Test dependencies imports
    dep_status = test_dependencies()
    
    # Webcam check (only if cv2 is installed successfully)
    if dep_status.get("cv2", (False, "", ""))[0]:
        webcam_ok, webcam_msg = test_webcam()
    else:
        print("\n[-] Skipping webcam check because 'cv2' (OpenCV) failed to import.")
        webcam_ok, webcam_msg = False, "Skipped - OpenCV is not installed"
        
    # Generate and print the setup report
    generate_report((py_ok, py_ver, py_msg), dep_status, (webcam_ok, webcam_msg))

if __name__ == "__main__":
    main()
