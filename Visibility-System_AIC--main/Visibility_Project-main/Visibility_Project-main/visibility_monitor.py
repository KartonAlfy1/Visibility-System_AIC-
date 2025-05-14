import cv2
import numpy as np
from mss import mss
import win32gui
import win32con
import time
import json
from datetime import datetime

# Global variables
drawing = False
bbox_list = []
current_bbox = []
reference_values = {}
monitoring = False
frame = None
target_window = None
window_rect = None

def get_average_colors(frame, bbox):
    """Get both RGB and grayscale intensity for a region."""
    x1, y1, x2, y2 = bbox
    x = min(x1, x2)
    y = min(y1, y2)
    w = abs(x2 - x1)
    h = abs(y2 - y1)
    
    roi = frame[y:y+h, x:x+w]
    rgb_means = np.mean(roi, axis=(0, 1))
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    intensity = np.mean(gray)
    
    return rgb_means, intensity

def save_reference_values():
    """Save reference values to a file."""
    data = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'boxes': bbox_list,  # Saves box coordinates
        'values': reference_values  # Saves intensity values
    }
    with open('reference_values.json', 'w') as f:
        json.dump(data, f)
    print("Reference values saved")

def compute_visibility_change(current, reference):
    """Compute visibility change from reference."""
    if abs(reference) < 1e-6:
        return 0
    
    change = abs((current - reference) / reference) * 100
    return change

def get_window_by_title(title_pattern):
    """Find window by partial title match."""
    def callback(hwnd, windows):
        if win32gui.IsWindowVisible(hwnd):
            window_text = win32gui.GetWindowText(hwnd)
            if title_pattern.lower() in window_text.lower():
                windows.append((hwnd, window_text))
        return True
    
    windows = []
    win32gui.EnumWindows(callback, windows)
    return windows

def set_window_position():
    """Set the target window position."""
    global window_rect
    if target_window:
        try:
            window_rect = win32gui.GetWindowRect(target_window)
        except Exception as e:
            print(f"Error getting window position: {e}")
            return False
    return True

def mouse_callback(event, x, y, flags, param):
    """Handle mouse events for drawing boxes."""
    global drawing, current_bbox, bbox_list, frame
    
    if event == cv2.EVENT_LBUTTONDOWN and not monitoring:
        drawing = True
        current_bbox = [(x, y)]
    
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        if frame is not None:
            temp_frame = frame.copy()
            cv2.rectangle(temp_frame, current_bbox[0], (x, y), (0, 255, 0), 2)
            cv2.imshow("Visibility Monitor", temp_frame)
    
    elif event == cv2.EVENT_LBUTTONUP and drawing:
        drawing = False
        current_bbox.append((x, y))
        x1, y1 = current_bbox[0]
        x2, y2 = current_bbox[1]
        bbox_list.append((x1, y1, x2, y2))
        
        if frame is not None:
            rgb, intensity = get_average_colors(frame, (x1, y1, x2, y2))
            reference_values[len(bbox_list)-1] = {
                'rgb': rgb.tolist(),
                'intensity': intensity
            }
            print(f"Box {len(bbox_list)} created with intensity: {intensity:.2f}")
        
        current_bbox = []

def main():
    global frame, monitoring, target_window, window_rect
    
    # Initialize screen capture
    sct = mss()

    # Find v380 window
    windows = get_window_by_title("edge")
    if not windows:
        print("No Edge window found! Please open v380 first.")
        return
    
    # Let user select which window if multiple found
    if len(windows) > 1:
        print("Multiple v380 windows found. Please select one:")
        for i, (_, title) in enumerate(windows):
            print(f"{i}: {title}")
        selection = int(input("Enter number: "))
        target_window = windows[selection][0]
    else:
        target_window = windows[0][0]

    # Create window
    cv2.namedWindow("Visibility Monitor")
    cv2.setMouseCallback("Visibility Monitor", mouse_callback)
    
    print("\nControls:")
    print("Click and drag to create boxes")
    print("m: Start/stop monitoring")
    print("r: Reset boxes")
    print("s: Save reference values")
    print("q: Quit\n")
    
    while True:
        if not set_window_position() or not window_rect:
            print("Waiting for window...")
            time.sleep(1)
            continue

        try:
            # Capture specific window
            screenshot = sct.grab({
                'left': window_rect[0],
                'top': window_rect[1],
                'width': window_rect[2] - window_rect[0],
                'height': window_rect[3] - window_rect[1]
            })
            
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            
            # Draw boxes and show measurements
            for i, bbox in enumerate(bbox_list):
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                #   current values
                rgb, intensity = get_average_colors(frame, bbox)
                
                if monitoring and i in reference_values:
                    # Compare with reference
                    ref_intensity = reference_values[i]['intensity']
                    change = compute_visibility_change(intensity, ref_intensity)
                    
                    # Display status
                    color = (0, 255, 0) if change < 20 else (0, 0, 255)
                    status = f"Change: {change:.1f}%"
                    cv2.putText(frame, status, (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                else:
                    # Show current intensity
                    cv2.putText(frame, f"I: {intensity:.1f}", (x1, y1-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            cv2.imshow("Visibility Monitor", frame)

        except Exception as e:
            print(f"Error capturing window: {e}")
            time.sleep(1)
            continue
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('m'):
            monitoring = not monitoring
            print("Monitoring:", "Started" if monitoring else "Stopped")
        elif key == ord('r'):
            bbox_list.clear()
            reference_values.clear()
            monitoring = False
            print("Reset complete")
        elif key == ord('s'):
            save_reference_values()
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
