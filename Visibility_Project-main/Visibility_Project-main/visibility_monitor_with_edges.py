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
color_change_monitoring = False
distance_to_structure = 0.0  # Distance in meters
background_color = np.array([0, 0, 0])  # Default background color (black)
setting_background_color = False
sct = None  # Global mss instance

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

def get_edges(frame, bbox, threshold1=100, threshold2=200):
    """Apply Canny Edge Detection to the region inside the bounding box."""
    x1, y1, x2, y2 = bbox
    x = min(x1, x2)
    y = min(y1, y2)
    w = abs(x2 - x1)
    h = abs(y2 - y1)
    
    roi = frame[y:y+h, x:x+w]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1, threshold2)
    edge_count = np.sum(edges > 0)
    
    return edge_count, edges

def save_reference_values():
    """Save reference values to a file."""
    data = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'boxes': bbox_list,
        'values': {k: {'rgb': v['rgb'], 'intensity': float(v['intensity']), 'distance': float(v['distance']), 'edges': int(v['edges'])} for k, v in reference_values.items()}
    }
    with open('reference_values.json', 'w') as f:
        json.dump(data, f)
    print("Reference values saved")

def compute_visibility_change(current, reference, threshold=0.5):
    """Compute visibility change from reference."""
    if abs(reference) < 1e-6:
        return 0
    
    change = abs((current - reference) / reference) * 100
    return change if change >= threshold else 0

def compute_rgb_change(current_rgb, reference_rgb, threshold=5.0):
    """Compute RGB change from reference."""
    change = np.linalg.norm(current_rgb - reference_rgb) / np.linalg.norm(reference_rgb) * 100
    return change if change >= threshold else 0

def compute_color_similarity(current_rgb, background_rgb):
    """Compute color similarity to the background color."""
    similarity = np.linalg.norm(current_rgb - background_rgb) / np.linalg.norm(background_rgb) * 100
    return similarity

def compute_visibility_percentage(intensity_change, edge_change, color_change):
    """Compute visibility percentage based on changes."""
    visibility_percentage = max(0, 100 - (intensity_change + edge_change + color_change) / 3)
    return visibility_percentage

def mouse_callback(event, x, y, flags, param):
    """Handle mouse events for drawing boxes."""
    global drawing, current_bbox, bbox_list, frame, distance_to_structure, setting_background_color, background_color
    
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
        
        if setting_background_color:
            background_color, _ = get_average_colors(frame, (x1, y1, x2, y2))
            print(f"Background color set to: {background_color}")
            setting_background_color = False
        else:
            bbox_list.append((x1, y1, x2, y2))
            
            if frame is not None:
                rgb, intensity = get_average_colors(frame, (x1, y1, x2, y2))
                edge_count, _ = get_edges(frame, (x1, y1, x2, y2), threshold1=50, threshold2=150)
                distance_to_structure = float(input(f"Enter the distance from the camera to the structure for box {len(bbox_list)} (in meters): "))
                reference_values[len(bbox_list)-1] = {
                    'rgb': rgb.tolist(),
                    'intensity': float(intensity),
                    'distance': distance_to_structure,
                    'edges': edge_count
                }
                print(f"Box {len(bbox_list)} created with intensity: {intensity:.2f}, edges: {edge_count}, and distance: {distance_to_structure} meters")
        
        current_bbox = []

def close_all_edge_windows(edge_windows):
    """Close all edge detection windows."""
    for window_name in edge_windows.values():
        try:
            cv2.destroyWindow(window_name)
        except:
            pass
    edge_windows.clear()

def initialize_camera(camera_choice):
    """Initialize the camera based on the user's choice."""
    if camera_choice == 0:
        return cv2.VideoCapture(0)
    elif camera_choice == 1:
        return cv2.VideoCapture(1)
    elif camera_choice == 2:
        return cv2.VideoCapture(2)
    elif camera_choice == 3:
        # List of possible URLs to try
        urls = [
            "rtsp://C320WS9213:visibility@192.168.0.102:554/stream1"
        ]
        for url in urls:
            cap = cv2.VideoCapture(url)
            if cap.isOpened():
                return cap
    return None

def main():
    global frame, monitoring, target_window, window_rect, color_change_monitoring, background_color, setting_background_color, sct
    
    # Ask the user to choose between screen capture or camera
    print("Choose input source:")
    print("0: Default Webcam")
    print("1: Secondary Webcam")
    print("2: USB Webcam")
    print("3: RTSP Stream")
    print("4: Screen Capture")
    input_choice = int(input("Enter your choice: "))
    
    if input_choice in [0, 1, 2, 3]:
        cap = initialize_camera(input_choice)
        if not cap or not cap.isOpened():
            print("Failed to initialize camera. Exiting...")
            return
    elif input_choice == 4:
        try:
            # Initialize screen capture
            sct = mss()
            if not sct:
                raise Exception("Failed to initialize screen capture")
            
            # Get the screen size
            monitors = sct.monitors
            if not monitors or len(monitors) < 2:  # monitors[0] is all monitors combined
                raise Exception("No monitors detected")
                
            # Use the primary monitor (monitors[1] is the first actual monitor)
            monitor = monitors[1]
            window_rect = {'left': monitor['left'], 'top': monitor['top'],
                          'width': monitor['width'], 'height': monitor['height']}
            print(f"Screen capture initialized for monitor: {monitor['width']}x{monitor['height']}")
        except Exception as e:
            print(f"Failed to initialize screen capture: {e}")
            return
    else:
        print("Invalid input choice. Exiting...")
        return

    # Create window
    cv2.namedWindow("Visibility Monitor")
    cv2.setMouseCallback("Visibility Monitor", mouse_callback)
    
    print("\nControls:")
    print("Click and drag to create boxes")
    print("m: Start/stop monitoring")
    print("r: Reset boxes")
    print("s: Save reference values")
    print("c: Toggle color change monitoring")
    print("b: Set background color")
    print("q: Quit\n")
    
    # Dictionary to store edge windows and their information
    edge_windows = {}
    
    while True:
        if input_choice == 4:  # Screen capture
            try:
                # Capture the screen using mss
                screenshot = sct.grab(window_rect)
                frame = np.array(screenshot)
                if frame.size == 0:
                    continue
                # Convert from BGRA to BGR
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            except Exception as e:
                print(f"Error capturing screen: {e}")
                time.sleep(1)
                continue
        else:  # Camera feed
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame from camera. Exiting...")
                break

        # Draw boxes and show measurements
        for i, bbox in enumerate(bbox_list):
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Get current values
            rgb, intensity = get_average_colors(frame, bbox)
            
            if monitoring and i in reference_values:
                # Compare with reference
                ref_rgb = np.array(reference_values[i]['rgb'])
                ref_intensity = reference_values[i]['intensity']
                ref_edges = reference_values[i]['edges']
                distance = reference_values[i]['distance']
                change_rgb = compute_rgb_change(rgb, ref_rgb)
                change_intensity = compute_visibility_change(intensity, ref_intensity, 4.0)
                edge_count, _ = get_edges(frame, bbox, threshold1=50, threshold2=150)
                change_edges = compute_visibility_change(edge_count, ref_edges, 4.0)
                color_similarity = compute_color_similarity(rgb, background_color)
                
                # Display status
                color = (0, 255, 0) if change_intensity < 20 else (0, 0, 255)
                status = f"Change: {change_intensity:.1f}%"
                cv2.putText(frame, status, (x1, y1-20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                if color_change_monitoring and change_rgb > 0:
                    # Compute visibility percentage
                    visibility_percentage = compute_visibility_percentage(change_intensity, change_edges, color_similarity)
                    cv2.putText(frame, f"Distance: {distance:.1f}m", (x1, y2+30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    cv2.putText(frame, f"V Ratio: {visibility_percentage:.1f}%", (x1, y2+50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    cv2.putText(frame, f"Edges: {edge_count}", (x1, y2+70),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                
                # Show current intensity
                cv2.putText(frame, f"I: {intensity:.1f}", (x1, y1-40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Add color monitoring if enabled
            if color_change_monitoring:
                roi = frame[min(y1, y2):max(y1, y2), min(x1, x2):max(x1, x2)]
                if roi.size > 0:  # Ensure ROI is not empty
                    rgb_means = np.mean(roi, axis=(0, 1)).astype(int)
                    cv2.putText(frame, f"{rgb_means}", (x1, y2+10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        cv2.imshow("Visibility Monitor", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('m'):
            monitoring = not monitoring
            print("Monitoring:", "Started" if monitoring else "Stopped")
        elif key == ord('r'):
            # Close all edge windows before clearing boxes
            close_all_edge_windows(edge_windows)
            bbox_list.clear()
            reference_values.clear()
            monitoring = False
            print("Reset complete")
        elif key == ord('s'):
            save_reference_values()
        elif key == ord('c'):
            # Toggle color change monitoring
            color_change_monitoring = not color_change_monitoring
            print("Color Change Monitoring:", "Activated" if color_change_monitoring else "Deactivated")
        elif key == ord('b'):
            # Set background color
            setting_background_color = True
            print("Draw a box to set the background color")

    # Clean up
    if input_choice in [0, 1, 2, 3]:
        cap.release()
    close_all_edge_windows(edge_windows)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 