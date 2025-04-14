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
edge_comparison = {}  # Store edge count comparisons
edge_history = {}  # Store edge count history for each box

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
                # Initialize edge comparison immediately with the same edge count
                edge_comparison[len(bbox_list)-1] = {
                    'initial': edge_count,
                    'current': edge_count
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
            "rtsp://buth:4ytkfe@192.168.1.210/live/ch00_1"
        ]
        for url in urls:
            cap = cv2.VideoCapture(url)
            if cap.isOpened():
                return cap
    return None

def show_edge_comparison():
    """Display edge count comparison matrix with simple bar chart comparison."""
    if not edge_comparison:
        # Create a blank image with a message
        img = np.zeros((200, 600, 3), dtype=np.uint8)
        cv2.putText(img, "No edge comparisons available yet.", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, "Start monitoring first.", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    else:
        # Calculate dimensions
        row_height = 40
        header_height = 90
        matrix_height = header_height + (len(edge_comparison) * row_height) + 200  # Added height for comparison bars
        matrix_width = 1200
        
        # Create a blank image for the matrix with dark gray background
        img = np.ones((matrix_height, matrix_width, 3), dtype=np.uint8) * 40
        
        # Draw title background
        cv2.rectangle(img, (0, 0), (matrix_width, 60), (60, 60, 60), -1)
        
        # Draw the header with black outline
        cv2.putText(img, "Edge Count Comparison Matrix", (50, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)
        cv2.putText(img, "Edge Count Comparison Matrix", (50, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Draw header background
        cv2.rectangle(img, (0, 60), (matrix_width, 90), (50, 50, 50), -1)
        
        # Draw column headers with black outline
        headers = [("Box", 50), ("Distance (m)", 150), ("Change %", 300),
                  ("Initial Edges (Clear Day)", 450), ("Current Edges (Non-Clear Day)", 600)]
        
        for text, x in headers:
            cv2.putText(img, text, (x, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
            cv2.putText(img, text, (x, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Draw separator line
        cv2.line(img, (50, 90), (matrix_width-50, 90), (100, 100, 100), 1)
        
        # Draw the data rows
        y_pos = 120
        
        # Define colors for each box (up to 8 distinct colors)
        box_colors = [
            (0, 255, 0),    # Green
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
            (255, 128, 0),  # Orange
            (128, 0, 255),  # Purple
            (0, 255, 128)   # Mint
        ]
        
        # Find max edge count for scaling bars
        max_edge_count = 0
        for data in edge_comparison.values():
            max_edge_count = max(max_edge_count, data['initial'], data['current'])
        
        bar_section_top = y_pos + (len(edge_comparison) * row_height) + 20
        bar_height = 30
        bar_spacing = 40
        bar_max_width = 400
        
        for box_idx, data in edge_comparison.items():
            initial_edges = data['initial']
            current_edges = data['current']
            distance = reference_values[box_idx]['distance']
            change_percent = ((current_edges - initial_edges) / initial_edges * 100) if initial_edges > 0 else 0
            
            # Get color for this box
            color = box_colors[box_idx % len(box_colors)]
            
            # Draw alternating row backgrounds
            if box_idx % 2 == 0:
                cv2.rectangle(img, (0, y_pos-20), (matrix_width, y_pos+20), (45, 45, 45), -1)
            
            # Draw the data with black outlines
            texts = [
                (str(box_idx), 50, color),
                (f"{distance:.1f}", 150, (255, 255, 255)),
                (f"{change_percent:+.1f}%", 300, color if abs(change_percent) < 20 else (0, 0, 255)),
                (str(initial_edges), 450, (255, 255, 255)),
                (str(current_edges), 600, (255, 255, 255))
            ]
            
            for text, x, color in texts:
                cv2.putText(img, text, (x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
                cv2.putText(img, text, (x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw comparison bars
            bar_y = bar_section_top + (box_idx * bar_spacing)
            
            # Draw bar labels with black outline
            cv2.putText(img, f"Box {box_idx}", (50, bar_y + bar_height//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
            cv2.putText(img, f"Box {box_idx}", (50, bar_y + bar_height//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw initial edge count bar
            initial_width = int((initial_edges / max_edge_count) * bar_max_width)
            cv2.rectangle(img, (200, bar_y), (200 + initial_width, bar_y + bar_height),
                         (100, 100, 100), -1)  # Gray for initial
            
            # Draw current edge count bar
            current_width = int((current_edges / max_edge_count) * bar_max_width)
            bar_color = (0, 255, 0) if current_edges >= initial_edges else (0, 0, 255)
            cv2.rectangle(img, (200, bar_y + bar_height//2),
                         (200 + current_width, bar_y + bar_height),
                         bar_color, -1)
            
            # Add edge count labels with black outline
            cv2.putText(img, f"Initial: {initial_edges}", (210 + initial_width, bar_y + bar_height//2-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3)
            cv2.putText(img, f"Initial: {initial_edges}", (210 + initial_width, bar_y + bar_height//2-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            cv2.putText(img, f"Current: {current_edges}", (210 + current_width, bar_y + bar_height-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3)
            cv2.putText(img, f"Current: {current_edges}", (210 + current_width, bar_y + bar_height-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            y_pos += row_height
        
        # Add instructions with background
        cv2.rectangle(img, (0, matrix_height-40), (matrix_width, matrix_height), (60, 60, 60), -1)
        cv2.putText(img, "Press any key to close", (50, matrix_height-15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(img, "Press any key to close", (50, matrix_height-15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
    
    # Show the window
    cv2.imshow("Edge Comparison", img)
    cv2.waitKey(0)
    cv2.destroyWindow("Edge Comparison")

def load_reference_values():
    """Load reference values from a file."""
    try:
        with open('reference_values.json', 'r') as f:
            data = json.load(f)
            global bbox_list, reference_values, edge_comparison
            bbox_list = data['boxes']
            reference_values = {k: {'rgb': np.array(v['rgb']), 'intensity': float(v['intensity']), 
                                  'distance': float(v['distance']), 'edges': int(v['edges'])} 
                              for k, v in data['values'].items()}
            # Initialize edge comparison with loaded values
            edge_comparison = {k: {'initial': v['edges'], 'current': v['edges']} 
                             for k, v in reference_values.items()}
            print("Reference values loaded successfully")
            print(f"Loaded {len(bbox_list)} boxes with their reference values")
    except FileNotFoundError:
        print("No saved reference values found")
    except Exception as e:
        print(f"Error loading reference values: {e}")

def main():
    global frame, monitoring, target_window, window_rect, color_change_monitoring, background_color, setting_background_color, sct, edge_comparison, edge_history
    
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
    print("l: Load reference values")
    print("c: Toggle color change monitoring")
    print("b: Set background color")
    print("i: Show edge count comparison matrix")
    print("q: Quit\n")
    
    # Dictionary to store edge windows and their information
    edge_windows = {}
    
    while True:
        if input_choice == 4:  # Screen capture
            try:
                screenshot = sct.grab(window_rect)
                frame = np.array(screenshot)
                if frame.size == 0:
                    continue
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
                
                # Store edge count comparison
                if i not in edge_comparison:
                    edge_comparison[i] = {'initial': ref_edges, 'current': edge_count}
                else:
                    edge_comparison[i]['current'] = edge_count
                
                # Display status with black outline
                color = (0, 255, 0) if change_intensity < 20 else (0, 0, 255)
                status = f"Change: {change_intensity:.1f}%"
                # Draw black outline
                cv2.putText(frame, status, (x1, y1-20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
                # Draw colored text
                cv2.putText(frame, status, (x1, y1-20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Show current intensity with black outline
                cv2.putText(frame, f"I: {intensity:.1f}", (x1, y1-40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
                cv2.putText(frame, f"I: {intensity:.1f}", (x1, y1-40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # Show edge count with black outline
                cv2.putText(frame, f"Edges: {edge_count}", (x1, y2+30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
                cv2.putText(frame, f"Edges: {edge_count}", (x1, y2+30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                
                if color_change_monitoring and change_rgb > 0:
                    # Compute visibility percentage
                    visibility_percentage = compute_visibility_percentage(change_intensity, change_edges, color_similarity)
                    
                    # Distance with black outline
                    cv2.putText(frame, f"Distance: {distance:.1f}m", (x1, y2+50),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
                    cv2.putText(frame, f"Distance: {distance:.1f}m", (x1, y2+50),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    
                    # V Ratio with black outline
                    cv2.putText(frame, f"V Ratio: {visibility_percentage:.1f}%", (x1, y2+70),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
                    cv2.putText(frame, f"V Ratio: {visibility_percentage:.1f}%", (x1, y2+70),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                
                # Update edge history for matrix display
                if i not in edge_history:
                    edge_history[i] = []
                edge_history[i] = [edge_count]  # Only store current value
            
            # Add color monitoring if enabled
            if color_change_monitoring:
                roi = frame[min(y1, y2):max(y1, y2), min(x1, x2):max(x1, x2)]
                if roi.size > 0:  # Ensure ROI is not empty
                    rgb_means = np.mean(roi, axis=(0, 1)).astype(int)
                    # RGB values with black outline
                    cv2.putText(frame, f"{rgb_means}", (x1, y2+10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
                    cv2.putText(frame, f"{rgb_means}", (x1, y2+10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # Show the main frame
        cv2.imshow("Visibility Monitor", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('m'):
            monitoring = not monitoring
            if not monitoring:
                edge_history.clear()
            print("Monitoring:", "Started" if monitoring else "Stopped")
        elif key == ord('r'):
            close_all_edge_windows(edge_windows)
            bbox_list.clear()
            reference_values.clear()
            edge_comparison.clear()
            edge_history.clear()
            monitoring = False
            print("Reset complete")
        elif key == ord('s'):
            save_reference_values()
        elif key == ord('l'):
            load_reference_values()
        elif key == ord('c'):
            # Toggle color change monitoring
            color_change_monitoring = not color_change_monitoring
            print("Color Change Monitoring:", "Activated" if color_change_monitoring else "Deactivated")
        elif key == ord('b'):
            # Set background color
            setting_background_color = True
            print("Draw a box to set the background color")
        elif key == ord('i'):
            # Show edge count comparison matrix
            show_edge_comparison()

    # Clean up
    if input_choice in [0, 1, 2, 3]:
        cap.release()
    close_all_edge_windows(edge_windows)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 