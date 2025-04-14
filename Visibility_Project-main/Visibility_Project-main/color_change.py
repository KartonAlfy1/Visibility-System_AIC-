import cv2
import numpy as np
from mss import mss
import win32gui
import win32con
import time

# Global variables
drawing = False
bbox_list = []
current_bbox = []
mode = None
target_window = None
window_rect = None
frame = None  # Add frame as global variable

def get_pixel_distance(bbox1, bbox2):
    """
    Calculate the pixel distance between the centers of two bounding boxes.
    """
    # Unpack coordinates
    x1, y1, x2, y2 = bbox1
    x3, y3, x4, y4 = bbox2
    
    # Calculate centers
    center1 = ((x1 + x2) / 2, (y1 + y2) / 2)
    center2 = ((x3 + x4) / 2, (y3 + y4) / 2)
    
    # Calculate distance between centers
    return np.sqrt((center2[0] - center1[0])**2 + (center2[1] - center1[1])**2)

def get_average_intensity(gray_image, bbox):
    """Calculates the average intensity of the region defined by the bounding box."""
    x, y, w, h = bbox
    # Ensure the bounding box is within the image bounds
    x = max(0, x)
    y = max(0, y)
    w = max(1, w)  # Ensure width is at least 1
    h = max(1, h)  # Ensure height is at least 1
    
    # Extract the region from the gray image
    face_region = gray_image[y:y+h, x:x+w]
    
    if face_region.size == 0:
        return 0  # If the region is empty, return 0 to avoid NaN

    return int(face_region.mean())

def compute_visibility(I1, I2, d1, d2):
    """
    Compute the visibility using Koschmieder's law and contrast threshold.
    
    Parameters:
    I1: Intensity of closer target
    I2: Intensity of farther target
    d1: Distance to closer target (meters)
    d2: Distance to farther target (meters)
    """
    try:
        # Calculate contrast
        contrast = abs(I1 - I2) / max(I1, I2)
        
        # Calculate extinction coefficient
        # Using contrast threshold of 0.02 (standard meteorological visibility)
        beta = -np.log(0.02) / (d2 - d1)
        
        # Detailed debug output
        print("\nVisibility Calculation Details:")
        print(f"I1 (closer): {I1:.2f}")
        print(f"I2 (farther): {I2:.2f}")
        print(f"Contrast: {contrast:.4f}")
        print(f"Distance difference: {d2-d1}m")
        print(f"Beta: {beta:.6f}")
        
        # Calculate meteorological visibility
        visibility = 3.0 / beta  # Using standard meteorological constant
        
        # Adjust visibility based on contrast
        adjusted_visibility = visibility * (0.02 / contrast) if contrast > 0.02 else visibility
        
        print(f"Raw visibility: {visibility:.2f}m")
        print(f"Adjusted visibility: {adjusted_visibility:.2f}m")
        
        return adjusted_visibility
        
    except Exception as e:
        print(f"Error in visibility calculation: {e}")
        return float("inf")

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
    """Handle mouse events for drawing bounding boxes."""
    global drawing, current_bbox, bbox_list, mode, frame

    if event == cv2.EVENT_LBUTTONDOWN:
        if mode is not None:
            drawing = True
            current_bbox = [(x, y)]
    
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing and frame is not None:
            temp_frame = frame.copy()
            cv2.rectangle(temp_frame, current_bbox[0], (x, y), (0, 255, 0), 2)
            cv2.imshow("Window Capture", temp_frame)
    
    elif event == cv2.EVENT_LBUTTONUP:
        if drawing:
            drawing = False
            if mode is not None:
                current_bbox.append((x, y))
                x1, y1 = current_bbox[0]
                x2, y2 = current_bbox[1]
                bbox_list.append((x1, y1, x2, y2))
                current_bbox = []
                print(f"Box {len(bbox_list)} created")

def get_color_info(frame, bbox):
    """Get both intensity and color information for a region."""
    x1, y1, x2, y2 = bbox
    x = min(x1, x2)
    y = min(y1, y2)
    w = abs(x2 - x1)
    h = abs(y2 - y1)
    
    roi = frame[y:y+h, x:x+w]
    # Get BGR means
    bgr_means = np.mean(roi, axis=(0, 1))
    # Get grayscale intensity
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    intensity = np.mean(gray_roi)
    
    return bgr_means, intensity

def main():
    global mode, bbox_list, target_window, window_rect, frame

    # Initialize screen capture
    sct = mss()

    # Find Edge window
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

    # Create window for display
    window_name = "Window Capture"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    print("\nControls:")
    print("1: Start drawing first box")
    print("2: Start drawing second box")
    print("d: Delete all boxes")
    print("r: Refresh window position")
    print("q: Quit\n")

    while True:
        if not set_window_position() or not window_rect:
            print("Waiting for window...")
            time.sleep(1)
            continue

        # Capture specific window
        try:
            screenshot = sct.grab({
                'left': window_rect[0],
                'top': window_rect[1],
                'width': window_rect[2] - window_rect[0],
                'height': window_rect[3] - window_rect[1]
            })
            
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Draw existing boxes
            for bbox in bbox_list:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Get both color and intensity
                bgr, intensity = get_color_info(frame, bbox)
                
                # Display both intensity and color values
                cv2.putText(frame, f'I: {intensity:.1f}', (x1, y1 - 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(frame, f'BGR: ({bgr[0]:.0f},{bgr[1]:.0f},{bgr[2]:.0f})', 
                           (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Calculate visibility if two boxes exist
            if len(bbox_list) == 2:
                try:
                    # Get color and intensity for both boxes
                    bgr1, I1 = get_color_info(frame, bbox_list[0])
                    bgr2, I2 = get_color_info(frame, bbox_list[1])
                    
                    # Print detailed color information
                    print("\nBox 1 (Far):")
                    print(f"Intensity: {I1:.1f}")
                    print(f"BGR: ({bgr1[0]:.0f}, {bgr1[1]:.0f}, {bgr1[2]:.0f})")
                    print("\nBox 2 (Near):")
                    print(f"Intensity: {I2:.1f}")
                    print(f"BGR: ({bgr2[0]:.0f}, {bgr2[1]:.0f}, {bgr2[2]:.0f})")
                    
                    # Calculate pixel distance between boxes (optional)
                    pixel_dist = get_pixel_distance((bbox_list[0][0], bbox_list[0][1], 
                                                   bbox_list[0][2], bbox_list[0][3]),
                                                  (bbox_list[1][0], bbox_list[1][1], 
                                                   bbox_list[1][2], bbox_list[1][3]))
                    
                    # You might want to add a way to input real distances here
                    d1 = 500  # Distance to first target
                    d2 = 1000 # Distance to second target
                    
                    visibility = compute_visibility(I1, I2, d1, d2)
                    
                    # Display results
                    if visibility > 5000:
                        status = "Clear conditions"
                    elif visibility > 1000:
                        status = f"Good visibility: {visibility:.0f}m"
                    else:
                        status = f"Reduced visibility: {visibility:.0f}m"
                        
                    cv2.putText(frame, status, (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    print(f"\nResults:")
                    print(f"Target 1: {I1:.1f} at {d1}m")
                    print(f"Target 2: {I2:.1f} at {d2}m")
                    print(f"Status: {status}")
                    
                except Exception as e:
                    print(f"Error: {e}")

            cv2.imshow(window_name, frame)

        except Exception as e:
            print(f"Error capturing window: {e}")
            time.sleep(1)
            continue

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('1'):
            mode = 1
            bbox_list = []
            print("Drawing box 1...")
        elif key == ord('2'):
            if len(bbox_list) == 1:
                mode = 2
                print("Drawing box 2...")
        elif key == ord('d'):
            bbox_list = []
            mode = None
            print("Boxes cleared")
        elif key == ord('r'):
            set_window_position()
            print("Window position refreshed")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
