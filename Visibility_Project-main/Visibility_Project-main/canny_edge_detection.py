import cv2
import numpy as np
from mss import mss
import win32gui
import time
2
# Global variables
drawing = False
bbox = None
current_bbox = []

def process_frame(frame, bbox):
    """Apply Canny Edge Detection to the region inside the bounding box."""
    x1, y1 = bbox[0]
    x2, y2 = bbox[1]
    x = min(x1, x2)
    y = min(y1, y2)
    w = abs(x2 - x1)
    h = abs(y2 - y1)
    
    roi = frame[y:y+h, x:x+w]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    
    # Create an empty image and place the edges in the corresponding region
    edge_frame = np.zeros_like(frame)
    edge_frame[y:y+h, x:x+w] = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    return edge_frame

def mouse_callback(event, x, y, flags, param):
    """Handle mouse events for drawing bounding boxes."""
    global drawing, current_bbox, bbox
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        current_bbox = [(x, y)]
    
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            temp_frame = param.copy()
            cv2.rectangle(temp_frame, current_bbox[0], (x, y), (0, 255, 0), 2)
            cv2.imshow("Original", temp_frame)
    
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        current_bbox.append((x, y))
        bbox = current_bbox
        current_bbox = []

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

def capture_from_window(window_title):
    """Capture frames from a specific window."""
    sct = mss()
    windows = get_window_by_title(window_title)
    if not windows:
        print(f"No window found with title containing '{window_title}'")
        return

    if len(windows) > 1:
        print("Multiple windows found. Please select one:")
        for i, (_, title) in enumerate(windows):
            print(f"{i}: {title}")
        selection = int(input("Enter number: "))
        target_window = windows[selection][0]
    else:
        target_window = windows[0][0]

    window_rect = win32gui.GetWindowRect(target_window)

    cv2.namedWindow("Original")
    cv2.setMouseCallback("Original", mouse_callback)

    while True:
        screenshot = sct.grab({
            'left': window_rect[0],
            'top': window_rect[1],
            'width': window_rect[2] - window_rect[0],
            'height': window_rect[3] - window_rect[1]
        })
        
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        
        if bbox:
            edges = process_frame(frame, bbox)
            cv2.imshow('Canny Edges', edges)
        
        cv2.imshow('Original', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

def capture_from_video(video_path):
    """Capture frames from a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    cv2.namedWindow("Original")
    cv2.setMouseCallback("Original", mouse_callback)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if bbox:
            edges = process_frame(frame, bbox)
            cv2.imshow('Canny Edges', edges)
        
        cv2.imshow('Original', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def capture_from_camera(camera_index=0):
    """Capture frames from an attached camera."""
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    cv2.namedWindow("Original")
    cv2.setMouseCallback("Original", mouse_callback)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if bbox:
            edges = process_frame(frame, bbox)
            cv2.imshow('Canny Edges', edges)
        
        cv2.imshow('Original', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    print("Select input source:")
    print("1: Video file")
    print("2: Window")
    print("3: Camera")
    choice = input("Enter choice (1/2/3): ")

    if choice == '1':
        video_path = input("Enter video file path: ")
        capture_from_video(video_path)
    elif choice == '2':
        window_title = input("Enter part of the window title: ")
        capture_from_window(window_title)
    elif choice == '3':
        camera_index = int(input("Enter camera index (default 0): ") or 0)
        capture_from_camera(camera_index)
    else:
        print("Invalid choice. Exiting.")

if __name__ == "__main__":
    main()
