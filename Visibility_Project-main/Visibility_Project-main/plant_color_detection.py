import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import time

# Set dark mode style for the plot
plt.style.use('dark_background')

# RTSP URL for the webcam
RTSP_URL = "rtsp://buth:4ytkfe@192.168.1.210/live/ch00_1"

# Video Parameters
FRAME_WIDTH, FRAME_HEIGHT = 1280, 720

# Initialize the camera using RTSP URL
cap = cv2.VideoCapture(RTSP_URL)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

# Define the color ranges for detection in HSV
color_ranges = {
    'yellow': ((20, 100, 100), (30, 255, 255), 'yellow'),  # Lower and upper range for yellow
    'brown': ((10, 100, 20), (20, 255, 200), 'brown'),  # Lower and upper range for brown
    'white': ((0, 0, 200), (180, 20, 255), 'white'),  # Lower and upper range for white
    'blue': ((90, 50, 50), (130, 255, 255), 'blue')  # Lower and upper range for blue
}

# Define thresholds for each color
thresholds = {
    'yellow': 0.1,
    'brown': 0.1,
    'white': 0.02,
    'blue': 0.2
}

# Initialize deque to store color counts
color_counts = {color: deque(maxlen=100) for color in color_ranges}
frame_buffer = deque(maxlen=100)  # Initialize frame buffer

# Initialize plots
plt.ion()
fig, (ax1, ax2) = plt.subplots(2, 1)
lines_live = {
    color: ax1.plot([], [], label=color, color=color_ranges[color][2])[0]
    for color in color_ranges
}
lines_static = {
    color: ax2.plot([], [], label=color, color=color_ranges[color][2])[0]
    for color in color_ranges
}
ax1.legend(loc='upper right')
ax1.set_xlim(0, 100)
ax1.set_ylim(0, 1)  # Set y-axis limit to 0.5 for normalized values
ax2.legend(loc='upper right')
ax2.set_xlim(0, 100)
ax2.set_ylim(0, 1)  # Set y-axis limit to 0.5 for normalized values

last_alert_time = time.time()
last_static_update_time = time.time()

# Variables for drawing the rectangle
drawing = False
box_start = (0, 0)
box_end = (0, 0)

# Initialize lists to store the highest values for the static plot
static_counts = {color: [] for color in color_ranges}

# Variable to track the current detection mode
detect_all_colors = True
colors_to_detect = color_ranges.keys()

def draw_rectangle(event, x, y, flags, param):
    global box_start, box_end, drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        box_start = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            box_end = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        box_end = (x, y)

cv2.namedWindow('Frame')
cv2.setMouseCallback('Frame', draw_rectangle)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    frame_buffer.append(frame)

    # Draw the box on the frame
    if box_start != box_end:
        cv2.rectangle(frame, box_start, box_end, (0, 255, 0), 2)

    # Extract the region of interest (ROI) within the box
    if box_start != box_end:
        roi = frame[min(box_start[1], box_end[1]):max(box_start[1], box_end[1]), 
                    min(box_start[0], box_end[0]):max(box_start[0], box_end[0])]
        hsv_frame = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        for color in colors_to_detect:
            lower, upper, _ = color_ranges[color]
            mask = cv2.inRange(hsv_frame, lower, upper)
            color_count = cv2.countNonZero(mask)
            normalized_count = color_count / (roi.shape[0] * roi.shape[1])  # Normalize the count
            color_counts[color].append(normalized_count)
            
            # Check if the normalized count exceeds the threshold
            current_time = time.time()
            if normalized_count > thresholds[color] and (current_time - last_alert_time) >= 20:
                print(f"Alert: {color} count exceeded threshold with value {normalized_count:.2f}")
                last_alert_time = current_time
            
            # Update live plot data
            lines_live[color].set_xdata(range(len(color_counts[color])))
            lines_live[color].set_ydata(color_counts[color])
        
        ax1.relim()
        ax1.autoscale_view()
        plt.draw()
        plt.pause(0.01)
        
        # Update static plot data every 20 seconds
        if (current_time - last_static_update_time) >= 20:
            for color in colors_to_detect:
                highest_value = max(color_counts[color]) if color_counts[color] else 0
                static_counts[color].append(highest_value)
                lines_static[color].set_xdata(range(len(static_counts[color])))
                lines_static[color].set_ydata(static_counts[color])
            ax2.relim()
            ax2.autoscale_view()
            last_static_update_time = current_time
    
    cv2.imshow('Frame', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('b'):
        detect_all_colors = False
        colors_to_detect = ['blue']
    elif key == ord('a'):
        detect_all_colors = True
        colors_to_detect = color_ranges.keys()
    elif key == ord('w'):
        detect_all_colors = False
        colors_to_detect = ['white']
    elif key == ord('y'):
        detect_all_colors = False
        colors_to_detect = ['yellow']
    elif key == ord('r'):
        detect_all_colors = False
        colors_to_detect = ['brown']
    
    # Slowly remove other colors from the graph
    if not detect_all_colors:
        for color in color_ranges:
            if color not in colors_to_detect:
                if len(color_counts[color]) > 0:
                    color_counts[color].popleft()
                    lines_live[color].set_xdata(range(len(color_counts[color])))
                    lines_live[color].set_ydata(color_counts[color])
        ax1.relim()
        ax1.autoscale_view()
        plt.draw()
        plt.pause(0.01)

cap.release()
cv2.destroyAllWindows()
plt.ioff()
plt.show()
