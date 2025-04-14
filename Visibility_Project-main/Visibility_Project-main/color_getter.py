import cv2
import numpy as np

# Function to calculate average intensity in a region
def get_average_intensity(gray, bbox):
    x, y, w, h = bbox
    return np.mean(gray[y:y+h, x:x+w])

# Function to compute visibility distance based on Beer-Lambert law
def compute_visibility(I1, I2, d1, d2):
    """
    Compute the visibility distance given intensity
    measurements at two distances.
    

    Parameters:
    I1 : float -> Intensity of the closer building
    I2 : float -> Intensity of the farther building
    d1 : float -> Distance to the closer building (meters)
    d2 : float -> Distance to the farther building (meters)

    Returns:
    float -> Estimated visibility distance in meters
    """
    if I1 <= 0 or I2 <= 0 or d1 <= 0 or d2 <= 0 or d2 <= d1:
        raise ValueError("Invalid input values. Ensure positive intensities and correct distances.")

    # Calculate the extinction coefficient beta
    beta = np.log(I2 / I1) / (d2 - d1)

    # Calculate visibility (V = 1 / beta)
    visibility = 1 / beta if beta > 0 else float("inf")

    return visibility

# Initialize the webcam feed
cap = cv2.VideoCapture(0)

# List to store the bounding boxes
bbox_list = []
drawing = False
current_bbox = None

# Mouse callback function to draw bounding boxes
def draw_bbox(event, x, y, flags, param):
    global drawing, current_bbox, bbox_list
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(bbox_list) < 2:  # Allow only 2 bounding boxes
            drawing = True
            current_bbox = [x, y]
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            frame_copy = frame.copy()
            cv2.rectangle(frame_copy, (current_bbox[0], current_bbox[1]), (x, y), (0, 255, 0), 2)
            cv2.imshow("Webcam Feed", frame_copy)
    elif event == cv2.EVENT_LBUTTONUP:
        if len(bbox_list) < 2:  # Allow only 2 bounding boxes
            drawing = False
            bbox_list.append([current_bbox[0], current_bbox[1], x, y])  # Add the rectangle to the list
            current_bbox = None

# Set up the window and set the mouse callback
cv2.namedWindow("Webcam Feed")
cv2.setMouseCallback("Webcam Feed", draw_bbox)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_copy = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert frame to grayscale

    # Draw the bounding boxes if they exist
    for bbox in bbox_list:
        if len(bbox) == 4:  # Ensure we have 4 values (x1, y1, x2, y2)
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Calculate intensity for the manually selected region
            x = min(x1, x2)
            y = min(y1, y2)
            w = abs(x2 - x1)
            h = abs(y2 - y1)
            intensity = get_average_intensity(gray, (x, y, w, h))

            # Display intensity value next to the bounding box
            cv2.putText(frame_copy, f'Intensity: {intensity:.2f}', (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Calculate visibility if there are two bounding boxes
    if len(bbox_list) >= 2:
        # Get intensities for two bounding boxes
        I1 = get_average_intensity(gray, (min(bbox_list[0][0], bbox_list[0][2]), min(bbox_list[0][1], bbox_list[0][3]),
                                          abs(bbox_list[0][2] - bbox_list[0][0]), abs(bbox_list[0][3] - bbox_list[0][1])))
        I2 = get_average_intensity(gray, (min(bbox_list[1][0], bbox_list[1][2]), min(bbox_list[1][1], bbox_list[1][3]),
                                          abs(bbox_list[1][2] - bbox_list[1][0]), abs(bbox_list[1][3] - bbox_list[1][1])))

        # Define arbitrary distances for the two buildings (or use real data if available)
        d1 = 500  # Distance to closer building in meters
        d2 = 1000  # Distance to farther building in meters

        try:
            # Compute the visibility using the formula
            visibility = compute_visibility(I1, I2, d1, d2)
            cv2.putText(frame_copy, f'Visibility: {visibility:.2f} meters', (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        except ValueError as e:
            print(e)

    # Display the webcam feed
    cv2.imshow("Webcam Feed", frame_copy)

    # Wait for the user to press 'q' to quit or 'd' to delete all boxes
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('d'):  # Delete all bounding boxes when 'd' is pressed
        bbox_list.clear()

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
