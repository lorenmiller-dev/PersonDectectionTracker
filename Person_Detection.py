import cv2
import imutils
import datetime
import numpy as np

# Constants
FPS_DISPLAY_POS = (30, 30)  # Position to display FPS text
NORMALIZATION_SCALE = 0.007843  # Normalization scale for blobFromImage
MEAN_SUBTRACTION_VALUE = 127.5  # Mean subtraction value for blobFromImage

# Color
BOUNDING_BOX_COLOR = (0, 255, 255)  # BGR Format


# Calculate frames per second (FPS) based on the total frames processed and the start time
def calculate_fps(total_frames, fps_start):
    fps_end = datetime.datetime.now()
    time_diff = fps_end - fps_start

    if time_diff.seconds > 0:
        fps = (total_frames / time_diff.seconds)
    else:
        fps = 0.0

    return fps


# Perform person detection on the given frame using the specified detector
def detect_person(frame, detector):
    # Person Detection
    # get image dimensions
    (H, W) = frame.shape[:2]

    # Create a blob from the image for input to the model
    blob = cv2.dnn.blobFromImage(frame, NORMALIZATION_SCALE, (W, H), MEAN_SUBTRACTION_VALUE)

    # Set the blob as input to the detector
    detector.setInput(blob)

    # Perform object detection using the MobileNetSSD model
    person_detections = detector.forward()

    return person_detections


# Draw bounding boxes around the detected persons in the frame
def draw_boxes(frame, detections, W, H):
    for detection in detections[0, 0]:
        # Extract confidence and index for detection
        confidence = detection[2]
        index = int(detection[1])

        # Check if detection is a person class with sufficient confidence
        if confidence > 0.5 and CLASSES[index] == "person":
            # Extract box coordinates for person detection
            person_box = detection[3:7] * np.array([W, H, W, H])
            (startX, startY, endX, endY) = person_box.astype("int")

            # Expand the bounding box by a certain percentage
            box_width = endX - startX
            box_height = endY - startY
            expansion = 0.1  # 10% expansion
            startX = max(0, startX - int(expansion * box_width))
            startY = max(0, startY - int(expansion * box_height))
            endX = min(W, endX + int(expansion * box_width))
            endY = min(H, endY + int(expansion * box_height))

            # Draw rectangle around the person in the image
            cv2.rectangle(frame, (startX, startY), (endX, endY), BOUNDING_BOX_COLOR, 2)

    return frame


# Paths to the MobileNetSSD model files
protopath = "MobileNetSSD_deploy.prototxt"
modelpath = "MobileNetSSD_deploy_1.caffemodel"

# Load the pre-trained MobileNetSSD model
detector = cv2.dnn.readNetFromCaffe(prototxt=protopath, caffeModel=modelpath)

# List of class labels recognized by the MobileNetSSD model
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
           "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
           "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]


def main():
    # Open video file
    cap = cv2.VideoCapture(0)

    fps_start = datetime.datetime.now()  # start time for calculating FPS
    total_frames = 0

    while True:
        # If camera is not open, break
        if not cap.isOpened():
            print("Camera disconnected")
            break

        # Read frame from video
        ret, frame = cap.read()  # reads next frame from video

        # If frame retrieval fails, break
        if not ret:
            print("Failed to retrieve frame")
            break

        # Calculate FPS
        total_frames += 1
        fps = calculate_fps(total_frames, fps_start)

        # Resize frame
        width = 700
        frame = imutils.resize(frame, width)

        # Person Detection
        person_detections = detect_person(frame, detector)

        # Draw bounding boxes around people
        frame = draw_boxes(frame, person_detections, frame.shape[1], frame.shape[0])

        # Display text and FPS
        fps_text = "FPS: {:.1f}".format(fps)
        cv2.putText(frame, fps_text, FPS_DISPLAY_POS, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow('Application', frame)  # Display frames

        key = cv2.waitKey(1)  # waits for key press
        if key == ord('q'):  # press q to exit loop
            break

    # Release video capture and close windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

