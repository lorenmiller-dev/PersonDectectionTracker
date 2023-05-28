import cv2
import imutils
import datetime
import numpy as np

# Paths to the MobileNetSSD model files
protopath = "MobileNetSSD_deploy.prototxt"
modelpath = "MobileNetSSD_deploy_1.caffemodel"

# Load the pre-trained MobileNetSSD model
detector = cv2.dnn.readNetFromCaffe(prototxt=protopath, caffeModel=modelpath)

# List of class labels recognized by the MobileNetSSD model
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
           "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
           "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]


# Test Text

def main():
    # Open video file
    cap = cv2.VideoCapture(0)

    fps_start = datetime.datetime.now()  # start time for calculating FPS
    fps = 0
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
        fps_end = datetime.datetime.now()  # end time for calculating FPS
        time_diff = fps_end - fps_start

        if time_diff.seconds > 0:
            fps = (total_frames / time_diff.seconds)
        else:
            fps = 0.0

        # Resize frame
        width = 700
        frame = imutils.resize(frame, width)

        # Person Detection
        # get image dimensions
        (H, W) = frame.shape[:2]

        # Create a blob from the image for input to the model
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)

        # Set the blob as input to the detector
        detector.setInput(blob)

        # Perform object detection using the MobileNetSSD model
        person_detections = detector.forward()

        for detection in person_detections[0, 0]:
            # Extract confidence and index for detection
            confidence = detection[2]
            index = int(detection[1])

            # Check if detection = person class with sufficient confidence
            if confidence > 0.5 and CLASSES[index] == "person":
                # Extract box coordinates for person detection
                person_box = detection[3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = person_box.astype("int")

                # Draws rectangle around the person in the image
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 255), 2)

        # Display text and FPS
        fps_text = "FPS: {:.1f}".format(fps)
        cv2.putText(frame, fps_text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow('Application', frame)  # Display frames

        key = cv2.waitKey(1)  # waits for key press
        if key == ord('q'):  # press q to exit loop
            break

    cv2.destroyAllWindows()  # Close all windows


main()
