# Motion Detection Application

This repository contains a motion detection application that uses computer vision techniques to detect and track movement in a video stream. When movement is detected, the application sends a text message notification using Twilio.

## Features

- Real-time motion detection using the MobileNetSSD model
- Bounding box visualization around detected persons
- Text message notification for movement events
- Frames per second (FPS) display on the video stream

## Prerequisites

Before running the application, make sure you have the following installed:

- Python 3
- OpenCV
- imutils
- Twilio Python library

You also need to set up a Twilio account and obtain your account SID, authentication token, and phone numbers for sending and receiving text messages. Update the corresponding variables in the code with your Twilio credentials.

## Usage

1. Clone the repository:

   ```shell
   git clone https://github.com/lorenmiller.dev/PersonDetectionTracker.git
   
2. Navigate to the project directory:

  cd motion-detection
  
3. Install the required dependencies:

pip install -r requirements.txt

4. Run the application:

python motion_detection.py

5. The application will open a video stream and start detecting motion. When movement is detected, a text message notification will be sent to the specified phone number.

6. Press 'q' to exit the application.

License
This project is licensed under the MIT License.

Acknowledgments
This project utilizes the MobileNetSSD model for person detection. The model files (.prototxt and .caffemodel) are obtained from the official OpenCV repository.

Contact
For any questions or suggestions, please feel free to reach out to me at lorenmiller.dev@gmail.com

