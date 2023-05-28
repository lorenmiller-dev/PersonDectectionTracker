# PersonDectectionTracker
Simple detection program that can intake objects user wants to track. Creates boundry boxes around objects and utilizes OpenCV and MobileNetSSD

Default video capture on device is 0 (laptop camera), import camera of choice through usb.
Person detection can also be done through .mp4 video files, to do so change cap.cv2.VideoCapture("#.mp4 file name#")
