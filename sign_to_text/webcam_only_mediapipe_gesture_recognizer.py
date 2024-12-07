import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np
import time
import queue
import threading

# Initialize a thread-safe queue to store gesture recognition results
result_queue = queue.Queue()

# Callback function for handling the result from gesture recognition
def gesture_callback(result: vision.GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    """
    Callback function invoked by MediaPipe when a gesture is recognized.

    Args:
        result (GestureRecognizerResult): The result of gesture recognition.
        output_image (mp.Image): The image output from MediaPipe (unused in this context).
        timestamp_ms (int): The timestamp of the frame in milliseconds.
    """
    # Enqueue the result and its timestamp for processing in the main loop
    result_queue.put((result, timestamp_ms))

# Path to the gesture recognition model
model_path = '/home/adel/Documents/project1/Arsl_gesture_recognizer.task'

# Set up MediaPipe components for gesture recognition
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Drawing utilities for landmarks
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Configure the gesture recognizer options
options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=1,
    result_callback=gesture_callback
)

# Initialize the Gesture Recognizer with the specified options
recognizer = GestureRecognizer.create_from_options(options)

# Initialize the Hands object for drawing landmarks (outside the loop to avoid reinitialization)
hands = mp_hands.Hands(
    static_image_mode=False,
    min_detection_confidence=0.3,
    max_num_hands=1
)

# Initialize video capture from the default webcam
video = cv2.VideoCapture(0)

# Verify that the webcam has been opened successfully
if not video.isOpened():
    print("Error: Could not open video stream.")
    exit()

timestamp = 0  # Initialize timestamp for gesture recognition

try:
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            print("Ignoring empty frame.")
            break

        # Flip the frame horizontally for a mirror-like effect (optional)
        frame = cv2.flip(frame, 1)

        # Convert the frame to RGB as MediaPipe uses RGB images
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create a MediaPipe Image from the RGB frame
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # Send the frame to the Gesture Recognizer asynchronously
        recognizer.recognize_async(mp_image, timestamp)
        timestamp += 1  # Increment timestamp for the next frame

        # Process the frame with MediaPipe Hands to draw landmarks
        results = hands.process(frame_rgb)

        # Draw hand landmarks on the frame if any are detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

        # Process any available gesture recognition results from the queue
        while not result_queue.empty():
            result, result_timestamp = result_queue.get()
            if result.gestures:
                # Extract the first detected gesture
                gesture = result.gestures[0][0]
                category_name = gesture.category_name
                confidence = gesture.score
                print(f'Gesture: {category_name}, Confidence: {confidence:.2f}')

                # Overlay the gesture information on the frame
                cv2.putText(
                    frame,
                    f'Gesture: {category_name}',
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

        # Display the processed frame in a window
        cv2.imshow('MediaPipe Hands', frame)

        # Exit the loop when the 'ESC' key is pressed
        if cv2.waitKey(5) & 0xFF == 27:
            break

except KeyboardInterrupt:
    print("Interrupted by user.")

finally:
    # Release the webcam and close all OpenCV windows
    video.release()
    recognizer.close()
    hands.close()
    cv2.destroyAllWindows()
