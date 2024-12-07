import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np
import time
import queue
import threading
from PIL import Image, ImageDraw, ImageFont

# Initialize a thread-safe queue to store gesture recognition results
result_queue = queue.Queue()

def map_english_to_arabic(english_name):
    """
    Maps English gesture names to their corresponding Arabic letters.
    
    Args:
        english_name (str): The English name of the gesture.
    
    Returns:
        str: The corresponding Arabic letter, a space for 'Laa', or "Letter not found" if the gesture is unrecognized.
    """
    arabic_mapping = {
        'Alef': 'ا', 'Beh': 'ب', 'Teh': 'ت', 'Theh': 'ث', 'Jeem': 'ج', 
        'Hah': 'ح', 'Khah': 'خ', 'Dal': 'د', 'Thal': 'ذ', 'Reh': 'ر', 
        'Zain': 'ز', 'Seen': 'س', 'Sheen': 'ش', 'Sad': 'ص', 'Dad': 'ض', 
        'Tah': 'ط', 'Zah': 'ظ', 'Ain': 'ع', 'Ghain': 'غ', 'Feh': 'ف', 
        'Qaf': 'ق', 'Kaf': 'ك', 'Lam': 'ل', 'Meem': 'م', 'Noon': 'ن', 
        'Heh': 'ه', 'Waw': 'و', 'Yeh': 'ي', 'Al': 'ال', 'Teh_Marbuta': 'ة', 
        'Laa': ' '  # Mapping 'Laa' to a space
    }
    
    return arabic_mapping.get(english_name, "Letter not found")

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

# Variables for gesture confirmation logic
current_gesture = None
gesture_count = 0
gesture_threshold = 20  # Increased number of consecutive frames required to confirm a gesture
letter_added_for_current_gesture = False
sentence = ""  # The constructed sentence

"""
Gesture Confirmation Logic:
When a gesture is detected, the system starts counting consecutive frames where the same gesture appears.
Once the gesture has been consistently recognized for a set number of frames (`gesture_threshold`),
it confirms the gesture and appends the corresponding Arabic letter to the sentence.
If the gesture changes, the count resets, and the new gesture must also be consistently detected
over the threshold before adding its letter.
This ensures that only stable and intentional gestures are added, minimizing false detections
and building accurate sentences.
"""

# Load a font that supports Arabic characters
# Ensure the font path is correct and the font supports Arabic glyphs
# You can download an Arabic font like 'Arial.ttf' or 'Amiri-Regular.ttf' and provide its path
font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'  # Example path; replace with your Arabic-supporting font
try:
    arabic_font = ImageFont.truetype(font_path, 32)
except IOError:
    print(f"Error: Font file not found at {font_path}. Please provide a valid font path.")
    exit()

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
        recognizer.recognize_async(mp_image, timestamp_ms=int(time.time() * 1000))
        
        # Process any available gesture recognition results from the queue
        while not result_queue.empty():
            result, result_timestamp = result_queue.get()
            if result.gestures:
                # Assuming only one gesture is detected per frame
                gesture = result.gestures[0][0]
                category_name = gesture.category_name
                confidence = gesture.score

                # Debug print (optional)
                print(f'Gesture: {category_name}, Confidence: {confidence:.2f}')

                # Check if it's a new gesture or the same as the current one
                if category_name != current_gesture:
                    # New gesture detected, reset the counting logic
                    current_gesture = category_name
                    gesture_count = 1
                    letter_added_for_current_gesture = False
                else:
                    # Same gesture continues
                    gesture_count += 1

                # Check if we've reached the threshold and haven't added a letter yet for this gesture
                if gesture_count == gesture_threshold and not letter_added_for_current_gesture:
                    # Map the English gesture name to Arabic letter
                    arabic_letter = map_english_to_arabic(category_name)
                    
                    if arabic_letter != "Letter not found":
                        # Append the Arabic letter to the sentence
                        sentence += arabic_letter
                        print(f'Added Arabic Letter: {arabic_letter}')
                    else:
                        print(f'No mapping found for gesture: {category_name}')
                    
                    # Mark that we've added the letter for this continuous gesture run
                    letter_added_for_current_gesture = True

            else:
                # Reset if no gesture is detected
                current_gesture = None
                gesture_count = 0
                letter_added_for_current_gesture = False

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

        # Convert the OpenCV frame (BGR) to PIL Image (RGB)
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Initialize ImageDraw
        draw = ImageDraw.Draw(pil_image)

        # Define the position for the sentence (e.g., bottom-left corner)
        text_position = (10, pil_image.height - 50)

        # Define text properties
        text_color = (255, 0, 0)  # Blue color in RGB
        text_font = arabic_font  # Use the loaded Arabic font

        # Draw the sentence on the PIL image
        draw.text(text_position, f'Sentence: {sentence}', font=text_font, fill=text_color)

        # Optionally, display the current gesture and confirmation status
        if current_gesture is not None:
            status_text = f'Current Gesture: {current_gesture} ({gesture_count}/{gesture_threshold})'
            if letter_added_for_current_gesture:
                status_text += " [Confirmed]"
            status_position = (10, pil_image.height - 90)
            status_color = (0, 255, 255)  # Yellow color in RGB
            status_font = ImageFont.truetype(font_path, 24)
            draw.text(status_position, status_text, font=status_font, fill=status_color)

        # Convert the PIL image back to OpenCV format (BGR)
        frame_with_text = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        # Display the processed frame in a window
        cv2.imshow('MediaPipe Hands', frame_with_text)

        # Press 'ESC' to exit the loop or 'c' to clear the sentence
        key = cv2.waitKey(5) & 0xFF
        if key == 27:
            print("Exiting on user command.")
            break
        elif key == ord('c'):
            # Press 'c' to clear the sentence
            sentence = ""
            print("Sentence cleared.")

except KeyboardInterrupt:
    print("Interrupted by user.")

finally:
    # Release the webcam and close all OpenCV windows
    video.release()
    recognizer.close()
    hands.close()
    cv2.destroyAllWindows()
