import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np
import time
import queue
import threading
from PIL import Image, ImageDraw, ImageFont
import os

# Define input and output paths
input_video_path = '/home/adel/Documents/project1/sign_to_text/sahla.mp4'
output_video_path = 'output.mp4'
text_file = 'translation.txt'

# Desired output properties
desired_fps = 5  # Set FPS to 5 as per your comment
desired_height = 720  # Set height to 720 pixels

gesture_threshold = 10  # Increased number of consecutive frames required to confirm a gesture

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
    running_mode=VisionRunningMode.LIVE_STREAM,  # Keep LIVE_STREAM mode for similar logic
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

# Verify that the input video file exists
if not os.path.isfile(input_video_path):
    print(f"Error: Input video file '{input_video_path}' not found.")
    exit()

# Initialize video capture from the input video file
video = cv2.VideoCapture(input_video_path)
if not video.isOpened():
    print("Error: Could not open input video.")
    exit()

# Get original video properties
original_fps = video.get(cv2.CAP_PROP_FPS)
original_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Calculate sampling interval to maintain playback speed
# sampling_interval = original_fps / desired_fps
sampling_interval = int(original_fps / desired_fps)
if sampling_interval < 1:
    sampling_interval = 1  # Ensure at least every frame is processed

# Calculate new width to maintain aspect ratio
scale_factor = desired_height / original_height
desired_width = int(original_width * scale_factor)

# Initialize the VideoWriter with the desired FPS and frame size
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_video = cv2.VideoWriter(output_video_path, fourcc, desired_fps, (desired_width, desired_height))

# Variables for gesture confirmation logic
current_gesture = None
gesture_count = 0
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
# Ensure the font file 'Amiri-Regular.ttf' is in the same directory as this script or provide the correct path
font_path = 'Amiri-Regular.ttf'  # Update with an Arabic-supporting font path if necessary
if not os.path.isfile(font_path):
    print(f"Error: Font file not found at {font_path}. Please provide a valid Arabic-supporting font.")
    out_video.release()
    video.release()
    exit()

try:
    # Increase font size for the sentence
    arabic_font = ImageFont.truetype(font_path, 40)  # Increased font size from 32 to 40
except IOError:
    print(f"Error: Unable to load font at {font_path}.")
    out_video.release()
    video.release()
    exit()

try:
    frame_counter = 0  # To keep track of frame sampling
    while True:
        ret, frame = video.read()
        if not ret:
            # Reached the end of the video
            break

        frame_counter += 1

        # Sample frames based on the sampling_interval
        if (frame_counter % sampling_interval) != 0:
            continue  # Skip this frame

        # Flip the frame horizontally for a mirror-like effect (optional)
        frame = cv2.flip(frame, 1)

        # Resize the frame to the desired dimensions
        frame = cv2.resize(frame, (desired_width, desired_height))

        # Convert the frame to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create a MediaPipe Image from the RGB frame
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # Send the frame to the Gesture Recognizer asynchronously
        recognizer.recognize_async(mp_image, timestamp_ms=int(time.time() * 1000))
        
        # Process gesture results from the queue
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

        # Process hand landmarks
        results = hands.process(frame_rgb)

        # Draw hand landmarks
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

        # Define text properties
        text_color = (0, 0, 0)  # Blue color in RGB (changed from (255, 0, 0) to blue)
        # You can choose another color by updating the RGB tuple

        # Define text positions for top-left corner
        sentence_position = (10, 10)  # Top-left corner
        status_position = (10, 60)    # Below the sentence

        # Draw the constructed sentence on the PIL image
        draw.text(sentence_position, f'Sentence: {sentence}', font=arabic_font, fill=text_color)

        # Define a font for status text (making it larger for consistency)
        status_font = ImageFont.truetype(font_path, 30)  # Increased font size from 24 to 30

        # Optionally, display the current gesture and confirmation status
        if current_gesture is not None:
            status_text = f'Current Gesture: {current_gesture} ({gesture_count}/{gesture_threshold})'
            if letter_added_for_current_gesture:
                status_text += " [Confirmed]"
            draw.text(status_position, status_text, font=status_font, fill=text_color)

        # Convert the PIL image back to OpenCV format (BGR)
        frame_with_text = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        # Write the processed frame to the output video
        out_video.write(frame_with_text)

        # Display the processed frame in a window (optional)
        # You can comment this out if you don't need to see the video in real-time
        cv2.imshow('MediaPipe Hands', frame_with_text)

        # Press 'ESC' to exit early or 'c' to clear the sentence
        key = cv2.waitKey(1) & 0xFF
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
    # Clean up resources
    video.release()
    out_video.release()
    recognizer.close()
    hands.close()
    cv2.destroyAllWindows()

    # Save the final sentence to a text file
    with open(text_file, 'w', encoding='utf-8') as f:
        f.write(sentence)

    print("Processing complete.")
    print(f"Final sentence saved to '{text_file}'.")
    print(f"Processed video saved to '{output_video_path}'.")
