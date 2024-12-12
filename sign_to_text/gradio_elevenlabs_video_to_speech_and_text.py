import mediapipe as mp
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import time
import gradio as gr
from google.cloud import texttospeech
from elevenlabs import ElevenLabs


client = ElevenLabs(
    api_key="",
)

# Desired output properties
desired_fps = 5  # Set FPS to 5 as per your comment
desired_height = 720  # Set height to 720 pixels
gesture_threshold = 10  # Increased number of consecutive frames required to confirm a gesture


# Function to convert text to speech using Google Cloud Text-to-Speech API
def text_to_speech_elevenlabs(text):
    
    # Define the text and settings for the API request
    response = client.text_to_speech.convert(
        voice_id="iP95p4xoKVk53GoZ742B", # Cris
        model_id="eleven_multilingual_v2",
        text=text,
        output_format="mp3_44100_128"  # Set the desired output format
    )
    
    audio_file_path = "output.mp3"
    
    # Save the audio response to a file
    with open(audio_file_path, "wb") as audio_file:
        # Iterate over the generator and write the content to the file
        for chunk in response:
            audio_file.write(chunk)
    return audio_file_path

# Define mapping from English gesture names to Arabic
def map_english_to_arabic(english_name):
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

# Initialize MediaPipe components for gesture recognition
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Configure the gesture recognizer options
model_path = '/home/adel/Documents/project1/Arsl_gesture_recognizer.task'

options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,  # Set to VIDEO for processing video frames natively
    num_hands=1,
)

# Initialize the Gesture Recognizer
recognizer = GestureRecognizer.create_from_options(options)

# Drawing utilities for landmarks
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize the Hands object for drawing landmarks (outside the loop to avoid reinitialization)
hands = mp_hands.Hands(
    static_image_mode=False,
    min_detection_confidence=0.3,
    max_num_hands=1
)

# Gradio Interface function to process video
def process_video(input_video_path):
    output_video_path = 'output.mp4'

    # Initialize video capture from the input video file
    video = cv2.VideoCapture(input_video_path)
    if not video.isOpened():
        return "Error: Could not open input video.", None, None

    # Get original video properties
    original_fps = video.get(cv2.CAP_PROP_FPS)
    original_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Calculate sampling interval to maintain playback speed
    sampling_interval = int(original_fps / desired_fps)  # FPS is set to 5
    if sampling_interval < 1:
        sampling_interval = 1  # Ensure at least every frame is processed

    # Calculate new width to maintain aspect ratio
    scale_factor = desired_height / original_height 
    desired_width = int(original_width * scale_factor)

    # Initialize VideoWriter with desired FPS and frame size
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(output_video_path, fourcc, desired_fps, (desired_width, desired_height))

    # Variables for gesture confirmation logic
    current_gesture = None
    gesture_count = 0
    letter_added_for_current_gesture = False
    sentence = ""  # The constructed sentence

    # Load a font that supports Arabic characters
    font_path = 'Amiri-Regular.ttf'   # Ensure this path is correct
    if not os.path.isfile(font_path):
        return f"Error: Font file not found at {font_path}.", None, None

    try:
        arabic_font = ImageFont.truetype(font_path, 40)
    except IOError:
        return f"Error: Unable to load font at {font_path}.", None, None

    # Process video frames
    frame_counter = 0
    while True:
        ret, frame = video.read()
        if not ret:
            # Reached the end of the video
            break

        frame_counter += 1

        # Sample frames based on the sampling_interval
        if (frame_counter % sampling_interval) != 0:
            continue  # Skip this frame

        # Resize the frame to the desired dimensions
        frame = cv2.resize(frame, (desired_width, desired_height))

        # Convert the frame to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create a MediaPipe Image from the RGB frame
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # Perform gesture recognition on the provided single image
        frame_timestamp_ms = int(time.time() * 1000)  # Timestamp in milliseconds
        gesture_recognition_result = recognizer.recognize_for_video(mp_image, frame_timestamp_ms)

        # Process the gesture recognition result
        if gesture_recognition_result.gestures:
            # Assuming only one gesture is detected per frame
            gesture = gesture_recognition_result.gestures[0][0]
            category_name = gesture.category_name
            confidence = gesture.score

            print(f'Gesture: {category_name}, Confidence: {confidence:.2f}')
            
            # Check if it's a new gesture or the same as the current one
            if category_name != current_gesture:
                # New gesture detected, reset the counting logic
                current_gesture = category_name
                gesture_count = 1
                letter_added_for_current_gesture = False
            else:
                gesture_count += 1

            # Check if we've reached the threshold and haven't added a letter yet for this gesture
            if gesture_count == gesture_threshold and not letter_added_for_current_gesture:
                arabic_letter = map_english_to_arabic(category_name)
                if arabic_letter != "Letter not found":
                    sentence += arabic_letter
                    print(f'Added Arabic Letter: {arabic_letter}')
                else:
                    print(f'No mapping found for gesture: {category_name}')

                letter_added_for_current_gesture = True
        else:
            # Reset if no gesture is detected
            current_gesture = None
            gesture_count = 0
            letter_added_for_current_gesture = False

        # Process hand landmarks (optional, if you want to draw hand landmarks)
        results = hands.process(frame_rgb)
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

        # Draw the constructed sentence on the PIL image
        text_color = (0, 0, 0)  # Black color in RGB
        sentence_position = (10, 10)  # Top-left corner
        status_position = (10, 60)    # Below the sentence
        draw.text(sentence_position, f'Sentence: {sentence}', font=arabic_font, fill=text_color)

        # Optionally, display the current gesture and confirmation status
        status_font = ImageFont.truetype(font_path, 30)  # Increased font size from 24 to 30
        if current_gesture is not None:
            status_text = f'Current Gesture: {current_gesture} ({gesture_count}/{gesture_threshold})'
            if letter_added_for_current_gesture:
                status_text += " [Confirmed]"
            draw.text(status_position, status_text, font=status_font, fill=text_color)

        # Convert back to OpenCV image
        frame_with_text = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        # Write the frame to the output video
        out_video.write(frame_with_text)

    # Release video resources
    video.release()
    out_video.release()

    # Generate speech from the sentence
    audio_file_path = text_to_speech_elevenlabs(sentence)

    return sentence, audio_file_path, output_video_path

# Define Gradio interface
demo = gr.Interface(
    fn=process_video,
    inputs=gr.Video(),
    outputs=[gr.Textbox(), gr.Audio(), gr.Video()]
)

demo.launch(share=True, show_error=True)
