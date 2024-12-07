import gradio as gr
import mediapipe as mp
import cv2
import numpy as np
import time
import queue
import threading
from PIL import Image, ImageDraw, ImageFont
import os

# Your existing functions
def map_english_to_arabic(english_name):
    arabic_mapping = {
        'Alef': 'ا', 'Beh': 'ب', 'Teh': 'ت', 'Theh': 'ث', 'Jeem': 'ج', 
        'Hah': 'ح', 'Khah': 'خ', 'Dal': 'د', 'Thal': 'ذ', 'Reh': 'ر', 
        'Zain': 'ز', 'Seen': 'س', 'Sheen': 'ش', 'Sad': 'ص', 'Dad': 'ض', 
        'Tah': 'ط', 'Zah': 'ظ', 'Ain': 'ع', 'Ghain': 'غ', 'Feh': 'ف', 
        'Qaf': 'ق', 'Kaf': 'ك', 'Lam': 'ل', 'Meem': 'م', 'Noon': 'ن', 
        'Heh': 'ه', 'Waw': 'و', 'Yeh': 'ي', 'Al': 'ال', 'Teh_Marbuta': 'ة', 
        'Laa': ' '  
    }
    return arabic_mapping.get(english_name, "Letter not found")

# Path to the gesture recognition model
model_path = '/home/adel/Documents/project1/Arsl_gesture_recognizer.task'

# Initialize MediaPipe components
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

def process_video(input_video):
    # Initialize components for gesture recognition
    options = GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.LIVE_STREAM,
        num_hands=1,
        result_callback=gesture_callback  # Pass the gesture callback here
    )

    recognizer = GestureRecognizer.create_from_options(options)
    hands = mp_hands.Hands(
        static_image_mode=False,
        min_detection_confidence=0.3,
        max_num_hands=1
    )

    result_queue = queue.Queue()
    sentence = ""

    # Load the font
    font_path = 'Amiri-Regular.ttf'
    arabic_font = ImageFont.truetype(font_path, 32)

    # Prepare output video and text file
    processed_frames = []
    frame_counter = 0

    video = cv2.VideoCapture(input_video)
    original_fps = video.get(cv2.CAP_PROP_FPS)
    original_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    desired_height = 720
    scale_factor = desired_height / original_height
    desired_width = int(original_width * scale_factor)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    temp_output_video_path = 'temp_output.mp4'
    out_video = cv2.VideoWriter(temp_output_video_path, fourcc, original_fps, (desired_width, desired_height))

    try:
        while True:
            ret, frame = video.read()
            if not ret:
                break  # End of video

            frame_counter += 1
            frame = cv2.flip(frame, 1)  # Mirror effect
            frame = cv2.resize(frame, (desired_width, desired_height))
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            recognizer.recognize_async(mp_image, timestamp_ms=int(time.time() * 1000))

            while not result_queue.empty():
                result, _ = result_queue.get()
                if result.gestures:
                    gesture = result.gestures[0][0]
                    category_name = gesture.category_name
                    confidence = gesture.score
                    if category_name != current_gesture:
                        current_gesture = category_name
                        gesture_count = 1
                    else:
                        gesture_count += 1

                    if gesture_count == gesture_threshold:
                        arabic_letter = map_english_to_arabic(category_name)
                        sentence += arabic_letter

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

            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_image)
            text_position = (10, pil_image.height - 50)
            text_color = (255, 0, 0)
            draw.text(text_position, f'Sentence: {sentence}', font=arabic_font, fill=text_color)

            frame_with_text = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            out_video.write(frame_with_text)
            processed_frames.append(frame_with_text)

        video.release()
        out_video.release()

        return processed_frames, sentence

    except Exception as e:
        print(f"Error: {e}")
        video.release()
        out_video.release()
        
def gradio_interface(input_video):
    # Process video and return the processed video and the text
    processed_video, sentence = process_video(input_video)
    return processed_video, sentence


# Create Gradio interface
interface = gr.Interface(
    fn=gradio_interface,
    inputs=gr.Video(label="Input Video"),
    outputs=[gr.Video(label="Processed Video"), gr.Textbox(label="Translated Text")],
    live=True
)

interface.launch()
