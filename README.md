
## Installation Instructions

### Step 1: Create a Virtual Environment

First, navigate to the root directory of the repository. Open your terminal or command prompt and run the following command to create a virtual environment.

#### On Windows:
```bash
python -m venv env
```

#### On Linux/MacOS:
```bash
python3 -m venv env
```

### Step 2: Activate the Virtual Environment

Activate the virtual environment you just created:

#### On Windows:
```bash
.\env\Scripts\activate
```

#### On Linux/MacOS:
```bash
source env/bin/activate
```

Once the virtual environment is activated, your command prompt or terminal should show the environment name (e.g., `(env)`).

### Step 3: Install Dependencies

Now that the virtual environment is active, install the necessary Python packages listed in `requirements.txt` by running:

```bash
pip install -r requirements.txt
```

This will install all the dependencies required for the project.

---

## File Overview

### `speech_to_text/gradio_speech_to_text.py`
This Gradio app uses the Google API to convert speech into text. It provides a simple user interface where you can speak into a microphone, and the app will transcribe your speech to text in real-time.

### `speech_to_text/gradio_save_audio.py`
This Gradio script allows you to record an audio file via a microphone. The audio is saved to a file for further processing, such as speech-to-text conversion or other audio-related tasks.

### `speech_to_text/gradio_api_test_speech_to_text.py`
This script is designed to send an API call to any Gradio app that provides speech-to-text functionality. It acts as a client that interacts with a Gradio app's API to test or retrieve speech-to-text outputs.

### `sign_to_text/webcam_only_mediapipe_gesture_recognizer.py`
This script uses the MediaPipe framework to recognize and detect Arabic letter signs from hand gestures via a webcam. It is specifically designed for Arabic sign language.

### `sign_to_text/webcam_with_textbox_mediapipe_gesture_recognizer.py`
Similar to the previous script, but with an added feature: this script not only detects Arabic letter gestures but also generates and displays a sentence by combining the detected letters in sequence.

### `sign_to_text/video_to_text_mediapipe_gesture_recognizer.py`
This script works with pre-recorded videos instead of live webcam streams. It processes the video, detects gestures, and outputs both the video with overlaid gesture annotations and the corresponding Arabic sentence created from the gestures.

### `sign_to_text/gradio_video_to_text.py`
This is a Gradio app that attempts to wrap the previous gesture recognition logic (from the `video_to_text_mediapipe_gesture_recognizer.py` script) into a user-friendly interface. Currently, this script is not working.

---

## Running the Project

Once the dependencies are installed, you can run the project according to its specific usage instructions.

---

## Additional Notes

- If you're using a custom Python version, make sure it's set up correctly in your environment.
- If you face any issues, ensure that all the dependencies are listed correctly in the `requirements.txt`.

---

## License

Include your license details here, if applicable.
```

This structure clearly explains the setup process and provides an overview of each file in the project. The `File Overview` section is now part of the `README.md`, making it easier for others to understand the purpose of each script in your repository.