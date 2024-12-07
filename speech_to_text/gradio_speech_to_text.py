import gradio as gr
from google.cloud import speech
import io
from scipy.io.wavfile import write
import numpy as np
import os
import json
from google.oauth2 import service_account

# Set up Google Cloud authentication
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/adel/Documents/project1/speech_to_text/service_account_speech_to_text.json"
# Load the credentials JSON as a dictionary
credentials_json = """

"""  # Paste the full content of the JSON here

credentials_info = json.loads(credentials_json)
credentials = service_account.Credentials.from_service_account_info(credentials_info)
client = speech.SpeechClient(credentials=credentials)


def transcribe_audio(sound):
    print("Received sound:", sound)
    sample_rate, audio_data = sound  # Unpack the audio data

    # for local testing the audio has two channels, for production use 1 channel
    # gradio live link interface doesn't work with any type of audio 

    # Keep it for production and remove it for local testing
    # if audio_data.ndim > 1:
    #     audio_data = audio_data.mean(axis=1)

    # Ensure audio_data is in int16 format for LINEAR16 encoding
    if audio_data.dtype != np.int16:
        audio_data = (audio_data * 32767).astype(np.int16)

    # Create an in-memory buffer to hold the WAV data
    with io.BytesIO() as wav_buffer:
        write(wav_buffer, sample_rate, audio_data)
        wav_buffer.seek(0)
        content = wav_buffer.read()

    # Initialize the Google Cloud client
    # client = speech.SpeechClient()
    client = speech.SpeechClient(credentials=credentials)

    # Prepare the audio data for the API
    audio = speech.RecognitionAudio(content=content)

    # Configure the transcription settings
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=sample_rate,
        language_code="ar-EG",  # Change to your target language code
        audio_channel_count = 2 # for production use 1 for local testing use 2
    )

    # Perform the transcription
    response = client.recognize(config=config, audio=audio)

    # Extract the transcribed text
    transcript = ''
    for result in response.results:
        transcript += result.alternatives[0].transcript + ' '

    return transcript.strip()

# Create the Gradio interface
demo = gr.Interface(
    fn=transcribe_audio, 
    inputs=gr.Audio(type="numpy"), 
    outputs="textbox",
    title="Speech-to-Text Transcription",
    description="Record your voice and get a transcription using Google Cloud Speech-to-Text API."
)

demo.launch(share=True, show_error=True)
