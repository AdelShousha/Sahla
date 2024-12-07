# code for using the Google Cloud Speech-to-Text API
from google.cloud import speech
import io

# Set up Google Cloud authentication
import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/adel/Documents/project1/speech_to_text/service_account_speech_to_text.json"

def transcribe_audio(file_path):
    # Instantiates a client
    client = speech.SpeechClient()

    # Loads the audio into memory
    with io.open(file_path, "rb") as audio_file:
        content = audio_file.read()
        audio = speech.RecognitionAudio(content=content)

    # Configure request for Arabic transcription
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16, # Use appropriate encoding for your audio
        sample_rate_hertz=44100,  # Adjust to match your audio file's sample rate
        language_code="ar-EG",  # For Egyptian Arabic; use "ar-SA" for Saudi Arabic or "ar" for general Arabic
        audio_channel_count = 2,
    )

    # Sends the request and receives the response
    response = client.recognize(config=config, audio=audio)

    # Process and print the transcription
    for result in response.results:
        print("Transcript: {}".format(result.alternatives[0].transcript))

# Run the function with your audio file path
transcribe_audio("/home/adel/Documents/project1/speech_to_text/test_audio_2channel.wav")
