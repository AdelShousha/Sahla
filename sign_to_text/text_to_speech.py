from google.cloud import texttospeech
import json
from google.oauth2 import service_account
import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/adel/Documents/project1/sign_to_text/service_account_text_to_speech.json"

# credentials_json = """
# 

# """  # Paste the full content of the JSON here

# credentials_info = json.loads(credentials_json)
# credentials = service_account.Credentials.from_service_account_info(credentials_info)
# client = texttospeech.TextToSpeechClient(credentials=credentials)

client = texttospeech.TextToSpeechClient()

text = "انا اسمي عادل"

# Set the text input to be synthesized
synthesis_input = texttospeech.SynthesisInput(text=text)


voice = texttospeech.VoiceSelectionParams(
    language_code="ar-XA",
    name="ar-XA-Standard-C",
)

# Select the type of audio file you want returned
audio_config = texttospeech.AudioConfig(
    audio_encoding=texttospeech.AudioEncoding.MP3,
    speaking_rate=1.0,
    pitch=-2.0,
)

# Perform the text-to-speech request on the text input with the selected
# voice parameters and audio file type
response = client.synthesize_speech(
    input=synthesis_input, 
    voice=voice, 
    audio_config=audio_config
)

# The response's audio_content is binary.
with open("output.mp3", "wb") as out:
    out.write(response.audio_content)
    print('Audio content written to file "output.mp3"')