from elevenlabs import ElevenLabs


client = ElevenLabs(
    api_key="",
)

# Define the text and settings for the API request
response = client.text_to_speech.convert(
    # voice_id="nPczCjzI2devNBz1zQrb", #brian
    # voice_id="pqHfZKP75CvOlQylNhV4", # bill
    # voice_id="N2lVS1w4EtoT3dr4eOWO", # cull
    # voice_id="IKne3meq5aSn9XLyUdCD", # charlie
    # voice_id="onwK4e9ZLuTAKqWW03F9", # Daniel
    voice_id="iP95p4xoKVk53GoZ742B", # Cris
    # voice_id="JBFqnCBsd6RMkjVDRZzb", # goerge
    model_id="eleven_multilingual_v2",
    # text="سهلة انا اسمي عادل",
    text="ازَّيك عامل ايه انا كويس ان شاء الله انت تمام",
    output_format="mp3_44100_128"  # Set the desired output format
)

# Save the audio response to a file
with open("brian1_output_audio.mp3", "wb") as audio_file:
    # Iterate over the generator and write the content to the file
    for chunk in response:
        audio_file.write(chunk)