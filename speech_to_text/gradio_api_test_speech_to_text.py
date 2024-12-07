from gradio_client import Client, handle_file



client = Client("https://872b33b81fc616a1cd.gradio.live")
result = client.predict(
		sound=handle_file('test_audio_1channel_reactnative.wav'),
		api_name="/predict"
)
print(result)