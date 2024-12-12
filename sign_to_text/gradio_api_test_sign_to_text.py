from gradio_client import Client, handle_file

client = Client("https://9af321f5fd212cb666.gradio.live/")
result = client.predict(
		input_video_path={"video":handle_file('/home/adel/Documents/project1/sign_to_text/sahla.mp4')},
		api_name="/predict"
)
print(result)