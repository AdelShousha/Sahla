# code for saving the audio file
import gradio as gr
from scipy.io.wavfile import write
import numpy as np

def save_audio(sound):
    sample_rate, audio_data = sound
    output_file = "output.wav"
    write(output_file, sample_rate, audio_data)
    return output_file

demo = gr.Interface(fn=save_audio, inputs="audio", outputs="file")

demo.launch()
