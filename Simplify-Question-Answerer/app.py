import gradio as gr
import subprocess
import os
import ffmpeg
# import pymedia.audio.acodec as acodec
# import pymedia.muxer as muxer
import random
import string
import spaces
from openai import OpenAI
import os
import re
from math import floor
import subprocess

ACCESS_TOKEN = os.getenv("HF_TOKEN")

client = OpenAI(
    base_url="https://api-inference.huggingface.co/v1/",
    api_key=ACCESS_TOKEN,
)

# val = None
@spaces.GPU(duration=1)
def random_name_generator():
    length = random.randint(10, 15)  # Random length between 10 and 15
    characters = string.ascii_letters + string.digits  # All alphanumeric characters
    random_name = ''.join(random.choice(characters) for _ in range(length))
    return random_name

# Example usage:
# print(random_name_generator())


def outputProducer(inputVideo):
    print(inputVideo)
    input_file = ffmpeg.input(inputVideo)
    name_random = random_name_generator()
    input_file.output('audio'+name_random+'.mp3', acodec='mp3').run()
    command2 = ["whisper",'./audio'+name_random+'.mp3']
    try:
        retVal = subprocess.check_output(command2)
    except:
        retVal = subprocess.check_output("ls")
    subprocess.run(['rm', 'audio'+name_random+'.mp3'], check=True) 
    return retVal

def subtitle_it(subtitle_str):
    # Regular expression to extract time and text
    pattern = re.compile(
      r'\[(\d{2}):(\d{2})\.(\d{3})\s*-->\s*(\d{2}):(\d{2})\.(\d{3})\]\s*(.*)'
    )
    # List to hold subtitle entries as tuples: (start_time, end_time, text)
    subtitles = []
    subtitle_str = subtitle_str.decode('utf-8')  # or replace 'utf-8' with the appropriate encoding if needed
    max_second = 0  # To determine the size of the list L
    
    sub_string = ""
    # Parse each line
    for line in subtitle_str.strip().split('\n'):
        match = pattern.match(line)
        if match:
          (
              start_min, start_sec, start_ms,
              end_min, end_sec, end_ms,
              text
          ) = match.groups()
          
          # Convert start and end times to total seconds          
          sub_string+=text
          
          # Update maximum second
        else:
          print(f"Line didn't match pattern: {line}")
    return sub_string
  # Initialize list L with empty strings

def respond(
    message,
    history: list[tuple([str, str])],
    system_message,
    reprocess,
    max_tokens,
    temperature,
    top_p,
):
    # global val
    # if ((val is None) or reprocess):
    subtitles = outputProducer(system_message)
    val = subtitle_it(subtitles)
        # reprocess-=1
    messages = [{"role": "system", "content": "Answer by using the transcript"+val}]

    for val in history:
        if val[0]:
            messages.append({"role": "user", "content": val[0]})
        if val[1]:
            messages.append({"role": "assistant", "content": val[1]})

    messages.append({"role": "user", "content": message})

    response = ""
    
    for message in  client.chat.completions.create(
        model="Qwen/Qwen2.5-72B-Instruct",
        max_tokens=max_tokens,
        stream=True,
        temperature=temperature,
        top_p=top_p,
        messages=messages,
    ):
        token = message.choices[0].delta.content
        
        response += token
        yield response
        
chatbot = gr.Chatbot(height=600)

demo = gr.ChatInterface(
    respond,
    additional_inputs=[
        gr.Video(value=None, label="System message"),
        gr.Slider(minimum=0, maximum=1, value=1, step=1, label="Reprocess"),
        gr.Slider(minimum=1, maximum=4098, value=1024, step=1, label="Max new tokens"),
        gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(
            minimum=0.1,
            maximum=1.0,
            value=0.95,
            step=0.05,
            label="Top-P",
        ),
        
    ],
    fill_height=True,
    chatbot=chatbot
)
if __name__ == "__main__":
    demo.launch()
