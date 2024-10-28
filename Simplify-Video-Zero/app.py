import gradio as gr
import subprocess
import os
import ffmpeg
# import pymedia.audio.acodec as acodec
# import pymedia.muxer as muxer
import random
import string
import spaces

def random_name_generator():
    length = random.randint(10, 15)  # Random length between 10 and 15
    characters = string.ascii_letters + string.digits  # All alphanumeric characters
    random_name = ''.join(random.choice(characters) for _ in range(length))
    return random_name

# Example usage:
# print(random_name_generator())

@spaces.GPU(duration = 100)
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
exampleList = [["examples/" + example] for example in os.listdir("examples")]
demo = gr.Interface(fn=outputProducer,
                    inputs = [gr.Video()],
                    outputs= [gr.Textbox()],
                    examples=exampleList, 
                    title = 'Simplify')
demo.launch()
