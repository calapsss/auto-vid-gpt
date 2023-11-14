import io
from dotenv import load_dotenv
from IPython.display import display, Image, Audio
import streamlit as st
import os
import tempfile
from moviepy.editor import VideoFileClip, concatenate_audioclips, concatenate_videoclips, AudioFileClip

import math
import cv2
import base64
import requests
import openai
import time
import re

load_dotenv()

#convert video to frame
def video_to_frame(video_filename):
    print(f"CONVERTING VIDEO: {video_filename} TO FRAMES...")
    video = cv2.VideoCapture(video_filename)
    base64Frames = []
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        _, buffer = cv2.imencode(".jpg", frame)
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
    video.release()
    print(len(base64Frames), "frames read.")
    return base64Frames

#video to chunks
def video_to_frames_in_chunks(video_file, chunk_length_in_seconds=4):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmpfile:
        tmpfile.write(video_file.read())
        video_filename = tmpfile.name
    video_duration = VideoFileClip(video_filename).duration

    print(f"CONVERTING VIDEO: {video_filename} TO FRAMES...")
    video = cv2.VideoCapture(video_filename)
    # Find the frame rate of the video
    fps = video.get(cv2.CAP_PROP_FPS)
    
    # Calculate the number of frames that make up the chunk length
    frames_per_chunk = int(chunk_length_in_seconds * fps)
    
    # Initialize variables
    chunks_base64Frames = []
    current_chunk = []
    frame_count = 0
    
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
            
        _, buffer = cv2.imencode(".jpg", frame)
        base64_frame = base64.b64encode(buffer).decode("utf-8")
        current_chunk.append(base64_frame)
        frame_count += 1
        
        # If we've reached the number of frames for the chunk, start a new chunk
        if frame_count == frames_per_chunk:
            chunks_base64Frames.append(current_chunk)
            current_chunk = []
            frame_count = 0
    
    # Make sure to add the last chunk if there are any frames left
    if current_chunk:
        chunks_base64Frames.append(current_chunk)
    
    video.release()
    
    # Print information about the chunks
    for i, chunk in enumerate(chunks_base64Frames):
        print(f"Chunk {i+1}: {len(chunk)} frames")
    
    return chunks_base64Frames, video_filename, video_duration

#frames to story
def frames_to_story(base64Frames, prompt):
    PROMPT_MESSAGES = [
        {
            "role": "user",
            "content": [
                prompt,
                *map(lambda x: {"image": x, "resize": 768},
                     base64Frames[0::25]),
            ],
        },
    ]
    params = {
        "model": "gpt-4-vision-preview",
        "messages": PROMPT_MESSAGES,
        "api_key": os.environ["OPENAI_API_KEY"],
        "headers": {"Openai-Version": "2020-11-07"},
        "max_tokens": 500,
    }

    result = openai.ChatCompletion.create(**params)
    print(result.choices[0].message.content)
    return result.choices[0].message.content

#text to audio
def text_to_audio(text):
    response = requests.post(
        "https://api.openai.com/v1/audio/speech",
        headers={
            "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
        },
        json={
            "model": "tts-1-1106",
            "input": text,
            "voice": "onyx",
        },
    )

    if response.status_code != 200:
        raise Exception("Request failed with status code", response.status_code)
    # ...
    # Create an in-memory bytes buffer
    audio_bytes_io = io.BytesIO()

    # Write audio data to the in-memory bytes buffer
    for chunk in response.iter_content(chunk_size=1024 * 1024):
        audio_bytes_io.write(chunk)

    # Important: Seek to the start of the BytesIO buffer before returning
    audio_bytes_io.seek(0)

    # Save audio to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmpfile:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            tmpfile.write(chunk)
        audio_filename = tmpfile.name

    return audio_filename, audio_bytes_io

#Main app
def main():
    print("STARTING APP...")
    st.title("Auto-Vid GPT")
    uploaded_file = st.file_uploader("Upload a video...")

    if uploaded_file is not None:
        st.video(uploaded_file)
        prompt = st.text_area("Prompt", value="These are frames of a [VIDEO DESCRIPTION]. Create a short voiceover [ADD ADDITIONAL INFO]")
    
    if st.button('Generate', type = "primary") and uploaded_file is not None:
        with st.spinner('Processing...'):
            time_start = time.time()
            #Split video
            chunks, video_filename, video_duration = video_to_frames_in_chunks(uploaded_file)
            #Generate story
            stories = []
            for chunk in chunks:
                print("Sending to OpenAI API...")
                video_duration = 4
                est_word_count = video_duration * 2
                final_prompt = prompt + f"(This video is only {video_duration} seconds long, so make sure the voice over MUST be able to be explained in less than {est_word_count} words.)"
                #Try story again from api if fails (bypass rate limits)
                while True:
                    try:
                        story = frames_to_story(chunk, final_prompt)
                        break
                    except Exception as e:
                        print(f"Failed to get story from OpenAI: {e}")
                        wait_time = 10
                        print(f"Waiting {wait_time} seconds before retrying...")
                        time.sleep(wait_time)
                        continue
                stories.append(story)
            #Combine stories audio
            print("Text to audio...")
            audio_files = []
            for story in stories:
                #try story again from api if fails  wait
                while True:
                    try:
                        audio_filename, audio_bytes_io = text_to_audio(story)
                        audio_files.append(audio_filename)
                        break
                    except Exception as e:
                        print(f"Failed to get audio from OpenAI: {e}")
                        wait_time = 10
                        print(f"Waiting {wait_time} seconds before retrying...")
                        time.sleep(wait_time)
                        continue
            #Combine audio files
            print("Combining audio files...")
            audio_clips = [AudioFileClip(audio_file) for audio_file in audio_files]
            final_audio = concatenate_audioclips(audio_clips)

            #Combine audio with vid file
            final_video = VideoFileClip(video_filename)
            video_with_audio = final_video.set_audio(final_audio)
            #Save video to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmpfile:
                video_with_audio.write_videofile(tmpfile.name)
                final_video_filename = tmpfile.name

            time_end = time.time() 
            runtime = f"Done it took {time_end - time_start} seconds"
            print(runtime)
            st.video(final_video_filename)
            st.success(runtime)

if __name__ == "__main__":
    main()

