import streamlit as st
from openai import OpenAI
import os
import cv2
from moviepy.editor import VideoFileClip
import time
import base64

# Set your OpenAI API key
api_key = ''
client = OpenAI(api_key=api_key)
MODEL = "gpt-4o"

# CSS to inject contained in a string
css = """
<style>
    .stApp {
        background-color: black;
        color: white;
    }
    .stVideo {
        border: 3px solid #f0f2f6;
        border-radius: 10px;
    }
    .stSpinner {
        color: red;
    }
    h1 {
        color: #4b6ef5;
        text-align: center;
    }
    .stButton button {
        background-color: #4b6ef5;
        color: #ffffff;
    }
    .stTextInput input {
        border: 1px solid #4b6ef5;
    }
</style>
"""

# Inject CSS with markdown
st.markdown(css, unsafe_allow_html=True)

def process_video(video_path, seconds_per_frame=2):
    base64frames = []
    base_video_path, _ = os.path.splitext(video_path)
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    frames_to_skip = int(fps * seconds_per_frame)
    curr_frame = 0

    while curr_frame < total_frames - 1:
        video.set(cv2.CAP_PROP_POS_FRAMES, curr_frame)
        success, frame = video.read()
        if not success:
            break
        _, buffer = cv2.imencode(".jpg", frame)
        base64frames.append(base64.b64encode(buffer).decode("utf-8"))
        curr_frame += frames_to_skip

    video.release()

    audio_path = f"{base_video_path}.mp3"
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(audio_path, bitrate="32k")
    clip.audio.close()
    clip.close()

    return base64frames, audio_path

def main():
    st.markdown("<h1>Video Summary Generator</h1>", unsafe_allow_html=True)
    uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

    if uploaded_video:
        video_path = "uploaded_video.mp4"
        with open(video_path, "wb") as f:
            f.write(uploaded_video.getbuffer())

        st.video(uploaded_video)

        with st.spinner("Processing video..."):
            base64_frames, audio_path = process_video(video_path)

        base64_frames = base64_frames[::20]

        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are generating a video summary. Please provide a summary of the video, respond in Markdown."
                },
                {
                    "role": "user",
                    "content": [
                        "These are the frames from the video",
                        *map(lambda x: {"type": "image_url", "image_url": {"url": f'data:image/jpg;base64,{x}', "detail": "low"}}, base64_frames)
                    ]
                }
            ],
            temperature=0,
        )

        st.markdown(response.choices[0].message.content)

        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=open(audio_path, "rb")
        )

        st.write("Transcript:", transcription.text)

        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are generating a transcript summary. Create a summary of the provided transcription."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"the audio transcription is: {transcription.text}"
                        }
                    ]
                }
            ],
            temperature=0,
        )

        st.markdown(response.choices[0].message.content)

        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are generating a video summary. Create a summary of the provided video and its transcript. Respond in Markdown."
                },
                {
                    "role": "user",
                    "content": [
                        "These are the frames from the video",
                        *map(lambda x: {"type": "image_url", "image_url": {"url": f'data:image/jpg;base64,{x}', "detail": "low"}}, base64_frames),
                        {
                            "type": "text",
                            "text": f"the audio transcription is: {transcription.text}"
                        }
                    ]
                }
            ],
            temperature=0,
        )

        st.markdown(response.choices[0].message.content)
        QUESTION = st.text_input("Ask a question about the video")
        if QUESTION:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "Use the video to answer the provided question. Respond in Markdown."
                    },
                    {
                        "role": "user",
                        "content": [
                            "These are the frames from the video",
                            *map(lambda x: {"type": "image_url", "image_url": {"url": f'data:image/jpg;base64,{x}', "detail": "low"}}, base64_frames),
                            QUESTION
                        ],
                    }
                ],
                temperature=0,
            )
            st.markdown(response.choices[0].message.content)
if __name__ == "__main__":
    main()
