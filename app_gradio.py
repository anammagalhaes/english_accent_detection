import gradio as gr
import os
import subprocess
import librosa
from transformers import pipeline
import requests
import yt_dlp

MODEL_ID = "HamzaSidhu786/speech-accent-detection"
TEMP_VIDEO = "temp_video.mp4"
TEMP_AUDIO = "temp_audio.wav"
FFMPEG_PATH = "ffmpeg"  # Uses system path in Hugging Face Spaces

clf = pipeline("audio-classification", model=MODEL_ID)

def download_from_url(url):
    if url.startswith("http"):
        if "youtube.com" in url or "youtu.be" in url:
            ydl_opts = {"format": "mp4", "outtmpl": TEMP_VIDEO, "quiet": True}
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
        else:
            r = requests.get(url, stream=True)
            with open(TEMP_VIDEO, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    elif os.path.exists(url):
        return url
    else:
        raise Exception("Invalid path or URL.")
    return TEMP_VIDEO

def extract_audio(video_path):
    cmd = [FFMPEG_PATH, "-i", video_path, "-t", "40", "-ac", "1", "-ar", "16000", "-y", TEMP_AUDIO]
    subprocess.run(cmd, check=True)
    return TEMP_AUDIO

def predict_accent(audio_path):
    audio, sr = librosa.load(audio_path, sr=16000)
    clip = audio[:sr * 10]
    results = clf(clip)
    top = max(results, key=lambda x: x["score"])
    return f"Accent: {top['label']} | Confidence: {top['score']*100:.2f}%"

def accent_from_upload(file):
    try:
        extract_audio(file)
        return predict_accent(TEMP_AUDIO)
    finally:
        cleanup()

def accent_from_url(url):
    try:
        video_path = download_from_url(url)
        extract_audio(video_path)
        return predict_accent(TEMP_AUDIO)
    finally:
        cleanup()

def cleanup():
    for f in (TEMP_VIDEO, TEMP_AUDIO):
        if os.path.exists(f):
            os.remove(f)

with gr.Blocks() as demo:
    gr.Markdown("## English Accent Detection (Upload, Path or URL)")

    with gr.Tab("Upload File"):
        file_input = gr.File(type="filepath")
        output1 = gr.Textbox(label="Detected Accent")
        btn1 = gr.Button("Detect")
        btn1.click(fn=accent_from_upload, inputs=file_input, outputs=output1)

    with gr.Tab("From URL or Path"):
        url_input = gr.Textbox(label="Enter a video URL or local path")
        output2 = gr.Textbox(label="Detected Accent")
        btn2 = gr.Button("Detect")
        btn2.click(fn=accent_from_url, inputs=url_input, outputs=output2)

if __name__ == "__main__":
    demo.launch()
