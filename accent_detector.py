from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import requests, os, subprocess, shutil
import yt_dlp
import librosa
from transformers import pipeline

app = FastAPI(title="Accent Detection API")

MODEL_ID = "HamzaSidhu786/speech-accent-detection"
TEMP_VIDEO = "video_input.mp4"
TEMP_AUDIO = "audio.wav"
FFMPEG_PATH = r"C:\\Users\\Ana\\Downloads\\ffmpeg-7.1.1-essentials_build\\ffmpeg-7.1.1-essentials_build\\bin\\ffmpeg.exe"

clf = pipeline("audio-classification", model=MODEL_ID)

class VideoURL(BaseModel):
    url: str

class LocalPath(BaseModel):
    path: str

def extract_audio(input_path):
    cmd = [
        FFMPEG_PATH,
        "-ss", "00:00:30",
        "-t", "30",
        "-i", input_path,
        "-ac", "1",
        "-ar", "16000",
        "-y", TEMP_AUDIO
    ]
    subprocess.run(cmd, check=True)
    if not os.path.exists(TEMP_AUDIO):
        raise Exception("Audio extraction failed.")

def predict_accent():
    audio, sr = librosa.load(TEMP_AUDIO, sr=16000)
    clip = audio[:sr * 10]
    result = clf(clip)
    top = max(result, key=lambda x: x["score"])
    return {
        "accent": top["label"],
        "confidence": round(top["score"] * 100, 2)
    }

@app.post("/detect-accent-url")
def detect_accent_from_url(data: VideoURL):
    try:
        if "youtube.com" in data.url or "youtu.be" in data.url:
            ydl_opts = {'format': 'mp4', 'outtmpl': TEMP_VIDEO}
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([data.url])
        else:
            r = requests.get(data.url, stream=True)
            if r.status_code != 200:
                raise Exception("Failed to download video.")
            with open(TEMP_VIDEO, 'wb') as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)

        extract_audio(TEMP_VIDEO)
        return predict_accent()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        for f in (TEMP_VIDEO, TEMP_AUDIO):
            if os.path.exists(f): os.remove(f)

@app.post("/detect-accent-local")
def detect_accent_from_local(data: LocalPath):
    try:
        if not os.path.exists(data.path):
            raise HTTPException(status_code=400, detail="File not found.")
        extract_audio(data.path)
        return predict_accent()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(TEMP_AUDIO):
            os.remove(TEMP_AUDIO)

@app.post("/detect-accent-upload")
def detect_accent_from_upload(file: UploadFile = File(...)):
    try:
        with open(TEMP_VIDEO, "wb") as f:
            shutil.copyfileobj(file.file, f)
        extract_audio(TEMP_VIDEO)
        return predict_accent()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        for f in (TEMP_VIDEO, TEMP_AUDIO):
            if os.path.exists(f): os.remove(f)





#"C:\\Users\\Ana\\Desktop\\PROJ_VOICE\\temp_video.mp4.webm"