from fastapi import FastAPI, HTTPException, UploadFile, File
import os, subprocess
import librosa
import numpy as np
from transformers import pipeline

app = FastAPI(title="Accent Detection API")

MODEL_ID = "HamzaSidhu786/speech-accent-detection"
AUDIO_PATH = "temp_audio.wav"
FFMPEG_PATH = r"C:\Users\Ana\Downloads\ffmpeg-7.1.1-essentials_build\ffmpeg-7.1.1-essentials_build\bin\ffmpeg.exe"

clf = pipeline("audio-classification", model=MODEL_ID)

def convert_to_wav(input_path):
    print("[INFO] Converting to WAV...")
    cmd = [FFMPEG_PATH, "-i", input_path, "-ac", "1", "-ar", "16000", "-y", AUDIO_PATH]
    subprocess.run(cmd, check=True)
    if not os.path.exists(AUDIO_PATH):
        raise Exception("Conversion to WAV failed.")
    print("[OK] Audio ready.")

def predict_accent():
    audio, sr = librosa.load(AUDIO_PATH, sr=16000)
    clip = audio[:sr * 5]
    clip = np.asarray(clip, dtype=np.float32)
    result = clf(clip)
    top = max(result, key=lambda x: x["score"])
    return {
        "accent": top["label"],
        "confidence": round(top["score"] * 100, 2)
    }

@app.post("/detect-accent-file")
async def detect_accent_from_file(file: UploadFile = File(...)):
    try:
        file_path = f"temp_input_{file.filename}"
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        convert_to_wav(file_path)
        return predict_accent()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        for f in (file_path, AUDIO_PATH):
            if os.path.exists(f):
                os.remove(f)
