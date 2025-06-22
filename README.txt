# Accent Detection App (Gradio)

This application detects the **English accent** used in a video (e.g., American, British, etc.), using a Gradio interface.

## Features

- Accepts video input in 3 ways:
  - Direct **upload** of local video files (MP4, WEBM, etc.)
  - URL to video file online (e.g., GitHub raw link)
  - Local file **path** (only when running locally)

- Extracts the first 40 seconds of audio from the video
- Runs accent classification using `HamzaSidhu786/speech-accent-detection`
- Returns the most likely accent with a confidence score

## How to Use

### Online (Hugging Face Spaces)
Once deployed on Hugging Face, just open the link in your browser. No setup is needed.

### Local

1. Clone the repository
2. Install dependencies:

```
pip install -r requirements.txt
```

3. Run the app:

```
python app_gradio.py
```

4. A browser window will open automatically.

## Dependencies

- Python 3.10+
- `gradio`, `transformers`, `yt_dlp`, `librosa`, `ffmpeg-python`

> Make sure `ffmpeg` is installed and accessible via system PATH. On Windows, download from [ffmpeg.org](https://ffmpeg.org/download.html).

## Limitations


- The model sometimes returns generic `"English"` rather than more specific accents.
- Bias exists in the model, e.g., overpredicting "Canadian" or "Scottish" due to dataset imbalance.
- YouTube links are not supported due to scraping blocks (403 Forbidden).
