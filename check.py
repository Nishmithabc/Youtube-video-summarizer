import yt_dlp
import whisper
import numpy as np
from pydub import AudioSegment
import tempfile
import os
import uuid
from pydub.effects import normalize, strip_silence
import warnings


# Suppress FutureWarnings from torch
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

#load Whisper model
model = whisper.load_model("base")

def transcribe_from_youtube(youtube_url):
    # Download audio as WAV directly and avoid extra lossy encoding.
    temp_dir = tempfile.gettempdir()
    temp_audio_base = os.path.join(temp_dir, f"temp_audio_{uuid.uuid4().hex}")
    temp_audio_file = temp_audio_base + ".wav"

    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
        }],
        'outtmpl': temp_audio_base,
    }

    try:
        # Download audio from YouTube
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])


        # Process audio in chunks
        audio = AudioSegment.from_file(temp_audio_file, format="wav")
        total_duration = len(audio)
        chunk_duration = 60000  # 60 seconds chunks
        overlap = 3000  # 3 seconds overlap
        transcripts = []

        for start in range(0, total_duration, chunk_duration - overlap):
            end = min(start + chunk_duration, total_duration)
            chunk = audio[start:end]
            chunk=normalize(chunk)
            chunk=strip_silence(chunk,silence_len=1000,silence_thresh=-40)

            # Export chunk to WAV
            chunk_path = os.path.join(temp_dir, f"chunk_{start // 1000}.wav")
            chunk.export(chunk_path, format="wav")

            # Load the audio data as a NumPy array
            audio_array = whisper.audio.load_audio(chunk_path)

            # Transcribe chunk using Whisper
            result = model.transcribe(audio_array, language="en", task="transcribe", temperature=0.0)
            cleaned_text=" ".join(segment["text"] for segment in result["segments"] if not segment.get("no_speech_prob",0)>0.5)
            transcripts.append(cleaned_text)

            # Remove temporary chunk file
            os.remove(chunk_path)

        # Join all transcripts into one
        return " ".join(transcripts)
    except Exception as e:
        print(f"Error while searching for the video: {e}")
    finally:
        # Ensure the temporary file is deleted after processing
        if os.path.exists(temp_audio_file):
            os.remove(temp_audio_file)


if __name__ == "__main__":
    youtube_url = input("Enter YouTube URL: ").strip()

    transcript = transcribe_from_youtube(youtube_url)

    if transcript:
        print("\n" + "=" * 50)
        print("TRANSCRIPT")
        print("=" * 50)
        print(transcript)
    else:
        print("Failed to generate transcript.")