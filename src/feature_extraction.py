import yt_dlp
import subprocess
import os
import numpy as np
import librosa
import torch
from transformers import AutoProcessor, AutoModelForAudioClassification

# NEW: Import the official yt-dlp filename sanitizer
from yt_dlp.utils import sanitize_filename
from src.config import FFMPEG_PATH

# --- Model Loading (Done once when the script starts) ---
print("Loading Audio Classification Model...")
processor = AutoProcessor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593", use_fast=True)
model = AutoModelForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
print("Model loaded.")


def process_and_extract_features(artist, title, playlist_label, base_dir="data/audio", keep_audio=False):
    """
    This is the main worker function. It handles the entire pipeline for a single song:
    1. Checks if the MP3 already exists (for resuming).
    2. If not, downloads it efficiently (direct-to-MP3).
    3. Extracts Librosa and Transformer-based features.
    4. Deletes the MP3 file to save space (optional).
    """
    # --- 1. SETUP PATHS AND TRIMMING RULES ---
    output_dir = os.path.join(base_dir, playlist_label)
    os.makedirs(output_dir, exist_ok=True)
    
    safe_title = sanitize_filename(f"{artist} - {title}")[:150]
    final_mp3_path = os.path.join(output_dir, f"{safe_title}.mp3")

    # Define which playlists should have shorter downloads
    TRIM_PLAYLISTS = {'lofi-downtempo', 'instrumental-happy', 'ambient-focus'}
    trim_duration = 120 if playlist_label in TRIM_PLAYLISTS else None
    
    # --- 2. RESUME LOGIC & EFFICIENT DOWNLOAD ---
    if not os.path.exists(final_mp3_path):
        print(f"\nMP3 not found for '{safe_title}'. Starting efficient download...")
        
        search_query = f"ytsearch1:{artist} {title}"
        
        # This is the direct-to-mp3 streaming logic
        success = _download_via_stream(
            query=search_query,
            output_path=final_mp3_path,
            ffmpeg_path=FFMPEG_PATH,
            trim_duration=trim_duration
        )
        if not success:
            return None # Download failed, skip this song
    else:
        print(f"Found existing MP3 for '{safe_title}'. Skipping download.")

    # --- 3. FEATURE EXTRACTION ---
    if os.path.exists(final_mp3_path):
        instrument_features = _get_instrument_probabilities(final_mp3_path)
        librosa_features = _get_librosa_features(final_mp3_path)
        
        # --- 4. CLEANUP ---
        if not keep_audio:
            try:
                os.remove(final_mp3_path)
            except OSError as e:
                print(f"Error removing audio file {final_mp3_path}: {e}")
        
        if librosa_features and instrument_features:
            return {**librosa_features, **instrument_features}
            
    return None

# --- HELPER FUNCTIONS (prefixed with _ for internal use) ---

def _download_via_stream(query, output_path, ffmpeg_path, trim_duration=None, trim_start=30):
    ydl_opts = {'format': 'bestaudio/best', 'cookiefile': 'cookies.txt', 'quiet': True}
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(query, download=False)['entries'][0]
            audio_url = info['url']
    except Exception as e:
        print(f"-> yt-dlp couldn't get video info: {e}")
        return False

    ffmpeg_command = [ffmpeg_path, '-i', audio_url]
    if trim_duration:
        ffmpeg_command.extend(['-ss', str(trim_start), '-t', str(trim_duration)])
    ffmpeg_command.extend(['-codec:a', 'libmp3lame', '-b:a', '192k', '-y', output_path])

    try:
        subprocess.run(ffmpeg_command, check=True, capture_output=True, text=True, encoding='utf-8')
        return True
    except subprocess.CalledProcessError as e:
        print(f"-> ffmpeg conversion failed: {e.stderr}")
        return False

def _get_instrument_probabilities(audio_file_path):
    try:
        y, sr = librosa.load(audio_file_path, sr=16000)
        inputs = processor(y, sampling_rate=sr, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits
        probabilities = torch.sigmoid(logits[0])
        results = {model.config.id2label[i]: probabilities[i].item() for i in range(len(probabilities))}
        return results
    except Exception as e:
        print(f"-> Could not get instrument probabilities: {e}")
        return None

def _get_librosa_features(audio_file_path):
    features = {}
    try:
        y, sr = librosa.load(audio_file_path, mono=True, duration=60)
        features['tempo'] = librosa.feature.tempo(y=y, sr=sr)[0]
        rms = librosa.feature.rms(y=y)
        features['rms_mean'], features['rms_std'] = np.mean(rms), np.std(rms)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        features['spectral_centroid_mean'], features['spectral_centroid_std'] = np.mean(spec_cent), np.std(spec_cent)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        features['spectral_bandwidth_mean'], features['spectral_bandwidth_std'] = np.mean(spec_bw), np.std(spec_bw)
        zcr = librosa.feature.zero_crossing_rate(y)
        features['zero_crossing_rate_mean'], features['zero_crossing_rate_std'] = np.mean(zcr), np.std(zcr)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        for i in range(20):
            features[f'mfcc_{i+1}_mean'] = np.mean(mfccs[i])
            features[f'mfcc_{i+1}_std'] = np.std(mfccs[i])
        return features
    except Exception as e:
        print(f"-> Error processing with librosa: {e}")
        return None