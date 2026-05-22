import yt_dlp
import os
import numpy as np
import librosa
import torch
from transformers import AutoProcessor, AutoModelForAudioClassification
from yt_dlp.utils import sanitize_filename

# --- Model Loading (Done once) ---
print("Loading Audio Classification Model...")
processor = AutoProcessor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593", use_fast=True)
model = AutoModelForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
print("Model loaded.")

def process_and_extract_features(artist, title):
    """
    Downloads a song and extracts full features (Librosa + AST).
    """
    safe_title = sanitize_filename(f"{artist} - {title}")[:150]
    
    # --- ROUTE TO SPECIFIC FOLDER ---
    # This automatically builds the path: data/audio/temp_classification
    temp_dir = os.path.join("data", "audio", "temp_classification")
    
    # This ensures the folder exists so the app doesn't crash if you accidentally delete it
    os.makedirs(temp_dir, exist_ok=True) 
    
    # Final path: data/audio/temp_classification/SongName.mp3
    temp_path = os.path.join(temp_dir, f"{safe_title}.mp3")
    # --------------------------------
    
    # 1. Download via yt-dlp to disk
    success = _download_to_disk(f"ytsearch1:{artist} {title}", temp_path)
    
    if not success:
        return None

    try:
        # 2. AST Model Features (Sampling rate 16k)
        # We limit duration during load to save memory/processing time, similar to the old -t 120 flag
        y_ast, sr_ast = librosa.load(temp_path, sr=16000, duration=120) 
        inputs = processor(y_ast, sampling_rate=sr_ast, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits
        probabilities = torch.sigmoid(logits[0])
        ast_features = {model.config.id2label[i]: probabilities[i].item() for i in range(len(probabilities))}

        # 3. Full Librosa Features
        y, sr = librosa.load(temp_path, mono=True, duration=60)
        lib_features = {}
        
        lib_features['tempo'] = librosa.feature.tempo(y=y, sr=sr)[0]
        
        rms = librosa.feature.rms(y=y)
        lib_features['rms_mean'], lib_features['rms_std'] = np.mean(rms), np.std(rms)
        
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        lib_features['spectral_centroid_mean'], lib_features['spectral_centroid_std'] = np.mean(spec_cent), np.std(spec_cent)
        
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        lib_features['spectral_bandwidth_mean'], lib_features['spectral_bandwidth_std'] = np.mean(spec_bw), np.std(spec_bw)
        
        zcr = librosa.feature.zero_crossing_rate(y)
        lib_features['zero_crossing_rate_mean'], lib_features['zero_crossing_rate_std'] = np.mean(zcr), np.std(zcr)
        
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        for i in range(20):
            lib_features[f'mfcc_{i+1}_mean'] = np.mean(mfccs[i])
            lib_features[f'mfcc_{i+1}_std'] = np.std(mfccs[i])

        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        return {**lib_features, **ast_features}

    except Exception as e:
        print(f"-> Extraction error for {title}: {e}")
        if os.path.exists(temp_path): os.remove(temp_path)
        return None

def _download_to_disk(query, output_path):
    base_path = output_path.rsplit('.mp3', 1)[0]
    
    ydl_opts = {
        'format': 'bestaudio/best', 
        'outtmpl': base_path, 
        
        # Keep using Firefox cookies
        'cookiesfrombrowser': ('firefox',), 
        
        # --- THE FIX: Explicitly tell yt-dlp to use Node.js for puzzles ---
        'js_runtimes': {'node': {}},
        # ------------------------------------------------------------------
        
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'quiet': True, 
        'no_warnings': True,
    }
    
    try:
        # Since we are running this inside a Python script, 
        # it will automatically use the correct, updated yt-dlp package!
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([query])
            return True
    except Exception as e:
        print(f"\n[DOWNLOAD ERROR] yt-dlp failed on '{query}': {e}")
        return False

"""
def _download_via_stream(query, output_path):
    ydl_opts = {
        'format': 'bestaudio/best',
        'cookiefile': 'cookies.txt',
        'quiet': False,       
        'no_warnings': False, 
        'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36'
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(query, download=False)['entries'][0]
            audio_url = info['url']
            
            # THE FIX: Tell FFmpeg to use the cookies and a fake User-Agent
            user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            
            cmd = [
                FFMPEG_PATH, 
                '-cookies', 'all', '-cookies_file', 'cookies.txt',  # Pass cookies to FFmpeg
                '-user_agent', user_agent,                          # Pass User-Agent to FFmpeg
                '-i', audio_url, 
                '-t', '120', 
                '-codec:a', 'libmp3lame', 
                '-b:a', '192k', 
                '-y', output_path
            ]
            
            subprocess.run(cmd, check=True)
            return True
            
    except Exception as e:
        print(f"\n[DOWNLOAD ERROR] Failed on '{query}': {e}")
        return False
"""