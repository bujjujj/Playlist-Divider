import yt_dlp
import os
import re
import numpy as np
import librosa
import torch
from transformers import AutoProcessor, AutoModelForAudioClassification

processor = AutoProcessor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
model = AutoModelForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")

def download_song_as_mp3(artist, title, playlist_label, base_dir="data/audio"):
    """
    Finds a song, sanitizes the filename, downloads it as an MP3,
    and returns the EXACT final file path from yt-dlp.
    """
    # Create the specific subfolder for the playlist
    output_dir = os.path.join(base_dir, playlist_label)
    os.makedirs(output_dir, exist_ok=True)

    search_query = f"ytsearch1:{artist} - {title} lyrics"
    base_filename = f"{artist} - {title}"
    safe_filename = re.sub(r'[\\/*?:"<>|]', "", base_filename).strip(' .')[:150]
    output_template = os.path.join(output_dir, safe_filename)

    final_mp3_path = os.path.join(output_dir, f"{safe_filename}.mp3")
    # If a file already exists at that exact path, return the path and stop.
    if os.path.exists(final_mp3_path):
        return final_mp3_path

    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': output_template,
        'noplaylist': True,
        'quiet': True,
        'no_warnings': True,
        'keepvideo': False
    }

    try:
        # Use extract_info to get metadata, including the final file path
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(search_query, download=True)
            # The actual path of the post-processed file
            final_filepath = info_dict['entries'][0]['requested_downloads'][0]['filepath']
            
            # If the file already existed, the path might not be in the dict, so we build it
            if not os.path.exists(final_filepath):
                 final_filepath = output_template + ".mp3"

            return final_filepath
            
    except Exception as e:
        return None
    
def get_instrument_probabilities(audio_file_path):
    """
    Analyzes an audio file and returns the model's confidence for various instruments/sounds.
    """
    try:
        # Load audio file. The model expects a 16kHz sample rate.
        y, sr = librosa.load(audio_file_path, sr=16000)

        # Process the audio waveform
        inputs = processor(y, sampling_rate=sr, return_tensors="pt")

        # Make a prediction
        with torch.no_grad():
            logits = model(**inputs).logits

        # Get the top 5 predicted classes
        predicted_class_ids = torch.topk(logits, k=5).indices.tolist()[0]
        
        # Apply sigmoid to logits to get probabilities
        probabilities = torch.sigmoid(logits[0])
        
        print(f"\n--- Top 5 Predictions for: {audio_file_path.split('/')[-1]} ---")
        results = {}
        for i in predicted_class_ids:
            label = model.config.id2label[i]
            score = probabilities[i].item()
            results[label] = score
            print(f"{label}: {score:.4f}")
        
        # This dictionary of label:probability is what you'd use as features
        return results

    except Exception as e:
        print(f"Could not analyze {audio_file_path}. Error: {e}")
        return None

def get_librosa_features(audio_file_path):
    """
    Loads an audio file and extracts a dictionary of features using librosa.
    Analyzes the first 60 seconds for efficiency.
    """
    features = {}
    try:
        # Load the first 60 seconds of the audio file
        # mono=True converts the signal to mono, which is standard for most feature extraction
        y, sr = librosa.load(audio_file_path, mono=True, duration=60)

        # --- RHYTHMIC FEATURES ---
        # Tempo: The speed of the music in Beats Per Minute (BPM)
        features['tempo'] = librosa.feature.tempo(y=y, sr=sr)[0]

        # --- DYNAMIC FEATURES ---
        # Root-Mean-Square (RMS) Energy: The average loudness of the song
        rms = librosa.feature.rms(y=y)
        features['rms_mean'] = np.mean(rms)
        features['rms_std'] = np.std(rms) # Standard deviation of energy (how dynamic is it?)

        # --- TIMBRAL/TONAL FEATURES ---
        # Spectral Centroid: The "center of mass" of the spectrum. Correlates to "brightness".
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        features['spectral_centroid_mean'] = np.mean(spec_cent)
        features['spectral_centroid_std'] = np.std(spec_cent)

        # Spectral Bandwidth: The width of the frequency band.
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        features['spectral_bandwidth_mean'] = np.mean(spec_bw)
        features['spectral_bandwidth_std'] = np.std(spec_bw)

        # Spectral Rolloff: The frequency below which a specified percentage of the total spectral energy lies.
        spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        features['spectral_rolloff_mean'] = np.mean(spec_rolloff)
        features['spectral_rolloff_std'] = np.std(spec_rolloff)
        
        # Zero-Crossing Rate: The rate of sign-changes in the signal. Correlates to percussiveness.
        zcr = librosa.feature.zero_crossing_rate(y)
        features['zero_crossing_rate_mean'] = np.mean(zcr)
        features['zero_crossing_rate_std'] = np.std(zcr)

        # Mel-Frequency Cepstral Coefficients (MFCCs): The "gold standard" for timbre.
        # We'll take the average and standard deviation of the first 20 MFCCs.
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        for i in range(20):
            features[f'mfcc_{i+1}_mean'] = np.mean(mfccs[i])
            features[f'mfcc_{i+1}_std'] = np.std(mfccs[i])
            
        return features

    except Exception as e:
        print(f"Error processing {audio_file_path} with librosa: {e}")
        return None

# You could even combine them into one main function
def extract_all_features_for_song(artist, title, playlist_label):
    # 1. Download the audio
    audio_file = download_song_as_mp3(artist, title, playlist_label)
    
    if audio_file and os.path.exists(audio_file):
        # 2. Get instrument features
        instrument_features = get_instrument_probabilities(audio_file)
        
        # 3. Get librosa features
        librosa_features = get_librosa_features(audio_file)
        
        # 4. Combine and return them
        all_features = {**librosa_features, **instrument_features}
        return all_features
        
    return None