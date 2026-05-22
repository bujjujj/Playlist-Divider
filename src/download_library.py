import os
import random
import time
import subprocess
import tqdm
import spotipy
import yt_dlp
from spotipy.oauth2 import SpotifyOAuth
from yt_dlp.utils import sanitize_filename

# Import credentials from your config file
from src.config import SPOTIPY_CLIENT_ID, SPOTIPY_CLIENT_SECRET, SPOTIPY_REDIRECT_URI, FFMPEG_PATH

# --- CONFIGURATION ---
OUTPUT_DIR = 'data/library'
COOKIES_FILE = 'cookies.txt'
MAX_SONGS_PER_PLAYLIST = 35 

TRAINING_PLAYLISTS = {
    'shazam-library': 'spotify:playlist:5ph0zF40yAuw05p5PyvHGT'
}

def download_via_stream(query, output_path):
    """
    Uses yt-dlp to find the URL and ffmpeg to stream/convert it to MP3.
    """
    ydl_opts = {
        'format': 'bestaudio/best',
        'cookiefile': COOKIES_FILE,
        'quiet': True,
        'no_warnings': True,
        'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36'
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # 1. Get the direct stream URL
            # Note: We added 'official audio' to the search to improve accuracy
            info = ydl.extract_info(query, download=False)
            
            if 'entries' in info:
                info = info['entries'][0]
                
            audio_url = info['url']
            
            # 2. Use FFmpeg to stream download directly to MP3
            # FIX: Removed '-t', '120' so it downloads the whole song
            cmd = [
                FFMPEG_PATH, 
                '-i', audio_url, 
                '-codec:a', 'libmp3lame', 
                '-b:a', '192k', 
                '-y', # Overwrite if exists
                output_path
            ]
            
            # We use text=True to capture error messages if it fails
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            return True
            
    except subprocess.CalledProcessError as e:
        print(f"\n[FFmpeg Error] {e.stderr}")
        return False
    except Exception as e:
        if "403" in str(e):
            print(f"\n[!] Access Blocked (403). Your cookies.txt might be expired.")
        else:
            print(f"\n[!] Error finding URL: {e}")
        return False

def gather_audio_library():
    # 1. Setup Directories
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 2. Connect to Spotify
    print("Connecting to Spotify...")
    sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
        client_id=SPOTIPY_CLIENT_ID,
        client_secret=SPOTIPY_CLIENT_SECRET,
        redirect_uri=SPOTIPY_REDIRECT_URI,
        scope="playlist-read-private"
    ))
    print("Connected.")

    # 3. Iterate Playlists
    for label, playlist_id in TRAINING_PLAYLISTS.items():
        print(f"\n--- Processing Playlist: '{label}' ---")
        
        try:
            results = sp.playlist_items(playlist_id)
            items = results['items']
            while results['next']:
                results = sp.next(results)
                items.extend(results['items'])
        except Exception as e:
            print(f"Could not fetch playlist '{label}': {e}")
            continue

        if len(items) > MAX_SONGS_PER_PLAYLIST:
            random.seed(42)
            items = random.sample(items, MAX_SONGS_PER_PLAYLIST)

        for item in tqdm.tqdm(items, desc=f"Downloading {label}"):
            try:
                track = item.get('track')
                if not track or not track.get('artists'): continue

                artist = track['artists'][0]['name']
                name = track['name']

                safe_name = sanitize_filename(f"{artist} - {name}")
                file_path = os.path.join(OUTPUT_DIR, f"{safe_name}.mp3")

                if os.path.exists(file_path):
                    # Check file size to ensure previous download wasn't a 0kb error
                    if os.path.getsize(file_path) > 100000: 
                        continue
                    else:
                        os.remove(file_path) # Delete corrupted file

                # IMPROVEMENT: refined search query for better accuracy
                # Adding "official audio" helps avoid music videos with long intros
                query = f"ytsearch1:{artist} - {name} official audio"
                
                success = download_via_stream(query, file_path)
                
                if success:
                    time.sleep(random.uniform(1.0, 3.0))
                else:
                    print(f"Failed to download: {name}")

            except Exception as e:
                print(f"Error processing item: {e}")
                continue

if __name__ == '__main__':
    gather_audio_library()