import os
import pandas as pd
import tqdm
import random
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import time

from src.feature_extraction import process_and_extract_features
from src.config import SPOTIPY_CLIENT_ID, SPOTIPY_CLIENT_SECRET, SPOTIPY_REDIRECT_URI

def gather_training_data():
    """
    A resumable script to gather training data from Spotify playlists.
    It orchestrates the download, feature extraction, and data saving.
    """
    # --- 1. SETUP ---
    TRAINING_PLAYLISTS = {
        'rap-adjacent': 'spotify:playlist:7AYWhrfuCo9MRscfTvBEC1',
        'makeout': 'spotify:playlist:4qZs86VNq1kXaRtCE5lcSr',
        'chase': 'spotify:playlist:74s26XHRLZ716UvUj3hL4S',
        'lofi-downtempo': 'spotify:playlist:2DRvUsr4TnWlAvFYv5B1xi',
        'instrumental-happy': 'spotify:playlist:3L3ChTfSqTO6QdEfCd7l0s',
        'ambient-focus': 'spotify:playlist:70S8eB9yATWo90aQny9oGb',
        'room': 'spotify:playlist:36vNl3AjU4sbCQcsQUOq3K',
        'canonsburg': 'spotify:playlist:60qXIMg2QYAIjo5TQVp3mi',
        'citypop': 'spotify:playlist:5drMgosoieMPSYbq46ugqa',
        'upbeat': 'spotify:playlist:5hU1rIGdbFsQxftzlcGpA2', #stopped here
        'edm-club': 'spotify:playlist:2Fl0AxmDN4BPYvgZrtQSZF'
    }
    MAX_SONGS_PER_PLAYLIST = 150
    output_filename = 'training_features.csv'
    processed_songs_memory = set()

    # --- 2. LOAD MEMORY (From previously saved CSV) ---
    print("Loading script memory from CSV...")
    if os.path.exists(output_filename):
        try:
            df_existing = pd.read_csv(output_filename)
            for index, row in df_existing.iterrows():
                processed_songs_memory.add((row['artist'], row['track']))
            print(f"Found {len(processed_songs_memory)} songs with features already saved.")
        except pd.errors.EmptyDataError:
            print("Found an empty existing CSV. Starting fresh.")
    else:
        print("No existing data found. Starting a new training set.")

    # --- 3. AUTHENTICATION ---
    print("Connecting to Spotify...")
    sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
        client_id=SPOTIPY_CLIENT_ID,
        client_secret=SPOTIPY_CLIENT_SECRET,
        redirect_uri=SPOTIPY_REDIRECT_URI,
        scope="playlist-read-private",
        requests_timeout=30
    ))
    print("Successfully connected to Spotify!")

    # --- 4. MAIN DATA GATHERING LOOP ---
    for label, playlist_id in TRAINING_PLAYLISTS.items():
        print(f"\n--- Starting Playlist: '{label}' ---")
        
        try:
            results = sp.playlist_items(playlist_id)
            items = results['items']
            # Continue fetching all songs from the playlist
            while results['next']:
                results = sp.next(results)
                items.extend(results['items'])
        except Exception as e:
            print(f"CRITICAL ERROR: Could not fetch playlist '{label}'. Skipping. Error: {e}")
            continue

        # --- UPDATED: Changed from taking the first X songs to a random sample ---
        if len(items) > MAX_SONGS_PER_PLAYLIST:
            print(f"Playlist has {len(items)} songs. Taking a reproducible random sample of {MAX_SONGS_PER_PLAYLIST}.")
            random.seed(42) # Use a seed for reproducible "random" sampling
            items = random.sample(items, MAX_SONGS_PER_PLAYLIST)

        # --- Loop through each song in the playlist ---
        for item in tqdm.tqdm(items, desc=f"Processing '{label}'"):
            try:
                track = item.get('track')
                if not (track and track.get('artists')):
                    continue

                artist = track['artists'][0]['name']
                name = track['name']

                # --- Check against CSV memory ---
                if (artist, name) in processed_songs_memory:
                    continue 

                # --- DELEGATE ALL WORK TO THE WORKER FUNCTION ---
                features = process_and_extract_features(
                    artist=artist,
                    title=name,
                    playlist_label=label,
                    keep_audio=False
                )
                
                # --- INSTANT SAVE ---
                if features:
                    features.update({'artist': artist, 'track': name, 'label': label})
                    new_row_df = pd.DataFrame([features])
                    
                    write_header = not os.path.exists(output_filename)
                    new_row_df.to_csv(
                        output_filename, 
                        mode='a', 
                        header=write_header, 
                        index=False, 
                        encoding='utf-8-sig'
                    )
                    
                    processed_songs_memory.add((artist, name))
                
                time.sleep(random.uniform(1.2, 2.3))

            except Exception as e:
                print(f"\nSkipping a song due to an unexpected error in the main loop: {e}")
                continue

        print(f"Finished processing playlist '{label}'.")

    print("\n--- Script Complete: All Playlists Processed ---")

if __name__ == '__main__':
    gather_training_data()