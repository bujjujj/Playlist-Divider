import os
import pandas as pd
import tqdm
import random
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import time

# Make sure these imports are correct based on your file structure
from src.feature_extraction import extract_all_features_for_song
from src.config import SPOTIPY_CLIENT_ID, SPOTIPY_CLIENT_SECRET, SPOTIPY_REDIRECT_URI

def gather_training_data():
    """
    A resumable script to gather training data from Spotify playlists.
    """
    # --- 1. SETUP ---
    TRAINING_PLAYLISTS = {
        'hiphop-workout': 'spotify:playlist:6nZs9qkLc2Pg6Q2iZOSgbk',
        'makeout': 'spotify:playlist:4qZs86VNq1kXaRtCE5lcSr',
        'chase': 'spotify:playlist:74s26XHRLZ716UvUj3hL4S',
        'lofi-downtempo': 'spotify:playlist:2DRvUsr4TnWlAvFYv5B1xi', #very huge playlist, only take first 200 songs
        'instrumental-happy': 'spotify:playlist:3L3ChTfSqTO6QdEfCd7l0s',
        'ambient-focus': 'spotify:playlist:70S8eB9yATWo90aQny9oGb', #very huge playlist, only take first 200 songs
        'atmospheric-room': 'spotify:playlist:36vNl3AjU4sbCQcsQUOq3K',
        'acoustic-guitar': 'spotify:playlist:3ru5tc8HpvzsOklw78rKnf',
        'citypop': 'spotify:playlist:5drMgosoieMPSYbq46ugqa',
        'feel-good': 'spotify:playlist:4bY71u1Mc66zxbWSmPjBeF',
        'edm-club': 'spotify:playlist:2Fl0AxmDN4BPYvgZrtQSZF'
    }
    MAX_SONGS_PER_PLAYLIST = 200
    output_filename = 'training_features.csv'
    processed_songs = set()

    # --- 2. LOAD MEMORY (Previously Processed Songs) ---
    print("Loading script memory...")
    if os.path.exists(output_filename):
        try:
            df_existing = pd.read_csv(output_filename)
            # Create a unique identifier for each song (artist, track)
            for index, row in df_existing.iterrows():
                processed_songs.add((row['artist'], row['track']))
            print(f"Found {len(processed_songs)} songs that have already been processed.")
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
            while results['next']:
                results = sp.next(results)
                items.extend(results['items'])
        except Exception as e:
            print(f"CRITICAL ERROR: Could not fetch playlist '{label}'. Skipping. Error: {e}")
            continue

        if len(items) > MAX_SONGS_PER_PLAYLIST:
            print(f"Playlist has {len(items)} songs. Taking a reproducible random sample of {MAX_SONGS_PER_PLAYLIST}.")
            random.seed(42)
            items = random.sample(items, MAX_SONGS_PER_PLAYLIST)
        
        # --- Loop through each song in the playlist ---
        for item in tqdm.tqdm(items, desc=f"Processing '{label}'"):
            try:
                track = item['track']
                if not (track and track['artists']):
                    continue

                artist = track['artists'][0]['name']
                name = track['name']

                # --- THE RESUME CHECK ---
                if (artist, name) in processed_songs:
                    continue # Instantly skip this song

                # --- Main work is done here ---
                features = extract_all_features_for_song(artist, name, label)
                
                # --- INSTANT SAVE ---
                if features:
                    # Add identifiers for our "memory"
                    features['artist'] = artist
                    features['track'] = name
                    features['label'] = label
                    
                    new_row_df = pd.DataFrame([features])
                    
                    # Append new row to the CSV. Write header only if file is new.
                    write_header = not os.path.exists(output_filename)
                    new_row_df.to_csv(output_filename, mode='a', header=write_header, index=False)
                    
                    # Update our in-memory set
                    processed_songs.add((artist, name))
                
                time.sleep(0.1) # Small delay to be kind to APIs

            except Exception as e:
                # This will catch any unexpected error for a single song and move on
                print(f"\nSkipping a song due to an unexpected error: {e}")
                continue

        print(f"Finished processing playlist '{label}'.")

    print("\n--- All Playlists Processed ---")


if __name__ == '__main__':
    gather_training_data()