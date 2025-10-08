# In src/classify_playlist.py

import time
import joblib
import json
from collections import defaultdict
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from tqdm import tqdm
import pandas as pd

# Imports from your project files
from src.feature_extraction import extract_all_features_for_song
from src.config import SPOTIPY_CLIENT_ID, SPOTIPY_CLIENT_SECRET, SPOTIPY_REDIRECT_URI

def classify_playlist():
    """
    Analyzes a user-selected Spotify playlist and saves the classification results to a JSON file.
    """
    # --- AUTHENTICATION (Scope is for reading playlists only) ---
    print("Connecting to Spotify...")
    sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
        client_id=SPOTIPY_CLIENT_ID,
        client_secret=SPOTIPY_CLIENT_SECRET,
        redirect_uri=SPOTIPY_REDIRECT_URI,
        scope="playlist-read-private",
        requests_timeout=15
    ))
    print("Successfully connected to Spotify!")

    # --- Loading the model ---
    print("Loading the trained song classifier model...")
    try:
        model = joblib.load('models/song_classifier.joblib')
    except FileNotFoundError:
        print("Error: 'models/song_classifier.joblib' not found. Please run the training script first.")
        return # Use return instead of exit() in a function

    # --- Playlist Selection ---
    chosen_playlist = None
    try:
        playlists = sp.current_user_playlists()['items']
        if not playlists:
            print("No playlists found in your Spotify account.")
            return

        print("\nYour Spotify Playlists:")
        for i, playlist in enumerate(playlists):
            print(f"  {i+1}. {playlist['name']} ({playlist['tracks']['total']} tracks)")

        while True:
            try:
                choice = int(input("\nEnter the number of the playlist you want to sort: "))
                if 1 <= choice <= len(playlists):
                    chosen_playlist = playlists[choice - 1]
                    break
                else:
                    print("Invalid number. Please try again.")
            except ValueError:
                print("Please enter a valid number.")

    except Exception as e:
        print(f"An error occurred while fetching playlists: {e}")
        return

    # --- Classification Loop ---
    if chosen_playlist:
        playlist_id_to_sort = chosen_playlist['id']
        print(f"\nFetching tracks from '{chosen_playlist['name']}'...")
        results = sp.playlist_items(playlist_id_to_sort)
        tracks_to_classify = results['items']
        while results['next']:
            results = sp.next(results)
            tracks_to_classify.extend(results['items'])

        sorted_songs = defaultdict(list)
        print("Classifying songs...")
        for item in tqdm(tracks_to_classify, desc="Classifying"):
            try:
                track = item['track']
                
                # Check if the track data is valid
                if track and track['artists']:
                    artist = track['artists'][0]['name']
                    name = track['name']
                    
                    # This is the main function that can fail (downloading, analysis, etc.)
                    features = extract_all_features_for_song(artist, name)
                    
                    if features:
                        # Convert the single dictionary of features into a DataFrame
                        # so it has the same format as the training data
                        df_features = pd.DataFrame([features])

                        # Use the model to predict the label (category)
                        prediction = model.predict(df_features)
                        predicted_label = prediction[0]
                        
                        # Add the song to our sorted dictionary
                        sorted_songs[predicted_label].append(f"{artist} - {name}")
                
                # Rate limit your API calls
                time.sleep(0.1)

            except Exception as e:
                # If ANY error occurs for this one song, we just continue.
                # You can uncomment the line below for debugging to see which songs are failing.
                # print(f"\nSkipping a song due to an error: {e}")
                continue # This is the key command to "just move on" to the next song.

        # --- Display and Save Results to JSON ---
        print("\n--- Classification Complete ---")
        output_filename = 'classification_results.json'
        for label, songs in sorted_songs.items():
            print(f"\nCategory: {label} ({len(songs)} songs)")
            for song_title in songs[:5]: # Preview first 5
                print(f"   - {song_title}")
            if len(songs) > 5:
                print(f"   - ... and {len(songs) - 5} more.")
        
        with open(output_filename, 'w') as f:
            json.dump(sorted_songs, f, indent=4)
        print(f"\nFull results saved to '{output_filename}'.")
        print("Run the 'create_spotify_playlists.py' script to build these playlists on Spotify.")

if __name__ == '__main__':
    classify_playlist()