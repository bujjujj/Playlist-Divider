# In src/create_spotify_playlists.py

import json
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from tqdm import tqdm
import time

# Imports from project files
from src.config import SPOTIPY_CLIENT_ID, SPOTIPY_CLIENT_SECRET, SPOTIPY_REDIRECT_URI

def create_playlists():
    """
    Loads classification results from a JSON file and creates the corresponding
    playlists on Spotify.
    """
    # --- AUTHENTICATION (Scope is for creating/modifying playlists) ---
    print("Connecting to Spotify...")
    sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
        client_id=SPOTIPY_CLIENT_ID,
        client_secret=SPOTIPY_CLIENT_SECRET,
        redirect_uri=SPOTIPY_REDIRECT_URI,
        scope="playlist-modify-public", # This scope allows playlist creation
        requests_timeout=15
    ))
    user_id = sp.current_user()['id']
    print("Successfully connected to Spotify!")

    # --- Load Classification Results ---
    input_filename = 'classification_results.json'
    print(f"Loading song data from '{input_filename}'...")
    try:
        with open(input_filename, 'r') as f:
            playlists_to_create = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find '{input_filename}'. Please run 'classify_playlist.py' first.")
        return

    # --- Playlist Creation Loop ---
    print("\n--- Starting Playlist Creation ---")
    for playlist_name, tracks in playlists_to_create.items():
        print(f"\nProcessing playlist: '{playlist_name}'...")

        new_playlist = sp.user_playlist_create(
            user=user_id,
            name=f"Sorted by ML: {playlist_name}",
            public=True,
            description=f"Songs classified as '{playlist_name}' by my personal ML model on {time.strftime('%Y-%m-%d')}."
        )
        playlist_id = new_playlist['id']
        
        track_uris_to_add = []
        print(f"Searching for {len(tracks)} tracks on Spotify...")
        for song_title in tqdm(tracks, desc=f"Finding '{playlist_name}' tracks"):
            try:
                artist, track_name = song_title.split(' - ', 1)
                results = sp.search(q=f"artist:{artist.strip()} track:{track_name.strip()}", type="track", limit=1)
                if results['tracks']['items']:
                    uri = results['tracks']['items'][0]['uri']
                    track_uris_to_add.append(uri)
            except ValueError:
                continue
                
        if track_uris_to_add:
            print(f"Adding {len(track_uris_to_add)} songs to the new playlist...")
            for i in range(0, len(track_uris_to_add), 100):
                batch = track_uris_to_add[i:i+100]
                sp.playlist_add_items(playlist_id, batch)
        else:
            print("No tracks were found on Spotify for this category.")

    print("\nAll playlists have been created! Check your Spotify account.")

if __name__ == '__main__':
    create_playlists()