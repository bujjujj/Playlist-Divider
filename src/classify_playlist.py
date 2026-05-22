import pandas as pd
import os
import time
import joblib
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from src.feature_extraction import process_and_extract_features
from src.config import SPOTIPY_CLIENT_ID, SPOTIPY_CLIENT_SECRET, SPOTIPY_REDIRECT_URI

CSV_PATH = 'training_features.csv'

def get_spotify_client():
    return spotipy.Spotify(auth_manager=SpotifyOAuth(
        client_id=SPOTIPY_CLIENT_ID, 
        client_secret=SPOTIPY_CLIENT_SECRET,
        redirect_uri=SPOTIPY_REDIRECT_URI,
        scope="playlist-read-private playlist-modify-public playlist-modify-private"
    ))

def get_user_playlists():
    """Returns all user playlists for the frontend dropdown."""
    sp = get_spotify_client()
    playlists = []
    results = sp.current_user_playlists()
    playlists.extend(results['items'])
    while results['next']:
        results = sp.next(results)
        playlists.extend(results['items'])
    return [{'id': p['id'], 'name': p['name'], 'total': p['tracks']['total']} for p in playlists]

def is_song_in_playlist(sp, playlist_id, track_id):
    """Checks if a song is already in a playlist to prevent duplicates."""
    results = sp.playlist_items(playlist_id, fields="items(track(id)),next")
    items = results['items']
    while results['next']:
        results = sp.next(results)
        items.extend(results['items'])
    return any(item['track']['id'] == track_id for item in items if item['track'])

def save_final_result(artist, name, track_id, label, features, add_repeats):
    """Adds the track to Spotify and updates the local CSV cache."""
    sp = get_spotify_client()
    
    all_p = sp.current_user_playlists()['items']
    target_id = next((p['id'] for p in all_p if p['name'] == label), None)

    if target_id:
        if not is_song_in_playlist(sp, target_id, track_id) or add_repeats:
            sp.playlist_add_items(target_id, [track_id])
            print(f"Added {name} to {label}")
            
            # Append to CSV Memory so we don't need to re-extract in the future
            new_row = {**features, 'artist': artist, 'track': name, 'label': label}
            pd.DataFrame([new_row]).to_csv(CSV_PATH, mode='a', header=not os.path.exists(CSV_PATH), index=False)

def classify_and_create(playlist_id, add_repeats=False, callback=None):
    """The main worker function."""
    print("[1/5] Authenticating Spotify...")
    sp = get_spotify_client()
    
    print("[2/5] Loading Machine Learning Model...")
    model = joblib.load('models/song_classifier.joblib')
    
    print("[3/5] Loading local CSV memory...")
    cache_df = pd.read_csv(CSV_PATH) if os.path.exists(CSV_PATH) else pd.DataFrame()

    print(f"[4/5] Fetching tracks for playlist ID {playlist_id}...")
    results = sp.playlist_items(playlist_id)
    tracks = results['items']
    while results['next']:
        results = sp.next(results)
        tracks.extend(results['items'])

    print(f"[5/5] Processing {len(tracks)} tracks. Starting loop...")
    for item in tracks:
        track = item['track']
        if not track: continue
        
        artist, name, tid = track['artists'][0]['name'], track['name'], track['id']
        print(f" -> Analyzing: {artist} - {name}")
        
        # 1. Memory Check
        match = cache_df[(cache_df['artist'] == artist) & (cache_df['track'] == name)]
        
        if not match.empty:
            print("    (Found in cache, skipping extraction)")
            features = match.iloc[0].drop(['label', 'artist', 'track']).to_dict()
        else:
            print("    (Extracting new audio features via yt-dlp & librosa...)")
            features = process_and_extract_features(artist, name)

        # 2. Prediction & Assignment
        if features:
            probs = model.predict_proba(pd.DataFrame([features]))[0]
            classes = model.classes_
            
            top_idx = probs.argmax()
            best_label = classes[top_idx]
            best_conf = probs[top_idx]

            if callback:
                callback(artist, name, tid, best_label, best_conf)

            save_final_result(artist, name, tid, best_label, features, add_repeats)
            
        time.sleep(0.1)