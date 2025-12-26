""" import time
import joblib
import pandas as pd
from collections import defaultdict
import spotipy
from spotipy.oauth2 import SpotifyOAuth

# Import your extraction logic
from src.feature_extraction import process_and_extract_features as extract_all_features_for_song
from src.config import SPOTIPY_CLIENT_ID, SPOTIPY_CLIENT_SECRET, SPOTIPY_REDIRECT_URI

def get_spotify_client():
    return spotipy.Spotify(auth_manager=SpotifyOAuth(
        client_id=SPOTIPY_CLIENT_ID,
        client_secret=SPOTIPY_CLIENT_SECRET,
        redirect_uri=SPOTIPY_REDIRECT_URI,
        scope="playlist-read-private playlist-modify-public playlist-modify-private",
        requests_timeout=15
    ))

def get_user_playlists():
    #Returns all user playlists for the frontend dropdown.
    sp = get_spotify_client()
    playlists = []
    results = sp.current_user_playlists()
    playlists.extend(results['items'])
    while results['next']:
        results = sp.next(results)
        playlists.extend(results['items'])
    return [{'id': p['id'], 'name': p['name'], 'total': p['tracks']['total']} for p in playlists]

def is_song_in_playlist(sp, playlist_id, track_id):
    #Checks if a song is already in a playlist to prevent duplicates.
    results = sp.playlist_items(playlist_id, fields="items(track(id)),next")
    items = results['items']
    while results['next']:
        results = sp.next(results)
        items.extend(results['items'])
    return any(item['track']['id'] == track_id for item in items if item['track'])

def classify_and_create(playlist_id, approval_mode=False, add_repeats=False, callback=None):
    sp = get_spotify_client()
    
    try:
        model = joblib.load('models/song_classifier.joblib')
    except Exception as e:
        print(f"Model load error: {e}")
        return

    # Map playlist names to IDs for the current user
    user_p_data = sp.current_user_playlists()
    user_playlists = {p['name']: p['id'] for p in user_p_data['items']}
    
    # Get all tracks from the source playlist (handling pagination)
    results = sp.playlist_items(playlist_id)
    tracks = results['items']
    while results['next']:
        results = sp.next(results)
        tracks.extend(results['items'])

    for item in tracks:
        track = item['track']
        if not (track and track['artists']): continue
        
        artist = track['artists'][0]['name']
        name = track['name']
        track_id = track['id']
        
        # 1. Feature Extraction
        features = extract_all_features_for_song(artist, name, "temp_classification")
        
        if features:
            df_features = pd.DataFrame([features])
            probs = model.predict_proba(df_features)[0]
            classes = model.classes_
            
            # 2. 70% Confidence Logic
            high_conf_idx = [i for i, p in enumerate(probs) if p >= 0.7]
            
            if high_conf_idx:
                # Add to all playlists meeting 70% threshold
                final_assignments = [(classes[i], probs[i]) for i in high_conf_idx]
            else:
                # If none hit 70%, take only the top one
                top_idx = probs.argmax()
                final_assignments = [(classes[top_idx], probs[top_idx])]

            # 3. Handle Assignments
            for label, conf in final_assignments:
                if callback:
                    callback(artist, name, track_id, label, conf, features)

                if not approval_mode:
                    target_id = user_playlists.get(label)
                    if target_id:
                        already_exists = is_song_in_playlist(sp, target_id, track_id)
                        if not already_exists or add_repeats:
                            sp.playlist_add_items(target_id, [track_id])
                            print(f"Auto-added {name} to {label}")
        
        time.sleep(0.1) """
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

# --- ADD THIS FUNCTION BACK ---
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
    results = sp.playlist_items(playlist_id, fields="items(track(id)),next")
    items = results['items']
    while results['next']:
        results = sp.next(results)
        items.extend(results['items'])
    return any(item['track']['id'] == track_id for item in items if item['track'])

# --- ENSURE THESE MATCH THE APP.PY IMPORTS ---
def save_final_results(artist, name, track_id, label, features, add_repeats):
    sp = get_spotify_client()
    # Find the target playlist ID by searching user playlists
    all_p = sp.current_user_playlists()['items']
    target_id = next((p['id'] for p in all_p if p['name'] == label), None)

    if target_id:
        if not is_song_in_playlist(sp, target_id, track_id) or add_repeats:
            sp.playlist_add_items(target_id, [track_id])
            
            # Append to CSV Memory
            new_row = {**features, 'artist': artist, 'track': name, 'label': label}
            pd.DataFrame([new_row]).to_csv(CSV_PATH, mode='a', header=not os.path.exists(CSV_PATH), index=False)

def classify_and_create(playlist_id, approval_mode=False, add_repeats=False, callback=None):
    sp = get_spotify_client()
    model = joblib.load('models/song_classifier.joblib')
    cache_df = pd.read_csv(CSV_PATH) if os.path.exists(CSV_PATH) else pd.DataFrame()

    results = sp.playlist_items(playlist_id)
    tracks = results['items']
    while results['next']:
        results = sp.next(results)
        tracks.extend(results['items'])

    for item in tracks:
        track = item['track']
        if not track: continue
        artist, name, tid = track['artists'][0]['name'], track['name'], track['id']
        
        # 1. Memory Check
        match = cache_df[(cache_df['artist'] == artist) & (cache_df['track'] == name)]
        
        if not match.empty:
            # Drop metadata to get pure features
            features = match.iloc[0].drop(['label', 'artist', 'track']).to_dict()
        else:
            features = process_and_extract_features(artist, name)

        if features:
            probs = model.predict_proba(pd.DataFrame([features]))[0]
            classes = model.classes_
            
            # 70% Logic
            high_conf_idx = [i for i, p in enumerate(probs) if p >= 0.7]
            if high_conf_idx:
                final_labels = [(classes[i], probs[i]) for i in high_conf_idx]
            else:
                top_idx = probs.argmax()
                final_labels = [(classes[top_idx], probs[top_idx])]

            for label, conf in final_labels:
                if callback:
                    callback(artist, name, tid, label, conf, features)

                if not approval_mode:
                    save_final_results(artist, name, tid, label, features, add_repeats)
        time.sleep(0.1)