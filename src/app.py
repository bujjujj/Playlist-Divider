""" import os
import threading
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
from .classify_playlist import get_user_playlists, classify_and_create, get_spotify_client

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

@app.route('/')
def index():
    playlists = get_user_playlists()
    return render_template('index.html', playlists=playlists)

@socketio.on('start_classification')
def handle_start(data):
    playlist_id = data['playlist_id']
    app_mode = data.get('approval_mode', False)
    repeats = data.get('add_repeats', False)
    
    def progress_callback(artist, name, tid, label, conf, features):
        socketio.emit('update', {
            'song': f"{artist} - {name}",
            'track_id': tid,
            'label': label,
            'confidence': f"{conf * 100:.1f}%",
            'features': features
        })

    # Run in thread so the UI doesn't freeze
    t = threading.Thread(target=classify_and_create, 
                         args=(playlist_id, app_mode, repeats, progress_callback))
    t.daemon = True
    t.start()

@socketio.on('manual_action')
def handle_manual(data):
    sp = get_spotify_client()
    track_id = data['track_id']
    label = data['label']
    
    # Refresh playlists to get the latest IDs
    user_playlists = {p['name']: p['id'] for p in sp.current_user_playlists()['items']}
    target_id = user_playlists.get(label)
    
    if target_id:
        sp.playlist_add_items(target_id, [track_id])
        emit('status', {'msg': f"Successfully added to {label}"})
    else:
        emit('status', {'msg': f"Error: Playlist '{label}' not found."})

if __name__ == '__main__':
    socketio.run(app, debug=True, port=5000) """
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import threading
from .classify_playlist import get_user_playlists, classify_and_create, save_final_results

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

@app.route('/')
def index():
    return render_template('index.html', playlists=get_user_playlists())

@socketio.on('start_classification')
def handle_start(data):
    def progress(artist, name, tid, label, conf, feat):
        socketio.emit('update', {
            # Make sure this matches what index.html expects!
            'song': f"{artist} - {name}", 
            'track_id': tid,
            'label': label,
            'confidence': f"{conf*100:.1f}%",
            'features': feat,
            'artist': artist, # Keep these for the Approve button logic
            'name': name
        })

    t = threading.Thread(target=classify_and_create, 
                         args=(data['playlist_id'], data['approval_mode'], data['add_repeats'], progress))
    t.start()

@socketio.on('approve_song')
def on_approve(data):
    # This is triggered when you click Approve in the browser
    save_final_results(data['artist'], data['name'], data['track_id'], data['label'], data['features'], True)
    emit('status', {'msg': f"Saved {data['name']} to {data['label']}"})

if __name__ == '__main__':
    socketio.run(app, debug=True)