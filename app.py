from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import threading
import traceback # NEW: For printing exact error lines

from src.classify_playlist import get_user_playlists, classify_and_create

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

@app.route('/')
def index():
    return render_template('index.html', playlists=get_user_playlists())

@socketio.on('start_classification')
def handle_start(data):
    print(f"\n[SERVER] Received request to sort playlist: {data['playlist_id']}")
    
    def progress(artist, name, tid, label, conf):
        socketio.emit('update', {
            'song': f"{artist} - {name}", 
            'label': label,
            'confidence': f"{conf*100:.1f}%"
        })

    # Wrapper function to catch and print thread errors
    def run_worker():
        try:
            print("[THREAD] Starting worker...")
            classify_and_create(data['playlist_id'], data['add_repeats'], progress)
            print("[THREAD] Worker finished successfully.")
        except Exception as e:
            print(f"\n❌ [CRITICAL THREAD ERROR]: {e}")
            traceback.print_exc() # Prints the exact line that failed
            socketio.emit('status', {'msg': f"Error: {str(e)}"}) # Send error to UI

    t = threading.Thread(target=run_worker)
    t.daemon = True # Ensures thread dies if you close the Flask app
    t.start()

if __name__ == '__main__':
    socketio.run(app, debug=True)