# personal_playlist_divider

An automated, ML-powered music curation tool that takes an unsorted Spotify playlist and routes each track into specific mood/genre-based personal playlists — without touching metadata. Instead it downloads raw audio, runs it through a deep learning pipeline, and classifies it using a trained Random Forest model.

Target playlists include genres like `lofi-downtempo`, `citypop`, and `edm-club`.

## Quick Start

```bash
# Setup
cd playlist_divider_project
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux
pip install --upgrade pip
pip install -r requirements.txt

# Configure environment variables (see Prerequisites below)
cp .env.example .env

# Run the app
python app.py
```

Then open `http://localhost:5000` in your browser.

**Every subsequent session:**
```bash
cd playlist_divider_project
venv\Scripts\activate
python app.py
```

## Project Structure

```
playlist_divider_project/
├── app.py                           # Flask & SocketIO entry point
├── training_features.csv            # Persistent local audio feature database
├── data/
│   └── audio/
│       └── temp_classification/     # Temporary storage for downloaded MP3s
├── models/
│   └── song_classifier.joblib       # Serialized Random Forest classifier
├── templates/
│   └── index.html                   # Glassmorphism web UI
└── src/
    ├── config.py                    # Environment variable loader
    ├── classify_playlist.py         # Main orchestration and inference logic
    ├── feature_extraction.py        # yt-dlp, FFmpeg, librosa, and AST processing
    ├── gather_training_data.py      # Offline: builds initial training datasets
    └── run_training.py              # Offline: trains and evaluates the RF model
```

## How It Works

```
User selects source playlist on dashboard
        │
        ▼
┌───────────────────┐
│  classify_        │  Fetch all tracks from Spotify playlist
│  playlist.py      │  Check training_features.csv cache
└────────┬──────────┘
         │ (cache miss — new song)
         ▼
┌───────────────────┐
│  feature_         │  yt-dlp borrows live Firefox session cookies
│  extraction.py    │  Downloads best available audio stream from YouTube
│                   │  FFmpeg extracts 120s MP3 snippet → temp_classification/
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  librosa + AST    │  Extracts MFCCs, tempo, RMS, spectral centroid,
│                   │  zero-crossing rate + runs Hugging Face Audio
│                   │  Spectrogram Transformer → instrument/genre matrix
└────────┬──────────┘
         │ combined feature vector
         ▼
┌───────────────────┐
│  Random Forest    │  Predicts target playlist via .argmax()
│  Classifier       │  Confidence score computed per class
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Spotipy          │  Adds track to predicted Spotify playlist
│                   │  Appends features + label to training_features.csv
│                   │  WebSocket emits result + confidence to UI table
└───────────────────┘
```

Background threading keeps Flask responsive during processing — classification runs in a daemon thread so the server never freezes.

## Prerequisites

The following must be installed and configured before running the app.

**FFmpeg** — audio format conversion. Expected by default at:
```
C:\FFmpeg\bin\ffmpeg.exe
```

**Node.js v20+** — required to solve YouTube JavaScript cryptography challenges used by yt-dlp. Must be added to system PATH.

**Mozilla Firefox** — must be installed and actively logged into YouTube. yt-dlp borrows the live browser session to bypass bot protection.

**Spotify Developer Account** — create an app at [developer.spotify.com](https://developer.spotify.com) to obtain API credentials.

## Environment Variables

Copy `.env.example` to `.env` and fill in your values. The following are read by `src/config.py`:

```bash
SPOTIPY_CLIENT_ID=        # Your Spotify app Client ID
SPOTIPY_CLIENT_SECRET=    # Your Spotify app Client Secret
SPOTIPY_REDIRECT_URI=     # Whitelisted redirect URI
                          # e.g. http://127.0.0.1:5000/
```

The redirect URI must also be added to the allowlist in your Spotify Developer Dashboard under app settings.

## Offline: Training the Model

To retrain the classifier from scratch or expand the training dataset:

```bash
# Step 1 — gather audio features for labeled tracks
python src/gather_training_data.py

# Step 2 — train and evaluate the Random Forest model
python src/run_training.py
```

Output is saved to `models/song_classifier.joblib` and replaces the existing model.

## Tech Stack

**Backend & Web Server**
- **Python 3.x**: Core application logic
- **Flask + Flask-SocketIO**: Async web server and WebSocket management for real-time UI updates

**Machine Learning & Data Science**
- **scikit-learn**: Random Forest Classifier pipeline (scaling, imputation, class balancing)
- **pandas**: Local CSV feature memory and dataframe handling during inference
- **joblib**: Model serialization and loading

**Audio Processing & Feature Extraction**
- **librosa**: MFCCs, spectral centroid, zero-crossing rate, tempo, RMS
- **Hugging Face `transformers` + PyTorch**: `MIT/ast-finetuned-audioset-10-10-0.4593` Audio Spectrogram Transformer — outputs instrument/genre probability matrix
- **FFmpeg**: Audio format conversion, video stream stripping to raw MP3

**APIs & Data Acquisition**
- **Spotipy**: Spotify Web API wrapper — OAuth authentication, playlist reading and writing
- **yt-dlp**: YouTube audio acquisition via live Firefox session cookies and Node.js n-sig challenge solving

**Frontend**
- **HTML5 / CSS3 / Vanilla JS**: Glassmorphism UI with CSS animations and Google Fonts (*Cormorant Garamond* & *DM Mono*)
- **Socket.IO Client**: Real-time processing events streamed to UI without page reloads
