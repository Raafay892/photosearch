# Photo Search (Face Match)

This app lets a user:
- Upload a photo **or** capture a live camera snapshot
- Search for matching face photos inside the local `photos/` folder

## Requirements

- Python 3.10+
- Linux build tools for `face-recognition` (dlib dependency)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open: `http://127.0.0.1:5000`

## Usage

1. Put your image collection inside `photos/`
2. Open the web app
3. Upload an image or use **Start Camera** + **Capture Snapshot**
4. Click **Search Matching Photos**

## Notes

- The app matches the first detected face from your input image.
- It scans all image files in `photos/` recursively (`jpg`, `jpeg`, `png`, `bmp`, `webp`).
- If no face is detected in input image, it shows an error.
- Face encodings are cached in `.face_index.pkl` so repeated searches are much faster.
- If you add or edit photos, the cache updates automatically for changed files.
