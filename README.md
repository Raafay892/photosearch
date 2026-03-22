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
- For speed, input images are resized before face encoding.

## Hosting Performance Tips

- Use a stable production profile from the same venv: `python -m gunicorn -c gunicorn.conf.py app:app`
- Keep `photos/` on fast SSD storage.
- First request may be slower while index warmup happens; later searches use an in-memory encoding matrix.
- Index refresh is throttled to avoid frequent full-folder rescans.

## Deploy Options

### Render

- Push this repo to GitHub.
- In Render, create a **Web Service** from the repo.
- Render will use `render.yaml` automatically (Docker mode).
- Build uses `Dockerfile`, so native tools (`cmake`, compiler, BLAS/LAPACK) are guaranteed.

Why this works well here:
- Flask supported directly
- Native Python package builds (CMake/dlib) supported
- Easy GitHub auto-deploy

### Railway

- Push this repo to GitHub.
- In Railway, create a new project from the repo.
- Railway will read `nixpacks.toml` and install required native packages.
- `nixpacks.toml` is configured to use `pip` instead of `uv` and runs `python -m gunicorn -c gunicorn.conf.py app:app`.
- If Railway still uses `uv`, deploy using the `Dockerfile` path.

Why this works well here:
- Very simple setup
- Good support for Python apps

### Heroku (classic)

- Push this repo to GitHub.
- Create Heroku app and connect your repo.
- Ensure Python buildpack is enabled.
- Add Heroku Apt buildpack (before Python) to use `Aptfile` for native deps.
- Heroku runs `Procfile` automatically.
- `Procfile` uses `gunicorn.conf.py`, which reads `PORT` in Python (no shell expansion needed).

Files used by Heroku:
- `Procfile`
- `runtime.txt`
- `Aptfile`

## Important for all platforms

- This app expects images in `photos/` at runtime.
- If you need persistent shared storage for uploaded/managed photos, attach a volume/object storage and point the app to it.

## Troubleshooting deploy errors

If you see an error like:

- `Failed to build dlib`
- `CMake is not installed on your system`
- `uv sync ... failed while building dlib`

Root cause:
- `face-recognition` depends on `dlib`, and `dlib` needs native build tools (`cmake`, compiler, BLAS/LAPACK headers).

Fix by platform:

- Render: already handled in `render.yaml` build command (installs `cmake`, `build-essential`, `python3-dev`, `libopenblas-dev`, `liblapack-dev`).
- Railway: `nixpacks.toml` installs same packages via `aptPkgs`.
- Heroku: use Apt buildpack + `Aptfile` before Python buildpack.

If your platform still tries `uv sync` and fails:
- Ensure it is building from this repo root (so `render.yaml` / `nixpacks.toml` / `Aptfile` are detected).
- Re-deploy after clearing build cache.
- Prefer Docker deploy (uses `Dockerfile`) to fully bypass `uv sync` behavior.
