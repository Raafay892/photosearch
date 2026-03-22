import base64
import io
import importlib
import pickle
from pathlib import Path

import numpy as np
from flask import Flask, abort, render_template, request, send_file

BASE_DIR = Path(__file__).resolve().parent
PHOTOS_DIR = BASE_DIR / "photos"
CACHE_FILE = BASE_DIR / ".face_index.pkl"
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

app = Flask(__name__)
PHOTO_INDEX: dict[str, dict] = {}
INDEX_LOADED = False


def get_face_recognition_module():
    try:
        return importlib.import_module("face_recognition")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Dependency missing: install packages with 'pip install -r requirements.txt'"
        ) from exc


def list_photo_files() -> list[Path]:
    if not PHOTOS_DIR.exists():
        return []
    files: list[Path] = []
    for path in PHOTOS_DIR.rglob("*"):
        if path.is_file() and path.suffix.lower() in ALLOWED_EXTENSIONS:
            files.append(path)
    return files


def load_index_cache() -> dict[str, dict]:
    if not CACHE_FILE.exists():
        return {}

    try:
        with CACHE_FILE.open("rb") as file:
            data = pickle.load(file)
        if isinstance(data, dict):
            return data
    except Exception:
        return {}
    return {}


def save_index_cache(index_data: dict[str, dict]) -> None:
    try:
        with CACHE_FILE.open("wb") as file:
            pickle.dump(index_data, file)
    except Exception:
        pass


def build_or_update_photo_index() -> dict[str, dict]:
    global PHOTO_INDEX, INDEX_LOADED
    face_recognition = get_face_recognition_module()

    if not INDEX_LOADED:
        PHOTO_INDEX = load_index_cache()
        INDEX_LOADED = True

    photo_files = list_photo_files()
    current_rel_paths = {str(path.relative_to(PHOTOS_DIR)): path for path in photo_files}
    changed = False

    removed_paths = set(PHOTO_INDEX) - set(current_rel_paths)
    for rel_path in removed_paths:
        PHOTO_INDEX.pop(rel_path, None)
        changed = True

    for rel_path, photo_path in current_rel_paths.items():
        try:
            stat = photo_path.stat()
        except OSError:
            continue

        mtime_ns = stat.st_mtime_ns
        size = stat.st_size
        cached = PHOTO_INDEX.get(rel_path)
        if cached and cached.get("mtime_ns") == mtime_ns and cached.get("size") == size:
            continue

        try:
            image = face_recognition.load_image_file(str(photo_path))
            photo_encodings = face_recognition.face_encodings(image, model="small")
            first_encoding = photo_encodings[0].tolist() if photo_encodings else None
            PHOTO_INDEX[rel_path] = {
                "mtime_ns": mtime_ns,
                "size": size,
                "encoding": first_encoding,
            }
            changed = True
        except Exception:
            PHOTO_INDEX[rel_path] = {
                "mtime_ns": mtime_ns,
                "size": size,
                "encoding": None,
            }
            changed = True

    if changed:
        save_index_cache(PHOTO_INDEX)

    return PHOTO_INDEX


def extract_face_encoding_from_bytes(image_bytes: bytes):
    try:
        face_recognition = get_face_recognition_module()
        image = face_recognition.load_image_file(io.BytesIO(image_bytes))
        encodings = face_recognition.face_encodings(image, model="small")
        if not encodings:
            return None
        return encodings[0]
    except Exception:
        return None


def find_matching_photos(query_encoding, tolerance: float = 0.5) -> list[Path]:
    face_recognition = get_face_recognition_module()
    index_data = build_or_update_photo_index()
    known_encodings: list[list[float]] = []
    relative_paths: list[str] = []

    for rel_path, metadata in index_data.items():
        encoding = metadata.get("encoding")
        if encoding is None:
            continue
        known_encodings.append(encoding)
        relative_paths.append(rel_path)

    if not known_encodings:
        return []

    distances = face_recognition.face_distance(np.array(known_encodings), query_encoding)
    ranked_matches = sorted(
        (
            (float(distance), rel_path)
            for distance, rel_path in zip(distances, relative_paths)
            if distance <= tolerance
        ),
        key=lambda item: item[0],
    )

    return [PHOTOS_DIR / rel_path for _, rel_path in ranked_matches]


def parse_input_image_bytes() -> bytes | None:
    uploaded_file = request.files.get("photo")
    if uploaded_file and uploaded_file.filename:
        return uploaded_file.read()

    captured_data_url = request.form.get("captured_photo", "")
    if not captured_data_url:
        return None

    try:
        _, base64_data = captured_data_url.split(",", 1
        )  # e.g. data:image/png;base64,....
        return base64.b64decode(base64_data)
    except Exception:
        return None


@app.route("/", methods=["GET", "POST"])
def index():
    matched_rel_paths: list[str] = []
    error: str | None = None

    if request.method == "POST":
        image_bytes = parse_input_image_bytes()
        if not image_bytes:
            error = "Please upload a photo or capture one from your camera."
        else:
            try:
                query_encoding = extract_face_encoding_from_bytes(image_bytes)
                if query_encoding is None:
                    error = "No clear face found in the submitted image."
                else:
                    matches = find_matching_photos(query_encoding)
                    matched_rel_paths = [str(path.relative_to(PHOTOS_DIR)) for path in matches]
            except RuntimeError as exc:
                error = str(exc)

    return render_template(
        "index.html",
        matches=matched_rel_paths,
        error=error,
        photos_count=len(list_photo_files()),
    )


@app.route("/photos/<path:relative_path>")
def serve_photo(relative_path: str):
    file_path = (PHOTOS_DIR / relative_path).resolve()
    photos_root = PHOTOS_DIR.resolve()

    if not str(file_path).startswith(str(photos_root)):
        abort(403)
    if not file_path.exists() or not file_path.is_file():
        abort(404)

    return send_file(file_path)


if __name__ == "__main__":
    PHOTOS_DIR.mkdir(parents=True, exist_ok=True)
    app.run(debug=True)
