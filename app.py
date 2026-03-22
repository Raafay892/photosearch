import base64
import io
import importlib
import os
import pickle
import time
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
from flask import Flask, abort, render_template, request, send_file
from werkzeug.exceptions import RequestEntityTooLarge
from PIL import Image, ImageOps

warnings.filterwarnings(
    "ignore",
    message=r"pkg_resources is deprecated as an API.*",
    category=UserWarning,
)

BASE_DIR = Path(__file__).resolve().parent
PHOTOS_DIR = BASE_DIR / "photos"
CACHE_FILE = BASE_DIR / ".face_index.pkl"
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
INDEX_REFRESH_SECONDS = 60
QUERY_MAX_SIZE = (960, 960)
INDEX_MAX_SIZE = (1280, 1280)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
app.config["MAX_FORM_MEMORY_SIZE"] = 16 * 1024 * 1024
PHOTO_INDEX: dict[str, dict] = {}
INDEX_LOADED = False
LAST_INDEX_SCAN_TS = 0.0
INDEX_WARMUP_STARTED = False
ENCODING_PATHS: list[str] = []
ENCODINGS_MATRIX = np.empty((0, 128), dtype=np.float64)


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


def rebuild_runtime_encoding_matrix() -> None:
    global ENCODING_PATHS, ENCODINGS_MATRIX
    encoding_paths: list[str] = []
    encodings: list[list[float]] = []

    for rel_path, metadata in PHOTO_INDEX.items():
        encoding = metadata.get("encoding")
        if encoding is None:
            continue
        encoding_paths.append(rel_path)
        encodings.append(encoding)

    if encodings:
        ENCODINGS_MATRIX = np.array(encodings, dtype=np.float64)
    else:
        ENCODINGS_MATRIX = np.empty((0, 128), dtype=np.float64)
    ENCODING_PATHS = encoding_paths


def build_or_update_photo_index() -> dict[str, dict]:
    global PHOTO_INDEX, INDEX_LOADED, LAST_INDEX_SCAN_TS
    face_recognition = get_face_recognition_module()

    if not INDEX_LOADED:
        PHOTO_INDEX = load_index_cache()
        INDEX_LOADED = True
        rebuild_runtime_encoding_matrix()
        if PHOTO_INDEX:
            LAST_INDEX_SCAN_TS = time.time()

    now = time.time()
    if PHOTO_INDEX and (now - LAST_INDEX_SCAN_TS) < INDEX_REFRESH_SECONDS:
        return PHOTO_INDEX

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
            with Image.open(photo_path) as pil_image:
                oriented = ImageOps.exif_transpose(pil_image).convert("RGB")
                oriented.thumbnail(INDEX_MAX_SIZE)
                image = np.array(oriented)

            face_locations = face_recognition.face_locations(
                image,
                number_of_times_to_upsample=0,
                model="hog",
            )
            if face_locations:
                selected_location = max(
                    face_locations,
                    key=lambda loc: (loc[2] - loc[0]) * (loc[1] - loc[3]),
                )
                photo_encodings = face_recognition.face_encodings(
                    image,
                    known_face_locations=[selected_location],
                    num_jitters=1,
                    model="small",
                )
                first_encoding = photo_encodings[0].tolist() if photo_encodings else None
            else:
                first_encoding = None

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
        rebuild_runtime_encoding_matrix()

    LAST_INDEX_SCAN_TS = time.time()

    return PHOTO_INDEX


def refresh_index_if_needed() -> None:
    if not PHOTO_INDEX:
        return

    now = time.time()
    if (now - LAST_INDEX_SCAN_TS) < INDEX_REFRESH_SECONDS:
        return

    build_or_update_photo_index()


def ensure_index_warmup_started() -> None:
    global INDEX_WARMUP_STARTED
    if INDEX_WARMUP_STARTED:
        return

    INDEX_WARMUP_STARTED = True
    try:
        build_or_update_photo_index()
    except RuntimeError as exc:
        app.logger.warning("Index warmup skipped: %s", exc)
    except Exception as exc:
        app.logger.exception("Unexpected error during index warmup: %s", exc)


def extract_face_encoding_from_bytes(image_bytes: bytes):
    try:
        face_recognition = get_face_recognition_module()

        with Image.open(io.BytesIO(image_bytes)) as pil_image:
            oriented = ImageOps.exif_transpose(pil_image).convert("RGB")
            full_image = np.array(oriented)

            resized = oriented.copy()
            resized.thumbnail(QUERY_MAX_SIZE)
            resized_image = np.array(resized)

        detection_attempts = [
            (resized_image, "hog", "small", 1),
            (full_image, "hog", "small", 1),
            (full_image, "hog", "small", 2),
            (full_image, "hog", "large", 2),
        ]

        for image_data, detector_model, encoding_model, upsample_times in detection_attempts:
            face_locations = face_recognition.face_locations(
                image_data,
                number_of_times_to_upsample=upsample_times,
                model=detector_model,
            )
            if not face_locations:
                continue

            selected_location = max(
                face_locations,
                key=lambda loc: (loc[2] - loc[0]) * (loc[1] - loc[3]),
            )
            encodings = face_recognition.face_encodings(
                image_data,
                known_face_locations=[selected_location],
                num_jitters=1,
                model=encoding_model,
            )
            if encodings:
                return encodings[0]

        mean_brightness = float(np.mean(full_image)) if full_image.size else 0.0
        app.logger.warning(
            "No face detected. image_shape=%s mean_brightness=%.2f bytes=%d",
            tuple(full_image.shape),
            mean_brightness,
            len(image_bytes),
        )
        return None
    except Exception:
        return None


def find_matching_photos(query_encoding, tolerance: float = 0.5) -> list[Path]:
    face_recognition = get_face_recognition_module()
    if not INDEX_LOADED:
        build_or_update_photo_index()

    refresh_index_if_needed()

    if ENCODINGS_MATRIX.shape[0] == 0:
        return []

    distances = face_recognition.face_distance(ENCODINGS_MATRIX, query_encoding)
    ranked_matches = sorted(
        (
            (float(distance), rel_path)
            for distance, rel_path in zip(distances, ENCODING_PATHS)
            if distance <= tolerance
        ),
        key=lambda item: item[0],
    )

    return [PHOTOS_DIR / rel_path for _, rel_path in ranked_matches]


def parse_input_image_bytes() -> bytes | None:
    uploaded_file = request.files.get("photo")
    if uploaded_file and uploaded_file.filename:
        file_bytes = uploaded_file.read()
        if file_bytes:
            return file_bytes

    captured_data_url = request.form.get("captured_photo", "").strip()
    if captured_data_url:
        try:
            if captured_data_url.startswith("data:") and "," in captured_data_url:
                _, base64_data = captured_data_url.split(",", 1)
            else:
                base64_data = captured_data_url

            normalized = "".join(base64_data.split())
            padding = (-len(normalized)) % 4
            if padding:
                normalized += "=" * padding

            try:
                return base64.b64decode(normalized, validate=True)
            except Exception:
                return base64.urlsafe_b64decode(normalized)
        except Exception:
            return None

    return None


@app.route("/", methods=["GET", "POST"])
def index():
    matched_rel_paths: list[str] = []
    error: str | None = None
    photos_count = len(list_photo_files())
    indexed_count = len(PHOTO_INDEX) if INDEX_LOADED else 0

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
        photos_count=photos_count,
        indexed_count=indexed_count,
    )


@app.errorhandler(RequestEntityTooLarge)
def handle_request_too_large(_error):
    photos_count = len(list_photo_files())
    indexed_count = len(PHOTO_INDEX) if INDEX_LOADED else 0
    return render_template(
        "index.html",
        matches=[],
        error="Submitted image is too large. Please try again with a smaller capture or upload.",
        photos_count=photos_count,
        indexed_count=indexed_count,
    ), 413


@app.route("/photos/<path:relative_path>")
def serve_photo(relative_path: str):
    file_path = (PHOTOS_DIR / relative_path).resolve()
    photos_root = PHOTOS_DIR.resolve()

    if not str(file_path).startswith(str(photos_root)):
        abort(403)
    if not file_path.exists() or not file_path.is_file():
        abort(404)

    download_flag = request.args.get("download", "0").lower() in {"1", "true", "yes"}
    return send_file(
        file_path,
        as_attachment=download_flag,
        download_name=file_path.name if download_flag else None,
    )


@app.route("/all-photos")
def all_photos():
    photo_paths = sorted(list_photo_files())
    all_rel_paths = [str(path.relative_to(PHOTOS_DIR)) for path in photo_paths]
    return render_template("all_photos.html", all_photos=all_rel_paths, photos_count=len(all_rel_paths))


if __name__ == "__main__":
    PHOTOS_DIR.mkdir(parents=True, exist_ok=True)
    app.run(debug=True)
