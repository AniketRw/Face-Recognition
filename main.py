
from fastapi import FastAPI, Form, UploadFile, File
from typing import Optional
#from deepface import DeepFace
from insightface.app import FaceAnalysis
from concurrent.futures import ThreadPoolExecutor
from fastapi.responses import RedirectResponse
import socket
import faiss
import numpy as np
from typing import Optional, List, Union
import os
from fastapi.middleware.cors import CORSMiddleware
import json
import cv2
import configparser
import sys
import pyodbc
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
import uuid
import cv2
import numpy as np
import secrets
import time

#DATA_DIR = "/app/data"

#os.makedirs(DATA_DIR, exist_ok=True)

DATA_DIR="data"
os.makedirs(DATA_DIR,exist_ok=True)

app = FastAPI()

app.mount("/static", StaticFiles(directory="."), name="static")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
DIMENSION = 512
# INDEX_PATH = "face_index.faiss"
# MAPPING_PATH = "user_mapping.json"
#BASE_STORAGE = "/app/data"


print("STEP 20")
face_app = FaceAnalysis(
    name='buffalo_l',
    allowed_modules=['detection', 'recognition', 'landmark_2d_106'],
    providers=["CPUExecutionProvider"]
)

face_app.prepare(
    ctx_id=-1,
    det_size=(320, 320)
)
print("STEP 3")

# --- Model Warm-up Sequence ---
# This prevents the "Cold Start" delay on the first registration

try:
    print("Warming up face recognition model...")
    dummy_img = np.ones((160, 160, 3), dtype=np.uint8) * 200
    dummy_img[40:120, 30:130] = [180, 140, 100]
    dummy_img[55:70, 45:70]   = [40, 30, 20]
    dummy_img[55:70, 90:115]  = [40, 30, 20]
    dummy_img[80:100, 70:90]  = [160, 120, 90]
    noise = np.random.randint(0, 30, (160, 160, 3), dtype=np.uint8)
    dummy_img = cv2.add(dummy_img, noise)
    face_app.get(dummy_img)
    face_app.get(dummy_img)
    print("Model warm-up complete.")
except Exception as e:
        print(f"Warm-up failed (non-critical): {e}")
        print("Model warm-up complete. Ready for instant registration.")

# ------------------------------

MATCH_DISTANCE_THRESHOLD = 0.65
MATCH_MARGIN = 0.06
MIN_USER_MATCHES = 2
MIN_REGISTRATION_PHOTOS = 3
MIN_LOGIN_FRAME_MATCHES = 2




if getattr(sys, "frozen", False):
    BASE_DIR = os.path.dirname(
        sys.executable
    )
else:
    BASE_DIR = os.path.dirname(
        os.path.abspath(__file__)
    )

#BASE_STORAGE = "/app/data"
BASE_STORAGE=os.path.join(BASE_DIR,"data")


os.makedirs(
    BASE_STORAGE,
    exist_ok=True
)
INDEX_PATH = os.path.join(
    BASE_STORAGE,
    "face_index.faiss"
)

MAPPING_PATH = os.path.join(
    BASE_STORAGE,
    "user_mapping.json"
)

CONFIG_PATH = os.path.join(
    BASE_DIR,
    "config.ini"
)

config = configparser.ConfigParser()
config.read(CONFIG_PATH)

print("STEP 100") 

MATCH_DISTANCE_THRESHOLD = float(
    config.get(
        "APP",
        "MATCH_DISTANCE_THRESHOLD",
        fallback="1.00"
    )
)

MATCH_MARGIN = float(
    config.get(
        "APP",
        "MATCH_MARGIN",
        fallback="0.06"
    )
)

MIN_USER_MATCHES = int(
    config.get(
        "APP",
        "MIN_USER_MATCHES",
        fallback="1"
    )
)

MIN_REGISTRATION_PHOTOS = int(
    config.get(
        "APP",
        "MIN_REGISTRATION_PHOTOS",
        fallback="3"
    )
)

MIN_LOGIN_FRAME_MATCHES = int(
    config.get(
        "APP",
        "MIN_LOGIN_FRAME_MATCHES",
        fallback="1"
    )
)

MAX_FACE_ATTEMPTS = int(
    config.get(
        "APP",
        "MAX_FACE_ATTEMPTS",
        fallback="3"
    )
)
failed_attempts = {}
# FAISS in-memory cache
_db_cache: dict = {}

def get_client_db(client_id: str):
    paths        = get_client_paths(client_id)
    index_path   = paths["faiss"]
    mapping_path = paths["mapping"]

    if not os.path.exists(index_path):
        return faiss.IndexFlatL2(DIMENSION), {}

    try:
        mtime = os.path.getmtime(index_path)
    except Exception:
        return faiss.IndexFlatL2(DIMENSION), {}

    cached = _db_cache.get(client_id)
    if cached and cached["mtime"] == mtime:
        print(f"✅ CACHE HIT: {client_id}")
        return cached["index"], cached["mapping"]

    print(f"🔄 CACHE MISS: {client_id}")
    idx = faiss.read_index(index_path)
    with open(mapping_path) as f:
        mapping = json.load(f)

    _db_cache[client_id] = {"index": idx, "mapping": mapping, "mtime": mtime}
    return idx, mapping
challenge_store = {}                    
CHALLENGE_EXPIRY_SECONDS = 30




# if os.path.exists(INDEX_PATH):
#     index = faiss.read_index(INDEX_PATH)
#     print("FAISS LOADED")
# else:
#     index = faiss.IndexFlatL2(DIMENSION)
#     print("NEW FAISS CREATED")

# if os.path.exists(MAPPING_PATH):
#     with open(MAPPING_PATH, "r") as file:
#         user_mapping = json.load(file)
#         print("User Mapping Loaded")
# else:
#     user_mapping = {}
#     print("New User Mapping")
# def get_client_paths(client_id):

#     base_path = os.path.join(
#         BASE_STORAGE,
#         str(client_id)
#     )

#     os.makedirs(
#         base_path,
#         exist_ok=True
#     )

#     return {
#         "faiss": os.path.join(
#             base_path,
#             "face_index.faiss"
#         ),
#         "mapping": os.path.join(
#             base_path,
#             "user_mapping.json"
#         )
#     }

def get_client_paths(client_id):

    print("RAW CLIENT ID:", repr(client_id))

    clean_client_id = str(client_id).strip()

    print("CLEAN CLIENT ID:", repr(clean_client_id))

    base_path = os.path.join(
        BASE_STORAGE,
        clean_client_id
    )

    print("FINAL BASE PATH:", repr(base_path))

    os.makedirs(
        base_path,
        exist_ok=True
    )

    return {
        "faiss": os.path.join(
            base_path,
            "face_index.faiss"
        ),
        "mapping": os.path.join(
            base_path,
            "user_mapping.json"
        )
    }
def read_upload_image(photo: UploadFile):

    image_bytes = photo.file.read()

    np_arr = np.frombuffer(
        image_bytes,
        np.uint8
    )

    image = cv2.imdecode(
        np_arr,
        cv2.IMREAD_COLOR
    )

    if image is None:
        raise ValueError(
            "Invalid image file"
        )
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8,8)
    )

    l = clahe.apply(l)

    lab = cv2.merge((l,a,b))

    image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Reduce image size for CPU optimization only if needed
    height, width = image.shape[:2]
    max_width = 480

    if width > max_width:
        scale = max_width / width
        new_height = int(height * scale)
        image = cv2.resize(image, (max_width, new_height))

    return image




def get_face_vector(image, return_box=False, return_raw=False):

    faces = face_app.get(image)

    if not faces:
        raise ValueError("No face detected")

    face = max(
        faces,
        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
    )

    embedding = np.array(face.embedding, dtype=np.float32).reshape(1, -1)
    norm = np.linalg.norm(embedding)

    if norm == 0:
        raise ValueError("Invalid face embedding")

    normalized_vector = embedding / norm

    x1, y1, x2, y2 = face.bbox.astype(int)

    face_box = {
        "x":      x1 / image.shape[1],
        "y":      y1 / image.shape[0],
        "width":  (x2 - x1) / image.shape[1],
        "height": (y2 - y1) / image.shape[0]
    }

    if return_box and return_raw:
        return normalized_vector, face_box, face    # returns raw face for landmarks
    if return_box:
        return normalized_vector, face_box
    return normalized_vector

# Removed global index and user_mapping for multi-tenant safety
# Each request now loads its own client-specific database

def get_registered_user(vector_id: int, user_mapping: dict):
    user = user_mapping.get(str(vector_id))

    # Supports the old mapping format: {"0": "Aniket"}.
    if isinstance(user, str):
        return {
            "userid": None,
            "username": user,
        }

    return user


def user_key(user):
    return f"{user.get('userid')}::{user.get('username')}"


def is_live_face(image: np.ndarray, face_box: dict) -> tuple[bool, str]:
    """
    Detects if the face is from a live person or a screen/print.
    Returns (is_live: bool, reason: str)
    """
    h, w = image.shape[:2]
    
    # Crop the face region
    x = int(face_box["x"] * w)
    y = int(face_box["y"] * h)
    fw = int(face_box["width"] * w)
    fh = int(face_box["height"] * h)
    face_crop = image[y:y+fh, x:x+fw]
    
    if face_crop.size == 0:
        return False, "Invalid face region"
    
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    
    # --- Check 1: Laplacian Variance (blur detection) ---
    # Real faces have higher texture variance; screens/prints are often smoother
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if lap_var < 50:  # Tune this threshold
        return False, f"Image too blurry or flat (score: {lap_var:.1f})"
    
    # --- Check 2: Moire Pattern Detection (screen artifact) ---
    # Screens have pixel grid patterns detectable via FFT
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = 20 * np.log(np.abs(fshift) + 1)
    
    rows, cols = gray.shape
    center_r, center_c = rows // 2, cols // 2
    mask = np.zeros((rows, cols), np.uint8)
    cv2.circle(mask, (center_c, center_r), 30, 1, -1)  # block DC center
    
    outer_energy = magnitude[mask == 0].mean()
    inner_energy = magnitude[mask == 1].mean()
    
    # Screens show abnormally high periodic energy in outer frequencies
    freq_ratio = outer_energy / (inner_energy + 1e-5)
    if freq_ratio > 0.85:  # Tune this threshold
        return False, f"Screen pattern detected (freq ratio: {freq_ratio:.2f})"
    
    # --- Check 3: Color Channel Uniformity ---
    # Phone screens have very uniform RGB channels; real skin varies
    b, g, r = cv2.split(face_crop)
    channel_stds = [np.std(c) for c in [b, g, r]]
    channel_diff = max(channel_stds) - min(channel_stds)
    if channel_diff < 5:  # Real skin has uneven channel distribution
        return False, f"Uniform color channels (diff: {channel_diff:.1f})"
    
    return True, "Live face"

def find_best_user_match(face_vector, index, user_mapping):
    search_count = min(index.ntotal, 10)
    distances, indices = index.search(face_vector.astype(np.float32), search_count)
    user_distances = {}
    user_data = {}
    nearest_matches = []

    for raw_distance, raw_index in zip(distances[0], indices[0]):
        vector_id = int(raw_index)
        if vector_id < 0:
            continue

        #user = get_registered_user(vector_id, user_mapping)
        user = None

        for data in user_mapping.values():
            if (
                isinstance(data, dict)
                and data.get("faiss_pos") == vector_id
            ):
                user = data
                break
        if not user:
            continue

        distance = float(raw_distance)
        nearest_matches.append({
            "vector_id": vector_id,
            "user": user,
            "distance": distance,
        })
        key = user_key(user)
        user_distances.setdefault(key, []).append(distance)
        user_data[key] = user

    print("NEAREST MATCHES:", nearest_matches[:5])

    candidates = []
    for key, values in user_distances.items():
        close_values = sorted(
            value for value in values
            if value <= MATCH_DISTANCE_THRESHOLD
        )
        if len(close_values) < MIN_USER_MATCHES:
            continue

        score = sum(close_values[:MIN_USER_MATCHES]) / MIN_USER_MATCHES
        candidates.append({
            "key": key,
            "user": user_data[key],
            "score": score,
            "distances": close_values,
        })

    candidates.sort(key=lambda candidate: candidate["score"])
    print("MATCH CANDIDATES:", candidates)

    if not candidates:
        return None, "no_match"

    if (
        len(candidates) > 1
        and candidates[1]["score"] - candidates[0]["score"] < MATCH_MARGIN
    ):
        return None, "ambiguous"

    return candidates[0], "matched"



def save_database(
    index,
    user_mapping,
    index_path,
    mapping_path
):
    try:

        os.makedirs(
            os.path.dirname(index_path),
            exist_ok=True
        )

        # Save FAISS directly
        faiss.write_index(
            index,
            index_path
        )

        # Save mapping directly
        with open(
            mapping_path,
            "w"
        ) as file:

            json.dump(
                user_mapping,
                file,
                indent=2
            )

            file.flush()
            os.fsync(file.fileno())

        print(
            "FAISS SAVED:",
            os.path.exists(index_path)
        )

        print(
            "MAPPING SAVED:",
            os.path.exists(mapping_path)
        )

    except Exception as e:

        print(
            "SAVE DATABASE ERROR:",
            str(e)
        )


def get_db_connection():
    config = configparser.ConfigParser()
    config.read(CONFIG_PATH)

    db_server = config["DATABASE"]["SERVER"]
    db_name = config["DATABASE"]["DATABASE"]
    db_user = config["DATABASE"]["USER"]
    db_pass = config["DATABASE"]["PASSWORD"]
    db_driver = config["DATABASE"]["DRIVER"]

    connection_string = (
        f"DRIVER={{{db_driver}}};"
        f"SERVER={db_server};"
        f"DATABASE={db_name};"
        f"UID={db_user};"
        f"PWD={db_pass};"
        "Encrypt=no;"
        "TrustServerCertificate=yes;"
        "Connection Timeout=30;"
    )

    return pyodbc.connect(connection_string, timeout=30)

@app.get("/db-config")
def db_config():
    return {
        "success": True,
        "db_server": "",
        "db_name": "",
        "db_user": "",
        "db_pass": "",
        "db_driver": ""
    }

@app.get("/get-user/{userid}", include_in_schema=False)
def get_user(userid: int):
    try:
        with get_db_connection() as connection:
            cursor = connection.cursor()
            cursor.execute(
                "SELECT name FROM usermaster WHERE userid = ?",
                userid
            )
            user = cursor.fetchone()

        if user:
            return {
                "success": True,
                "username": str(user[0])
            }

        return {
            "success": False,
            "message": "User ID not found"
        }

    except Exception as e:
        print("GET USER ERROR:", e)
        return {
            "success": False,
            "message": str(e)
        }

# @app.get("/")
# def home():
#     return FileResponse(
#         "login.html",
#         headers={
#             "Cache-Control":
#             "no-cache, no-store, must-revalidate"
#         }
#     )

@app.get("/")
def home():
    return RedirectResponse(
        url="/login.html"
    )
@app.get("/index.html",include_in_schema=False)
def index_page():
    return FileResponse("index.html")


@app.get("/welcome.html",include_in_schema=False)
def welcome_page():
    return FileResponse("welcome.html")


@app.get("/login.html", include_in_schema=False)
def login_page():
    return FileResponse(
        "login.html",
        headers={
            "Cache-Control":
            "no-cache, no-store, must-revalidate"
        }
    )

@app.post("/upload-entity")
def upload_entity(
    clientid: str = Form(...),
    userid: int = Form(...),
    username: str = Form(...),
    photos: list[UploadFile] = File(...),
    vector_id: list[str] = Form(...)   # VB.NET kadun ekek photo sathi SrNo yeईल
):
    registered_images = []
    import time
    start_time = time.time()

    client_id = str(clientid).strip()
    print(f"--- STARTING REGISTRATION FOR {username} (ID: {userid}) ---")
    paths = get_client_paths(client_id)

    index_path = paths["faiss"]
    mapping_path = paths["mapping"]

    # Load local database for this client
    if os.path.exists(index_path):
        current_index = faiss.read_index(index_path)
    else:
        current_index = faiss.IndexFlatL2(DIMENSION)

    if os.path.exists(mapping_path):
        with open(mapping_path, "r") as file:
            current_mapping = json.load(file)
    else:
        current_mapping = {}

    print("LOGIN CLIENT:", client_id)
    print("LOGIN INDEX:", index_path)
    print("INDEX EXISTS:", os.path.exists(index_path))
    print("RECEIVED PHOTOS COUNT:", len(photos))
    print("RECEIVED VECTOR IDS:", vector_id)

    # Safety check: photos ani vector_id chi count match zali pahije
    if len(photos) != len(vector_id):
        return {
            "success": False,
            "message": f"Mismatch: {len(photos)} photos received but {len(vector_id)} vector_id values received.",
            "userid": userid,
            "username": username,
        }

    # ------------------------------------------------------------------
    # Build set of vector_ids ALREADY registered for THIS userid
    # (works for any userid — not hardcoded to a specific user)
    # ------------------------------------------------------------------
    existing_vector_ids_for_user = set()
    for data in current_mapping.values():
        if isinstance(data, dict) and str(data.get("userid")) == str(userid):
            existing_vector_ids_for_user.add(str(data.get("vector_id")))

    print(f"EXISTING VECTOR IDS FOR userid={userid}:", existing_vector_ids_for_user)

    face_vectors = []
    skipped_photos = []
    duplicate_vector_ids = []   # track skipped duplicates for response

    for photo_index, (photo, vid) in enumerate(zip(photos, vector_id), start=1):
        try:
            # ------------------------------------------------------------
            # Skip this photo entirely if vector_id already exists
            # for this userid — no need to even run face detection on it
            # ------------------------------------------------------------
            display_vid = str(vid).strip()
            if display_vid in existing_vector_ids_for_user:
                print(f"SKIPPING duplicate vector_id={display_vid} for userid={userid}")
                duplicate_vector_ids.append(display_vid)
                continue

            step_start = time.time()
            image = read_upload_image(photo)
            face_vector = get_face_vector(image, return_box=False)
            print(f"  Photo {photo_index} (vector_id={vid}) processed in {time.time() - step_start:.3f}s")

            photo.file.seek(0)
            face_vectors.append({
                "vector":     face_vector,
                "filename":   photo.filename,
                "photo_file": photo.file,
                "vector_id":  vid
            })

            # Mark as used so duplicates WITHIN the same upload batch
            # (e.g. same vector_id sent twice in one call) are also skipped
            existing_vector_ids_for_user.add(display_vid)

        except Exception as e:
            skipped_photos.append({
                "photo_number": photo_index,
                "filename": photo.filename,
                "reason": str(e),
            })
            continue

    if skipped_photos:
        return {
            "success": False,
            "message": f"Registration failed. Photo {skipped_photos[0]['photo_number']} issues.",
            "userid": userid,
            "username": username,
            "photos_registered": 0,
            "photos_skipped": len(skipped_photos),
        }

    # ------------------------------------------------------------------
    # If everything sent was already registered (all duplicates),
    # tell the client clearly instead of failing on MIN_REGISTRATION_PHOTOS
    # ------------------------------------------------------------------
    if not face_vectors and duplicate_vector_ids:
        return {
            "success": False,
            "message": "All submitted vector_ids are already registered for this user. Nothing new to add.",
            "userid": userid,
            "username": username,
            "duplicate_vector_ids_skipped": duplicate_vector_ids,
            "photos_registered": 0,
        }

    if len(face_vectors) < MIN_REGISTRATION_PHOTOS:
        return {
            "success": False,
            "message": f"Need at least {MIN_REGISTRATION_PHOTOS} new photos (excluding duplicates).",
            "userid": userid,
            "username": username,
            "duplicate_vector_ids_skipped": duplicate_vector_ids,
        }

    duplicate_found = None
    for item in face_vectors:
        if current_index.ntotal == 0:
            break
        distances, indices = current_index.search(
            item["vector"].astype(np.float32),
            1
        )
        nearest_distance = float(distances[0][0])
        nearest_id = int(indices[0][0])

        if nearest_distance > MATCH_DISTANCE_THRESHOLD:
            continue

        existing_user = None
        for data in current_mapping.values():
            if (
                isinstance(data, dict)
                and data.get("faiss_pos") == nearest_id
            ):
                existing_user = data
                break
        if not existing_user:
            continue

        existing_userid = (
            existing_user.get("userid")
            if isinstance(existing_user, dict)
            else None
        )

        if existing_userid != userid:
            duplicate_found = existing_user
            break

    if duplicate_found:
        existing_name = (
            duplicate_found.get("username")
            if isinstance(duplicate_found, dict)
            else duplicate_found
        )
        return {
            "success": False,
            "message": f"This face is already registered under a different user. Registration blocked.",
            "duplicate_username": existing_name,
            "duplicate_userid": duplicate_found.get("userid") if isinstance(duplicate_found, dict) else None,
        }

    IMAGES_DIR = os.path.join(BASE_STORAGE, client_id, "images")
    os.makedirs(IMAGES_DIR, exist_ok=True)

    for item in face_vectors:

        current_index.add(
            item["vector"].astype(np.float32)
        )

        if current_mapping:
            internal_id = max(map(int, current_mapping.keys())) + 1
        else:
            internal_id = 0

        # Client kadun aलेला vector_id (SrNo) direct vaparला, auto-increment nahi
        display_vector_id = item["vector_id"]

        all_srnos = [
            data.get("srno", 0)
            for data in current_mapping.values()
            if isinstance(data, dict)
        ]

        if all_srnos:
            global_srno = max(all_srnos) + 1
        else:
            global_srno = 1

        image_id = str(uuid.uuid4())

        current_mapping[str(internal_id)] = {
            "userid": userid,
            "vector_id": display_vector_id,
            "srno": global_srno,
            "faiss_pos": current_index.ntotal - 1,
            "username": username,
            "image_id": image_id,
            "filename": item["filename"]
        }

        ext = os.path.splitext(item["filename"])[-1] or ".jpg"
        save_path = os.path.join(IMAGES_DIR, f"{image_id}{ext}")
        with open(save_path, "wb") as f:
            f.write(item["photo_file"].read())
        print(f"IMAGE SAVED: {save_path}")

        registered_images.append({

            "vector_id":  display_vector_id,
            "image_id":   image_id,
            "filename":   item["filename"],
            "image_path": save_path
        })

    print("CURRENT MAPPING:")
    print(json.dumps(current_mapping, indent=2))
    save_database(current_index, current_mapping, index_path, mapping_path)
    _db_cache.pop(client_id, None)

    total_time = time.time() - start_time
    print(f"--- REGISTRATION COMPLETE | TOTAL TIME: {total_time:.3f}s ---")
    print("SAVED INDEX:", index_path)
    print("INDEX EXISTS AFTER SAVE:", os.path.exists(index_path))
    print("TOTAL FACES AFTER SAVE:", current_index.ntotal)
    print("REGISTERED IMAGES:")
    print(json.dumps(registered_images, indent=2))
    if duplicate_vector_ids:
        print("DUPLICATE VECTOR IDS SKIPPED:", duplicate_vector_ids)

    response_data = {
        "success": True,
        "message": f"Face registered successfully for {username}.",
        "userid": userid,
        "username": username,
        "photos_registered": len(face_vectors),
        "duplicate_vector_ids_skipped": duplicate_vector_ids,
        "images": registered_images
    }

    print("FINAL RESPONSE:")
    print(json.dumps(response_data, indent=2))

    return response_data

# @app.post("/upload-entity")
# def upload_entity(
#     clientid: str = Form(...),

#     userid: int = Form(...),
#     username: str = Form(...),
#     photos: list[UploadFile] = File(...)
# ):
#     registered_images = []
#     import time
#     start_time = time.time()
    
#     #client_id = str(clientid)
#     client_id = str(clientid).strip()
#     print(f"--- STARTING REGISTRATION FOR {username} (ID: {userid}) ---")
#     paths = get_client_paths(client_id)

#     index_path = paths["faiss"]
#     mapping_path = paths["mapping"]



#     # Load local database for this client
#     if os.path.exists(index_path):
#         current_index = faiss.read_index(index_path)
#     else:
#         current_index = faiss.IndexFlatL2(DIMENSION)

#     if os.path.exists(mapping_path):
#         with open(mapping_path, "r") as file:
#             current_mapping = json.load(file)
#     else:
#         current_mapping = {}

#     print("LOGIN CLIENT:", client_id)
#     print("LOGIN INDEX:", index_path)
#     print("INDEX EXISTS:", os.path.exists(index_path))
#     #print("TOTAL FACES:", current_index.ntotal)
#     face_vectors = []
#     skipped_photos = []

#     for photo_index, photo in enumerate(photos, start=1):
#         try:
#             step_start = time.time()
#             image = read_upload_image(photo)
#             face_vector = get_face_vector(image, return_box=False)
#             print(f"  Photo {photo_index} processed in {time.time() - step_start:.3f}s")
#             #face_vectors.append(face_vector)
#             # face_vectors.append({
#             #     "vector": face_vector,
#             #     "filename": photo.filename
#             # })
#             photo.file.seek(0)
#             face_vectors.append({
#                 "vector":     face_vector,
#                 "filename":   photo.filename,
#                 "photo_file": photo.file
#             })
#         except Exception as e:
#             skipped_photos.append({
#                 "photo_number": photo_index,
#                 "filename": photo.filename,
#                 "reason": str(e),
#             })
#             continue

#     if skipped_photos:
#         return {
#             "success": False,
#             "message": f"Registration failed. Photo {skipped_photos[0]['photo_number']} issues.",
#             "userid": userid,
#             "username": username,
#             "photos_registered": 0,
#             "photos_skipped": len(skipped_photos),
#         }

#     if len(face_vectors) < MIN_REGISTRATION_PHOTOS:
#         return {
#             "success": False,
#             "message": f"Need at least {MIN_REGISTRATION_PHOTOS} photos.",
#             "userid": userid,
#             "username": username,
#         }
    
#     duplicate_found = None
#     # for face_vector in face_vectors:
#     #     if current_index.ntotal == 0:
#     #         break

#     #     distances, indices = current_index.search(
#     #         face_vector.astype(np.float32), 1
#     #     )
#     for item in face_vectors:
#         if current_index.ntotal == 0:
#             break
#         distances, indices = current_index.search(
#             item["vector"].astype(np.float32),
#             1
#         )
#         nearest_distance = float(distances[0][0])
#         nearest_id = int(indices[0][0])

#         if nearest_distance > MATCH_DISTANCE_THRESHOLD:
#             continue

#         #existing_user = current_mapping.get(str(nearest_id))
#         existing_user = None

#         for data in current_mapping.values():
#             if (
#                 isinstance(data, dict)
#                 and data.get("faiss_pos") == nearest_id
#             ):
#                 existing_user = data
#                 break
#         if not existing_user:
#             continue

#         existing_userid = (
#             existing_user.get("userid")
#             if isinstance(existing_user, dict)
#             else None
#         )

#         if existing_userid != userid:
#             duplicate_found = existing_user
#             break

#     if duplicate_found:
#         existing_name = (
#             duplicate_found.get("username")
#             if isinstance(duplicate_found, dict)
#             else duplicate_found
#         )
#         return {
#             "success": False,
#             "message": f"This face is already registered under a different user. Registration blocked.",
#             "duplicate_username": existing_name,
#             "duplicate_userid": duplicate_found.get("userid") if isinstance(duplicate_found, dict) else None,
#         }

#     # for face_vector in face_vectors:
#     #     current_index.add(face_vector.astype(np.float32))
#     #     vector_id = current_index.ntotal - 1
#     #     image_id = str(uuid.uuid4())

#     #     current_mapping[str(vector_id)] = {
#     #         "userid": userid,
#     #         "username": username,
#     #         "image_id": image_id,
#     #         "filename": photo.filename
#     #     }
#     # for item in face_vectors:

#     #     current_index.add(
#     #         item["vector"].astype(np.float32)
#     #     )

#     #     vector_id = current_index.ntotal - 1

#     #     image_id = str(uuid.uuid4())

#     #     current_mapping[str(vector_id)] = {
#     #         "userid": userid,
#     #         "username": username,
#     #         "image_id": image_id,
#     #         "filename": item["filename"]
#     #     }

#     #     registered_images.append({
#     #         "image_id": image_id,
#     #         "filename": item["filename"]
#     #     })
#     # IMAGES_DIR = os.path.join(BASE_STORAGE, client_id, "images")
#     # os.makedirs(IMAGES_DIR, exist_ok=True)

#     # for item in face_vectors:

#     #     # current_index.add(
#     #     #     item["vector"].astype(np.float32)
#     #     # )

#     #     # vector_id = current_index.ntotal

#     #     # image_id = str(uuid.uuid4())
#     #     # Internal FAISS position
#     #     current_index.add(
#     #         item["vector"].astype(np.float32)
#     #     )

#     #     #internal_id = current_index.ntotal - 1
#     #     if current_mapping:
#     #         internal_id = max(map(int, current_mapping.keys())) + 1
#     #     else:
#     #         internal_id = 0

#     #     user_vector_ids = [
#     #         data.get("vector_id", 0)
#     #         for data in current_mapping.values()
#     #             if isinstance(data, dict)
#     #             and str(data.get("userid")) == str(userid)
#     #     ]

#     #     if user_vector_ids:
#     #         display_vector_id = max(user_vector_ids) + 1
#     #     else:
#     #         display_vector_id = 1

#     #     image_id = str(uuid.uuid4())

#     #     # current_mapping[str(vector_id)] = {
#     #     #     "userid": userid,
#     #     #     "username": username,
#     #     #     "image_id": image_id,
#     #     #     "filename": item["filename"]
#     #     # }
#     #     current_mapping[str(internal_id)] = {
#     #         "userid": userid,
#     #         "vector_id": display_vector_id,
#     #         "faiss_pos": current_index.ntotal - 1,
#     #         "username": username,
#     #         "image_id": image_id,
#     #         "filename": item["filename"]
#     #     }

#     #     ext = os.path.splitext(item["filename"])[-1] or ".jpg"
#     #     save_path = os.path.join(IMAGES_DIR, f"{image_id}{ext}")
#     #     with open(save_path, "wb") as f:
#     #         f.write(item["photo_file"].read())
#     #     print(f"IMAGE SAVED: {save_path}")

#     #     registered_images.append({
#     #         "vector_id": display_vector_id,
#     #         "image_id":  image_id,
#     #         "filename":  item["filename"],
#     #         "image_path": save_path
#     #     })
# IMAGES_DIR = os.path.join(BASE_STORAGE, client_id, "images")
# os.makedirs(IMAGES_DIR, exist_ok=True)

# for item in face_vectors:

#     current_index.add(
#         item["vector"].astype(np.float32)
#     )

#     if current_mapping:
#         internal_id = max(map(int, current_mapping.keys())) + 1
#     else:
#         internal_id = 0

#     user_vector_ids = [
#         data.get("vector_id", 0)
#         for data in current_mapping.values()
#             if isinstance(data, dict)
#             and str(data.get("userid")) == str(userid)
#     ]

#     if user_vector_ids:
#         display_vector_id = max(user_vector_ids) + 1
#     else:
#         display_vector_id = 1

#     image_id = str(uuid.uuid4())

#     current_mapping[str(internal_id)] = {
#         "userid": userid,
#         "vector_id": display_vector_id,
#         "faiss_pos": current_index.ntotal - 1,
#         "username": username,
#         "image_id": image_id,
#         "filename": item["filename"]
#     }

#     ext = os.path.splitext(item["filename"])[-1] or ".jpg"
#     save_path = os.path.join(IMAGES_DIR, f"{image_id}{ext}")
#     with open(save_path, "wb") as f:
#         f.write(item["photo_file"].read())
#     print(f"IMAGE SAVED: {save_path}")

#     registered_images.append({
#         "srno":       display_vector_id,
#         "vector_id":  display_vector_id,
#         "image_id":   image_id,
#         "filename":   item["filename"],
#         "image_path": save_path
#     })

#     print("CURRENT MAPPING:")
#     print(json.dumps(current_mapping, indent=2))
#     save_database(current_index, current_mapping, index_path, mapping_path)
#     _db_cache.pop(client_id, None) 
    
#     total_time = time.time() - start_time
#     print(f"--- REGISTRATION COMPLETE | TOTAL TIME: {total_time:.3f}s ---")
#     print("SAVED INDEX:", index_path)
#     print("INDEX EXISTS AFTER SAVE:", os.path.exists(index_path))
#     print("TOTAL FACES AFTER SAVE:", current_index.ntotal)
#     # return {
#     #     "success": True,
#     #     "message": f"Face registered successfully for {username}.",
#     #     "userid": userid,
#     #     "username": username,
#     #     "photos_registered": len(face_vectors),
#     # }
#     print("REGISTERED IMAGES:")
#     print(json.dumps(registered_images, indent=2))
#     return {
#         "success": True,
#         "message": f"Face registered successfully for {username}.",
#         "userid": userid,
#         "username": username,
#         "photos_registered": len(face_vectors),
#         "images": registered_images
#     }
    
from typing import Optional, List

def check_motion_across_frames(face_boxes: list) -> bool:
    if len(face_boxes) < 2:
        return True  # single frame — can't check motion, allow through

    movements = []
    for i in range(1, len(face_boxes)):
        dx = abs(face_boxes[i]["x"] - face_boxes[i-1]["x"])
        dy = abs(face_boxes[i]["y"] - face_boxes[i-1]["y"])
        movements.append(dx + dy)

    avg_movement = sum(movements) / len(movements)
    print(f"MOTION CHECK: avg_movement={avg_movement:.4f}")

    return avg_movement >= 0.003
def verify_blink(raw_faces: list) -> tuple[bool, str]:
    ear_values = []
    for face in raw_faces:
        lm = getattr(face, 'landmark_2d_106', None)
        if lm is None:
            continue
        left_v  = abs(lm[35][1] - lm[40][1])
        left_h  = abs(lm[33][0] - lm[39][0])
        right_v = abs(lm[89][1] - lm[94][1])
        right_h = abs(lm[87][0] - lm[93][0])
        left_ear  = left_v  / (left_h  + 1e-5)
        right_ear = right_v / (right_h + 1e-5)
        ear_values.append((left_ear + right_ear) / 2.0)

    if len(ear_values) < 3:
        return False, "Could not read eye landmarks"

    variation = max(ear_values) - min(ear_values)
    print(f"BLINK EAR variation: {variation:.4f}")
    return (variation >= 0.08), f"Blink variation={variation:.3f}"

def verify_head_turn(raw_faces: list, direction: str) -> tuple[bool, str]:
    nose_offsets = []
    for face in raw_faces:
        lm = getattr(face, 'landmark_2d_106', None)
        if lm is None:
            continue
        nose_x       = lm[86][0]
        eye_center_x = (lm[36][0] + lm[90][0]) / 2
        nose_offsets.append(nose_x - eye_center_x)

    if len(nose_offsets) < 3:
        return False, "Could not read nose/eye landmarks"

    initial_offset = nose_offsets[0]
    max_change     = max(nose_offsets) - min(nose_offsets)
    TURN_THRESHOLD = 8.0

    if direction == "left":
        moved = min(nose_offsets) < initial_offset - TURN_THRESHOLD
    else:
        moved = max(nose_offsets) > initial_offset + TURN_THRESHOLD

    print(f"HEAD TURN {direction}: max_change={max_change:.2f}")
    return moved, f"Head turn {direction}: change={max_change:.2f}"

def verify_any_movement(raw_faces: list) -> tuple[bool, str]:
    """
    Detects ANY natural head movement across frames.
    No specific direction required — just checks that
    the face wasn't completely static (like a photo/video loop).
    """
    nose_x_values = []
    nose_y_values = []

    for face in raw_faces:
        lm = getattr(face, 'landmark_2d_106', None)
        if lm is None:
            continue
        nose_x_values.append(lm[86][0])
        nose_y_values.append(lm[86][1])

    if len(nose_x_values) < 3:
        return False, "Could not read landmarks"

    x_variation = max(nose_x_values) - min(nose_x_values)
    y_variation = max(nose_y_values) - min(nose_y_values)
    total_variation = x_variation + y_variation

    print(f"MOVEMENT variation x={x_variation:.2f} y={y_variation:.2f} total={total_variation:.2f}")

    return (total_variation >= 3.0), f"Movement variation={total_variation:.2f}px"


def verify_challenge(raw_faces: list, challenge: str) -> tuple[bool, str]:
    if not raw_faces or len(raw_faces) < 3:
        return False, "Not enough frames"

    # Silent mode — just check natural blink OR natural head movement
    blink_ok, blink_reason     = verify_blink(raw_faces)
    turn_ok,  turn_reason      = verify_any_movement(raw_faces)

    print(f"SILENT BLINK: {blink_reason}")
    print(f"SILENT MOVEMENT: {turn_reason}")

    # Pass if EITHER blink OR movement detected
    if blink_ok and turn_ok:
        return True, "Natural liveness detected"

    return False, "No natural movement detected"

@app.get("/auth-challenge")
def get_auth_challenge():
    # No random challenge — just issue a token for silent liveness
    token = secrets.token_hex(16)

    challenge_store[token] = {
        "challenge":  "silent",
        "expires_at": time.time() + CHALLENGE_EXPIRY_SECONDS
    }

    print(f"SILENT LIVENESS TOKEN ISSUED: {token}")

    return {
    "success": True,
    "token": token,
    "message": "Look at camera"
}


# @app.post("/authenticate")
# def authenticate(
#     clientid: str = Form(...),
#     token:    str = Form(...),              # NEW
#     photo:    Optional[UploadFile] = File(None),
#     photos:   List[UploadFile] = File(...)
# ):
#     print("\n========== AUTH START ==========")
#     client_id = str(clientid).strip()

#     # Validate challenge token first
#     challenge_data = challenge_store.pop(token, None)   # one-time use

#     if not challenge_data:
#         return {
#             "success": False,
#             "matched": False,
#             "message": "Invalid or expired challenge. Please try again."
#         }

#     if time.time() > challenge_data["expires_at"]:
#         return {
#             "success": False,
#             "matched": False,
#             "message": "Challenge expired. Please try again."
#         }

#     challenge = challenge_data["challenge"]
#     print(f"CHALLENGE TO VERIFY: {challenge}")

#     paths        = get_client_paths(client_id)
#     index_path   = paths["faiss"]
#     mapping_path = paths["mapping"]

#     if os.path.exists(index_path):
#         current_index = faiss.read_index(index_path)
#     else:
#         current_index = faiss.IndexFlatL2(DIMENSION)

#     if os.path.exists(mapping_path):
#         with open(mapping_path, "r") as file:
#             current_mapping = json.load(file)
#     else:
#         current_mapping = {}

#     try:
#         if current_index.ntotal == 0:
#             return {"success": False, "matched": False, "message": "No registered faces found"}

#         login_photos = photos if photos else ([photo] if photo else [])
#         if not login_photos:
#             return {"success": False, "matched": False, "message": "No login photo received"}

#         # --- Collect all frames first ---
#         face_detected_count    = 0
#         liveness_fail_count    = 0
#         face_box               = None
#         face_boxes_collected   = []
#         raw_faces_collected    = []
#         face_vectors_collected = []

#         for login_photo in login_photos:

#             # STEP 1: Read image
#             try:
#                 image = read_upload_image(login_photo)
#             except Exception as e:
#                 print(f"IMAGE READ ERROR: {e}")
#                 continue

#             # STEP 2: Detect face
#             try:
#                 face_vector, face_box, raw_face = get_face_vector(
#                     image, return_box=True, return_raw=True
#                 )
#                 face_detected_count += 1
#                 face_boxes_collected.append(face_box)
#                 raw_faces_collected.append(raw_face)
#                 face_vectors_collected.append(face_vector)
#             except Exception as e:
#                 print(f"FACE DETECT ERROR: {e}")
#                 continue

#             # STEP 3: Liveness — first frame only (saves time)
#             if face_detected_count == 1:
#                 is_live, liveness_reason = is_live_face(image, face_box)
#                 if not is_live:
#                     print(f"LIVENESS FAIL: {liveness_reason}")
#                     liveness_fail_count += 1

#         # --- All frames collected, now decide ---

#         if face_detected_count == 0:
#             return {
#                 "success": False, "matched": False,
#                 "show_password_login": False,
#                 "message": "Face Not Detected"
#             }

#         if liveness_fail_count > 0:
#             return {
#                 "success": False, "matched": False,
#                 "show_password_login": False,
#                 "message": "Liveness check failed. Please use a live camera."
#             }

#         # STEP 4: Motion check
#         if not check_motion_across_frames(face_boxes_collected):
#             return {
#                 "success": False, "matched": False,
#                 "show_password_login": False,
#                 "message": "Liveness check failed. Please move slightly and try again."
#             }

#         # STEP 5: Challenge check (blink + movement)
#         challenge_ok, challenge_reason = verify_challenge(
#             raw_faces_collected, challenge
#         )
#         print(f"CHALLENGE RESULT: {challenge_reason}")

#         if not challenge_ok:
#             return {
#                 "success": False, "matched": False,
#                 "show_password_login": False,
#                 "message": "Liveness check failed. Please face the camera and try again."
#             }

#         # STEP 6: Match on best single frame (middle frame)
#         best_vector = face_vectors_collected[len(face_vectors_collected) // 2]
#         match, match_status = find_best_user_match(
#             best_vector, current_index, current_mapping
#         )

#         if match_status == "matched":
#             user     = match["user"]
#             username = user["username"]
#             failed_attempts["camera_login"] = 0

#             return {
#                 "success":  True,
#                 "matched":  True,
#                 "message":  f"Welcome {username}",
#                 "userid":   user.get("userid"),
#                 "username": username,
#                 "face_box": face_box
#             }

#         # No match
#         failed_attempts["camera_login"] = failed_attempts.get("camera_login", 0) + 1
#         attempts  = failed_attempts["camera_login"]
#         remaining = MAX_FACE_ATTEMPTS - attempts

#         if attempts >= MAX_FACE_ATTEMPTS:
#             return {
#                 "success": False, "matched": False,
#                 "show_password_login": True,
#                 "message": "Face login failed 3 times. Use username and password."
#             }

#         return {
#             "success": False, "matched": False,
#             "show_password_login": False,
#             "message": f"Face not matched. {remaining} attempts left.",
#             "face_box": face_box
#         }

#     except Exception as e:
#         print("AUTH ERROR:", e)
#         return {
#             "success": False, "matched": False,
#             "show_password_login": False,
#             "message": "Face Not Detected"
#         }
#         if liveness_fail_count == face_detected_count:
#             return {
#                 "success": False, "matched": False,
#                 "show_password_login": False,
#                 "message": "Liveness check failed. Please use a live camera."
#             }

#         failed_attempts["camera_login"] = failed_attempts.get("camera_login", 0) + 1
#         attempts  = failed_attempts["camera_login"]
#         remaining = MAX_FACE_ATTEMPTS - attempts

#         if attempts >= MAX_FACE_ATTEMPTS:
#             return {
#                 "success": False, "matched": False,
#                 "show_password_login": True,
#                 "message": "Face login failed 3 times. Use username and password."
#             }

#         return {
#             "success": False, "matched": False,
#             "show_password_login": False,
#             "message": f"Face not matched. {remaining} attempts left.",
#             "face_box": face_box
#         }

#     except Exception as e:
#         print("AUTH ERROR:", e)
#         return {
#             "success": False, "matched": False,
#             "show_password_login": False,
#             "message": "Face Not Detected"
#         }

# ============================================================
# FAST AUTHENTICATE — blink detected = instant match
# ============================================================
# @app.post("/authenticate")
# def authenticate(
#     clientid: str = Form(...),
#     photo1: UploadFile = File(...),
#     photo2: UploadFile = File(...),
#     photo3: UploadFile = File(...)
# ):
#     t0 = time.time()
#     client_id = str(clientid).strip()

#     paths = get_client_paths(client_id)
#     index_path   = paths["faiss"]
#     mapping_path = paths["mapping"]

#     if os.path.exists(index_path):
#         current_index = faiss.read_index(index_path)
#     else:
#         return {"success": False, "matched": False, "message": "No face database found"}

#     if os.path.exists(mapping_path):
#         with open(mapping_path, "r") as f:
#             current_mapping = json.load(f)
#     else:
#         return {"success": False, "matched": False, "message": "No face database found"}

#     if current_index.ntotal == 0:
#         return {"success": False, "matched": False, "message": "No registered faces found"}

#     # --- 3 photos process --- (image resize आधीच होतं, fast)
#     raw_faces   = []
#     vectors     = []
#     boxes       = []
#     ear_values  = []

#     for photo in [photo1, photo2, photo3]:
#         try:
#             image = read_upload_image(photo)
#             vec, box, raw = get_face_vector(image, return_box=True, return_raw=True)
#             vectors.append(vec)
#             boxes.append(box)
#             raw_faces.append(raw)

#             lm = getattr(raw, 'landmark_2d_106', None)
#             if lm is not None:
#                 lv = abs(lm[35][1] - lm[40][1])
#                 lh = abs(lm[33][0] - lm[39][0])
#                 rv = abs(lm[89][1] - lm[94][1])
#                 rh = abs(lm[87][0] - lm[93][0])
#                 ear = ((lv / (lh + 1e-5)) + (rv / (rh + 1e-5))) / 2.0
#                 ear_values.append(ear)
#         except Exception as e:
#             print(f"SKIP: {e}")
#             continue

#     print(f"FACE DETECT: {time.time() - t0:.3f}s")

#     if not vectors:
#         return {"success": False, "matched": False,
#                 "show_password_login": False,
#                 "message": "Face Not Detected"}

#     # --- Blink check — FAST (no loop, direct index) ---
#     # ear_values = []
#     # for face in raw_faces:
#     #     lm = getattr(face, 'landmark_2d_106', None)
#     #     if lm is None:
#     #         continue
#     #     lv = abs(lm[35][1] - lm[40][1])
#     #     lh = abs(lm[33][0] - lm[39][0])
#     #     rv = abs(lm[89][1] - lm[94][1])
#     #     rh = abs(lm[87][0] - lm[93][0])
#     #     ear_values.append(((lv / (lh + 1e-5)) + (rv / (rh + 1e-5))) / 2.0)

#     print(f"EAR VALUES: {[f'{e:.3f}' for e in ear_values]}")

#     blink_ok = False
#     if len(ear_values) >= 2:
#         variation = max(ear_values) - min(ear_values)
#         min_ear   = min(ear_values)
#         print(f"EAR VARIATION: {variation:.4f}, MIN EAR: {min_ear:.4f}")
#         blink_ok = (variation >= 0.04) or (min_ear < 0.18)

#     print(f"BLINK CHECK: {time.time() - t0:.3f}s")

#     if not blink_ok:
#         failed_attempts["camera_login"] = failed_attempts.get("camera_login", 0) + 1
#         attempts  = failed_attempts["camera_login"]
#         remaining = MAX_FACE_ATTEMPTS - attempts

#         # if attempts >= MAX_FACE_ATTEMPTS:
#         #     failed_attempts["camera_login"] = 0
#         #     return {"success": False, "matched": False,
#         #             "show_password_login": True,
#         #             "message": "Face login failed 3 times. Use username and password."}

#         # return {"success": False, "matched": False,
#         #         "show_password_login": False,
#         #         "message": f"Blink not detected. {remaining} attempts left."}

#         return {
#         "success": False,
#         "matched": False,
#         "show_password_login": False,
#         "message": f"Blink not detected  Just Blink ."
#     }

#     # --- Blink OK → ONLY middle frame match (skip others) ---
#     best_vector  = vectors[len(vectors) // 2]
#     best_box     = boxes[len(boxes) // 2]

#     match, match_status = find_best_user_match(
#         best_vector, current_index, current_mapping
#     )

#     print(f"TOTAL AUTH: {time.time() - t0:.3f}s")

#     if match_status == "matched":
#         user = match["user"]
#         failed_attempts["camera_login"] = 0
#         return {
#             "success":  True, "matched": True,
#             "message":  f"Welcome {user['username']}",
#             "userid":   user.get("userid"),
#             "username": user["username"],
#             "face_box": best_box
#         }

#     failed_attempts["camera_login"] = failed_attempts.get("camera_login", 0) + 1
#     attempts  = failed_attempts["camera_login"]
#     remaining = MAX_FACE_ATTEMPTS - attempts

#     if attempts >= MAX_FACE_ATTEMPTS:
#         failed_attempts["camera_login"] = 0
#         return {"success": False, "matched": False,
#                 "show_password_login": True,
#                 "message": "Face login failed 3 times. Use username and password."}

#     return {"success": False, "matched": False,
#             "show_password_login": False,
#             "message": f"Face not matched. {remaining} attempts left.",
#             "face_box": best_box}
#     failed_attempts["camera_login"] = failed_attempts.get("camera_login", 0) + 1
#     attempts  = failed_attempts["camera_login"]
#     remaining = MAX_FACE_ATTEMPTS - attempts
#     face_box  = face_boxes_collected[len(face_boxes_collected) // 2] \
#                 if face_boxes_collected else None

#     if attempts >= MAX_FACE_ATTEMPTS:
#         return {"success": False, "matched": False,
#                 "show_password_login": True,
#                 "message": "Face login failed 3 times. Use username and password."}

#     return {"success": False, "matched": False,
#             "show_password_login": False,
#             "message": f"Face not matched. {remaining} attempts left.",
#             "face_box": face_box}

@app.post("/authenticate")
def authenticate(
    clientid: str = Form(...),
    photo1: UploadFile = File(...),
    photo2: UploadFile = File(...),
    photo3: UploadFile = File(...)
):
    import time
    T = {}  # timing dict

    t0 = time.time()
    client_id = str(clientid).strip()

    # --- FAISS Load ---
    t1 = time.time()
    paths = get_client_paths(client_id)
    index_path   = paths["faiss"]
    mapping_path = paths["mapping"]

    # if os.path.exists(index_path):
    #     current_index = faiss.read_index(index_path)
    # else:
    #     return {"success": False, "matched": False, "message": "No face database found"}

    # if os.path.exists(mapping_path):
    #     with open(mapping_path, "r") as f:
    #         current_mapping = json.load(f)
    # else:
    #     return {"success": False, "matched": False, "message": "No face database found"}

    # T["faiss_load"] = round((time.time() - t1) * 1000)
    # print(f"⏱ FAISS Load:     {T['faiss_load']}ms")
    current_index, current_mapping = get_client_db(client_id)
    T["faiss_load"] = round((time.time() - t1) * 1000)
    print(f"⏱ FAISS Load: {T['faiss_load']}ms")

    if current_index.ntotal == 0:
        return {"success": False, "matched": False, "message": "No registered faces found"}

    # --- Image Read ---
    t2 = time.time()
    raw_faces  = []
    vectors    = []
    boxes      = []
    ear_values = []

    # for i, photo in enumerate([photo1, photo2, photo3], 1):
    #     tp = time.time()
    #     try:
    #         image = read_upload_image(photo)
    #         T[f"img_read_{i}"] = round((time.time() - tp) * 1000)
    #         print(f"⏱ Image Read {i}:   {T[f'img_read_{i}']}ms")

    #         # --- Face Detection ---
    #         td = time.time()
    #         vec, box, raw = get_face_vector(image, return_box=True, return_raw=True)
    #         T[f"face_detect_{i}"] = round((time.time() - td) * 1000)
    #         print(f"⏱ Face Detect {i}:  {T[f'face_detect_{i}']}ms")

    #         vectors.append(vec)
    #         boxes.append(box)
    #         raw_faces.append(raw)

    #         lm = getattr(raw, 'landmark_2d_106', None)
    #         if lm is not None:
    #             lv = abs(lm[35][1] - lm[40][1])
    #             lh = abs(lm[33][0] - lm[39][0])
    #             rv = abs(lm[89][1] - lm[94][1])
    #             rh = abs(lm[87][0] - lm[93][0])
    #             ear = ((lv / (lh + 1e-5)) + (rv / (rh + 1e-5))) / 2.0
    #             ear_values.append(ear)

    #     except Exception as e:
    #         print(f"SKIP photo {i}: {e}")
    #         continue

    def process_one(photo):
        try:
            image = read_upload_image(photo)
            vec, box, raw = get_face_vector(image, return_box=True, return_raw=True)
            lm = getattr(raw, 'landmark_2d_106', None)
            ear = None
            if lm is not None:
                lv = abs(lm[35][1] - lm[40][1])
                lh = abs(lm[33][0] - lm[39][0])
                rv = abs(lm[89][1] - lm[94][1])
                rh = abs(lm[87][0] - lm[93][0])
                ear = ((lv / (lh + 1e-5)) + (rv / (rh + 1e-5))) / 2.0
            return vec, box, raw, ear
        except Exception as e:
            print(f"SKIP: {e}")
            return None

    t2 = time.time()
    with ThreadPoolExecutor(max_workers=3) as ex:
        results = list(ex.map(process_one, [photo1, photo2, photo3]))

    results = [r for r in results if r is not None]
    vectors    = [r[0] for r in results]
    boxes      = [r[1] for r in results]
    raw_faces  = [r[2] for r in results]
    ear_values = [r[3] for r in results if r[3] is not None]

    T["all_photos"] = round((time.time() - t2) * 1000)
    print(f"⏱ All Photos (parallel): {T['all_photos']}ms")


    # --- Blink Check ---
    t3 = time.time()
    print(f"EAR VALUES: {[f'{e:.3f}' for e in ear_values]}")
    #if len(ear_values) >= 2:
    #    variation = max(ear_values) - min(ear_values)
    #    min_ear   = min(ear_values)
    #    print(f"EAR VARIATION: {variation:.4f}")
    #    print(f"MIN EAR: {min_ear:.4f}")
    #    blink_ok  = (variation >= 0.10) and (min_ear < 0.22)
    #    print(f"BLINK OK: {blink_ok}")
    blink_ok = False
    if len(ear_values) >= 2:
        max_ear = max(ear_values)
        min_ear = min(ear_values)
        variation = max_ear - min_ear
        relative_drop = variation / (max_ear + 1e-5)
        print(f"EAR VARIATION: {variation:.4f}")
        print(f"MIN EAR: {min_ear:.4f}")
        print(f"RELATIVE DROP: {relative_drop:.2%}")
        blink_ok = relative_drop >= 0.25
        print(f"BLINK OK: {blink_ok}")

    T["blink_check"] = round((time.time() - t3) * 1000)
    print(f"⏱ Blink Check:    {T['blink_check']}ms")

    if not vectors:
        return {"success": False, "matched": False,
                "show_password_login": False,
                "message": "Face Not Detected"}

    if not blink_ok:
        return {"success": False, "matched": False,
                "show_password_login": False,
                "message": "Blink not detected. Just Blink."}

    # --- FAISS Match ---
    t4 = time.time()
    best_vector = vectors[len(vectors) // 2]
    best_box    = boxes[len(boxes) // 2]
    match, match_status = find_best_user_match(best_vector, current_index, current_mapping)
    T["faiss_match"] = round((time.time() - t4) * 1000)
    print(f"⏱ FAISS Match:    {T['faiss_match']}ms")

    # --- TOTAL ---
    T["total"] = round((time.time() - t0) * 1000)
    print(f"\n{'='*40}")
    print(f"📊 TIMING SUMMARY:")
    for k, v in T.items():
        bar = '█' * (v // 20)
        print(f"   {k:<20} {v:>5}ms  {bar}")
    print(f"{'='*40}\n")

    # --- Result ---
    if match_status == "matched":
        user = match["user"]
        failed_attempts["camera_login"] = 0
        return {
            "success":  True, "matched": True,
            "message":  f"Welcome {user['username']}",
            "userid":   user.get("userid"),
            "username": user["username"],
            "face_box": best_box
        
        }

    failed_attempts["camera_login"] = failed_attempts.get("camera_login", 0) + 1
    attempts  = failed_attempts["camera_login"]

    if attempts >= MAX_FACE_ATTEMPTS:
        failed_attempts["camera_login"] = 0
        return {"success": False, "matched": False,
                "show_password_login": True,
                "message": "Face login failed 3 times. Use username and password."}

    return {"success": False, "matched": False,
            "show_password_login": False,
            "message": f"Face not matched. {MAX_FACE_ATTEMPTS - attempts} attempts left.",
            "face_box": best_box,
            }

@app.post("/manual-login")
def manual_login(
    username: str = Form(...),
    password: str = Form(...)
):
    try:
        with get_db_connection() as connection:
            cursor = connection.cursor()
            cursor.execute(
                "SELECT userid, name FROM usermaster WHERE loginname = ? AND password = ?",
                username, password
            )
            user = cursor.fetchone()

        if user:
            return {
                "success": True,
                "message": f"Welcome {user[1]}",
                "userid": user[0],
                "username": str(user[1])
            }

        return {
            "success": False,
            "message": "Invalid username or password"
        }

    except Exception as e:
        print("MANUAL LOGIN ERROR:", e)
        return {
            "success": False,
            "message": "Login failed"
        }
@app.post("/remove-user")
def remove_user(
    clientid: str = Form(...),
    userid: str = Form(...)
):
    try:
        client_id = str(clientid).strip()
        paths = get_client_paths(client_id)

        index_path = paths["faiss"]
        mapping_path = paths["mapping"]

       
        if not os.path.exists(index_path) or not os.path.exists(mapping_path):
            return {
                "success": False,
                "message": "No face database found for this client"
            }

        current_index = faiss.read_index(index_path)

        with open(mapping_path, "r") as f:
            current_mapping = json.load(f)

       
        ids_to_remove = set()

        for vector_id_str, user_data in current_mapping.items():
            if not isinstance(user_data, dict):
                print(f"SKIPPING legacy/non-dict entry: {vector_id_str} -> {user_data}")
                continue

            stored_userid = user_data.get("userid")

            print(
                "VECTOR:", vector_id_str,
                "STORED:", stored_userid,
                type(stored_userid)
            )

            if str(stored_userid) == str(userid):
                print("MATCH FOUND:", vector_id_str)
                ids_to_remove.add(int(vector_id_str))

        if not ids_to_remove:
            return {
                "success": False,
                "message": f"No face data found for userid {userid} in client {client_id}"
            }

   
        ids_to_keep = [
            int(vid)
            for vid, data in current_mapping.items()
            if int(vid) not in ids_to_remove and isinstance(data, dict)
        ]

       
        kept_vectors = []

        for old_id in ids_to_keep:
            entry = current_mapping[str(old_id)]

            if "faiss_pos" not in entry:
                print(f"SKIPPING entry without faiss_pos: {old_id} -> {entry}")
                continue

            faiss_pos = entry["faiss_pos"]

            if faiss_pos < 0 or faiss_pos >= current_index.ntotal:
                print(f"SKIPPING out-of-range faiss_pos: {old_id} -> {faiss_pos}")
                continue

            vec = current_index.reconstruct(faiss_pos)
            kept_vectors.append((old_id, vec))

     
        new_index = faiss.IndexFlatL2(DIMENSION)
        new_mapping = {}

        for new_pos, (old_id, vec) in enumerate(kept_vectors):
            new_index.add(np.array([vec], dtype=np.float32))
            new_mapping[str(old_id)] = current_mapping[str(old_id)]
            new_mapping[str(old_id)]["faiss_pos"] = new_pos

        save_database(new_index, new_mapping, index_path, mapping_path)
        _db_cache.pop(client_id, None)

        return {
            "success": True,
            "message": f"Removed {len(ids_to_remove)} face vector(s) for userid {userid}",
            "userid": userid,
            "vectors_removed": len(ids_to_remove),
            "vectors_remaining": new_index.ntotal
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "message": f"Remove user failed: {str(e)}"
        }
@app.get("/debug-paths/{clientid}")
def debug_paths(clientid: str):
    paths = get_client_paths(clientid)
    return {
        "index_path": paths["faiss"],
        "mapping_path": paths["mapping"],
        "index_exists": os.path.exists(paths["faiss"]),
        "mapping_exists": os.path.exists(paths["mapping"]),
        "base_storage": BASE_STORAGE,
        "all_client_folders": os.listdir(BASE_STORAGE) if os.path.exists(BASE_STORAGE) else []
    }
@app.get("/debug-files/{clientid}")
def debug_files(clientid: str):
    base_path = os.path.join(BASE_STORAGE, str(clientid))
    return {
        "folder_exists": os.path.exists(base_path),
        "files_inside": os.listdir(base_path) if os.path.exists(base_path) else []
    }


@app.get("/server-check")
def server_check():
    return {
        "hostname": socket.gethostname(),
        "base_storage": BASE_STORAGE,
        "folders": os.listdir(BASE_STORAGE)
    }

@app.get("/debug-all-folders")
def debug_all_folders():
    result = {}

    for folder in os.listdir(BASE_STORAGE):
        path = os.path.join(BASE_STORAGE, folder)

        result[repr(folder)] = {
            "files": os.listdir(path)
        }

    return result

@app.get("/list-users/{clientid}")
def list_users(clientid: str):
    client_id = str(clientid).strip()
    paths = get_client_paths(client_id)

    mapping_path = paths["mapping"]

    if not os.path.exists(mapping_path):
        return {
            "success": False,
            "clientid": client_id,
            "message": "No face database found for this client"
        }

    with open(mapping_path, "r") as f:
        current_mapping = json.load(f)

    # Collect unique users by userid
    seen_userids = {}
    for vector_id_str, user_data in current_mapping.items():
        if isinstance(user_data, dict):
            userid = user_data.get("userid")
            username = user_data.get("username")
        else:
            # Old format: {"0": "Aniket"}
            userid = None
            username = user_data

        if userid not in seen_userids:
            seen_userids[userid] = {
                "userid": userid,
                "username": username,
                "vector_count": 1
            }
        else:
            seen_userids[userid]["vector_count"] += 1

    users = list(seen_userids.values())

    return {
        "success": True,
        "clientid": client_id,
        "total_users": len(users),
        "total_vectors": len(current_mapping),
        "users": users
    }
@app.post("/delete-face")
def delete_face(
    clientid: str = Form(...),
    userid: str = Form(...),
    vector_ids: str = Form(...)  # comma-separated, 1-based display IDs e.g. "1,2,3"
):
    client_id = str(clientid).strip()
    paths = get_client_paths(client_id)

    index_path   = paths["faiss"]
    mapping_path = paths["mapping"]

    if not os.path.exists(index_path) or not os.path.exists(mapping_path):
        return {
            "success": False,
            "message": "No face database found for this client"
        }

    current_index = faiss.read_index(index_path)
    with open(mapping_path, "r") as f:
        current_mapping = json.load(f)

    # Parse requested vector_ids (client-facing SrNo, comma-separated)
    try:
        ids_to_remove = set(
            int(vid.strip())
            for vid in vector_ids.split(",")
            if vid.strip().isdigit()
        )
    except Exception:
        return {
            "success": False,
            "message": "Invalid vector_ids format. Expected comma-separated integers."
        }

    print("REQUESTED REMOVE IDS:", ids_to_remove)
    print("CURRENT MAPPING:")
    print(json.dumps(current_mapping, indent=2))

    if not ids_to_remove or any(vid < 0 for vid in ids_to_remove):
        return {
            "success": False,
            "message": "Invalid vector_ids. IDs must be 1 or greater."
        }

    # Validate: each requested vector_id must exist and belong to this userid
    invalid_ids = []
    ids_to_remove_internal = set()

    for vid in ids_to_remove:
        found = False

        for internal_id, data in current_mapping.items():

            if not isinstance(data, dict):
                continue

            if (
                str(data.get("userid")) == str(userid)
                and str(data.get("vector_id")) == str(vid)   # ✅ FIXED: string comparison
            ):
                ids_to_remove_internal.add(int(internal_id))
                found = True
                break

        print("CHECKING VECTOR", vid, "-> FOUND:", found)

        if not found:
            invalid_ids.append({
                "vector_id": vid,
                "reason": "Vector ID not found for this user"
            })

    print("REQUESTED DISPLAY IDS:", ids_to_remove)
    print("INTERNAL IDS TO REMOVE:", ids_to_remove_internal)

    if invalid_ids:
        return {
            "success": False,
            "message": "Some vector IDs are invalid or belong to a different user.",
            "invalid_ids": invalid_ids
        }

    # IDs to keep (everything except the ones marked for removal)
    ids_to_keep = [
        int(vid)
        for vid in current_mapping.keys()
        if int(vid) not in ids_to_remove_internal
    ]

    # Reconstruct kept vectors from existing FAISS index
    kept_vectors = []
    for old_id in ids_to_keep:
        faiss_pos = current_mapping[str(old_id)]["faiss_pos"]
        vec = current_index.reconstruct(faiss_pos)
        kept_vectors.append(vec)

    # Rebuild index and mapping with remapped (re-sequenced) faiss_pos
    new_index   = faiss.IndexFlatL2(DIMENSION)
    new_mapping = {}

    for new_pos, (old_id, vec) in enumerate(zip(ids_to_keep, kept_vectors)):
        new_index.add(np.array([vec], dtype=np.float32))
        new_mapping[str(old_id)] = current_mapping[str(old_id)]
        new_mapping[str(old_id)]["faiss_pos"] = new_pos

    save_database(new_index, new_mapping, index_path, mapping_path)
    _db_cache.pop(client_id, None)

    return {
        "success": True,
        "message": f"Removed {len(ids_to_remove)} vector(s) for userid {userid}.",
        "userid": userid,
        "vectors_removed": len(ids_to_remove),
        "vectors_remaining": new_index.ntotal,
        "removed_vector_ids": sorted(ids_to_remove)
    }
@app.get("/list-user-vectors/{clientid}/{userid}")
def list_user_vectors(clientid: str, userid: int):
    client_id = str(clientid).strip()
    paths = get_client_paths(client_id)
    mapping_path = paths["mapping"]

    if not os.path.exists(mapping_path):
        return {
            "success": False,
            "message": "No face database found for this client"
        }

    with open(mapping_path, "r") as f:
        current_mapping = json.load(f)

    user_vectors = [
        {

            "vector_id": data.get("vector_id", int(vid) + 1),
            "username": data.get("username"),
            "image_id": data.get("image_id"),
            "filename": data.get("filename")
        }
        for vid, data in current_mapping.items()
        #if isinstance(data, dict) and data.get("userid") == userid
        if (
            isinstance(data, dict)
            and str(data.get("userid")) == str(userid)
            )
    ]

    if not user_vectors:
        return {
            "success": False,
            "message": f"No vectors found for userid {userid}"
        }
    print("USER VECTORS:")
    print(json.dumps(user_vectors, indent=2))
    return {
        "success": True,
        "userid": userid,
        "total_vectors": len(user_vectors),
        #"vectors": sorted(user_vectors, key=lambda v: v["vector_id"])
        "vectors": sorted(
            user_vectors,
            key=lambda v: int(v.get("vector_id") or 0)
        )
    }





if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
