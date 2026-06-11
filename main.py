
from fastapi import FastAPI, Form, UploadFile, File
from typing import Optional
#from deepface import DeepFace
from insightface.app import FaceAnalysis
from fastapi.responses import RedirectResponse

import faiss
import numpy as np
import os
from fastapi.middleware.cors import CORSMiddleware
import json
import cv2
import configparser
import sys
import pyodbc
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
print("STEP 1")
app = FastAPI()
print("STEP 5")
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
    name='buffalo_s',
    allowed_modules=['detection', 'recognition'],
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
    import numpy as np
    # Create a dummy blank image
    dummy_img = np.zeros((320, 320, 3), dtype=np.uint8)
    # Run a dummy detection to trigger model loading/optimization
    face_app.get(dummy_img)
    print("Model warm-up complete. Ready for instant registration.")
except Exception as e:
    print(f"Warm-up failed (non-critical): {e}")
# ------------------------------

MATCH_DISTANCE_THRESHOLD = 0.45
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

BASE_STORAGE = "/app/data"



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
def get_client_paths(client_id):

    base_path = os.path.join(
        BASE_STORAGE,
        str(client_id)
    )

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

    # Reduce image size for CPU optimization only if needed
    height, width = image.shape[:2]
    max_width = 480

    if width > max_width:
        scale = max_width / width
        new_height = int(height * scale)
        image = cv2.resize(image, (max_width, new_height))

    return image




def get_face_vector(
    image,
    return_box=False
):

    faces = face_app.get(image)

    if not faces:
        raise ValueError(
            "No face detected"
        )

    face = max(
        faces,
        key=lambda f:
        (
            f.bbox[2] - f.bbox[0]
        ) * (
            f.bbox[3] - f.bbox[1]
        )
    )

    embedding = np.array(
        face.embedding,
        dtype=np.float32
    ).reshape(1, -1)

    norm = np.linalg.norm(
        embedding
    )

    if norm == 0:
        raise ValueError(
            "Invalid face embedding"
        )

    normalized_vector = (
        embedding / norm
    )

    x1, y1, x2, y2 = (
        face.bbox.astype(int)
    )

    face_box = {
        "x":
        x1 / image.shape[1],

        "y":
        y1 / image.shape[0],

        "width":
        (x2 - x1)
        / image.shape[1],

        "height":
        (y2 - y1)
        / image.shape[0]
    }

    if return_box:
        return (
            normalized_vector,
            face_box
        )

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


def find_best_user_match(face_vector, index, user_mapping):
    search_count = min(index.ntotal, 20)
    distances, indices = index.search(face_vector.astype(np.float32), search_count)
    user_distances = {}
    user_data = {}
    nearest_matches = []

    for raw_distance, raw_index in zip(distances[0], indices[0]):
        vector_id = int(raw_index)
        if vector_id < 0:
            continue

        user = get_registered_user(vector_id, user_mapping)
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


@app.post("/delete-client-data")
def delete_client_data(
    clientid: str = Form(...)
):
    import shutil

    client_path = os.path.join(
        BASE_STORAGE,
        str(clientid)
    )

    print("BASE_DIR:", BASE_DIR)
    print("BASE_STORAGE:", BASE_STORAGE)
    print("CLIENT_PATH:", client_path)
    print("PATH_EXISTS:", os.path.exists(client_path))

    if not os.path.exists(client_path):
        return {
            "success": False,
            "message": "Client data not found",
            "base_dir": BASE_DIR,
            "base_storage": BASE_STORAGE,
            "client_path": client_path,
            "exists": os.path.exists(client_path)
        }

    shutil.rmtree(client_path)

    return {
        "success": True,
        "message":
        f"Deleted all data for client {clientid}",
        "deleted_path": client_path
    }


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
    photos: list[UploadFile] = File(...)
):
    import time
    start_time = time.time()
    
    client_id = str(clientid)
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
    #print("TOTAL FACES:", current_index.ntotal)
    face_vectors = []
    skipped_photos = []

    for photo_index, photo in enumerate(photos, start=1):
        try:
            step_start = time.time()
            image = read_upload_image(photo)
            face_vector = get_face_vector(image, return_box=False)
            print(f"  Photo {photo_index} processed in {time.time() - step_start:.3f}s")
            face_vectors.append(face_vector)
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

    if len(face_vectors) < MIN_REGISTRATION_PHOTOS:
        return {
            "success": False,
            "message": f"Need at least {MIN_REGISTRATION_PHOTOS} photos.",
            "userid": userid,
            "username": username,
        }

    for face_vector in face_vectors:
        current_index.add(face_vector.astype(np.float32))
        vector_id = current_index.ntotal - 1
        current_mapping[str(vector_id)] = {
            "userid": userid,
            "username": username,
        }

    save_database(current_index, current_mapping, index_path, mapping_path)
    
    total_time = time.time() - start_time
    print(f"--- REGISTRATION COMPLETE | TOTAL TIME: {total_time:.3f}s ---")
    print("SAVED INDEX:", index_path)
    print("INDEX EXISTS AFTER SAVE:", os.path.exists(index_path))
    print("TOTAL FACES AFTER SAVE:", current_index.ntotal)
    return {
        "success": True,
        "message": f"Face registered successfully for {username}.",
        "userid": userid,
        "username": username,
        "photos_registered": len(face_vectors),
    }

from typing import Optional, List


@app.post("/authenticate")
def authenticate(
    clientid: str = Form(...),
    photo: Optional[UploadFile] = File(None),
    photos: List[UploadFile] = File(...)
):
    client_id = str(clientid)
    paths = get_client_paths(client_id)

    index_path = paths["faiss"]
    mapping_path = paths["mapping"]

    print("CLIENT:", client_id)
    print("ACTUAL INDEX PATH:", index_path)
    print("INDEX EXISTS:", os.path.exists(index_path))
    #print("TOTAL FACES:", current_index.ntotal)

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

    try:
        if current_index.ntotal == 0:
            return {
                "success": False,
                "matched": False,
                "message": "No registered faces found"
            }

        login_photos = photos if photos else ([photo] if photo else [])
        if not login_photos:
            return {
                "success": False,
                "matched": False,
                "message": "No login photo received"
            }

        required_frame_matches = min(MIN_LOGIN_FRAME_MATCHES, len(login_photos))
        
        frame_matches = []
        face_detected_count = 0
        face_box = None

        for login_photo in login_photos:
            try:
                image = read_upload_image(login_photo)
                face_vector, face_box = get_face_vector(image, return_box=True)
                face_detected_count += 1
            except Exception as e:
                continue

            match, match_status = find_best_user_match(face_vector, current_index, current_mapping)

            if match_status == "matched":
                frame_matches.append(match)
                matched_keys = [m["key"] for m in frame_matches]

                if matched_keys.count(match["key"]) >= required_frame_matches:
                    user = match["user"]
                    username = user["username"]
                    failed_attempts["camera_login"] = 0

                    return {
                        "success": True,
                        "matched": True,
                        "message": f"Welcome {username}",
                        "userid": user.get("userid"),
                        "username": username,
                        "face_box": face_box
                    }

        if face_detected_count == 0:
            return {
                "success": False,
                "matched": False,
                "show_password_login": False,
                "message": "Face Not Detected"
            }

        print("FACE NOT MATCHED")

        failed_attempts["camera_login"] = (
            failed_attempts.get("camera_login", 0) + 1
        )
        attempts = failed_attempts["camera_login"]
        remaining = MAX_FACE_ATTEMPTS - attempts

        if attempts >= MAX_FACE_ATTEMPTS:
            return {
                "success": False,
                "matched": False,
                "show_password_login": True,
                "message": "Face login failed 3 times. Use username and password."
            }

        return {
            "success": False,
            "matched": False,
            "show_password_login": False,
            "message": f"Face not matched. {remaining} attempts left.",
            "face_box": face_box
        }

    except Exception as e:
        print("FACE NOT DETECTED", e)
        return {
            "success": False,
            "matched": False,
            "show_password_login": False,
            "message": "Face Not Detected"
        }



print("INDEX PATH:", INDEX_PATH)
print("INDEX EXISTS:", os.path.exists(INDEX_PATH))


@app.get("/debug-files")
def debug_files():

    import os

    base = "/app/data"

    result = {}

    for root, dirs, files in os.walk(base):

        result[root] = files

    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
