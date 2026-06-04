from fastapi import FastAPI, Form, UploadFile, File
from typing import Optional
#from deepface import DeepFace
from insightface.app import FaceAnalysis
import faiss
import numpy as np
import os
from fastapi.middleware.cors import CORSMiddleware
import json
import cv2
import configparser
import sys
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

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



face_app = FaceAnalysis(
    providers=["CPUExecutionProvider"]
)

face_app.prepare(
    ctx_id=-1,
    det_size=(640, 640)
)

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

BASE_STORAGE = os.path.join(
    BASE_DIR,
    "data"
)

os.makedirs(
    BASE_STORAGE,
    exist_ok=True
)
print("test redeploynew")
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
index = faiss.IndexFlatL2(DIMENSION)
user_mapping = {}



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

    # Reduce image size for CPU optimization
    height, width = image.shape[:2]

    max_width = 640

    if width > max_width:

        scale = (
            max_width / width
        )

        new_height = int(
            height * scale
        )

        image = cv2.resize(
            image,
            (
                max_width,
                new_height
            )
        )

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


def get_registered_user(vector_id: int):
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


def find_best_user_match(face_vector):
    search_count = min(index.ntotal, 20)
    distances, indices = index.search(face_vector.astype(np.float32), search_count)
    user_distances = {}
    user_data = {}
    nearest_matches = []

    for raw_distance, raw_index in zip(distances[0], indices[0]):
        vector_id = int(raw_index)
        if vector_id < 0:
            continue

        user = get_registered_user(vector_id)
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
    index_path=None,
    mapping_path=None
):

    index_path = (
        index_path
        or INDEX_PATH
    )

    mapping_path = (
        mapping_path
        or MAPPING_PATH
    )

    faiss.write_index(
        index,
        index_path
    )

    with open(
        mapping_path,
        "w"
    ) as file:

        json.dump(
            user_mapping,
            file,
            indent=2
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


@app.get("/")
def home():
    return FileResponse(
        "login.html",
        headers={
            "Cache-Control":
            "no-cache, no-store, must-revalidate"
        }
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

    # Read DB config from config.ini
   
    client_id = str(clientid)

    print("CLIENT ID:", client_id)
    paths = get_client_paths(client_id)

    global INDEX_PATH
    global MAPPING_PATH
    global index
    global user_mapping

    INDEX_PATH = paths["faiss"]
    MAPPING_PATH = paths["mapping"]

    print("CHECKING INDEX:", INDEX_PATH)
    if os.path.exists(INDEX_PATH):
        print("FAISS FOUND - LOADING EXISTING INDEX")
        index = faiss.read_index(INDEX_PATH)
    else:
        index = faiss.IndexFlatL2(DIMENSION)

    if os.path.exists(MAPPING_PATH):
        with open(MAPPING_PATH, "r") as file:
            user_mapping = json.load(file)
    else:
        user_mapping = {}

    
    face_vectors = []
    skipped_photos = []

    for photo_index, photo in enumerate(photos, start=1):

        try:
            image = read_upload_image(photo)
            face_vector = get_face_vector(image)

        except Exception as e:

            skipped_photos.append({
                "photo_number": photo_index,
                "filename": photo.filename,
                "reason": str(e),
            })

            print(f"REGISTRATION PHOTO SKIPPED: {e}")
            continue

        face_vectors.append(face_vector)

    if skipped_photos:

        failed_numbers = ", ".join(
            str(photo["photo_number"])
            for photo in skipped_photos
        )

        return {
            "success": False,
            "message": (
                f"Registration failed. Photo {failed_numbers} "
                "did not contain a clear front face. "
                "Please click again and recapture."
            ),
            "userid": userid,
            "username": username,
            "photos_registered": 0,
            "photos_skipped": len(skipped_photos),
            "skipped_photos": skipped_photos,
            "total_vectors": index.ntotal,
        }

    registered_count = len(face_vectors)

    if registered_count < MIN_REGISTRATION_PHOTOS:

        return {
            "success": False,
            "message": (
                f"Registration failed. "
                f"Need at least {MIN_REGISTRATION_PHOTOS} "
                f"clear face photos, got {registered_count}."
            ),
            "userid": userid,
            "username": username,
            "photos_registered": 0,
            "photos_skipped": 0,
            "total_vectors": index.ntotal,
        }

    for face_vector in face_vectors:

        index.add(face_vector.astype(np.float32))

        vector_id = index.ntotal - 1

        user_mapping[str(vector_id)] = {
            "userid": userid,
            "username": username,
        }

    save_database(
        INDEX_PATH,
        MAPPING_PATH
    )

    return {
        "success": True,
        "message": (
            f"Face registered successfully for {username}. "
            f"Saved {registered_count} clear face photos."
        ),
        "userid": userid,
        "username": username,
        "photos_registered": registered_count,
        "photos_skipped": 0,
        "vector_size": DIMENSION,
        "total_vectors": index.ntotal,
    }

from typing import Optional, List


@app.post("/authenticate")
def authenticate(
    clientid: str = Form(...),
    photo: Optional[UploadFile] = File(None),
    photos: List[UploadFile] = File(...)
):
    client_id = str(clientid)

    print("CLIENT ID:", client_id)

    paths = get_client_paths(client_id)

    global INDEX_PATH
    global MAPPING_PATH
    global index
    global user_mapping

    INDEX_PATH = paths["faiss"]
    MAPPING_PATH = paths["mapping"]

    if os.path.exists(INDEX_PATH):
        index = faiss.read_index(INDEX_PATH)
    else:
        index = faiss.IndexFlatL2(DIMENSION)

    if os.path.exists(MAPPING_PATH):
        with open(MAPPING_PATH, "r") as file:
            user_mapping = json.load(file)
    else:
        user_mapping = {}

    try:
        if index.ntotal == 0:
            return {
                "success": False,
                "matched": False,
                "message": "No registered faces found"
            }

        login_photos = (
            photos if photos
            else [photo] if photo
            else []
        )

        if not login_photos:
            return {
                "success": False,
                "matched": False,
                "message": "No login photo received"
            }

        required_frame_matches = min(
            MIN_LOGIN_FRAME_MATCHES,
            len(login_photos)
        )

        print(
            "LOGIN PHOTOS RECEIVED:",
            len(login_photos)
        )

        frame_matches = []
        face_detected_count = 0
        face_box = None

        for login_photo in login_photos:

            try:
                image = read_upload_image(login_photo)

                face_vector, face_box = (
                    get_face_vector(
                        image,
                        return_box=True
                    )
                )

                face_detected_count += 1

            except Exception as e:
                print(
                    "LOGIN FRAME SKIPPED:",
                    e
                )
                continue

            match, match_status = (
                find_best_user_match(
                    face_vector
                )
            )

            if match_status == "matched":

                frame_matches.append(match)

                matched_keys = [
                    existing_match["key"]
                    for existing_match
                    in frame_matches
                ]

                if (
                    matched_keys.count(
                        match["key"]
                    )
                    >= required_frame_matches
                ):

                    user = match["user"]

                    username = user["username"]

                    scores = [
                        existing_match["score"]
                        for existing_match
                        in frame_matches
                        if existing_match["key"]
                        == match["key"]
                    ]

                    score = (
                        sum(scores)
                        / len(scores)
                    )

                    failed_attempts[
                        "camera_login"
                    ] = 0

                    return {
                        "success": True,
                        "matched": True,
                        "message":
                        f"Welcome {username}",
                        "userid":
                        user.get("userid"),
                        "username":
                        username,
                        "distance":
                        score,
                        "frame_matches":
                        len(scores),
                        "face_box":
                        face_box
                    }

        if face_detected_count == 0:
            return {
                "success": False,
                "matched": False,
                "show_password_login":
                False,
                "message":
                "Face Not Detected"
            }

        print("FACE NOT MATCHED")

        failed_attempts["camera_login"] = (
            failed_attempts.get(
                "camera_login",
                0
            ) + 1
        )

        attempts = (
            failed_attempts[
                "camera_login"
            ]
        )

        remaining = (
            MAX_FACE_ATTEMPTS
            - attempts
        )

        if attempts >= MAX_FACE_ATTEMPTS:

            return {
                "success": False,
                "matched": False,
                "show_password_login":
                True,
                "message":
                "Face login failed 3 times. Use username and password."
            }

        return {
            "success": False,
            "matched": False,
            "show_password_login":
            False,
            "message":
            f"Face not matched. {remaining} attempts left.",
            "face_box":
            face_box
        }

    except Exception as e:

        print(
            "FACE NOT DETECTED"
        )

        print(e)

        return {
            "success": False,
            "matched": False,
            "show_password_login":
            False,
            "message":
            "Face Not Detected"
        }


print("INDEX PATH:", INDEX_PATH)
print("INDEX EXISTS:", os.path.exists(INDEX_PATH))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
