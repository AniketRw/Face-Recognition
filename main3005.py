from fastapi import FastAPI, Form, UploadFile, File
from typing import Optional
from deepface import DeepFace
import faiss
import numpy as np
import os
from fastapi.middleware.cors import CORSMiddleware
import json
import cv2
import pyodbc
import configparser
import sys
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "http://192.168.200.14:5500"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
DIMENSION = 128
INDEX_PATH = "face_index.faiss"
MAPPING_PATH = "user_mapping.json"
MODEL_NAME = "Facenet"
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
EYE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
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
        fallback="0.45"
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




if os.path.exists(INDEX_PATH):
    index = faiss.read_index(INDEX_PATH)
    print("FAISS LOADED")
else:
    index = faiss.IndexFlatL2(DIMENSION)
    print("NEW FAISS CREATED")

if os.path.exists(MAPPING_PATH):
    with open(MAPPING_PATH, "r") as file:
        user_mapping = json.load(file)
        print("User Mapping Loaded")
else:
    user_mapping = {}
    print("New User Mapping")


def get_db_connection(
    db_server,
    db_name,
    db_user,
    db_pass,
    db_driver
):
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


    return pyodbc.connect(
        connection_string,
        timeout=30
    )

def get_client_id(conn):

    cursor = conn.cursor()

    cursor.execute("""
        SELECT Value
        FROM Options2
        WHERE Name =
        'RSPL_ClientID'
    """)

    row = cursor.fetchone()

    if row:
        return str(row[0])

    return "default"

def get_client_paths(client_id):

    base_path = (
        f"/data/customers/"
        f"{client_id}"
    )

    os.makedirs(
        base_path,
        exist_ok=True
    )

    return {
        "faiss":
            f"{base_path}/face_index.faiss",

        "mapping":
            f"{base_path}/user_mapping.json"
    }

def get_user_from_db(
    userid: int,
    db_server,
    db_name,
    db_user,
    db_pass,
    db_driver
):
    with get_db_connection(
        db_server,
        db_name,
        db_user,
        db_pass,
        db_driver
    ) as connection:

        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT userid, name
            FROM usermaster
            WHERE userid = ?
            """,
            userid,
        )

        row = cursor.fetchone()

    if not row:
        return None

    return {
        "userid": int(row.userid),
        "username": str(row.name),
    }
def get_users_from_db(
    db_server,
    db_name,
    db_user,
    db_pass,
    db_driver
):
    with get_db_connection(
        db_server,
        db_name,
        db_user,
        db_pass,
        db_driver
    ) as connection:

        cursor = connection.cursor()

        cursor.execute(
            "SELECT userid, name FROM usermaster ORDER BY name"
        )

        rows = cursor.fetchall()

    return [
        {
            "userid": int(row.userid),
            "username": str(row.name),
        }
        for row in rows
    ]

def read_upload_image(photo: UploadFile):
    image_bytes = photo.file.read()
    np_arr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Invalid image file")

    return image


def crop_face(image):
    try:
        return crop_face_with_deepface_detector(image, backends=("ssd",))
    except Exception as e:
        print(f"SSD PRIMARY DETECTOR FAILED: {e}")

    height, width = image.shape[:2]
    if width > 640:
        new_width = 640
        new_height = int(height * (new_width / width))
        resized_image = cv2.resize(image, (new_width, new_height))
    else:
        resized_image = image.copy()

    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = []
    detection_settings = [
        (1.05, 5, (70, 70)),
        (1.05, 4, (60, 60)),
        (1.08, 4, (50, 50)),
    ]

    for scale_factor, min_neighbors, min_size in detection_settings:
        faces = list(FACE_CASCADE.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size,
        ))
        if faces:
            break

    if len(faces) == 0:
        return crop_face_with_deepface_detector(image, backends=("opencv",))

    x, y, w, h = max(faces, key=lambda face: face[2] * face[3])
    face_gray = gray[y:y + h, x:x + w]
    upper_face_gray = face_gray[: int(h * 0.65), :]
    eyes = EYE_CASCADE.detectMultiScale(
        upper_face_gray,
        scaleFactor=1.05,
        minNeighbors=3,
        minSize=(15, 15),
    )

    if len(eyes) < 1:
        return crop_face_with_deepface_detector(image, backends=("opencv",))

    margin = int(max(w, h) * 0.25)
    x1 = max(x - margin, 0)
    y1 = max(y - margin, 0)
    x2 = min(x + w + margin, resized_image.shape[1])
    y2 = min(y + h + margin, resized_image.shape[0])

    print("OPENCV FACE CROP FOUND")
    return resized_image[y1:y2, x1:x2], {
        "x": x1 / resized_image.shape[1],
        "y": y1 / resized_image.shape[0],
        "width": (x2 - x1) / resized_image.shape[1],
        "height": (y2 - y1) / resized_image.shape[0],
    }


def crop_face_with_deepface_detector(image, backends=("ssd", "opencv")):
    for backend in backends:
        try:
            faces = DeepFace.extract_faces(
                img_path=image,
                detector_backend=backend,
                enforce_detection=True,
                align=True,
            )

            if not faces:
                continue

            image_center_x = image.shape[1] / 2
            image_center_y = image.shape[0] / 2


            def face_priority(item):

                area = item.get(
                    "facial_area", {}
                )

                x = area.get("x", 0)
                y = area.get("y", 0)
                w = area.get("w", 0)
                h = area.get("h", 0)

                center_x = x + (w / 2)
                center_y = y + (h / 2)

                face_area = w * h

                distance = (
                    (
                        center_x -
                        image_center_x
                    ) ** 2
                    +
                    (
                        center_y -
                        image_center_y
                    ) ** 2
                ) ** 0.5

                return (
                    -distance,
                    face_area
                )


            selected_face = max(
                faces,
                key=face_priority
            )
            face = selected_face["face"]
            area = selected_face.get("facial_area", {})

            face = np.asarray(face)
            if face.dtype != np.uint8:
                face = np.clip(face * 255, 0, 255).astype(np.uint8)

            print(f"DEEPFACE {backend.upper()} FACE CROP FOUND")
            return face, {
                "x": area.get("x", 0) / image.shape[1],
                "y": area.get("y", 0) / image.shape[0],
                "width": area.get("w", image.shape[1]) / image.shape[1],
                "height": area.get("h", image.shape[0]) / image.shape[0],
            }

        except Exception as e:
            print(f"DEEPFACE {backend.upper()} DETECTOR FAILED: {e}")

    raise ValueError("No face detected")


def get_face_vector(image, return_box=False):
    face_image, face_box = crop_face(image)
    embedding = DeepFace.represent(
        face_image,
        model_name=MODEL_NAME,
        detector_backend="skip",
        enforce_detection=False,
    )

    face_vector = np.array(
        embedding[0]["embedding"],
        dtype=np.float32,
    ).reshape(1, -1)

    norm = np.linalg.norm(face_vector)
    if norm == 0:
        raise ValueError("Invalid face embedding")

    normalized_vector = face_vector / norm

    if return_box:
        return normalized_vector, face_box

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


def refresh_user_from_db(
    user,
    db_server,
    db_name,
    db_user,
    db_pass,
    db_driver
):
    userid = user.get("userid")

    if userid is None:
        return user

    try:
        db_user_data = get_user_from_db(
            int(userid),
            db_server,
            db_name,
            db_user,
            db_pass,
            db_driver
        )

        return db_user_data or user

    except Exception as e:
        print(f"DB USER REFRESH FAILED: {e}")
        return user


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

@app.get("/db-config")
def get_db_config():

    config = configparser.ConfigParser()
    config.read(CONFIG_PATH)

    return {
        "db_server":
            config["DATABASE"]["SERVER"],

        "db_name":
            config["DATABASE"]["DATABASE"],

        "db_user":
            config["DATABASE"]["USER"],

        "db_pass":
            config["DATABASE"]["PASSWORD"],

        "db_driver":
            config["DATABASE"]["DRIVER"]
    }




@app.post("/manual-login")
def manual_login(
    username: str = Form(...),
    password: str = Form(...),
    db_server: str = Form(...),
    db_name: str = Form(...),
    db_user: str = Form(...),
    db_pass: str = Form(...),
    db_driver: str = Form(...)
):

    try:

        connection = get_db_connection(
            db_server,
            db_name,
            db_user,
            db_pass,
            db_driver
    )

        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT
                userid,
                name,
                LoginName
            FROM usermaster
            WHERE LoginName = ?
            AND dbo.Fun_DecryptPwd(
                Password
            ) = ?
            """,
            (
               username,
                password
            )
        )

        user = cursor.fetchone()

        connection.close()

        # Login success
        if user:

            failed_attempts[
                "camera_login"
            ] = 0

            return {
                "success": True,
                "message":
                f"Welcome {user.name}",
                "userid":
                user.userid,
                "username":
                user.LoginName
            }

        # Wrong credentials
        return {
            "success": False,
            "message":
            "Invalid username or password"
        }

    except Exception as e:

        print(
            "MANUAL LOGIN ERROR:",
            e
        )

        return {
            "success": False,
            "message":
            str(e)
        }

@app.get("/")
def home():
    return {"message": "Face Recognition API Running"}
@app.get("/get-user/{userid}")
def get_user(userid: int):

    try:

        # Read DB config from config.ini
        config = configparser.ConfigParser()
        config.read(CONFIG_PATH)

        db_server = config["DATABASE"]["SERVER"]
        db_name = config["DATABASE"]["DATABASE"]
        db_user = config["DATABASE"]["USER"]
        db_pass = config["DATABASE"]["PASSWORD"]
        db_driver = config["DATABASE"]["DRIVER"]

        # Connect DB
        with get_db_connection(
            db_server,
            db_name,
            db_user,
            db_pass,
            db_driver
        ) as connection:

            cursor = connection.cursor()

            cursor.execute(
                """
                SELECT name
                FROM usermaster
                WHERE userid = ?
                """,
                userid
            )

            user = cursor.fetchone()

        # User found
        if user:
            return {
                "success": True,
                "username": str(user[0])
            }

        # User not found
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
    
@app.post("/users")
def users(
    db_server: str = Form(...),
    db_name: str = Form(...),
    db_user: str = Form(...),
    db_pass: str = Form(...),
    db_driver: str = Form(...)
):
    try:
        return {
            "success": True,
            "users": get_users_from_db(
                db_server,
                db_name,
                db_user,
                db_pass,
                db_driver
            ),
        }

    except Exception as e:
        return {
            "success": False,
            "message": str(e),
            "users": [],
        }
@app.post("/upload-entity")
def upload_entity(
    userid: int = Form(...),
    photos: list[UploadFile] = File(...)
):

    # Read DB config from config.ini
    config = configparser.ConfigParser()
    config.read(CONFIG_PATH)

    db_server = config["DATABASE"]["SERVER"]
    db_name = config["DATABASE"]["DATABASE"]
    db_user = config["DATABASE"]["USER"]
    db_pass = config["DATABASE"]["PASSWORD"]
    db_driver = config["DATABASE"]["DRIVER"]

    connection = get_db_connection(
        db_server,
        db_name,
        db_user,
        db_pass,
        db_driver
    )

    client_id = get_client_id(connection)
    connection.close()

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
        user_data = get_user_from_db(
            userid,
            db_server,
            db_name,
            db_user,
            db_pass,
            db_driver
        )

    except Exception as e:

        return {
            "success": False,
            "message": f"Database error: {e}",
            "userid": userid,
            "photos_registered": 0,
            "photos_skipped": 0,
            "total_vectors": index.ntotal,
        }

    if not user_data:
        return {
            "success": False,
            "message": f"User ID {userid} not found in usermaster.",
            "userid": userid,
            "photos_registered": 0,
            "photos_skipped": 0,
            "total_vectors": index.ntotal,
        }

    username = user_data["username"]
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


@app.post("/authenticate")
def authenticate(
    photo: Optional[UploadFile] = File(None),
    photos: Optional[list[UploadFile]] = File(None),

    db_server: str = Form(...),
    db_name: str = Form(...),
    db_user: str = Form(...),
    db_pass: str = Form(...),
    db_driver: str = Form(...)
):
    connection = get_db_connection(
        db_server,
        db_name,
        db_user,
        db_pass,
        db_driver
    )

    client_id = get_client_id(
        connection
    )

    connection.close()

    print(
        "CLIENT ID:",
        client_id
    )

    paths = get_client_paths(
        client_id
    )

    global INDEX_PATH
    global MAPPING_PATH
    global index
    global user_mapping

    INDEX_PATH = paths["faiss"]
    MAPPING_PATH = paths["mapping"]

    if os.path.exists(INDEX_PATH):
        index = faiss.read_index(
            INDEX_PATH
        )
    else:
        index = faiss.IndexFlatL2(
            DIMENSION
        )

    if os.path.exists(
        MAPPING_PATH
    ):
        with open(
            MAPPING_PATH,
            "r"
        ) as file:

            user_mapping = json.load(
                file
            )
    else:
        user_mapping = {}
    try:
        if index.ntotal == 0:
            return {
                "success": False,
                "matched": False,
                "message": "No registered faces found",
            }

        login_photos = photos if photos else ([photo] if photo else [])
        if not login_photos:
            return {
                "success": False,
                "matched": False,
                "message": "No login photo received",
            }

        required_frame_matches = min(MIN_LOGIN_FRAME_MATCHES, len(login_photos))
        print("LOGIN PHOTOS RECEIVED:", len(login_photos))

        frame_matches = []
        face_detected_count = 0

        for login_photo in login_photos:
            try:
                image = read_upload_image(login_photo)
                face_vector, face_box = get_face_vector(image, return_box=True)
                face_detected_count += 1
            except Exception as e:
                print("LOGIN FRAME SKIPPED:", e)
                continue

            match, match_status = find_best_user_match(face_vector)
            if match_status == "matched":
                frame_matches.append(match)

                matched_keys = [
                    existing_match["key"]
                    for existing_match in frame_matches
                ]
                if matched_keys.count(match["key"]) >= required_frame_matches:
                    user = refresh_user_from_db(
                        match["user"],
                        db_server,
                        db_name,
                        db_user,
                        db_pass,
                        db_driver
                    )
                    username = user["username"]
                    scores = [
                        existing_match["score"]
                        for existing_match in frame_matches
                        if existing_match["key"] == match["key"]
                    ]
                    score = sum(scores) / len(scores)
                    print(f"FACE MATCHED EARLY | USER SCORE: {score:.4f}")
                    failed_attempts["camera_login"] = 0

                    return {
                        "success": True,
                        "matched": True,
                        "message": f"Welcome {username}",
                        "userid": user.get("userid"),
                        "username": username,
                        "distance": score,
                        "frame_matches": len(scores),
                        "face_box": face_box,
                    }
            elif match_status == "ambiguous":
                print("LOGIN FRAME AMBIGUOUS")

        if face_detected_count == 0:
            return {
                "success": False,
                "matched": False,
                "show_password_login":
                False,
                "message":
                "Face Not Detected",
            }

        user_matches = {}
        for match in frame_matches:
            key = match["key"]
            user_matches.setdefault(key, {
                "user": match["user"],
                "scores": [],
            })
            user_matches[key]["scores"].append(match["score"])

        candidates = []

        for key, value in user_matches.items():

            score = (
                sum(value["scores"])
                / len(value["scores"])
            )

            if (
                len(value["scores"])
                >= required_frame_matches
                and score
                <= MATCH_DISTANCE_THRESHOLD
            ):

                candidates.append({
                    "key": key,
                    "user": value["user"],
                    "score": score,
                    "frame_matches":
                    len(value["scores"]),
                })

            candidates.sort(
                key=lambda candidate:
                candidate["score"]
            )

            print(
                "LOGIN USER CANDIDATES:",
                candidates
            )

        if (
            len(candidates) > 1
            and candidates[1]["score"] - candidates[0]["score"] < MATCH_MARGIN
        ):
            print("FACE MATCH AMBIGUOUS")
            return {
                "success": False,
                "matched": False,
                "message": "Face match is not clear. Please try again.",
            }

        if candidates:
            match = candidates[0]
            user = refresh_user_from_db(
                match["user"],
                db_server,
                db_name,
                db_user,
                db_pass,
                db_driver
            )
            username = user["username"]
            print(f"FACE MATCHED | USER SCORE: {match['score']:.4f}")
            failed_attempts["camera_login"] = 0

            return {
                "success": True,
                "matched": True,
                "message": f"Welcome {username}",
                "userid": user.get("userid"),
                "username": username,
                "distance": match["score"],
                "frame_matches": match["frame_matches"],
                "face_box": face_box if "face_box" in locals() else None,
            }

        print("FACE NOT MATCHED")

        failed_attempts["camera_login"] = (failed_attempts.get("camera_login",0)+ 1)
        attempts = (failed_attempts["camera_login"])
        remaining = (MAX_FACE_ATTEMPTS- attempts)
        print(f"FAILED ATTEMPT: {attempts}")
        if (attempts>=MAX_FACE_ATTEMPTS):
            return {
                "success": False,
            "matched": False,
            "show_password_login":
            True,
            "message":
            "Face login failed 3 times. Use username and password."
        }

# Still attempts left
        return {
            "success": False,
            "matched": False,
            "show_password_login":
            False,
            "message":
            f"Face not matched. {remaining} attempts left.",
            "face_box":
            face_box if "face_box" in locals() else None,
        }

    except Exception as e:
        print("FACE NOT DETECTED")
        print(e)

        return {
            "success": False,
            "matched": False,
            "show_password_login":
            False,
            "message":
            "Face Not Detected",
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
