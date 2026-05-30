import cv2
import requests
import pyodbc
import configparser
import tempfile
import os
import sys


API_URL = "https://facerecognition.apps.retailware.in"


def get_config_path():
    if getattr(sys, "frozen", False):
        return os.path.join(
            os.path.dirname(sys.executable),
            "config.ini"
        )

    return os.path.join(
        os.path.dirname(__file__),
        "config.ini"
    )


def get_db_connection():
    config = configparser.ConfigParser()

    config.read(get_config_path())

    db_server = config["DATABASE"]["SERVER"]
    db_name = config["DATABASE"]["DATABASE"]
    db_user = config["DATABASE"]["USER"]
    db_pass = config["DATABASE"]["PASSWORD"]
    db_driver = config["DATABASE"]["DRIVER"]

    conn = pyodbc.connect(
        f"DRIVER={{{db_driver}}};"
        f"SERVER={db_server};"
        f"DATABASE={db_name};"
        f"UID={db_user};"
        f"PWD={db_pass};"
        "Encrypt=no;"
        "TrustServerCertificate=yes;"
    )

    return conn, {
        "db_server": db_server,
        "db_name": db_name,
        "db_user": db_user,
        "db_pass": db_pass,
        "db_driver": db_driver
    }


def get_logged_user(login_name, password):
    conn, db = get_db_connection()

    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            userid,
            name
        FROM usermaster
        WHERE LoginName = ?
        AND dbo.Fun_DecryptPwd(
            Password
        ) = ?
    """, (login_name, password))

    row = cursor.fetchone()

    conn.close()

    if not row:
        return None

    return {
        "userid": int(row.userid),
        "username": str(row.name),
        "db": db
    }


def capture_faces():
    cap = cv2.VideoCapture(0)

    images = []

    print("Press SPACE 3 times to capture")

    while len(images) < 3:
        success, frame = cap.read()

        if not success:
            continue

        cv2.imshow(
            "Capture Face",
            frame
        )

        key = cv2.waitKey(1)

        if key == 32:
            temp_file = tempfile.NamedTemporaryFile(
                suffix=".jpg",
                delete=False
            )

            cv2.imwrite(
                temp_file.name,
                frame
            )

            images.append(
                temp_file.name
            )

            print(
                f"Captured {len(images)}/3"
            )

    cap.release()
    cv2.destroyAllWindows()

    return images


def upload_face(
    login_name,
    password
):
    user = get_logged_user(
        login_name,
        password
    )

    if not user:
        print(
            "Invalid Login"
        )
        return

    print(
        "USER:",
        user["username"]
    )

    image_paths = capture_faces()

    files = []

    for path in image_paths:
        files.append(
            (
                "photos",
                open(path, "rb")
            )
        )

    data = {
        "userid":
        user["userid"],

        **user["db"]
    }

    response = requests.post(
        f"{API_URL}/upload-entity",
        data=data,
        files=files,
        verify=False
    )

    print(
        response.text
    )

    for path in image_paths:
        os.remove(path)


login_name = input(
    "Enter Login Name: "
)

password = input(
    "Enter Password: "
)

upload_face(
    login_name,
    password
)

input("Press Enter to exit...")