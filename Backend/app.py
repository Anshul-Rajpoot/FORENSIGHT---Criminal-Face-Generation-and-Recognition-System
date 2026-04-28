import io
import os
import numpy as np
from datetime import datetime
import multiprocessing
import re
from functools import wraps
from concurrent.futures import ProcessPoolExecutor, TimeoutError
from concurrent.futures.process import BrokenProcessPool

# ✅ IMPORTANT for Render
multiprocessing.set_start_method("spawn", force=True)

from dotenv import load_dotenv
from flask import Flask, request, jsonify, g
from PIL import Image
from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired
import cloudinary
import cloudinary.uploader

# ==============================
# LOAD ENV
# ==============================
load_dotenv()

# ==============================
# CONFIG
# ==============================
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev")

_token_serializer = URLSafeTimedSerializer(app.secret_key, salt="auth")
TOKEN_MAX_AGE_SECONDS = 60 * 60 * 24 * 7

client = MongoClient(os.getenv("MONGO_CONNECTION_STRING"))
db = client["face_recognition_db"]
collection = db["criminals"]
users_collection = db["users"]

cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET")
)

THRESHOLD = float(os.getenv("MATCH_THRESHOLD", "0.5"))
TIMEOUT = int(os.getenv("EMBEDDING_TIMEOUT_SECONDS", "60"))

# ==============================
# AUTH HELPERS
# ==============================
def _issue_token(email, role):
    return _token_serializer.dumps({"email": email, "role": role})


def _verify_token(token):
    return _token_serializer.loads(token, max_age=TOKEN_MAX_AGE_SECONDS)


def require_auth(required_role=None):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            auth = request.headers.get("Authorization", "")
            if not auth.startswith("Bearer "):
                return jsonify({"message": "Missing token"}), 401

            token = auth.split(" ", 1)[1]
            try:
                payload = _verify_token(token)
            except SignatureExpired:
                return jsonify({"message": "Token expired"}), 401
            except BadSignature:
                return jsonify({"message": "Invalid token"}), 401

            if required_role and payload.get("role") != required_role:
                return jsonify({"message": "Forbidden"}), 403

            g.user = payload
            return fn(*args, **kwargs)

        return wrapper
    return decorator

# ==============================
# IMAGE PROCESSING
# ==============================
def file_to_numpy(file_bytes):
    return np.array(Image.open(io.BytesIO(file_bytes)).convert("RGB"))


def compute_embedding_worker(file_bytes):
    try:
        import face_recognition

        img = file_to_numpy(file_bytes)
        rgb_img = img[:, :, ::-1]

        encodings = face_recognition.face_encodings(rgb_img)

        if not encodings:
            return None

        emb = encodings[0]
        emb = emb / np.linalg.norm(emb)

        return emb.tolist()

    except Exception as e:
        print("Embedding error:", e)
        return None


_executor = None


def get_executor():
    global _executor
    if _executor is None:
        ctx = multiprocessing.get_context("spawn")
        _executor = ProcessPoolExecutor(max_workers=1, mp_context=ctx)
    return _executor


def get_embedding(file_bytes):
    try:
        fut = get_executor().submit(compute_embedding_worker, file_bytes)
        return fut.result(timeout=TIMEOUT)
    except TimeoutError:
        return None
    except BrokenProcessPool:
        return None


def upload_image(file_bytes):
    result = cloudinary.uploader.upload(io.BytesIO(file_bytes))
    return result["secure_url"]


def cosine_score(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# ==============================
# ROUTES
# ==============================

@app.route("/api/auth/signup", methods=["POST"])
def signup():
    data = request.get_json()
    email = data["email"]
    password = data["password"]

    if users_collection.find_one({"email": email}):
        return jsonify({"message": "User exists"}), 400

    users_collection.insert_one({
        "email": email,
        "password": generate_password_hash(password),
        "role": "NORMAL"
    })

    return jsonify({"message": "Signup success"})


@app.route("/api/auth/login", methods=["POST"])
def login():
    data = request.get_json()
    user = users_collection.find_one({"email": data["email"]})

    if not user or not check_password_hash(user["password"], data["password"]):
        return jsonify({"message": "Invalid credentials"}), 401

    token = _issue_token(user["email"], user["role"])
    return jsonify({"token": token})


@app.route("/api/upload", methods=["POST"])
@require_auth()
def upload_and_match():
    file = request.files.get("image")
    if not file:
        return jsonify({"error": "No image"}), 400

    emb = get_embedding(file.read())
    if emb is None:
        return jsonify({"error": "No face detected"}), 400

    results = []
    for doc in collection.find():
        if "embedding" not in doc:
            continue

        score = cosine_score(emb, doc["embedding"])

        if score >= THRESHOLD:
            results.append({
                "name": doc["name"],
                "crime": doc["crime"],
                "imageURL": doc["imageURL"],
                "score": float(score)
            })

    results.sort(key=lambda x: x["score"], reverse=True)
    return jsonify({"matches": results[:5]})


@app.route("/api/enroll", methods=["POST"])
@require_auth("ADMIN")
def enroll():
    file = request.files.get("image")
    data = request.form

    emb = get_embedding(file.read())
    if emb is None:
        return jsonify({"message": "Face not detected"}), 400

    image_url = upload_image(file.read())

    collection.insert_one({
        "name": data["name"],
        "crime": data["crime"],
        "imageURL": image_url,
        "embedding": emb
    })

    return jsonify({"message": "Added"})


@app.route("/")
def home():
    return "Backend running!"

# ==============================
# RUN
# ==============================
if __name__ == "__main__":
    app.run()
