import io
import os
import numpy as np
from datetime import datetime
import re

from dotenv import load_dotenv
from flask import Flask, request, jsonify, g
from PIL import Image
from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired
from flask_cors import CORS
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

CORS(app)  # simple CORS fix

_token_serializer = URLSafeTimedSerializer(app.secret_key, salt="auth")
TOKEN_MAX_AGE_SECONDS = 60 * 60 * 24 * 7

# MongoDB
client = MongoClient(os.getenv("MONGO_CONNECTION_STRING"))
db = client["face_recognition_db"]
collection = db["criminals"]
users_collection = db["users"]

# Cloudinary
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET")
)

# SETTINGS (LIGHTWEIGHT)
THRESHOLD = float(os.getenv("MATCH_THRESHOLD", "0.3"))
MODEL = "Facenet"           # lighter
DETECTOR = "opencv"         # lighter
ENFORCE = False             # avoid detection failure

# ==============================
# AUTH HELPERS
# ==============================
def _issue_token(email, role):
    return _token_serializer.dumps({"email": email, "role": role})

def _verify_token(token):
    return _token_serializer.loads(token, max_age=TOKEN_MAX_AGE_SECONDS)

def require_auth(required_role=None):
    def decorator(fn):
        def wrapper(*args, **kwargs):
            if request.method == "OPTIONS":
                return ("", 204)

            auth = request.headers.get("Authorization", "")
            if not auth.startswith("Bearer "):
                return jsonify({"message": "Missing token"}), 401

            token = auth.split(" ")[1]

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

        wrapper.__name__ = fn.__name__
        return wrapper
    return decorator

# ==============================
# UTILS
# ==============================
def file_to_numpy(file_bytes):
    return np.array(Image.open(io.BytesIO(file_bytes)).convert("RGB"))

def get_embedding(file_bytes):
    try:
        from deepface import DeepFace

        img = file_to_numpy(file_bytes)

        reps = DeepFace.represent(
            img_path=img,
            model_name=MODEL,
            detector_backend=DETECTOR,
            enforce_detection=ENFORCE,
        )

        if not reps:
            return None

        emb = np.array(reps[0]["embedding"], dtype=np.float32)
        emb = emb / np.linalg.norm(emb)
        return emb.tolist()

    except Exception as e:
        print("Embedding error:", e)
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

def _normalize_email(email: str | None) -> str:
    return (email or "").strip().lower()


def _get_user_password_hash(user: dict) -> str | None:
    if not isinstance(user, dict):
        return None
    for key in ("passwordHash", "password_hash", "password"):
        value = user.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _clamp_int(value: str | None, default: int, *, min_value: int, max_value: int) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return default
    return max(min_value, min(max_value, n))

@app.route("/api/auth/signup", methods=["POST"])
def signup():
    data = request.get_json()

    if not isinstance(data, dict):
        return jsonify({"message": "Invalid JSON"}), 400

    name = (data.get("name") or "").strip()
    raw_email = (data.get("email") or "").strip()
    email = _normalize_email(raw_email)
    password = data.get("password")
    requested_role = (data.get("role") or "NORMAL").strip().upper()
    admin_secret = (data.get("adminSecret") or "").strip()

    if not name or not email or not isinstance(password, str) or not password:
        return jsonify({"message": "Missing required fields"}), 400

    role = "NORMAL"
    if requested_role == "ADMIN":
        expected = (os.getenv("ADMIN_SECRET_KEY") or "").strip()
        if not expected or admin_secret != expected:
            return jsonify({"message": "Invalid admin secret"}), 403
        role = "ADMIN"

    existing = users_collection.find_one({
        "email": re.compile(rf"^{re.escape(raw_email)}$", re.IGNORECASE)
    })
    if existing:
        return jsonify({"message": "User exists"}), 409

    users_collection.insert_one({
        "name": name,
        "email": email,
        "passwordHash": generate_password_hash(password),
        "role": role,
        "createdAt": datetime.utcnow()
    })

    return jsonify({"message": "Signup successful"}), 201


@app.route("/api/auth/login", methods=["POST"])
def login():
    data = request.get_json()

    if not isinstance(data, dict):
        return jsonify({"message": "Invalid JSON"}), 400

    raw_email = (data.get("email") or "").strip()
    email = _normalize_email(raw_email)
    password = data.get("password")
    if not email or not isinstance(password, str) or not password:
        return jsonify({"message": "Missing email or password"}), 400

    user = users_collection.find_one({"email": email})
    if not user and raw_email:
        user = users_collection.find_one({
            "email": re.compile(rf"^{re.escape(raw_email)}$", re.IGNORECASE)
        })
    stored_hash = _get_user_password_hash(user or {})
    if not user or not stored_hash or not check_password_hash(stored_hash, password):
        return jsonify({"message": "Invalid credentials"}), 401

    role = user.get("role", "NORMAL")
    token = _issue_token(user.get("email", email), role)
    return jsonify({
        "token": token,
        "user": {
            "email": user.get("email", email),
            "name": user.get("name"),
            "role": role,
        }
    })


@app.route("/api/upload", methods=["POST"])
@require_auth()
def upload_and_match():
    file = request.files.get("image")

    if not file:
        return jsonify({"error": "No file"}), 400

    emb = get_embedding(file.read())

    if emb is None:
        return jsonify({"error": "Face not detected"}), 400

    results = []

    for doc in collection.find():
        if "embedding" not in doc:
            continue

        score = cosine_score(emb, doc["embedding"])

        if score >= THRESHOLD:
            results.append({
                "name": doc.get("name"),
                "crime": doc.get("crime"),
                "imageURL": doc.get("imageURL"),
                "score": float(score)
            })

    results.sort(key=lambda x: x["score"], reverse=True)

    return jsonify({"matches": results[:5]})


@app.route("/api/enroll", methods=["POST"])
@require_auth(required_role="ADMIN")
def enroll():
    file = request.files.get("image")

    if not file:
        return jsonify({"message": "No file"}), 400

    file_bytes = file.read()
    embedding = get_embedding(file_bytes)

    if embedding is None:
        return jsonify({"message": "Face not detected"}), 400

    image_url = upload_image(file_bytes)

    data = request.form

    doc = {
        "name": data.get("name"),
        "age": int(data.get("age")),
        "crime": data.get("crime"),
        "imageURL": image_url,
        "embedding": embedding,
        "createdAt": datetime.utcnow()
    }

    collection.insert_one(doc)

    return jsonify({"message": "Added"}), 201


@app.route("/api/latest-criminals", methods=["GET"])
def latest_criminals():
    limit = _clamp_int(request.args.get("limit"), 10, min_value=1, max_value=50)
    try:
        criminals = list(
            collection.find(
                {},
                {"_id": 0, "embedding": 0},
            ).sort("createdAt", -1).limit(limit)
        )
        return jsonify({"criminals": criminals})
    except Exception as e:
        print("latest_criminals error:", e)
        return jsonify({"message": "Database not reachable"}), 503


@app.route("/api/members", methods=["GET"])
@require_auth()
def members():
    name = (request.args.get("name") or "").strip()
    sex = (request.args.get("sex") or "").strip()
    limit = _clamp_int(request.args.get("limit"), 50, min_value=1, max_value=100)

    if not name:
        return jsonify({"message": "Missing name"}), 400

    query: dict = {"name": re.compile(re.escape(name), re.IGNORECASE)}
    if sex:
        query["sex"] = re.compile(rf"^{re.escape(sex)}$", re.IGNORECASE)

    try:
        results = list(
            collection.find(
                query,
                {"_id": 0, "embedding": 0},
            ).sort("createdAt", -1).limit(limit)
        )
        return jsonify({"members": results})
    except Exception as e:
        print("members error:", e)
        return jsonify({"message": "Database not reachable"}), 503


@app.route("/api/health", methods=["GET"])
def health():
    mongo_ok = False
    try:
        client.admin.command("ping")
        mongo_ok = True
    except Exception as e:
        print("health ping error:", e)

    return jsonify({
        "ok": True,
        "mongo": mongo_ok,
    })


@app.route("/")
def home():
    return "Backend Running 🚀"

# ==============================
# RUN
# ==============================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)