import io
import os
import numpy as np
from datetime import datetime
import multiprocessing
import re
from functools import wraps
from concurrent.futures import ProcessPoolExecutor, TimeoutError
from concurrent.futures.process import BrokenProcessPool

from dotenv import load_dotenv
from flask import Flask, flash, redirect, render_template, request, url_for, jsonify, g
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

# SETTINGS
THRESHOLD = float(os.getenv("MATCH_THRESHOLD", "0.3"))
MODEL = os.getenv("FACE_MODEL", "Facenet512")
DETECTOR = os.getenv("FACE_DETECTOR", "retinaface")
ENFORCE = os.getenv("ENFORCE_DETECTION", "true").lower() in {"1", "true", "yes"}
TIMEOUT = int(os.getenv("EMBEDDING_TIMEOUT_SECONDS", "180"))

# Frontend origins (Vite)
ALLOWED_ORIGINS = {
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:5174",
    "http://127.0.0.1:5174",
}


def _is_allowed_origin(origin: str | None) -> bool:
    if not origin:
        return False
    if origin in ALLOWED_ORIGINS:
        return True

    # Dev convenience: Vite can auto-switch ports if 5173 is occupied.
    # Allow a small port range on localhost/127.0.0.1.
    m = re.match(r"^http://(localhost|127\.0\.0\.1):(\d+)$", origin)
    if not m:
        return False
    try:
        port = int(m.group(2))
    except Exception:
        return False
    return 5173 <= port <= 5180


@app.after_request
def add_cors_headers(resp):
    origin = request.headers.get("Origin")
    if _is_allowed_origin(origin):
        resp.headers["Access-Control-Allow-Origin"] = origin
        resp.headers["Vary"] = "Origin"
        resp.headers["Access-Control-Allow-Credentials"] = "true"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return resp


def _issue_token(email, role):
    return _token_serializer.dumps({"email": email, "role": role})


def _verify_token(token):
    return _token_serializer.loads(token, max_age=TOKEN_MAX_AGE_SECONDS)


def require_auth(required_role=None):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            if request.method == "OPTIONS":
                return ("", 204)
            auth = request.headers.get("Authorization", "")
            if not auth.startswith("Bearer "):
                return jsonify({"message": "Missing token"}), 401
            token = auth.split(" ", 1)[1].strip()
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
# UTIL FUNCTIONS
# ==============================
def file_to_numpy(file_bytes):
    return np.array(Image.open(io.BytesIO(file_bytes)).convert("RGB"))


def compute_embedding_worker(file_bytes):
    """Runs inside subprocess (SAFE)"""
    try:
        from deepface import DeepFace

        img = file_to_numpy(file_bytes)

        # First attempt: configured detector and enforcement
        reps = []
        try:
            reps = DeepFace.represent(
                img_path=img,
                model_name=MODEL,
                detector_backend=DETECTOR,
                enforce_detection=ENFORCE,
            )
        except Exception as e:
            try:
                print("DeepFace represent primary failed:", repr(e))
            except Exception:
                pass
            reps = []

        # Fallback: relax detection (helps for low-res / non-photographic faces)
        if not reps and not ENFORCE:
            try:
                reps = DeepFace.represent(
                    img_path=img,
                    model_name=MODEL,
                    detector_backend="opencv",
                    enforce_detection=False,
                )
            except Exception as e:
                try:
                    print("DeepFace represent fallback failed:", repr(e))
                except Exception:
                    pass
                reps = []

        if not reps:
            try:
                print("DeepFace returned no representations")
            except Exception:
                pass
            return None

        emb = np.array(reps[0]["embedding"], dtype=np.float32)
        norm = float(np.linalg.norm(emb))
        if norm == 0:
            return None

        emb = emb / norm
        return emb.tolist()

    except Exception as e:
        # Keep worker robust; return None but log a hint to server stdout
        try:
            print("Embedding worker error:", repr(e))
        except Exception:
            pass
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
        ex = get_executor()
        fut = ex.submit(compute_embedding_worker, file_bytes)
        return fut.result(timeout=TIMEOUT)

    except TimeoutError:
        print("Embedding timeout")
        return None

    except BrokenProcessPool:
        global _executor
        if _executor:
            _executor.shutdown(wait=False, cancel_futures=True)
        _executor = None
        print("Worker crashed, restarted")
        return None

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
@app.route("/api/auth/signup", methods=["POST", "OPTIONS"])
def api_signup():
    if request.method == "OPTIONS":
        return ("", 204)

    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip()
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""
    role = (data.get("role") or "NORMAL").strip().upper()
    admin_secret = data.get("adminSecret")

    if not name or not email or not password:
        return jsonify({"message": "Missing required fields"}), 400

    if role not in {"NORMAL", "ADMIN"}:
        return jsonify({"message": "Invalid role"}), 400

    if role == "ADMIN":
        required = os.getenv("ADMIN_SECRET_KEY")
        if not required:
            return (
                jsonify({"message": "Admin signup is not configured"}),
                400,
            )
        if admin_secret != required:
            return jsonify({"message": "Invalid admin secret"}), 403

    if users_collection.find_one({"email": email}):
        return jsonify({"message": "User already exists"}), 409

    users_collection.insert_one(
        {
            "name": name,
            "email": email,
            "passwordHash": generate_password_hash(password),
            "role": role,
            "createdAt": datetime.utcnow(),
        }
    )

    return jsonify({"message": "Signup successful"}), 201


@app.route("/api/auth/login", methods=["POST", "OPTIONS"])
def api_login():
    if request.method == "OPTIONS":
        return ("", 204)

    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    if not email or not password:
        return jsonify({"message": "Missing email/password"}), 400

    user = users_collection.find_one({"email": email})
    if not user or not check_password_hash(user.get("passwordHash", ""), password):
        return jsonify({"message": "Invalid credentials"}), 401

    role = user.get("role") or "NORMAL"
    token = _issue_token(email=email, role=role)
    return jsonify({"token": token, "user": {"role": role}}), 200


@app.route("/api/upload", methods=["POST", "OPTIONS"])
@require_auth()
def api_upload_and_match():
    if request.method == "OPTIONS":
        return ("", 204)

    file = request.files.get("image")
    if not file:
        return jsonify({"error": "Please select an image"}), 400

    file_bytes = file.read()
    query_emb = get_embedding(file_bytes)
    if query_emb is None:
        return (
            jsonify(
                {
                    "error": "Could not detect a face in the uploaded image. Please use a clear, front-facing face image."
                }
            ),
            400,
        )

    sex_filter = (request.form.get("sex_filter") or "").strip()
    mongo_query = {}
    if sex_filter:
        mongo_query["sex"] = sex_filter

    results = []
    for doc in collection.find(mongo_query):
        if "embedding" not in doc:
            continue

        try:
            score = cosine_score(query_emb, doc["embedding"])
        except Exception:
            continue

        if score >= THRESHOLD:
            results.append(
                {
                    "name": doc.get("name"),
                    "age": doc.get("age"),
                    "sex": doc.get("sex"),
                    "crime": doc.get("crime"),
                    "status": doc.get("status"),
                    "imageURL": doc.get("imageURL"),
                    "score": float(score),
                }
            )

    results.sort(key=lambda x: x["score"], reverse=True)
    return jsonify({"matches": results[:6]}), 200


@app.route("/api/enroll", methods=["POST", "OPTIONS"])
@require_auth(required_role="ADMIN")
def api_enroll():
    if request.method == "OPTIONS":
        return ("", 204)

    data = request.form
    file = request.files.get("image")

    if not file:
        return jsonify({"message": "Upload image"}), 400

    file_bytes = file.read()

    embedding = get_embedding(file_bytes)
    if embedding is None:
        return (
            jsonify(
                {
                    "message": "Could not extract face features. If you just started the server, try again in 1–2 minutes. Otherwise use a clear, front-facing face image."
                }
            ),
            400,
        )

    image_url = upload_image(file_bytes)

    sex = (data.get("sex") or "").strip()
    status = (data.get("status") or "ARRESTED").strip()

    try:
        doc = {
            "name": data.get("name"),
            "age": int(data.get("age")),
            "sex": sex,
            "address": data.get("address"),
            "height": float(data.get("height")),
            "weight": float(data.get("weight")),
            "crime": data.get("crime"),
            "status": status,
            "imageURL": image_url,
            "embedding": embedding,
            "createdAt": datetime.utcnow(),
        }
    except Exception:
        return jsonify({"message": "Invalid form fields"}), 400

    inserted = collection.insert_one(doc)
    return jsonify({"message": "Criminal added", "id": str(inserted.inserted_id)}), 201


@app.route("/api/members", methods=["GET", "OPTIONS"])
@require_auth()
def api_members_search():
    if request.method == "OPTIONS":
        return ("", 204)

    name = (request.args.get("name") or "").strip()
    sex = (request.args.get("sex") or "").strip()

    if not name:
        return jsonify({"message": "Missing name"}), 400

    mongo_query = {
        "name": {"$regex": re.escape(name), "$options": "i"}
    }

    if sex and sex != "Any":
        mongo_query["sex"] = sex

    members = []
    for doc in collection.find(mongo_query).sort("createdAt", -1).limit(25):
        members.append(
            {
                "name": doc.get("name"),
                "age": doc.get("age"),
                "sex": doc.get("sex"),
                "crime": doc.get("crime"),
                "status": doc.get("status"),
                "imageURL": doc.get("imageURL"),
            }
        )

    return jsonify({"members": members}), 200


@app.route("/api/latest-criminals", methods=["GET", "OPTIONS"])
def api_latest_criminals():
    if request.method == "OPTIONS":
        return ("", 204)

    try:
        limit = int(request.args.get("limit") or 10)
    except Exception:
        limit = 10

    limit = max(1, min(limit, 10))

    criminals = []

    # Home carousel: return random criminals that have an image.
    # (Optional) allow status filtering via ?status=NOT%20ARRESTED
    status = (request.args.get("status") or "").strip()

    mongo_match = {
        "imageURL": {"$exists": True, "$nin": [None, ""]},
    }
    if status:
        mongo_match["status"] = {"$regex": rf"^\s*{re.escape(status)}\s*$", "$options": "i"}

    pipeline = [
        {"$match": mongo_match},
        {"$sample": {"size": limit}},
    ]

    for doc in collection.aggregate(pipeline):
        criminals.append(
            {
                "name": doc.get("name"),
                "imageURL": doc.get("imageURL"),
                "sex": doc.get("sex"),
                "crime": doc.get("crime"),
                "status": doc.get("status"),
            }
        )

    return jsonify({"criminals": criminals}), 200


@app.get("/")
def index():
    return render_template("index.html", matches=None)


@app.post("/enroll")
def enroll():
    data = request.form

    admin_secret_required = os.getenv("ADMIN_SECRET_KEY")
    if not admin_secret_required:
        flash("❌ Enrollment disabled. Set ADMIN_SECRET_KEY in your .env to enable admin-only enrollment.", "error")
        return redirect(url_for("index"))

    provided_secret = (data.get("adminSecret") or "").strip()
    if provided_secret != admin_secret_required:
        flash("❌ Invalid admin secret key", "error")
        return redirect(url_for("index"))

    file = request.files.get("image")

    if not file:
        flash("Upload image", "error")
        return redirect(url_for("index"))

    file_bytes = file.read()

    embedding = get_embedding(file_bytes)
    if embedding is None:
        flash(
            "❌ Could not extract face features (try a clearer front-facing image)",
            "error",
        )
        return redirect(url_for("index"))

    image_url = upload_image(file_bytes)

    sex = (data.get("sex") or "").strip()
    status = (data.get("status") or "ARRESTED").strip()

    doc = {
        "name": data.get("name"),
        "age": int(data.get("age")),
        "sex": sex,
        "address": data.get("address"),
        "height": float(data.get("height")),
        "weight": float(data.get("weight")),
        "crime": data.get("crime"),
        "status": status,
        "imageURL": image_url,
        "embedding": embedding,
        "createdAt": datetime.utcnow()
    }

    collection.insert_one(doc)

    flash("✅ Criminal added", "success")
    return redirect(url_for("index"))


@app.post("/search")
def search():
    file = request.files.get("image")
    sex_filter = (request.form.get("sex_filter") or "").strip()

    if not file:
        flash("Upload image", "error")
        return redirect(url_for("index"))

    file_bytes = file.read()

    query_emb = get_embedding(file_bytes)
    if query_emb is None:
        flash(
            "❌ Could not extract face features (try a clearer front-facing image)",
            "error",
        )
        return redirect(url_for("index"))

    results = []

    mongo_query = {}
    if sex_filter:
        mongo_query["sex"] = sex_filter

    for doc in collection.find(mongo_query):
        if "embedding" not in doc:
            continue

        score = cosine_score(query_emb, doc["embedding"])

        if score >= THRESHOLD:
            results.append({
                "name": doc["name"],
                "age": doc["age"],
                "sex": doc["sex"],
                "crime": doc["crime"],
                "status": doc["status"],
                "imageURL": doc["imageURL"],
                "score": score
            })

    results.sort(key=lambda x: x["score"], reverse=True)

    if results:
        flash(f"✅ {len(results)} matches found", "success")
    else:
        flash("❌ No matches", "warning")

    return render_template("index.html", matches=results[:6])


# ==============================
# RUN
# ==============================
if __name__ == "__main__":
    app.run(debug=True, port=8000, use_reloader=False)