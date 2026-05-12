# 🧠 ForenSight – Criminal Face Generation and Recognition System

## 📌 Overview
* Live- https://forensight-criminal-face-generation.vercel.app/

ForenSight is a full-stack Criminal Face Generation and Recognition System designed to assist in suspect visualization and identification. The application allows users to construct faces using modular facial components such as eyes, nose, hair, lips, and facial structure through an interactive drag-and-drop interface.

The generated facial composite can then be processed and compared with stored records for identification and analysis.

---

# 🚀 Features

## 🎨 Face Construction

* Build faces using modular facial assets
* Real-time canvas rendering
* Layer-based editing system
* Drag-and-drop style interaction

## 🔍 Search & Matching

* Search suspects by name
* Search using uploaded images
* Facial comparison workflow
* Result visualization interface

## 🖥️ Frontend Features

* Built using React + Vite
* Responsive and interactive UI
* Modular component architecture
* Custom hooks for canvas and toast handling

## ⚙️ Backend Features

* Python-based backend APIs
* Authentication support
* File upload and export handling
* Business logic and image processing

## 🗄️ Database & Storage

* Stores users, assets, and results
* Supports uploaded and generated images
* Read/write database operations

---

# 🏗️ System Architecture

```text
User
  ↓
Frontend (React + Vite)
  ↓
Canvas Face Builder
  ↓
REST API Requests
  ↓
Backend (Python)
  ↓
Face Matching Logic
  ↓
Database
  ↓
Results Display
```

---

# 🧰 Tech Stack

## Frontend

* React.js
* Vite
* CSS Modules
* JavaScript

## Backend

* Python
* Flask (API handling)

## Database

* SQLite / Database Storage

## AI & Enhancement (Planned)

* Hugging Face Models
* Stable Diffusion
* Face Enhancement Models

---

# 📂 Project Structure

```text
project/
├─ Backend/
│  ├─ .env
│  ├─ app.py
│  ├─ db_inspect.py
│
└─ Frontend/
   ├─ public/assets/
   ├─ src/components/
   ├─ src/pages/
   ├─ src/hooks/
   ├─ src/utils/
   ├─ App.jsx
   └─ main.jsx
```

---

# 📸 Facial Assets

The system uses categorized facial assets:

* Face Shapes
* Eyes
* Eyebrows
* Nose
* Lips
* Hair
* Beard
* Moustache
* Ears

These assets are layered dynamically to generate composite faces.

---

# ⚡ Installation & Setup

## 1️⃣ Clone Repository

```bash
git clone <repository-url>
cd project
```

---

## 2️⃣ Frontend Setup

```bash
cd Frontend
npm install
npm run dev
```

Frontend runs on:

```text
http://localhost:5173
```

---

## 3️⃣ Backend Setup

```bash
cd Backend
pip install -r requirements.txt
python app.py
```

Backend runs on:

```text
http://localhost:5000
```

---

# 🔐 Environment Variables

Create a `.env` file inside the Backend folder.

```env
HF_TOKEN=hf_xxxxxxxxxxxxxxxxx
HF_MODEL_ID=runwayml/stable-diffusion-v1-5
HF_PROVIDER=hf-inference
```

---

# 🤖 AI Integration (Future Enhancement)

The project is planned to integrate Hugging Face diffusion models to:

* Convert composite faces into realistic images
* Improve visual quality
* Enhance facial matching accuracy
* Support AI-based face generation

### Proposed Pipeline

```text
Generated Face
      ↓
AI Enhancement Model
      ↓
Realistic Face Output
      ↓
Face Embedding Extraction
      ↓
Database Comparison
```

---

# 🎯 Use Cases

* Criminal investigation assistance
* Digital suspect sketching
* Face visualization systems
* Educational and research purposes

---

# ⚠️ Limitations

* Asset-based face generation
* Limited realism in current version
* Matching accuracy depends on feature quality
* AI enhancement integration under development

---

# 🚀 Future Scope

* AI-based realistic face generation
* FaceNet embedding comparison
* Drag, resize, and rotate facial components
* Better blending and rendering
* Cloud database integration
* Advanced face recognition models

---

# 👨‍💻 Contributors

* Anshul Rajpoot

---

# 📄 License

This project is developed for educational and research purposes.

---

# 🙌 Acknowledgements

* React.js
* Vite
* Python
* Hugging Face
* Open Source Community
