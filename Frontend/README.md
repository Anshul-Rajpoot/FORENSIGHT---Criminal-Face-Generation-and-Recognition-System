
# 🧠 Facial Composite & Recognition System

A sophisticated full-stack web application designed to modernize suspect identification. This system allows users to **construct human faces using modular components** and perform **intelligent search & comparison** against stored records, significantly reducing the dependency on manual forensic sketching.

---

## 🚀 Key Features

* **🧑‍🎨 Precision Face Builder:** Construct faces using a wide library of modular facial features (eyes, nose, hair, etc.).
* **⚡ Real-time Rendering:** Live canvas-based construction with layer-based editing.
* **🔍 Dual-Mode Search:**
    * **Metadata Search:** Query by name or attributes from database records.
    * **Visual Matching:** Search by uploading an existing image or using a generated composite.
* **🔐 Secure Access:** Integrated user authentication (Login/Signup) and admin controls.
* **📤 Asset Management:** Seamless image uploading, export functionality, and result matching.

---

## 🏗️ System Architecture

<img width="1264" height="842" alt="architecture1 (1)" src="https://github.com/user-attachments/assets/0d877213-827c-46bf-be56-f2aeef88f95c" />


* **Frontend:** React + Vite (State management via Hooks, Canvas API for rendering).
* **Backend:** Python REST API (Handles business logic, file processing, and face matching).
* **Database:** Structured storage for user profiles, asset metadata, and identification results.
* **Communication:** Secure JSON-based communication over HTTP.

---

## 📁 Project Structure

```text
project/
├── Backend/
│   ├── app.py             # Main API Entry Point
│   ├── db_inspect.py      # Database utility scripts
│   └── .env               # Environment configurations
│
└── Frontend/
    ├── public/assets/     # High-res facial components
    ├── src/
    │   ├── components/    # Reusable UI elements (Canvas, Panels)
    │   ├── pages/         # View logic (Home, Editor, Results, Admin)
    │   ├── hooks/         # Custom React logic (useCanvas, useToast)
    │   ├── utils/         # Helper functions & API handlers
    │   ├── App.jsx        # Routing & Layout
    │   └── main.jsx       # Application bootstrap
    ├── package.json
    └── vite.config.js
````

-----

## ⚙️ Installation & Setup

### 1\. Clone the Repository

```bash
git clone [https://github.com/rajpootanshul92/FORENSIGHT.git)
cd your-repo-name
```

### 2\. Setup Backend

```bash
cd Backend
pip install -r requirements.txt
# Configure your .env file here
python app.py
```

### 3\. Setup Frontend

```bash
cd Frontend
npm install
npm run dev
```

-----

## 🛠️ Technologies Used

| Layer | Stack |
| :--- | :--- |
| **Frontend** | React, Vite, CSS Modules, Canvas API |
| **Backend** | Python (Flask/FastAPI) |
| **Database** | MongoDB / MySQL |
| **Tooling** | Axios, JWT, Postman |

-----

## 📈 Analysis

### **Advantages**

  * **Efficiency:** Drastically faster than traditional manual sketching.
  * **Consistency:** Reduces human error and subjective interpretation by artists.
  * **Scalability:** Modular design allows for easy addition of new facial assets.

### **Limitations**

  * **Asset-Based:** Realism is currently limited to the available component library.
  * **Alignment:** Component placement follows fixed alignment guides.

### **Future Roadmap**

  * 🤖 **AI Synthesis:** Implementing Generative Adversarial Networks (GANs) for hyper-realistic faces.
  * 🧠 **Deep Learning:** Integrating FaceNet for automated similarity scoring.
  * 🎨 **Advanced Manipulation:** Adding drag-to-resize, rotation, and skin-tone blending.

-----

## 📸 Screenshots

<img width="1919" height="881" alt="Screenshot 2026-04-23 010808" src="https://github.com/user-attachments/assets/88dc991c-975b-4bb1-8562-faca61446bad" />
<img width="1920" height="1080" alt="Screenshot 2026-04-22 222533" src="https://github.com/user-attachments/assets/927af575-85f0-44fb-85e7-56dce67a0a8f" />
<img width="1024" height="564" alt="Screenshot 2026-04-23 002550" src="https://github.com/user-attachments/assets/394ce8cf-6218-45cb-bbeb-3c2833d313e2" />
<img width="1919" height="1079" alt="Screenshot 2026-04-22 002040" src="https://github.com/user-attachments/assets/8123b33f-d6f0-4c89-a17e-11b02611e8ba" />


