# 🚧 DeepSign Vision: Road Sign Damage Detection

A full-stack AI-powered infrastructure monitoring system that uses **YOLOv8** to detect road signs, analyze their physical condition, and calculate severity scores based on visual degradation factors.

The system identifies various types of road signs and evaluates damage like fading, blurring, or occlusion, providing actionable maintenance data through a premium React dashboard.

This project includes:
- **Node.js/Express backend** API for model orchestration and file handling.
- **React (Vite) frontend** web interface with Framer Motion animations and CSS Glassmorphism.
- **YOLOv8 object detection engine** with ResNet50 classification refinement.
- **Automated damage analysis** and severity scoring pipeline.

---

## 📂 Project Structure
```
Road Sign Damage Detection/
│
├── backend-node/
│   ├── uploads/            (auto-created)
│   ├── server.js           (Express API)
│   ├── run_yolo.py         (Inference Engine - YOLOv8 + CNN)
│   ├── train_classifier.py  (CNN Training Script)
│   ├── package.json
│   └── sign_classifier_resnet50.pth
│
├── frontend-react/
│   ├── src/
│   │   ├── App.jsx         (Visual Dashboard)
│   │   └── App.css         (Glassmorphism Styling)
│   ├── index.html
│   └── package.json
│
├── Signs/                  (Dataset & Labels)
├── backend/                (Original FastAPI/Venv assets)
└── README.md
```

---

## ✨ Features
- **Object Detection using YOLOv8**: High-precision extraction of road signs from visual data.
- **CNN-Based Classification**: Secondary validation using a ResNet50 model to eliminate false positives and ensure deterministic labeling.
- **Multi-Factor Damage Analysis**:
  - **Blur Variance**: Laplacian depth analysis to detect unreadable or out-of-focus signs.
  - **Color Saturation**: HSV analysis to identify sun-faded or weathered signs.
  - **Structural Integrity**: Canny edge density checks to find physical cracks or deformation.
  - **Occlusion Ratio**: Content visibility checks to detect foliage or obstruction.
- **Severity Scoring (0-100)**: Automatically categorizes signs into **LOW**, **MEDIUM**, and **HIGH** priority for maintenance dispatch.
- **Premium Dashboard**: A futuristic, responsive UI with live telemetry feeds, performance metrics, and an interactive event stream.
- **Local Execution**: Runs fully on your local machine with no cloud dependencies.

---

## 🛠️ Technologies Used
**Backend:**
- **Node.js & Express** - Gateway and API orchestration.
- **Python 3.11** - Core ML execution.
- **YOLOv8 (Ultralytics)** - Object detection.
- **PyTorch** - Deep learning classifier.
- **OpenCV & NumPy** - Image processing and mathematical analysis.
- **Multer** - Secure file upload handling.

**Frontend:**
- **React 19 (Vite)** - Modern, fast Single Page Application.
- **Framer Motion** - Smooth micro-animations and transitions.
- **Lucide React** - High-quality iconography.
- **CSS3** - Premium glassmorphism and grid-based layouts.

---

## 💻 System Requirements
**Minimum:**
- **Python 3.10 – 3.12** (Recommended: 3.11)
- **Node.js v18+**
- **8 GB RAM** (16 GB for faster frame processing)
- **Windows / Linux / macOS**
- **GPU optional** (CUDA supported automatically if available)

---

## 🚀 Setup & Installation

Follow these steps to set up the development environment and run the application locally.

### 1. Prerequisites
Ensure you have the following installed:
- **Node.js** (v18.0.0 or higher)
- **Python** (v3.10 to v3.12)
- **Git**

### 2. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/Road-Sign-Damage-Detection
cd "Road Sign Damage Detection"
```

### 3. Backend Setup (Node.js & Python)
The backend requires both Node.js dependencies and a Python environment for the ML engine.

1. **Install Node.js Dependencies**:
   ```bash
   cd backend-node
   npm install
   ```

2. **Set up Python Environment**:
   It is highly recommended to use a virtual environment.
   ```bash
   # Create a venv (if not already present in root/backend)
   python -m venv venv
   # Activate on Windows:
   .\venv\Scripts\activate
   # Activate on Mac/Linux:
   source venv/bin/activate
   
   # Install ML dependencies
   pip install -r ../backend/requirements.txt
   ```

### 4. Configuration
Before running the servers, check the following configurations:

- **Python Path**: If your virtual environment is not located at `../backend/venv/`, you may need to update the `pythonExecutable` path in `backend-node/server.js` (Line 36) to point to your `python.exe`.
- **API Port**: The backend runs on port `8000` by default. If you change this, ensure you also update the `API_URL` in `frontend-react/src/App.jsx`.

### 5. Execution

#### Start the Backend API
In the `backend-node` directory:
```bash
npm start
```
*Wait for the console to log: "Node.js/Express server is running on http://localhost:8000"*

#### Start the Frontend Dashboard
Open a new terminal window and navigate to the `frontend-react` folder:
```bash
cd frontend-react
npm install
npm run dev
```
*The dashboard will be available at: `http://localhost:5173`*

---

## 🧠 How the System Works
1. **Data Ingest**: User uploads a traffic sign image via the "DeepSign Vision" dashboard.
2. **Processing**: Node.js receives the file and spawns a Python sub-process running `run_yolo.py`.
3. **Detection & Extraction**: YOLOv8 locates the sign, and the system crops it for high-resolution analysis.
4. **ResNet Validation**: A ResNet50 model performs secondary classification to confirm the sign type.
5. **Damage Diagnostic**: Four independent algorithms analyze blur, color, edges, and occlusion.
6. **Telemetry Delivery**: Results are sent back as JSON and rendered onto the React Canvas and event list.

---

## 📊 Damage Analysis Logic
The system evaluates condition based on four core metrics:
- **Blur Variance**: Calculated using the Laplacian variance. Values below a threshold indicate critical unreadability.
- **Color Quality**: Uses HSV saturation histograms to detect signs that have lost their reflective/colored properties.
- **Structural Integrity**: Analyzes the ratio of edges to surface area; deviations suggest physical sign damage.
- **Occlusion**: Monitors the color area ratio to detect if stickers, dirt, or trees are hiding the sign content.

---

## 🔧 Troubleshooting
- **Python Path Error**: If the server fails to find Python, ensure your virtual environment is active or update the path in `server.js`.
- **YOLOv8 Initialization**: On first run, YOLOv8 will download the `.pt` model weights. Ensure you have an internet connection.
- **CORS Issues**: If the frontend cannot reach the backend, confirm that port 8000 is open and not blocked by a firewall.

---

## 📝 License
This project is intended for educational and research purposes. Ensure compliance with infrastructure data usage laws in your local jurisdiction.

Developed as a Capstone for AI Infrastructure Monitoring.
