# AI-Powered Face Attendance System

A modern, web-based attendance system powered by **YOLOv8** and **FaceNet** for real-time face detection and recognition. This application logs entry and exit times automatically and includes an AI chatbot assistant for natural language queries about attendance.

## ✨ Features

- **Real-time Face Recognition**: accurate detection and identification using deep learning models.
- **Automated Attendance Logging**: Records entry and exit times in a CSV log.
- **Web Interface**: Beautiful glassmorphism UI built with Bootstrap 5.
- **User Management**:
  - Register unknown faces directly from the UI.
  - Delete users and their attendance records.
- **AI Assistant**: Built-in chatbot to ask questions like "Who is here?" or "How many visitors today?".
- **Data Export**: Download attendance logs as Excel files.
- **Live Status**: Real-time feedback on scanning status.

## 🛠️ Tech Stack

- **Backend**: Python (Flask)
- **Computer Vision**: OpenCV, Ultralytics YOLOv8, FaceNet PyTorch
- **Data Handling**: Pandas, OpenPyXL
- **Frontend**: HTML5, CSS3, JavaScript, Bootstrap 5

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8+ installed.
- A webcam or an IP camera (e.g., DroidCam).

### 1. Clone or Download the Project
Ensure you have all project files in your working directory.

### 2. Set Up a Virtual Environment (Recommended)
```bash
# Linux/macOS
python3 -m venv .venv
source .venv/bin/activate

# Windows
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

> **Note**: This may install PyTorch and other heavy libraries. Ensure you have a stable internet connection.

## ⚙️ Configuration

### Camera Setup
By default, the application is configured to use a camera source. You may need to change this to your local webcam or your own IP camera.

1. Open `app.py`.
2. Locate the `source` variable.
3. Set it to `0` for your default local webcam, or update it with your IP camera's stream URL (e.g., `"http://192.168.x.x:4747/video"`).
   ```python
   source = 0  # For local webcam
   ```

## 🏃 Usage

1. **Run the Application**:
   ```bash
   python app.py
   ```
2. **Access the Interface**:
   Open your browser and navigate to: `http://localhost:5000` (or `http://127.0.0.1:5000`).

3. **Taking Attendance**:
   - Click **▶ Start Scan**.
   - The camera feed will appear.
   - If your face is known, it will be recognized, and attendance marked. The scan stops automatically upon recognition to save resources.
   - If you are "Unknown", a red box will appear.

4. **Registering a New User**:
   - When an "Unknown" face is detected during a scan, type their name in the "Register Unknown Person" box below the video feed.
   - Click **Register User**.

5. **AI Assistant**:
   - Use the chat box in the bottom right to ask:
     - "Who is here?"
     - "How many people visited today?"
     - "List visitors"

6. **Export Logs**:
   - Click the **Download Excel** button in the Attendance Log section to get a report.

## 📁 Project Structure

```
.
├── app.py                  # Main Flask application
├── requirements.txt        # Python dependencies
├── yolov8n.pt              # YOLO model weights
├── attendance_log.csv      # Log file (auto-generated)
├── data.yaml               # Model configuration (if used)
├── known_faces/            # Directory storing registered user images
└── templates/
    └── index.html          # Frontend UI
```

## ⚠️ Troubleshooting

- **Camera Error / Blank Screen**: 
  - Ensure the `source` variable in `app.py` is correct. 
  - If using an IP camera, ensure your phone/camera and computer are on the **same Wi-Fi network**.
- **Model Download Issues**:
  - The first run might take time as it downloads YOLO/FaceNet weights.

## 🤝 Contributing
Feel free to fork this project and submit pull requests for improvements!
