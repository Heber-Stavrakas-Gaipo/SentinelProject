
# 🚨 Facial & License Plate Recognition System

Welcome to the **Facial & License Plate Recognition System**! This project leverages the latest in computer vision and deep learning to create a real-time surveillance tool capable of:

- 🔎 Recognizing faces from a database using YOLO models
- 🚗 Detecting and reading vehicle license plates with YOLOv8 and Tesseract OCR
- ⚠️ Emitting alerts for suspicious faces or plates

## 🎯 Project Highlights

- **Real-Time Video Processing**: Uses your webcam to scan and analyze frames on the fly.
- **YOLO-based Face Recognition**: Detects and recognizes known individuals using YOLO models for face detection and custom embeddings for recognition.
- **License Plate Detection (YOLOv8)**: Detects plates with YOLOv8 and reads them using Tesseract OCR.
- **Alert System**: Instantly notifies when a suspicious face or plate is detected.
- **Easy Database Management**: Just add images or plate numbers to the database folders/files.

## 🛠️ How It Works

1. **Load Known Faces**: Place images in `database/faces/` (filename = person's name).
2. **List Suspicious Plates**: Add plate numbers (one per line) in `database/placas_suspeitas.txt`.
3. **Run the App**: The system captures video, detects faces and plates with YOLO, recognizes faces, reads plates with Tesseract, and triggers alerts.

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- `opencv-python`
- `numpy`
- `torch` (for YOLO)
- `ultralytics` (YOLOv8)
- `pytesseract`
- `tesseract-ocr` installed on your system

Install dependencies:
```bash
pip install opencv-python numpy torch ultralytics pytesseract
```

> **Note:**
> - Download and install [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) for your OS and ensure it's in your PATH.
> - Place your YOLOv8 model weights (e.g., `license_plate_detector.pt`, `yolov5su.pt`) in the project root.

### Usage
```bash
python main.py
```
- Press `q` to quit the application.

## 📁 Project Structure
```
facial_recognition/
├── main.py                  # Main application script
├── database/
│   ├── faces/               # Folder with known face images
│   └── suspicious_plates.txt # List of suspicious plates
├── yolov5su.pt              # YOLO model for faces
├── license_plate_detector.pt# YOLOv8 model for license plates
├── ...
```

## 🧠 How to Add New Faces or Plates
- **Faces**: Add a clear photo (jpg/png) to `database/faces/` (filename = person's name).
- **Plates**: Add the plate (e.g., `ABC1234`) to `database/placas_suspeitas.txt` (one per line).

## 💡 Example Use Cases
- Parking lot security
- School or company access control
- Smart home surveillance
- Law enforcement monitoring

## 🤖 Tech Stack
- Python
- OpenCV
- YOLOv8 (Ultralytics)
- Tesseract OCR

## 📝 License
MIT License. See `LICENSE` for details.

## ✨ Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

## 🙌 Acknowledgements
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
- [OpenCV](https://opencv.org/)

---

> "AI is not just the future, it's the present. Make your security smarter!"
