# 🚗 Automatic License Plate Recognition (ALPR)

An AI-powered Automatic License Plate Recognition (ALPR) system that detects vehicle license plates from images/videos and extracts the license number using Optical Character Recognition (OCR). The project combines deep learning-based object detection with OCR to provide accurate and real-time license plate recognition.

---

## 📌 Features

- Vehicle license plate detection
- OCR-based text extraction
- Real-time image and video inference
- High detection accuracy
- Confidence score visualization
- Supports multiple vehicle images
- Easy-to-use Python interface

---

## 🛠️ Tech Stack

- Python
- YOLOv5
- OpenCV
- EasyOCR / PaddleOCR
- NumPy
- Ultralytics
- Matplotlib

---



## ⚙️ Installation

```bash
git clone https://github.com/yourusername/alpr.git

cd alpr

pip install -r requirements.txt
```

---

## ▶️ Usage

For image prediction:

```bash
python detect.py --source image.jpg
```

For video prediction:

```bash
python detect.py --source video.mp4
```

---

## Workflow

```
Input Image
      │
      ▼
YOLOv8 License Plate Detection
      │
      ▼
Crop License Plate
      │
      ▼
OCR (EasyOCR/PaddleOCR)
      │
      ▼
Recognized License Number
```

---

## Applications

- Smart Parking Systems
- Toll Collection
- Traffic Monitoring
- Vehicle Tracking
- Law Enforcement
- Security Surveillance

---

## Future Improvements

- Multi-language plate recognition
- Night-time enhancement
- Vehicle make/model detection
- Database integration
- REST API deployment

---

## Author

**Arpit Singh**

B.Tech CSE (AI & ML)
