
# 🅿️ Parking Lot Occupancy Detection

A real-time parking lot occupancy detection system built with OpenCV and a pre-trained SVM classifier. The system analyzes live CCTV footage, detects individual parking spots, and classifies each one as **empty** or **occupied** in real time.

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)
![scikit-learn](https://img.shields.io/badge/scikit--learn-SVM-orange)

---

## 🎯 How It Works

```
CCTV Video
    ↓
Binary Mask (parking spots)
    ↓
Connected Components → Bounding Boxes per Spot
    ↓
Frame Differencing (detect changed spots)
    ↓
SVM Classifier → Empty / Occupied
    ↓
Annotated Video Output
```

1. A **binary mask** defines parking spot regions (white = spot, black = background)
2. `cv2.connectedComponentsWithStats` extracts a bounding box for each spot
3. Every 90 frames (~3 seconds at 30 FPS), frame differencing identifies **which spots changed** — only those are reclassified (efficiency optimization)
4. Each changed spot crop is resized to 15×15×3 and fed into a pre-trained **SVM model**
5. Spots are drawn as green (empty) or red (occupied) rectangles on the frame

---

## 📁 Project Structure

```
Parking System/
├── main.py              # Main inference pipeline
├── util.py              # Helper functions (bbox extraction, SVM inference)
├── model.p              # Pre-trained SVM model (pickle)
├── requirements.txt     # Python dependencies
└── parking/
    ├── mask_1920_1080.png       # Binary mask (must match video resolution)
    ├── clf-data/
    │   ├── empty/               # Training images — empty spots
    │   └── not_empty/           # Training images — occupied spots
    ├── parking_1920_1080.mp4    # Original CCTV footage
    └── parking_output.mp4       # Annotated output with detection results
```

---

## ⚙️ Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/parking-lot-detection.git
cd parking-lot-detection
```

### 2. Create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the system

```bash
python main.py
```

Press **Q** to quit.

---

## 📦 Dependencies

```
opencv-python
numpy
scikit-learn
scikit-image
```

---

## 🔑 Key Technical Decisions

| Decision                           | Reason                                                              |
| ---------------------------------- | ------------------------------------------------------------------- |
| Frame differencing before SVM      | Avoids running classifier on every spot every frame — ~10× faster |
| 15×15 resize                      | Compact feature vector for SVM; captures texture, not fine detail   |
| Step = 90 frames                   | ~3 second interval; balances responsiveness vs. compute             |
| Binary mask + connected components | Clean, resolution-independent way to define spot regions            |

---

## 🎬 Demo

|                            | Video                                                             |
| -------------------------- | ----------------------------------------------------------------- |
| **Original footage** | [`parking/parking_1920_1080.mp4`](parking/parking_1920_1080.mp4) |
| **System output**    | [`parking/parking_output.mp4`](parking/parking_output.mp4)       |

The output video shows each spot annotated in real time — 🟢 green = empty, 🔴 red = occupied — with a live counter in the top-left corner.

---

## 🚀 Extending This Project

- **New parking lot**: create a new binary mask matching your video resolution, retrain or reuse `model.p`
- **Multi-camera**: run multiple instances with different masks and videos
- **Real-time dashboard**: add a web interface (Flask/FastAPI) to display occupancy stats
- **YOLOv8**: replace the SVM with a YOLO-based detector for higher accuracy across diverse conditions

---

## 👤 Author

**RAIS** — Computer Vision student
Part of a progressive CV learning journey: fundamentals → classical CV → deep learning → football analytics.
