# 🚦 Smart Traffic AI — Bengaluru v3.0

A unified, AI-powered smart traffic management system for Bengaluru's road network.

## Features

| Tab | Feature | Tech |
|-----|---------|------|
| 🗺️ Live Map | Real-time congestion heatmap across 15 junctions | Plotly Mapbox |
| 📈 Prediction | 30-min congestion forecast per junction | GAT + LSTM Fusion |
| 🚦 Signals | Adaptive signal timing optimisation | Q-Learning RL |
| 🚑 Emergency | Green corridor routing for emergency vehicles | Dijkstra + cascade |
| 📷 YOLO Detection | Vehicle detection & traffic density from camera images | YOLOv8 (ultralytics) |
| 🌊 Flood Risk | Waterlogging risk prediction + nearest safe route routing | LSTM + Dijkstra |

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## YOLO Model (optional)

Place `yolov8n.pt` inside a `models/` folder next to `app.py`.  
If absent, the app runs in **demo mode** with realistic mock detections.

Download: https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt

## Project Structure

```
smart_traffic/
├── app.py               # Main Streamlit application (6 tabs)
├── simulator.py         # Bengaluru road network + traffic data generator
├── models.py            # GAT, LSTM, Fusion, Q-Learning models
├── features.py          # Emergency Green Corridor logic
├── flood_predictor.py   # LSTM flood risk model + safe routing
├── yolo_detector.py     # YOLOv8 vehicle detector (with mock fallback)
├── requirements.txt
└── models/
    └── yolov8n.pt       # (optional — place here to enable live detection)
```

RESULTS:





<img width="600" height="300" alt="Screenshot 2026-04-07 013916" src="https://github.com/user-attachments/assets/e3c624c6-7a67-4283-a3cb-a34b52e30614" />

YOLO DETECTION

<img width="600" height="300" alt="Screenshot 2026-05-21 231246" src="https://github.com/user-attachments/assets/f0e135e8-7bf4-4a62-a161-0d8546a96e2c" />
<img width="600" height="300" alt="Screenshot 2026-05-21 231209" src="https://github.com/user-attachments/assets/cc2bfe19-31f3-44b7-95cb-768e3cb6174d" />
<img width="600" height="300" alt="Screenshot 2026-05-21 231246" src="https://github.com/user-attachments/assets/410dcf6f-081b-4922-8764-2ca147039acd" />
<img width="600" height="300" alt="Screenshot 2026-05-21 231448" src="https://github.com/user-attachments/assets/6fdcc850-ee7e-4a96-a014-23ddce0dc4ea" />





