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
