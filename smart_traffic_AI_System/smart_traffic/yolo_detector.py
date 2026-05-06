"""
yolo_detector.py
================
YOLOv8-based vehicle detector for traffic density estimation.
Works with uploaded images; falls back to a realistic mock if
ultralytics / the model weights are not available.

Detects: cars, trucks, buses, motorcycles, emergency vehicles
Outputs: vehicle counts, traffic density level, signal recommendations
"""

import cv2
import numpy as np
from pathlib import Path

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# Vehicle classes in COCO dataset
VEHICLE_CLASSES = {
    2: {"name": "car",        "color": (0, 220, 90),  "weight": 1.0},
    3: {"name": "motorcycle", "color": (255, 165, 0), "weight": 0.5},
    5: {"name": "bus",        "color": (0, 90, 255),  "weight": 2.5},
    7: {"name": "truck",      "color": (160, 32, 240),"weight": 2.0},
}

EMERGENCY_KEYWORDS = {"ambulance", "fire truck", "police car", "emergency vehicle"}

DENSITY_THRESHOLDS = {
    "Low":       (0,  10),
    "Medium":    (10, 25),
    "High":      (25, 50),
    "Very High": (50, 9999),
}

SIGNAL_RULES = {
    "Low":       {"action": "Normal Signal Cycle",         "extend_green": 0,  "priority": "none"},
    "Medium":    {"action": "Slightly Extend Green Phase", "extend_green": 10, "priority": "low"},
    "High":      {"action": "Extend Green Phase",          "extend_green": 20, "priority": "medium"},
    "Very High": {"action": "Maximum Green Phase",         "extend_green": 35, "priority": "high"},
}

DENSITY_COLORS = {
    "Low":       "#52b788",
    "Medium":    "#f4a261",
    "High":      "#e76f51",
    "Very High": "#ef233c",
}


class YOLODetector:
    """YOLOv8 vehicle detector. Falls back to mock if model unavailable."""

    def __init__(self, model_size: str = "yolov8n", conf_threshold: float = 0.35,
                 iou_threshold: float = 0.45):
        self.conf = conf_threshold
        self.iou = iou_threshold
        self.model = None
        self.using_mock = True

        if YOLO_AVAILABLE:
            model_path = Path(f"models/{model_size}.pt")
            alt_path = Path(f"{model_size}.pt")
            try:
                if model_path.exists():
                    self.model = YOLO(str(model_path))
                    self.using_mock = False
                elif alt_path.exists():
                    self.model = YOLO(str(alt_path))
                    self.using_mock = False
            except Exception:
                self.model = None
                self.using_mock = True

    def detect(self, frame: np.ndarray) -> dict:
        if self.model is not None:
            return self._detect_yolo(frame)
        return self._detect_mock(frame)

    def _detect_yolo(self, frame: np.ndarray) -> dict:
        results = self.model(frame, conf=self.conf, iou=self.iou, verbose=False)
        annotated = frame.copy()
        detections, vehicle_count, emergency_count, weighted_count = [], 0, 0, 0.0

        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cls_name = self.model.names[cls_id].lower()
                is_vehicle = cls_id in VEHICLE_CLASSES
                is_emergency = any(kw in cls_name for kw in EMERGENCY_KEYWORDS)
                if not (is_vehicle or is_emergency):
                    continue

                info = VEHICLE_CLASSES.get(cls_id, {"name": cls_name, "color": (0, 255, 255), "weight": 1.0})
                color = (0, 0, 255) if is_emergency else info["color"]
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated, f"{'🚨' if is_emergency else ''}{cls_name} {conf:.2f}",
                            (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

                vehicle_count += 1
                weighted_count += info.get("weight", 1.0)
                if is_emergency:
                    emergency_count += 1
                detections.append({"class_name": cls_name, "confidence": round(conf, 3),
                                   "bbox": [x1, y1, x2, y2], "is_emergency": is_emergency})

        density_label, density_score = self._compute_density(vehicle_count, frame.shape)
        signal_rec = self._signal_recommendation(density_label, emergency_count)
        self._draw_overlay(annotated, vehicle_count, emergency_count, density_label, density_score)

        return {
            "vehicle_count": vehicle_count, "emergency_count": emergency_count,
            "weighted_vehicle_count": round(weighted_count, 1),
            "traffic_density": round(density_score, 4),
            "density_label": density_label, "detections": detections,
            "signal_recommendation": signal_rec, "annotated_frame": annotated,
            "using_mock": False,
        }

    def _detect_mock(self, frame: np.ndarray) -> dict:
        """Realistic mock detector — used when YOLO weights are unavailable."""
        np.random.seed(int(frame.mean()) % 1000)
        h, w = frame.shape[:2]
        vehicle_count = np.random.randint(4, 28)
        emergency_count = 1 if np.random.random() < 0.06 else 0
        detections = []
        annotated = frame.copy()

        class_pool = [
            {"name": "car", "color": (0, 220, 90)},
            {"name": "truck", "color": (160, 32, 240)},
            {"name": "bus", "color": (0, 90, 255)},
            {"name": "motorcycle", "color": (255, 165, 0)},
        ]
        for i in range(vehicle_count):
            x1 = np.random.randint(0, max(1, w - 110))
            y1 = np.random.randint(0, max(1, h - 60))
            x2 = min(w, x1 + np.random.randint(55, 120))
            y2 = min(h, y1 + np.random.randint(30, 80))
            cls = class_pool[i % len(class_pool)]
            is_em = (i == 0 and emergency_count > 0)
            color = (0, 0, 255) if is_em else cls["color"]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            lbl = ("ambulance" if is_em else cls["name"])
            cv2.putText(annotated, lbl, (x1, max(10, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            detections.append({"class_name": lbl, "confidence": round(np.random.uniform(0.52, 0.97), 3),
                                "bbox": [x1, y1, x2, y2], "is_emergency": is_em})

        density_label, density_score = self._compute_density(vehicle_count, frame.shape)
        signal_rec = self._signal_recommendation(density_label, emergency_count)
        self._draw_overlay(annotated, vehicle_count, emergency_count, density_label, density_score)

        return {
            "vehicle_count": vehicle_count, "emergency_count": emergency_count,
            "weighted_vehicle_count": float(vehicle_count),
            "traffic_density": round(density_score, 4),
            "density_label": density_label, "detections": detections,
            "signal_recommendation": signal_rec, "annotated_frame": annotated,
            "using_mock": True,
        }

    def _compute_density(self, count: int, shape: tuple):
        for label, (lo, hi) in DENSITY_THRESHOLDS.items():
            if lo <= count < hi:
                return label, min(1.0, count / 50.0)
        return "Very High", 1.0

    def _signal_recommendation(self, density: str, emergency_count: int) -> dict:
        if emergency_count > 0:
            return {
                "action": "Emergency Vehicle Priority",
                "extend_green": 50, "priority": "critical",
                "description": f"🚨 {emergency_count} emergency vehicle(s) detected — override signal immediately",
            }
        rule = SIGNAL_RULES.get(density, SIGNAL_RULES["Low"])
        return {**rule, "description": f"Traffic is {density.lower()} — {rule['action']}"}

    def _draw_overlay(self, frame: np.ndarray, count: int, emerg: int,
                      label: str, score: float):
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (310, 115), (10, 10, 20), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
        color_map = {"Low": (0, 220, 90), "Medium": (0, 165, 255),
                     "High": (0, 80, 255), "Very High": (0, 0, 200)}
        col = color_map.get(label, (200, 200, 200))
        texts = [
            (f"Vehicles: {count}", (0, 220, 90)),
            (f"Density: {label} ({score:.0%})", col),
            (f"Emergency: {emerg}", (0, 0, 255) if emerg > 0 else (160, 160, 160)),
            (f"YOLO: {'ACTIVE' if not self.using_mock else 'DEMO MODE'}", (80, 180, 255)),
        ]
        for i, (txt, c) in enumerate(texts):
            cv2.putText(frame, txt, (10, 22 + i * 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, c, 2)
