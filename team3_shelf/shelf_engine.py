import time
import numpy as np
from typing import List, Dict, Tuple
from collections import Counter
from schemas.events import BaseEvent, ShelfAlertPayload

class ShelfEngine:
    RETAIL_CLASSES = {
        39: "bottle", 41: "cup", 40: "wine glass", 45: "bowl",
        46: "banana", 47: "apple", 48: "sandwich", 49: "orange",
        50: "broccoli", 51: "carrot", 52: "hot dog", 53: "pizza",
        54: "donut", 55: "cake", 73: "book", 74: "clock",
        75: "vase", 76: "scissors", 77: "teddy bear", 79: "toothbrush",
        64: "potted plant", 67: "cell phone"
    }

    def __init__(self, config: dict):
        self.config = config
        self.store_id = config.get("store_id", "store_001")
        self.shelves = config.get("shelves", [])
        self.model = None
        self.last_check_time = 0.0
        self.check_interval = 2.0 # Audit every 2 seconds
        self.consecutive_empty_frames: Dict[str, int] = {}
        self.active_tickets: Dict[str, str] = {}
        self._init_model()

    def _init_model(self):
        try:
            from ultralytics import YOLO
            self.model = YOLO("yolov8n.pt")
            print("[Team 3] Shelf detector initialized.")
        except Exception as e:
            print(f"[Team 3] Warning: YOLO shelf detector init ({e})")

    def auto_detect_shelves(self, frame: np.ndarray, padding: int = 15) -> List[dict]:
        """
        Dynamically discovers shelf tiers and SKU expectations from physical products
        in the camera view without hardcoded pixel coordinates.
        Clusters detected objects into horizontal tiers and fits shelf ROIs around them.
        """
        if self.model is None or frame is None:
            return self.shelves

        h, w, _ = frame.shape
        results = self.model(frame, verbose=False, conf=0.25)
        detected = []

        if results and len(results) > 0 and results[0].boxes is not None:
            for box in results[0].boxes:
                cls_id = int(box.cls[0].item())
                cls_name = self.model.names.get(cls_id, "item")
                xyxy = box.xyxy[0].cpu().numpy()
                cx = float((xyxy[0] + xyxy[2]) / 2.0)
                cy = float((xyxy[1] + xyxy[3]) / 2.0)
                detected.append({
                    "cls_name": cls_name,
                    "bbox": [float(x) for x in xyxy],
                    "centroid": (cx, cy)
                })

        if not detected:
            print("[ShelfEngine] No retail objects detected for auto-configuration.")
            return self.shelves

        # Sort detected items by vertical position (top to bottom)
        detected.sort(key=lambda d: d["centroid"][1])

        # Group items into vertical tiers (items whose cy is within 15% of frame height)
        threshold_y = max(30.0, h * 0.15)
        clusters = []
        current_cluster = [detected[0]]

        for item in detected[1:]:
            prev_cy = np.mean([x["centroid"][1] for x in current_cluster])
            if abs(item["centroid"][1] - prev_cy) <= threshold_y:
                current_cluster.append(item)
            else:
                clusters.append(current_cluster)
                current_cluster = [item]
        if current_cluster:
            clusters.append(current_cluster)

        # Build dynamic shelf configurations for each cluster/tier
        new_shelves = []
        for idx, cluster in enumerate(clusters, start=1):
            x1 = max(0, int(min(item["bbox"][0] for item in cluster) - padding))
            y1 = max(0, int(min(item["bbox"][1] for item in cluster) - padding))
            x2 = min(w, int(max(item["bbox"][2] for item in cluster) + padding))
            y2 = min(h, int(max(item["bbox"][3] for item in cluster) + padding))

            class_counts = Counter(item["cls_name"] for item in cluster)
            dominant_sku, count = class_counts.most_common(1)[0]

            shelf_def = {
                "shelf_id": f"auto_shelf_tier_{idx}",
                "zone_id": f"aisle_tier_{idx}",
                "bounding_box": [x1, y1, x2, y2],
                "expected_sku": dominant_sku,
                "sku_name": f"{dominant_sku.title()} Auto-Shelf (Tier {idx})",
                "target_count": count,
                "min_stock_alert": max(1, count // 2)
            }
            new_shelves.append(shelf_def)

        print(f"[ShelfEngine] Auto-configured {len(new_shelves)} shelf tier(s) successfully!")
        self.shelves = new_shelves
        self.config["shelves"] = new_shelves
        return new_shelves

    def process_frame(self, frame: np.ndarray, current_time: float = None) -> Tuple[List[dict], List[BaseEvent]]:
        if current_time is None:
            current_time = time.time()

        if (current_time - self.last_check_time) < self.check_interval:
            return [], []

        self.last_check_time = current_time
        detected_objects = []
        events = []

        if self.model is not None and frame is not None:
            try:
                results = self.model(frame, verbose=False, conf=0.3)
                if results and len(results) > 0 and results[0].boxes is not None:
                    for box in results[0].boxes:
                        cls_id = int(box.cls[0].item())
                        cls_name = self.model.names.get(cls_id, "unknown")
                        xyxy = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0].item())
                        cx = float((xyxy[0] + xyxy[2]) / 2.0)
                        cy = float((xyxy[1] + xyxy[3]) / 2.0)
                        detected_objects.append({
                            "class_id": cls_id,
                            "class_name": cls_name,
                            "bbox": [float(x) for x in xyxy],
                            "centroid": (cx, cy),
                            "confidence": conf
                        })
            except Exception:
                pass

        # Audit each shelf zone against planogram expectations
        for shelf in self.shelves:
            shelf_id = shelf["shelf_id"]
            zone_id = shelf.get("zone_id", "aisle_shelf")
            expected_sku = shelf.get("expected_sku", "item")
            target_count = shelf.get("target_count", 4)
            min_stock = shelf.get("min_stock_alert", 1)
            sx1, sy1, sx2, sy2 = shelf["bounding_box"]

            matching_count = 0
            other_sku_count = 0
            detected_sku_sample = None

            for obj in detected_objects:
                cx, cy = obj["centroid"]
                if sx1 <= cx <= sx2 and sy1 <= cy <= sy2:
                    if obj["class_name"] == expected_sku:
                        matching_count += 1
                    else:
                        other_sku_count += 1
                        detected_sku_sample = obj["class_name"]

            fill_ratio = matching_count / target_count if target_count > 0 else 1.0

            if matching_count <= 0:
                self.consecutive_empty_frames[shelf_id] = self.consecutive_empty_frames.get(shelf_id, 0) + 1
            else:
                self.consecutive_empty_frames[shelf_id] = 0

            alert_type = None
            severity = "low"
            if self.consecutive_empty_frames.get(shelf_id, 0) >= 2:
                alert_type = "shelf_oos"
                severity = "critical"
            elif matching_count <= min_stock and matching_count > 0:
                alert_type = "shelf_lowstock"
                severity = "high"
            elif other_sku_count > 0 and matching_count < target_count:
                alert_type = "planogram_violation"
                severity = "medium"

            if alert_type is not None:
                ticket_id = self.active_tickets.get(shelf_id)
                if not ticket_id:
                    ticket_id = f"WMS-TK-{int(time.time()) % 10000:04d}"
                    self.active_tickets[shelf_id] = ticket_id

                payload = ShelfAlertPayload(
                    shelf_id=shelf_id,
                    zone_id=zone_id,
                    expected_sku=shelf.get("sku_name", expected_sku),
                    detected_sku=detected_sku_sample or ("Empty" if matching_count == 0 else expected_sku),
                    stock_count=matching_count,
                    max_capacity=target_count,
                    fill_percentage=round(fill_ratio * 100.0, 1),
                    alert_type=alert_type,
                    restock_ticket_id=ticket_id,
                    severity=severity
                )

                events.append(BaseEvent(
                    event_type=alert_type,
                    store_id=self.store_id,
                    zone_id=zone_id,
                    confidence=0.90,
                    payload=payload.model_dump()
                ))
            else:
                if shelf_id in self.active_tickets:
                    del self.active_tickets[shelf_id]

        return detected_objects, events
