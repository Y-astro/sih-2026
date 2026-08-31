import time
import numpy as np
from typing import List, Dict, Tuple
from schemas.events import BaseEvent, ShelfAlertPayload

class ShelfEngine:
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
                # Detect retail objects (bottle: 39, cup: 41, book: 73, etc. in COCO)
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
            zone_id = shelf["zone_id"]
            expected_sku = shelf["expected_sku"]
            target_count = shelf["target_count"]
            min_stock = shelf.get("min_stock_alert", 1)
            sx1, sy1, sx2, sy2 = shelf["bounding_box"]

            # Count objects whose centroid falls inside shelf bounds
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

            # Debounce threshold check
            if matching_count <= 0:
                self.consecutive_empty_frames[shelf_id] = self.consecutive_empty_frames.get(shelf_id, 0) + 1
            else:
                self.consecutive_empty_frames[shelf_id] = 0

            # Alert evaluation
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
                # Resolved
                if shelf_id in self.active_tickets:
                    del self.active_tickets[shelf_id]

        return detected_objects, events
