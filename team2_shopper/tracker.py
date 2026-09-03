import time
import numpy as np
from typing import List, Dict, Tuple
from team2_shopper.footfall_engine import FootfallEngine
from team2_shopper.dwell_engine import DwellEngine
from team2_shopper.heatmap_engine import HeatmapEngine
from team2_shopper.queue_engine import QueueEngine
from schemas.events import BaseEvent, SystemStatusPayload

class ShopperTrackerPipeline:
    def __init__(self, config: dict):
        self.config = config
        self.store_id = config.get("store_id", "store_001")
        self.footfall_engine = FootfallEngine(config.get("virtual_lines", []), store_id=self.store_id)
        self.dwell_engine = DwellEngine(config.get("zones", []), store_id=self.store_id)
        self.heatmap_engine = HeatmapEngine(config.get("homography", {}), store_id=self.store_id)
        self.queue_engine = QueueEngine(config.get("zones", []), store_id=self.store_id)
        
        self.model = None
        self._init_model()

    def _init_model(self):
        try:
            from ultralytics import YOLO
            # Lightweight YOLOv8 nano model
            self.model = YOLO("yolov8n.pt")
            print("[Team 2] YOLOv8n initialized successfully.")
        except Exception as e:
            print(f"[Team 2] Warning: Could not initialize YOLO model directly ({e}). Fallback to simulation/mock.")

        self.next_fallback_id = 1000
        self.recent_centroids: Dict[int, Tuple[float, float, float]] = {}

    def _match_or_create_track_id(self, cx: float, cy: float) -> int:
        now = time.time()
        best_id = None
        best_dist = 100.0
        for tid, (px, py, t) in list(self.recent_centroids.items()):
            if (now - t) > 2.0:
                del self.recent_centroids[tid]
                continue
            dist = ((cx - px)**2 + (cy - py)**2)**0.5
            if dist < best_dist:
                best_dist = dist
                best_id = tid
        if best_id is not None:
            self.recent_centroids[best_id] = (cx, cy, now)
            return best_id
        self.next_fallback_id += 1
        new_id = self.next_fallback_id
        self.recent_centroids[new_id] = (cx, cy, now)
        return new_id

    def process_frame(self, frame: np.ndarray, timestamp: float = None) -> Tuple[List[dict], List[BaseEvent]]:
        if timestamp is None:
            timestamp = time.time()

        t0 = time.time()
        tracks = []
        events = []

        if self.model is not None and frame is not None:
            try:
                # Run tracking with ByteTrack (person class = 0 in COCO)
                results = self.model.track(
                    source=frame,
                    persist=True,
                    classes=[0],
                    tracker="bytetrack.yaml",
                    verbose=False,
                    conf=0.25,
                    iou=0.45
                )

                if results and len(results) > 0 and results[0].boxes is not None:
                    boxes = results[0].boxes
                    for box in boxes:
                        xyxy = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0].item())
                        cx = float((xyxy[0] + xyxy[2]) / 2.0)
                        cy = float((xyxy[1] + xyxy[3]) / 2.0)

                        if box.id is not None:
                            track_id = int(box.id.item())
                        else:
                            track_id = self._match_or_create_track_id(cx, cy)

                        tracks.append({
                            "track_id": track_id,
                            "bbox": [float(x) for x in xyxy],
                            "centroid": (cx, cy),
                            "confidence": conf
                        })
            except Exception as e:
                pass

        # Run Team 2 Sub-Engines
        ff_events = self.footfall_engine.update(tracks, timestamp)
        events.extend(ff_events)

        dwell_events = self.dwell_engine.update(tracks, timestamp)
        events.extend(dwell_events)

        heatmap_event = self.heatmap_engine.update(tracks, timestamp)
        if heatmap_event:
            events.append(heatmap_event)

        queue_events = self.queue_engine.update(tracks, timestamp)
        events.extend(queue_events)

        t_infer = (time.time() - t0) * 1000.0
        fps = (1000.0 / t_infer) if t_infer > 0 else 30.0

        # Emit system status event periodically
        events.append(BaseEvent(
            event_type="system_status",
            store_id=self.store_id,
            zone_id="system",
            payload=SystemStatusPayload(
                fps=round(fps, 1),
                inference_latency_ms=round(t_infer, 1),
                active_tracks=len(tracks),
                edge_device_online=True,
                camera_active=True
            ).model_dump()
        ))

        return tracks, events
