import time
from typing import List, Dict
from shapely.geometry import Point, Polygon
from schemas.events import BaseEvent, QueuePayload

class QueueEngine:
    def __init__(self, queue_zones_config: List[dict], store_id: str = "store_001"):
        self.store_id = store_id
        self.queue_zones = []
        for z in queue_zones_config:
            if z.get("type", "") == "queue_zone":
                self.queue_zones.append({
                    "id": z["id"],
                    "name": z["name"],
                    "polygon": Polygon(z["polygon"]),
                    "warning_threshold": z.get("warning_threshold", 3),
                    "congested_threshold": z.get("congested_threshold", 5),
                    "service_time_per_person": 30 # seconds estimate
                })
        self.last_emit_time: Dict[str, float] = {}

    def update(self, tracks: List[dict], current_time: float = None) -> List[BaseEvent]:
        if current_time is None:
            current_time = time.time()

        events = []

        for qz in self.queue_zones:
            zone_id = qz["id"]
            count = 0
            for t in tracks:
                cx, cy = t["centroid"]
                if qz["polygon"].contains(Point(cx, cy)):
                    count += 1

            est_wait = count * qz["service_time_per_person"]
            if count >= qz["congested_threshold"]:
                status = "congested"
                alert = True
                recommendation = "High queue congestion! Dispatch cashier to Open Counter 2 immediately."
            elif count >= qz["warning_threshold"]:
                status = "warning"
                alert = False
                recommendation = "Approaching peak threshold; prepare secondary billing station."
            else:
                status = "normal"
                alert = False
                recommendation = None

            last_time = self.last_emit_time.get(zone_id, 0.0)
            # Emit on status change or every 3 seconds for continuous monitoring
            if (current_time - last_time) >= 3.0 or alert:
                self.last_emit_time[zone_id] = current_time
                payload = QueuePayload(
                    counter_id=zone_id,
                    queue_length=count,
                    estimated_wait_seconds=est_wait,
                    congestion_status=status,
                    alert_triggered=alert,
                    recommended_action=recommendation
                )

                events.append(BaseEvent(
                    event_type="queue_state",
                    store_id=self.store_id,
                    zone_id=zone_id,
                    confidence=0.92,
                    payload=payload.model_dump()
                ))

        return events
