import time
import math
from typing import List, Dict, Tuple
from shapely.geometry import Point, Polygon
from schemas.events import BaseEvent, QueuePayload

class QueueEngine:
    def __init__(self, queue_zones_config: List[dict] = None, store_id: str = "store_001"):
        self.store_id = store_id
        self.queue_zones = []
        if queue_zones_config:
            for z in queue_zones_config:
                if z.get("type", "") == "queue_zone":
                    self.queue_zones.append({
                        "id": z["id"],
                        "name": z["name"],
                        "polygon": Polygon(z["polygon"]),
                        "warning_threshold": z.get("warning_threshold", 3),
                        "congested_threshold": z.get("congested_threshold", 5),
                        "service_time_per_person": 30
                    })
        self.last_emit_time: Dict[str, float] = {}
        self.dynamic_queues: List[dict] = []

    def _cluster_people(self, tracks: List[dict], max_distance: float = 160.0) -> List[List[dict]]:
        """
        Spatial clustering algorithm: groups people standing in close proximity (queuing/congregating)
        without needing any hardcoded polygon.
        """
        if len(tracks) < 2:
            return []

        clusters = []
        visited = set()

        for i, t1 in enumerate(tracks):
            if i in visited:
                continue
            c1 = t1["centroid"]
            current_cluster = [t1]
            visited.add(i)

            for j, t2 in enumerate(tracks):
                if j in visited:
                    continue
                c2 = t2["centroid"]
                dist = math.hypot(c1[0] - c2[0], c1[1] - c2[1])
                if dist <= max_distance:
                    current_cluster.append(t2)
                    visited.add(j)

            if len(current_cluster) >= 2:
                clusters.append(current_cluster)

        return clusters

    def update(self, tracks: List[dict], current_time: float = None) -> List[BaseEvent]:
        if current_time is None:
            current_time = time.time()

        events = []
        self.dynamic_queues = []

        # 1. Configured static queue zones (if defined by user calibration)
        if self.queue_zones:
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
                    rec = "High queue congestion! Open secondary counter."
                elif count >= qz["warning_threshold"]:
                    status = "warning"
                    alert = False
                    rec = "Queue building up; prepare next counter."
                else:
                    status = "normal"
                    alert = False
                    rec = None

                last_time = self.last_emit_time.get(zone_id, 0.0)
                if (current_time - last_time) >= 3.0 or alert:
                    self.last_emit_time[zone_id] = current_time
                    events.append(BaseEvent(
                        event_type="queue_state",
                        store_id=self.store_id,
                        zone_id=zone_id,
                        confidence=0.92,
                        payload=QueuePayload(
                            counter_id=zone_id,
                            queue_length=count,
                            estimated_wait_seconds=est_wait,
                            congestion_status=status,
                            alert_triggered=alert,
                            recommended_action=rec
                        ).model_dump()
                    ))

        # 2. DYNAMIC QUEUE DETECTION (Pure AI clustering - zero hardcoded polygons!)
        # Runs whenever 2 or more people cluster anywhere in the frame
        people_clusters = self._cluster_people(tracks, max_distance=160.0)
        for idx, cluster in enumerate(people_clusters, start=1):
            count = len(cluster)
            all_x1 = [p["bbox"][0] for p in cluster]
            all_y1 = [p["bbox"][1] for p in cluster]
            all_x2 = [p["bbox"][2] for p in cluster]
            all_y2 = [p["bbox"][3] for p in cluster]
            
            bx1, by1 = max(0, int(min(all_x1) - 10)), max(0, int(min(all_y1) - 10))
            bx2, by2 = int(max(all_x2) + 10), int(max(all_y2) + 10)

            est_wait = count * 30
            status = "congested" if count >= 4 else ("warning" if count >= 3 else "normal")
            rec = "Dynamic queue congestion detected! Open additional register." if status == "congested" else None

            qid = f"dynamic_queue_{idx}"
            self.dynamic_queues.append({
                "id": qid,
                "bbox": [bx1, by1, bx2, by2],
                "count": count,
                "status": status,
                "wait_time": est_wait
            })

            # Emit dynamic queue event if static zones are not configured
            if not self.queue_zones:
                last_time = self.last_emit_time.get(qid, 0.0)
                if (current_time - last_time) >= 3.0 or status == "congested":
                    self.last_emit_time[qid] = current_time
                    events.append(BaseEvent(
                        event_type="queue_state",
                        store_id=self.store_id,
                        zone_id=qid,
                        confidence=0.90,
                        payload=QueuePayload(
                            counter_id=qid,
                            queue_length=count,
                            estimated_wait_seconds=est_wait,
                            congestion_status=status,
                            alert_triggered=(status == "congested"),
                            recommended_action=rec
                        ).model_dump()
                    ))

        return events
