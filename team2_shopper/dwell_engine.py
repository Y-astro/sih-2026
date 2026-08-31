import time
from typing import Dict, List, Tuple
from shapely.geometry import Point, Polygon
from schemas.events import BaseEvent, DwellPayload, current_utc_iso

class DwellEngine:
    def __init__(self, zones_config: List[dict], store_id: str = "store_001"):
        self.store_id = store_id
        self.zones = []
        for z in zones_config:
            if z.get("type", "") == "shopping_aisle":
                self.zones.append({
                    "id": z["id"],
                    "name": z["name"],
                    "polygon": Polygon(z["polygon"])
                })
        self.track_zone_sessions: Dict[Tuple[int, str], dict] = {}

    def update(self, tracks: List[dict], current_time: float = None) -> List[BaseEvent]:
        if current_time is None:
            current_time = time.time()

        events = []
        active_track_zones = set()

        for t in tracks:
            track_id = t["track_id"]
            cx, cy = t["centroid"]
            pt = Point(cx, cy)

            for z in self.zones:
                zone_id = z["id"]
                if z["polygon"].contains(pt):
                    key = (track_id, zone_id)
                    active_track_zones.add(key)

                    if key not in self.track_zone_sessions:
                        self.track_zone_sessions[key] = {
                            "first_seen": current_time,
                            "last_seen": current_time,
                            "first_seen_iso": current_utc_iso(),
                            "last_reported_dwell": 0.0
                        }
                    else:
                        self.track_zone_sessions[key]["last_seen"] = current_time
                        sess = self.track_zone_sessions[key]
                        dwell_sec = current_time - sess["first_seen"]
                        
                        # Emit periodic heartbeat if dwelling > 10s
                        if dwell_sec - sess["last_reported_dwell"] >= 10.0:
                            sess["last_reported_dwell"] = dwell_sec
                            payload = DwellPayload(
                                zone_id=zone_id,
                                dwell_time_seconds=round(dwell_sec, 1),
                                status="in_progress",
                                first_seen=sess["first_seen_iso"],
                                last_seen=current_utc_iso()
                            )
                            events.append(BaseEvent(
                                event_type="dwell",
                                store_id=self.store_id,
                                zone_id=zone_id,
                                anonymous_track_id=f"sess_{track_id}",
                                confidence=t.get("confidence", 0.90),
                                payload=payload.model_dump()
                            ))

        # Check completed sessions (person left zone)
        finished_keys = [k for k in self.track_zone_sessions if k not in active_track_zones]
        for key in finished_keys:
            sess = self.track_zone_sessions.pop(key)
            track_id, zone_id = key
            dwell_sec = sess["last_seen"] - sess["first_seen"]
            if dwell_sec >= 3.0: # Filter out fleeting pass-throughs
                payload = DwellPayload(
                    zone_id=zone_id,
                    dwell_time_seconds=round(dwell_sec, 1),
                    status="completed",
                    first_seen=sess["first_seen_iso"],
                    last_seen=current_utc_iso()
                )
                events.append(BaseEvent(
                    event_type="dwell",
                    store_id=self.store_id,
                    zone_id=zone_id,
                    anonymous_track_id=f"sess_{track_id}",
                    confidence=0.92,
                    payload=payload.model_dump()
                ))

        return events
