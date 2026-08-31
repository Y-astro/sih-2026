import time
from typing import Dict, List, Any
from schemas.events import BaseEvent

class StoreState:
    def __init__(self, store_id: str = "store_001"):
        self.store_id = store_id
        self.total_footfall_in = 0
        self.total_footfall_out = 0
        self.current_occupancy = 0
        self.active_queue_states: Dict[str, dict] = {}
        self.active_shelf_alerts: Dict[str, dict] = {}
        self.zone_dwell_records: List[dict] = []
        self.recent_events_log: List[dict] = []
        self.density_matrix: List[List[float]] = [[0.0]*20 for _ in range(20)]
        self.system_status = {
            "fps": 30.0,
            "latency_ms": 18.5,
            "active_tracks": 0,
            "status": "ONLINE",
            "last_event_time": time.time()
        }
        self.wms_tickets: List[dict] = []

    def ingest_event(self, event: BaseEvent):
        self.system_status["last_event_time"] = time.time()
        event_dict = event.model_dump()
        self.recent_events_log.insert(0, event_dict)
        if len(self.recent_events_log) > 100:
            self.recent_events_log.pop()

        p = event.payload
        t = event.event_type

        if t == "footfall":
            self.total_footfall_in = p.get("running_total_in", self.total_footfall_in + p.get("in_delta", 0))
            self.total_footfall_out = p.get("running_total_out", self.total_footfall_out + p.get("out_delta", 0))
            self.current_occupancy = max(0, self.total_footfall_in - self.total_footfall_out)

        elif t == "queue_state":
            cid = p.get("counter_id", event.zone_id)
            self.active_queue_states[cid] = p

        elif t in ["shelf_oos", "shelf_lowstock", "planogram_violation"]:
            sid = p.get("shelf_id", event.zone_id)
            self.active_shelf_alerts[sid] = p

        elif t == "density_matrix":
            if "matrix" in p:
                self.density_matrix = p["matrix"]

        elif t == "dwell":
            self.zone_dwell_records.append(p)
            if len(self.zone_dwell_records) > 200:
                self.zone_dwell_records.pop(0)

        elif t == "system_status":
            self.system_status["fps"] = p.get("fps", self.system_status["fps"])
            self.system_status["latency_ms"] = p.get("inference_latency_ms", self.system_status["latency_ms"])
            self.system_status["active_tracks"] = p.get("active_tracks", self.system_status["active_tracks"])

    def create_wms_ticket(self, shelf_id: str, sku: str, reason: str) -> dict:
        ticket = {
            "ticket_id": f"WMS-TK-{len(self.wms_tickets) + 1001}",
            "shelf_id": shelf_id,
            "sku": sku,
            "status": "DISPATCHED_TO_FLOOR_STAFF",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "priority": "HIGH"
        }
        self.wms_tickets.insert(0, ticket)
        return ticket

    def get_summary_snapshot(self) -> dict:
        return {
            "store_id": self.store_id,
            "total_footfall_in": self.total_footfall_in,
            "total_footfall_out": self.total_footfall_out,
            "current_occupancy": self.current_occupancy,
            "queues": self.active_queue_states,
            "shelf_alerts": self.active_shelf_alerts,
            "density_matrix": self.density_matrix,
            "system_status": self.system_status,
            "recent_events": self.recent_events_log[:15],
            "wms_tickets": self.wms_tickets[:10]
        }
