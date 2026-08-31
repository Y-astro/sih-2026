"""
Shared Event Contract for Edge AI Retail Intelligence Platform.
Enforces standard schema across Team 1 (Edge/Sync), Team 2 (Shopper/Queue), and Team 3 (Shelf/Dashboard).
"""

from typing import Any, Dict, List, Literal, Optional
from datetime import datetime, timezone
import uuid
from pydantic import BaseModel, Field


def generate_event_id() -> str:
    return str(uuid.uuid4())


def current_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class BaseEvent(BaseModel):
    event_id: str = Field(default_factory=generate_event_id, description="Unique UUID for idempotency")
    event_type: Literal[
        "footfall",
        "dwell",
        "queue_state",
        "density_matrix",
        "shelf_oos",
        "shelf_lowstock",
        "planogram_violation",
        "system_status",
    ]
    store_id: str = Field(default="store_001", description="Store identifier for multi-store fleet")
    zone_id: str = Field(default="entrance_main", description="Zone, aisle, shelf, or checkout counter ID")
    timestamp: str = Field(default_factory=current_utc_iso, description="UTC ISO 8601 timestamp")
    anonymous_track_id: Optional[str] = Field(None, description="Ephemeral session ID (no biometric/PII)")
    confidence: float = Field(default=0.90, ge=0.0, le=1.0)
    payload: Dict[str, Any] = Field(default_factory=dict)


class FootfallPayload(BaseModel):
    direction: Literal["in", "out"]
    in_delta: int = 1
    out_delta: int = 0
    running_total_in: int = 0
    running_total_out: int = 0
    line_id: str = "entrance_line_1"


class DwellPayload(BaseModel):
    zone_id: str
    dwell_time_seconds: float
    status: Literal["in_progress", "completed"] = "completed"
    first_seen: str
    last_seen: str


class QueuePayload(BaseModel):
    counter_id: str
    queue_length: int
    estimated_wait_seconds: int
    congestion_status: Literal["normal", "warning", "congested"]
    alert_triggered: bool = False
    recommended_action: Optional[str] = None


class DensityMatrixPayload(BaseModel):
    grid_rows: int = 20
    grid_cols: int = 20
    matrix: List[List[float]] = Field(description="20x20 normalized float matrix 0.0 to 1.0")
    window_seconds: int = 300
    peak_zone: Optional[str] = None


class ShelfAlertPayload(BaseModel):
    shelf_id: str
    zone_id: str
    expected_sku: str
    detected_sku: Optional[str] = None
    stock_count: int
    max_capacity: int
    fill_percentage: float
    alert_type: Literal["shelf_oos", "shelf_lowstock", "planogram_violation"]
    restock_ticket_id: Optional[str] = None
    severity: Literal["low", "medium", "high", "critical"] = "high"


class SystemStatusPayload(BaseModel):
    fps: float
    inference_latency_ms: float
    active_tracks: int
    edge_device_online: bool = True
    camera_active: bool = True
