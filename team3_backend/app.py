import asyncio
import json
from typing import List, Set
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from schemas.events import BaseEvent
from team3_backend.state_store import StoreState

app = FastAPI(
    title="Edge AI Retail Intelligence Platform API",
    version="1.0.0",
    description="Unified API & WebSocket Event Hub for Shopper Analytics, Shelf Monitoring & Operations Dashboard"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Store State
store_state = StoreState(store_id="store_001")

# WebSocket Connection Manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
        # Send initial snapshot immediately upon connection
        await websocket.send_json({
            "type": "SNAPSHOT",
            "data": store_state.get_summary_snapshot()
        })

    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)

    async def broadcast(self, message: dict):
        dead_connections = set()
        for conn in list(self.active_connections):
            try:
                await conn.send_json(message)
            except Exception:
                dead_connections.add(conn)
        for conn in dead_connections:
            self.active_connections.discard(conn)

ws_manager = ConnectionManager()


@app.get("/")
def root():
    return {
        "status": "ONLINE",
        "service": "Retail Intelligence Platform - Team 3 Event Hub",
        "version": "1.0.0"
    }


@app.get("/api/v1/store/snapshot")
def get_snapshot():
    return store_state.get_summary_snapshot()


@app.post("/api/v1/events")
async def ingest_event(event: BaseEvent):
    """
    Ingestion endpoint for Team 2 (Shopper CV), Team 3 (Shelf CV), and Team 1 (Edge Sync Agent).
    """
    store_state.ingest_event(event)
    
    # Auto-dispatch WMS ticket on critical shelf OOS
    if event.event_type == "shelf_oos":
        p = event.payload
        store_state.create_wms_ticket(
            shelf_id=p.get("shelf_id", "shelf_unknown"),
            sku=p.get("expected_sku", "SKU"),
            reason="Zero stock detected on physical shelf."
        )

    # Broadcast real-time event to all connected dashboard clients
    await ws_manager.broadcast({
        "type": "EVENT",
        "event": event.model_dump(),
        "snapshot": store_state.get_summary_snapshot()
    })
    return {"status": "ACK", "event_id": event.event_id}


class WMSRestockRequest(BaseModel):
    shelf_id: str
    sku: str
    quantity: int = 10
    priority: str = "HIGH"

@app.post("/api/v1/wms/restock-ticket")
async def create_wms_restock_ticket(req: WMSRestockRequest):
    ticket = store_state.create_wms_ticket(
        shelf_id=req.shelf_id,
        sku=req.sku,
        reason=f"Manual/Auto Restock order of {req.quantity} units requested."
    )
    await ws_manager.broadcast({
        "type": "WMS_TICKET_CREATED",
        "ticket": ticket
    })
    return {"status": "SUCCESS", "ticket": ticket}


@app.websocket("/ws/feed")
async def websocket_endpoint(websocket: WebSocket):
    await ws_manager.connect(websocket)
    try:
        while True:
            # Keep connection alive
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
                if msg.get("action") == "PING":
                    await websocket.send_json({"type": "PONG"})
            except Exception:
                pass
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
