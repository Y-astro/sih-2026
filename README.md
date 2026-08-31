# Edge AI Retail Intelligence Platform (SIH 2026)

An offline-first, edge-native retail intelligence platform combining **Shopper Footfall & Queue Intelligence (Duo 2)** and **Shelf Monitoring & Operations Dashboard (Duo 3)** with zero cloud video transmission and zero PII storage.

---

## 🏛️ Architecture Overview

```
[Webcam / RTSP Video Feed]
          │
  ┌───────┴────────────────────────┐
  ▼                                ▼
[Team 2: Shopper AI]      [Team 3: Shelf AI]
• YOLOv8 + ByteTrack MOT  • Shelf Zone Object Detection
• Entry/Exit Line Counter • Planogram Compliance Check
• Aisle Dwell Time Engine • OOS & Low-Stock Detection
• Queue Wait Estimation   • Automated Restock Trigger
• 20x20 Density Matrix    
          │                                │
          └────────────────┬───────────────┘
                           │ Standard JSON Event Stream
                           ▼
          [Team 3: FastAPI Backend & Event Hub]
          • Ingestion API (`POST /api/v1/events`)
          • SQLite Event Store & State Store
          • Mock WMS / ERP Webhook Trigger
          • WebSocket Broadcaster (`/ws/feed`)
                           │
                           ▼
          [Team 3: Operations Dashboard & HUD]
          • Rich CLI Terminal Live TUI
          • Real-time 20x20 ASCII / Color Heatmap
          • Live Footfall, Queue & Alert Panels
          • OpenCV Visual HUD Overlay
```

---

## 🚀 Quick Start Guide

### 1. Environment Setup
The project uses a dedicated virtual environment with all required vision, geometry, and backend dependencies:
```bash
cd /home/astro/Projects/retail-intelligence-platform
source .venv/bin/activate
```

### 2. Running the System

#### Mode A: Zero-Camera Mock Simulation (Perfect for Quick Demos & UI Testing)
Runs the backend, simulates realistic footfall, queue congestion, out-of-stock events, and displays the live Rich CLI dashboard:
```bash
.venv/bin/python main.py --mode mock
```

#### Mode B: Live Webcam & Full Vision Pipeline
Runs real-time YOLOv8 + ByteTrack person tracking and shelf auditing directly on your laptop webcam with the visual OpenCV HUD and terminal dashboard:
```bash
.venv/bin/python main.py --mode live --camera 0
```

#### Mode C: Server / Dashboard Separate Terminals
```bash
# Terminal 1: Backend Server
.venv/bin/python main.py --mode server-only

# Terminal 2: CLI Operations Dashboard
.venv/bin/python main.py --mode dashboard-only
```

---

## 📦 Key File Structure

- `config/store_config.json`: Store layout, virtual crossing lines, aisle polygons, checkout zones, and expected planograms.
- `schemas/events.py`: Shared Pydantic contract for `footfall`, `dwell`, `queue_state`, `density_matrix`, `shelf_oos`, and `system_status`.
- `team2_shopper/`:
  - `tracker.py`: YOLOv8 + ByteTrack MOT tracking pipeline.
  - `footfall_engine.py`: Virtual line crossing with direction detection and debouncing.
  - `dwell_engine.py`: Zone residence time tracking.
  - `heatmap_engine.py`: Homography transform + 20x20 density grid matrix generator.
  - `queue_engine.py`: Checkout queue length and congestion alert evaluator.
- `team3_shelf/`:
  - `shelf_engine.py`: Shelf gap/OOS detector and planogram compliance auditor.
- `team3_backend/`:
  - `app.py`: FastAPI server, REST routes, and WebSocket connection manager.
  - `state_store.py`: In-memory & historical event state store.
- `dashboard/`:
  - `cli_dashboard.py`: Rich TUI terminal live dashboard with color heatmaps and KPI cards.
  - `preview_hud.py`: OpenCV graphical visual overlay window.
- `scripts/`:
  - `mock_event_stream.py`: Standalone mock generator for testing.
