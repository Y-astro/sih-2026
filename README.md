# Edge AI Retail Intelligence Platform (SIH 2026)

An offline-first, edge-native retail intelligence platform combining **Shopper Footfall & Queue Intelligence** and **Shelf Monitoring & Operations Dashboard** with zero raw video transmission and zero PII storage.

---

## 🏛️ System Architecture

```
[Webcam / RTSP Stream / Synthetic Video]
                  │
        ┌─────────┴─────────┐
        ▼                   ▼
[Shopper & Queue AI]   [Shelf & Planogram AI]
• YOLOv8 + ByteTrack   • Shelf Bin Object Detection
• In/Out Line Counter  • Planogram Compliance Check
• Aisle Dwell Time     • Out-of-Stock (OOS) Alerts
• Queue Congestion     • Automated WMS Restock Trigger
• 20x20 Density Matrix └───────────┬───────────┘
        │                          │
        └────────────┬─────────────┘
                     │ Standard JSON Event Stream
                     ▼
       [FastAPI Backend & WebSocket Hub]
       • Ingestion API (`POST /api/v1/events`)
       • Idempotent State Store (UUIDs + UTC ISO Timestamps)
       • WMS / ERP Restock Ticket Dispatcher
       • Real-Time WebSocket Broadcaster (`/ws/feed`)
                     │
                     ▼
       [Operations Dashboard & Visual HUD]
       • Rich CLI Terminal Live TUI
       • Real-Time 20x20 ASCII / ANSI Floor Heatmap
       • Live Occupancy, Queue, & Alert Panels
       • OpenCV Visual HUD Overlay Window
```

---

## ✨ Key Features

- **Privacy-by-Design & Zero PII:** All inference runs on-device. No facial recognition models or biometric embeddings are stored or transmitted. Track IDs reset on exit.
- **Multi-Object Tracking (MOT):** Real-time person tracking with ByteTrack, virtual line-crossing footfall counting (directional entry/exit), and aisle polygon dwell-time measurement.
- **Top-Down Spatial Analytics:** Angle-corrected perspective homography generating a continuous $20 \times 20$ normalized floor-plan density matrix.
- **Queue Intelligence:** Monitors checkout queues in real time, computes wait times using queueing principles (Little's Law approximation), and triggers proactive congestion alerts.
- **Automated Shelf Audit & WMS Dispatch:** Checks shelf bins against planogram configurations, detects zero-stock/low-stock states, and automatically dispatches restock tickets (`WMS-TK-XXXX`).
- **Low-Bandwidth & Offline Resilient:** Transmits only lightweight structured JSON events ($\\approx 1\\text{ KB}$ per payload) rather than streaming video feeds to the cloud.

---

## 🛠️ Prerequisites

- **Python:** 3.10+ (tested on Python 3.12)
- **OS:** Linux, macOS, or Windows
- **Camera:** Standard USB webcam, laptop integrated camera, or RTSP IP camera feed (optional for mock mode)
- **Hardware Acceleration (Optional):** NVIDIA GPU with CUDA for maximum inference FPS; runs efficiently on CPU as well.

---

## 🚀 Quick Start Guide

### 1. Clone the Repository
```bash
git clone https://github.com/Y-astro/sih-2026.git
cd sih-2026
```

### 2. Set Up Virtual Environment & Dependencies

Using standard `venv`:
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

*(Optional) Fast install using `uv`:*
```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### 3. Run the Application

#### Option A: Zero-Camera Mock Simulation (Instant Demo / UI Testing)
Runs the backend, simulates realistic store traffic, queue buildup, out-of-stock events, and displays the interactive terminal dashboard:
```bash
python main.py --mode mock
```

#### Option B: Live Webcam & Full Edge Vision Pipeline
Runs real-time YOLOv8 person tracking and shelf auditing on your webcam with the OpenCV visual HUD and live CLI dashboard:
```bash
python main.py --mode live --camera 0
```

#### Option C: Interactive Visual Store Calibrator (No Hardcoded Numbers!)
Launch the point-and-click visual GUI tool to interactively draw lines, polygons, and shelves on the camera feed:
```bash
python main.py --mode calibrate --camera 0
```
*Controls:*
- **`[1]`**: Click 2 points on screen to place the Entrance Footfall Line.
- **`[2]`**: Click points to draw an Aisle Polygon (`[C]` to close).
- **`[3]`**: Click points to draw a Checkout Queue Polygon (`[C]` to close).
- **`[4]`**: Click & drag a bounding box over a Shelf Tier.
- **`[5]`**: Click 4 floor corners for Bird\'s-Eye Homography calibration.
- **`[A]`**: **AI Auto-Discovery** — automatically clusters visible products into shelf tiers!
- **`[S]`**: Save configuration directly to `config/store_config.json`.
- **`[Q]`**: Exit calibrator.

#### Option D: AI Auto-Discovery on Startup
Point your camera at a shelf or table and automatically discover shelf tiers and target counts:
```bash
python main.py --mode live --camera 0 --auto-shelf
```

#### Option E: Run Server and Dashboard in Separate Terminals
```bash
# Terminal 1: Start FastAPI Backend
python main.py --mode server-only

# Terminal 2: Launch CLI Operations Dashboard
python main.py --mode dashboard-only
```

---

## 📂 Project Structure

```
.
├── config/
│   └── store_config.json      # Aisle zones, checkout polygons, lines & planogram configs
├── schemas/
│   └── events.py              # Pydantic schemas for footfall, dwell, queue, shelf & heatmap
├── team2_shopper/
│   ├── tracker.py             # YOLOv8 + ByteTrack tracking coordinator
│   ├── footfall_engine.py     # Directional virtual line crossing with debouncing
│   ├── dwell_engine.py        # Polygon-based aisle residence duration tracker
│   ├── heatmap_engine.py      # Perspective homography & 20x20 density grid generator
│   └── queue_engine.py        # Checkout occupancy & congestion alert evaluator
├── team3_shelf/
│   └── shelf_engine.py        # Shelf gap/OOS detector & planogram auditor
├── team3_backend/
│   ├── app.py                 # FastAPI server, REST ingestion API & WebSocket broadcaster
│   └── state_store.py         # Live state management & automated WMS ticket dispatch
├── dashboard/
│   ├── cli_dashboard.py       # Rich terminal live TUI dashboard
│   └── preview_hud.py         # OpenCV visual HUD overlay window
├── scripts/
│   └── mock_event_stream.py   # Standalone synthetic store event generator
├── docs/                      # Team briefs and master design plans
├── main.py                    # Unified CLI launcher
└── requirements.txt           # Python dependency specifications
```

---

## 📡 API Reference

- `GET /` — Health check and service status.
- `GET /api/v1/store/snapshot` — Retrieves current occupancy, queue metrics, shelf alert lists, and the $20 \times 20$ density matrix.
- `POST /api/v1/events` — Ingestion endpoint for edge devices and vision pipelines.
- `POST /api/v1/wms/restock-ticket` — Dispatches a restock ticket to store operations.
- `WS /ws/feed` — Persistent WebSocket channel pushing real-time event updates to connected clients.

---

## 📄 License

This project is developed for the Smart India Hackathon (SIH 2026).
