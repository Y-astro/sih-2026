# Duo 3 — Shelf & Inventory Monitoring + Operations Dashboard Team

**Members:** 2 Software engineers
**Mission:** Detect shelf gaps / out-of-stock (OOS) items and planogram violations, and build the real-time operations dashboard that visualizes all three modules' events, fires alerts, and demonstrates POS/ERP/WMS integration.

---

## Part A: Shelf & Inventory Monitoring

### 1. Topics & Concepts You Must Understand

#### 1.1 Object Detection for Shelves
- Same base concept as Duo 2 (YOLO-family detection), but here the classes are **products/SKUs or "empty space"** rather than "person."
- **Two viable approaches** — pick based on time and dataset availability:
  1. **Product/gap detection (recommended for hackathon):** train/fine-tune a detector to recognize "empty shelf space" vs "stocked shelf space" directly — simpler, doesn't need per-SKU labels. Public datasets like **SKU-110K** (dense retail shelf images) are useful for pretraining or as a demo dataset.
  2. **Per-SKU detection/classification (advanced):** detect individual products and classify by SKU — much harder, needs a labeled dataset per product, not realistic to train from scratch in a hackathon. If attempted, keep to a handful (3-5) of distinct product classes.

#### 1.2 Semantic Segmentation (for shelf-gap precision)
- Unlike bounding boxes, **segmentation** classifies every pixel (e.g., "shelf-with-product" vs "empty-shelf" vs "background/shelf-edge").
- Useful because shelf gaps are irregular shapes, not naturally box-shaped — a segmentation mask gives you % of shelf-space empty, which is a much stronger OOS signal than a bounding box count.
- Lightweight segmentation options: **U-Net (small)**, or simpler — a two-class semantic segmenter fine-tuned from a pretrained backbone (e.g., MobileNetV3 encoder). If time-constrained, a **classification-based heuristic** (e.g., color/edge-density difference between "product texture" and "flat empty shelf background") can be a scrappy but explainable fallback — mention this trade-off openly to judges.

#### 1.3 Planogram Compliance
- **Planogram** = the retailer's intended layout — which product should be in which shelf position.
- **Compliance auditing approach:**
  1. Define the "expected planogram" as a simple config (JSON): `{"shelf_zone_1": "product_A", "shelf_zone_2": "product_B", ...}`.
  2. Run detection/classification on each zone; compare detected product/class against expected.
  3. Flag violations: wrong product in zone, or zone empty when it shouldn't be.
- For a hackathon demo, 3-5 shelf zones with clear visual product differences is enough to convincingly demonstrate this.

#### 1.4 Missing SKU Alerts & Replenishment Notifications
- Business logic layer on top of detection: if a defined shelf zone has been below a stock threshold (e.g., >60% empty) for more than X minutes, generate a `shelf_lowstock` or `shelf_oos` event with `zone_id`, `severity`, and `duration`.
- This becomes a **staff notification** — surfaced on the dashboard as an actionable alert (see Part B).

---

### 2. Implementation Plan — Shelf Module (Step-by-Step)

#### Phase 1 — Setup (Hr 0–2)
1. Gather sample shelf images — search public datasets (e.g., SKU-110K, or simply take photos of a real shelf/bookshelf/pantry at the venue and stage "empty" vs "stocked" states for a controllable demo).
2. Set up detection env: `ultralytics` (YOLOv8) again works fine for "product region" vs "gap region" detection.

#### Phase 2 — Gap Detection Baseline (Hr 2–10)
1. Label a small dataset (even 30–50 images, staged shelf photos work) with two classes: `stocked` and `empty_gap`, using a quick annotation tool (Roboflow, LabelImg, or CVAT).
2. Fine-tune YOLOv8n on this small dataset (transfer learning from COCO pretrained weights — fast, few epochs needed for a hackathon-scale dataset).
3. Validate visually: run on held-out staged shelf photos, confirm gaps are detected.

#### Phase 3 — Zone Definition + Planogram Config (Hr 10–16)
1. Define shelf zones as polygons/regions in the camera frame (similar to Duo 2's zone approach).
2. Write the planogram config JSON (expected product per zone).
3. Implement compliance check logic: per zone, compare detected class vs expected; emit `planogram_violation` or `shelf_oos`/`shelf_lowstock` events per the shared schema.

#### Phase 4 — Alert Logic (Hr 16–20)
1. Add time-based thresholding (avoid false alerts from a single bad frame — require N consecutive detections or T seconds before firing an alert).
2. Emit structured alert events to the local event bus for the dashboard to consume.

#### Phase 5 — Integration (Hr 20–26)
1. Hand off trained model to Hardware Duo for ONNX/TensorRT conversion.
2. Test end-to-end: staged shelf demo (physically remove an item during demo) → alert appears on dashboard within seconds.

---

## Part B: Operations Dashboard & System Interoperability

### 3. Topics & Concepts You Must Understand

#### 3.1 Real-Time Data Push
- **WebSockets:** persistent bi-directional connection between backend and dashboard frontend — used so the dashboard updates instantly when a new event arrives (footfall count changes, alert fires) instead of the browser having to poll repeatedly.
- **Push notifications** (mention conceptually): for staff mobile alerts, e.g., Firebase Cloud Messaging — likely out of scope to fully implement, but good to mention as production path.

#### 3.2 Backend/Ingestion API
- Simple REST API (FastAPI/Flask/Express) that:
  - Receives buffered events from the Hardware Duo's sync agent (`POST /events`).
  - Stores them in a database (even SQLite/Postgres is fine for a hackathon).
  - Broadcasts new events to connected dashboard clients via WebSocket.

#### 3.3 Dashboard Frontend
- Real-time KPI panels: live footfall count, current queue length per counter, active alerts list, shelf compliance %, aisle heatmap image (from Duo 2).
- **Aggregate analytics**: hourly footfall trend chart, conversion-adjacent metrics (e.g., dwell time vs. footfall as a rough "engagement" proxy), if time allows.
- Suggested stack: React + a chart library (Recharts/Chart.js) + native WebSocket client, or even a simpler Streamlit dashboard if frontend time is limited (faster to build, still visually convincing for a demo).

#### 3.4 Multi-Store Fleet Management (concept-level, likely a "future work" slide unless time allows)
- Each store's edge device has a `store_id`; the dashboard backend simply groups/filters events by `store_id`.
- Demonstrating this with even 2 simulated `store_id`s (e.g., replaying two video feeds tagged differently) is an easy way to show "fleet-readiness" without needing real multi-store hardware.

#### 3.5 POS / ERP / WMS Integration
- You almost certainly cannot integrate with a real POS/ERP/WMS system in a hackathon — **mock/simulate it**:
  - Build a small mock REST endpoint representing "WMS replenishment API" that receives your `shelf_lowstock` events and returns a mock "restock ticket created" response.
  - This demonstrates the **integration pattern** (webhook/API call on alert) without needing a real enterprise system — be upfront that it's a mock/simulation in the demo, judges respect honesty about scope.

### 3.6 Privacy/Compliance on the Dashboard Side
- Dashboard should only ever display aggregate numbers, zone-level heatmaps, and anonymous alert events — never raw video or any personally identifying visual.
- If you show a demo video feed for visual effect, blur/anonymize it or clearly caption it as "illustrative only, not stored."

---

### 4. Implementation Plan — Dashboard/Backend (Step-by-Step)

#### Phase 1 — Setup (Hr 0–2)
1. Scaffold backend (FastAPI recommended — easy WebSocket support) and frontend (React or Streamlit).
2. Define REST endpoint `POST /events` matching the shared schema; define WebSocket channel for broadcasting.

#### Phase 2 — Mock Data First (Hr 2–10)
1. Before real model events exist, write a script that generates **fake but realistic events** matching the schema and pushes them to your backend — build and demo the UI against this while other modules are still in progress. This unblocks you completely from waiting on Duo 2/3's model progress.

#### Phase 3 — Core Dashboard UI (Hr 10–18)
1. Build live KPI cards: footfall count, current queue length, active alert count.
2. Build alert feed (list of recent `shelf_oos`, `queue_congestion`, etc. events with timestamps).
3. Build heatmap image display panel (renders the periodically-generated heatmap image from Duo 2).

#### Phase 4 — Threshold Alerts + Mock Integration (Hr 18–24)
1. Backend logic: when `shelf_lowstock` event received, call the mock WMS endpoint automatically → log "restock ticket created" and display on dashboard.
2. Same pattern for queue congestion → mock "open counter 3" recommendation displayed as an actionable suggestion.

#### Phase 5 — Real Integration (Hr 24–28)
1. Swap mock event generator for real events coming from Hardware Duo's sync agent.
2. End-to-end test: trigger a real event (walk past camera / remove shelf item) → confirm it appears on dashboard within a few seconds.

#### Phase 6 — Polish (Hr 28–30)
1. Add charts (hourly footfall trend), clean up UI, add "store offline/stale data" indicator (ties back to Hardware Duo's offline-sync work).

---

## 5. Suggested Tech Stack

| Task | Tool |
|---|---|
| Shelf detection | YOLOv8n fine-tuned (Ultralytics), Roboflow/LabelImg for annotation |
| Segmentation (optional/advanced) | Small U-Net or MobileNetV3-based segmenter |
| Backend API | FastAPI (Python) — native async + WebSocket support |
| Database | SQLite (hackathon-scale) or Postgres |
| Frontend | React + Recharts, or Streamlit for speed |
| Real-time push | WebSockets (native or `socket.io`) |
| Mock POS/ERP/WMS | Simple Flask/FastAPI mock endpoint returning canned JSON responses |

---

## 6. Critical Questions You Might Be Asked

1. **"How do you detect a shelf gap without knowing every possible product?"**
   → Explain the two-class approach (stocked vs. empty-gap) doesn't require per-SKU labels — it's a generic texture/space classification problem, so it generalizes across product types without retraining per-SKU.

2. **"What if lighting changes or shadows make an empty shelf look 'stocked'?"**
   → Acknowledge this as a real failure mode; mitigation is more diverse training data across lighting conditions, and segmentation-based approaches are more robust than simple heuristics — state what you'd add with more time.

3. **"Is your planogram compliance actually checking product identity, or just presence/absence?"**
   → Be precise about what you actually implemented — if it's presence/absence only (simpler), say so, and describe per-SKU classification as the natural extension.

4. **"Your POS/ERP/WMS integration — is that a real system or simulated?"**
   → Be upfront: it's a mock endpoint demonstrating the integration pattern (webhook-triggered ticket creation), since integrating a real enterprise system isn't feasible in a hackathon timeframe — the architecture is designed to plug into a real API the same way.

5. **"How do you avoid false alerts — e.g., a customer's hand briefly blocking the shelf triggering a false OOS alert?"**
   → Explain your time/consecutive-frame thresholding logic (require sustained detection over N seconds before firing an alert) specifically to filter out such transient false positives.

6. **"How does the dashboard stay useful if the store goes offline for hours?"**
   → Dashboard shows a clear "stale data / device offline" indicator (last-synced timestamp) rather than silently showing outdated numbers as if they were live — ties to Hardware Duo's offline buffering.

7. **"Why WebSockets instead of just refreshing/polling the dashboard every few seconds?"**
   → WebSockets give true real-time push with lower overhead — critical for time-sensitive alerts like queue congestion where even a 5-10 second polling delay could mean a missed staffing window; also reduces unnecessary network requests, relevant given the bandwidth-conscious design of the whole system.

8. **"How would multi-store fleet management actually work at scale?"**
   → Each event already carries a `store_id`; backend/dashboard simply filters/aggregates by that field — no architectural change needed to go from 1 store to 100, only additional edge devices publishing to the same ingestion API.
