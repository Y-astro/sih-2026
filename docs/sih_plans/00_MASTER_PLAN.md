# Edge AI Retail Intelligence Platform — Master Plan

**Team size:** 6 (2 Hardware + 4 Software) → split into **3 duos of 2**
**Goal:** Offline-first, edge-AI retail system for footfall tracking, shelf monitoring, and queue management — zero PII stored, all inference on-device.

---

## 1. Team Structure

| Duo | Members | Focus | File |
|---|---|---|---|
| **Duo 1 — Hardware/Edge** | 2 Hardware | Camera rigs, edge compute (Jetson/NPU), model optimization & deployment, edge↔cloud sync | `01_hardware_team.md` |
| **Duo 2 — Shopper & Queue Intelligence** | 2 Software | Multi-object tracking, footfall, dwell time, heatmaps, queue/crowd density | `02_shopper_queue_team.md` |
| **Duo 3 — Shelf & Dashboard/Backend** | 2 Software | Shelf detection, planogram compliance, dashboard, alerts, POS/ERP/WMS API integration | `03_shelf_dashboard_team.md` |

**Why this split works:** Hardware duo owns the physical + deployment pipeline that both software duos depend on. The two software duos split cleanly along **"people/queues" vs "shelves/business logic"** — different CV problems (tracking vs. detection/segmentation) and different downstream consumers, so they can work in parallel with minimal blocking. The shared contract between all three is a simple **JSON event schema** (see §4) — agree on this on Day 1 so nobody blocks anyone else.

---

## 2. System Architecture (high level)

```
[Camera(s)] → [Edge Device: Jetson/NPU]
                  ├─ Model 1: Person Detector + Tracker (MOT)  → footfall, dwell, heatmap events
                  ├─ Model 2: Shelf Object Detector/Segmenter  → OOS, low-stock, planogram events
                  ├─ Model 3: Queue/Crowd Density Estimator    → queue length, wait-time events
                  └─ Local Event Bus (MQTT/Redis) — no raw video leaves the device
                        ↓ (only structured JSON events, not video/faces)
              [Sync Agent — buffers offline, pushes when online]
                        ↓
        [Cloud/Server: Ingestion API → DB → Dashboard (WebSocket push) → POS/ERP/WMS connectors]
```

Key principle: **raw video and biometric data never leave the edge device.** Only anonymized structured events (counts, coordinates, timestamps, IDs that reset every session) go upstream.

---

## 3. Timeline (typical 24–36 hr hackathon)

| Phase | Time | Hardware Duo | Software Duo A (Shopper/Queue) | Software Duo B (Shelf/Dashboard) |
|---|---|---|---|---|
| Setup | Hr 0–2 | Unbox, flash Jetson/board, camera mount, network test | Set up repo, env, get sample retail video/dataset | Set up repo, env, get shelf image dataset, dashboard scaffold |
| Prototype | Hr 2–8 | Get a base detector (YOLO) running on-device at target FPS | Build MOT pipeline on laptop (YOLO+ByteTrack) with recorded video | Build shelf detector on laptop; define planogram JSON format |
| Integration v1 | Hr 8–14 | Convert Duo A/B models to TensorRT/ONNX/TFLite, deploy on edge | Add dwell-time + heatmap logic, define event schema | Build dashboard skeleton + WebSocket server; mock data first |
| Integration v2 | Hr 14–22 | Wire local event bus (MQTT), offline buffering, sync agent | Add queue/crowd density module; test end-to-end on edge feed | Connect dashboard to real events; build alert thresholds |
| Polish & Test | Hr 22–30 | Latency benchmarking, stress test offline→online switch | Accuracy validation, tune trackers, fix ID-switch bugs | Mock POS/ERP/WMS integration, polish UI, KPI charts |
| Demo Prep | Last 4–6 hrs | Live demo rig ready, fallback recorded video | Rehearse tracking demo | Rehearse dashboard + alert demo |

---

## 4. Shared Event Schema (agree on Day 1 — critical!)

All three duos must emit/consume events in this shape so integration doesn't break at the last minute:

```json
{
  "event_type": "footfall | dwell | heatmap | shelf_oos | shelf_lowstock | planogram_violation | queue_state",
  "store_id": "store_001",
  "zone_id": "aisle_3 | entrance | checkout_2",
  "timestamp": "2026-08-28T10:15:32Z",
  "anonymous_track_id": "sess_7f3a",   // resets every session, NOT a persistent identity
  "payload": { "...": "event-specific data, e.g. bbox, count, sku_zone, queue_length" },
  "confidence": 0.93
}
```

No face embeddings, no persistent cross-visit IDs, no images stored — only this JSON.

---

## 5. Core Concepts Every Team Member Should Know (overview)

Each duo file has its own deep-dive, but everyone on the team should be able to explain these at a high level for Q&A:

1. **Edge AI vs Cloud AI** — why on-device inference (latency, privacy, bandwidth resilience) vs. cloud (compute power, easy scaling). Trade-offs.
2. **Object Detection basics** — bounding boxes, YOLO family, confidence/NMS, mAP as a metric.
3. **Multi-Object Tracking (MOT)** — detection ≠ tracking; tracking-by-detection paradigm; ID persistence and ID switches.
4. **Semantic Segmentation** — pixel-level classification (used for shelf gap/space detection).
5. **Model Optimization** — quantization (FP32→INT8), pruning, ONNX/TensorRT/TFLite as deployment formats.
6. **Privacy-by-Design** — anonymization techniques (no facial recognition, ephemeral IDs, on-device processing, differential privacy concepts).
7. **Heatmaps & Spatial Analytics** — homography/perspective transform (camera view → floor-plan/bird's-eye view), density grids.
8. **Queueing Theory basics** — arrival rate, service rate, wait time estimation (even a simplified Little's Law explanation impresses judges).
9. **System Design** — event-driven architecture, message queues (MQTT), offline-first sync patterns, WebSockets for real-time push.
10. **Compliance** — GDPR/DPDP-style "privacy by design," data minimization principle.

---

## 6. Critical Cross-Team Questions (judges often ask these to the whole team)

1. **"How do you guarantee zero PII, and how would a judge/auditor verify that?"**
   → Emphasize: no facial recognition model in pipeline at all (not just "disabled"), only bounding boxes + anonymous session IDs that reset, no video ever leaves the device, schema is auditable JSON with no image data.

2. **"What happens when the network goes down for 2 hours — do you lose data?"**
   → Local buffering (SQLite/local queue) on edge device, sync agent retries with exponential backoff, no data loss, dashboard shows "stale/offline" indicator.

3. **"How does this differ from just buying an off-the-shelf people-counting camera?"**
   → Multi-modal (shopper + shelf + queue) in one unified platform, edge-native (no subscription cloud dependency), open integration with POS/ERP/WMS, and combines three intelligence layers into one KPI dashboard rather than three separate point solutions.

4. **"What's your accuracy and how did you measure it?"**
   → Be ready with actual numbers: precision/recall on your test video, or at minimum a manually-counted ground truth comparison ("we manually counted 50 people crossing the line, model got X correct").

5. **"Why edge and not cloud — isn't cloud more accurate/scalable?"**
   → Latency (<1s required for real-time alerts), bandwidth cost at scale (multi-store video streaming is expensive), privacy regulation (raw video shouldn't leave premises), and resilience (store operations shouldn't stop if internet drops).

6. **"How would this scale to 100 stores?"**
   → Fleet management: each edge device is autonomous, only sends lightweight events centrally, central dashboard aggregates — bandwidth stays flat regardless of store count since no video is transmitted.

7. **"What's your biggest technical limitation right now?"**
   → Be honest: e.g., occlusion in dense crowds causing ID switches, or lighting variation affecting shelf detection. Judges respect honesty + a stated next step more than false confidence.

---

## 7. Final Deliverable Checklist

- [ ] Working edge deployment (even if on a laptop simulating Jetson, have real optimized model files: `.onnx`/`.tflite`/`.engine`)
- [ ] Live or recorded demo of footfall + heatmap
- [ ] Live or recorded demo of shelf OOS detection
- [ ] Live or recorded demo of queue length estimation
- [ ] Dashboard showing all 3 modules with at least one real-time alert firing
- [ ] One slide explicitly on privacy architecture (no PII diagram)
- [ ] One slide on latency numbers (measured, not estimated)
- [ ] One slide on architecture diagram (edge → sync → cloud → dashboard)
- [ ] Fallback video recording in case live demo fails (ALWAYS have this)

Good luck — read your duo-specific file next for the deep technical breakdown.
