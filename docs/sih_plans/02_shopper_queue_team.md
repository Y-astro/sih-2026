# Duo 2 — Shopper Analytics, Footfall & Queue Intelligence Team

**Members:** 2 Software engineers
**Mission:** Detect and track people (anonymously) to produce footfall counts, dwell time, movement heatmaps, and queue/wait-time estimates — all without storing any identity information.

---

## 1. Topics & Concepts You Must Understand

### 1.1 Object Detection (foundation)
- **Bounding box detection:** model outputs `(x, y, w, h, class, confidence)` per detected object.
- **YOLO family (You Only Look Once):** single-shot detector, predicts boxes + classes in one forward pass — fast enough for real-time. Use **YOLOv8n / YOLOv11n** (nano = lightweight, edge-friendly).
- **NMS (Non-Max Suppression):** removes duplicate overlapping boxes for the same object.
- **mAP (mean Average Precision):** standard detection accuracy metric — know how to explain it simply: "how well predicted boxes overlap with true boxes across confidence thresholds."
- For this module you only need **one class: "person."** Use a pretrained COCO model (already includes "person") and fine-tune only if your specific camera angle/lighting needs it.

### 1.2 Multi-Object Tracking (MOT) — the core of your module
- **Detection ≠ Tracking.** Detection finds people in a single frame; tracking links the *same* person across frames over time by assigning a persistent (session-only) ID.
- **Tracking-by-detection paradigm:** run detector every frame (or every N frames) → match new detections to existing tracks.
- **Key algorithms to know:**
  - **SORT (Simple Online Realtime Tracking):** Kalman filter (motion prediction) + Hungarian algorithm (assignment) based on IoU overlap. Fast but loses tracks easily during occlusion.
  - **DeepSORT:** adds an appearance embedding (small CNN) to re-identify people after brief occlusion — better ID consistency, still no facial recognition (uses generic visual appearance, not identity).
  - **ByteTrack:** current best-practice, associates even low-confidence detections, handles occlusion very well, no deep appearance model needed (lighter weight — good for edge). **Recommended for this project.**
- **ID switches** — when the tracker mistakenly swaps IDs between two people (e.g., after they cross paths). This is your main accuracy failure mode — be ready to discuss it.
- **Re-entry problem:** if a person leaves camera view and re-enters, should they get a new ID or the same one? For footfall counting (not identity tracking), a **new session ID on re-entry is actually fine and privacy-preserving** — you're not trying to re-identify the same shopper across time.

### 1.3 Footfall Counting (Entry/Exit)
- **Virtual line-crossing method:** define a virtual line in the frame (e.g., at the store entrance); when a tracked person's centroid crosses the line, increment entry or exit counter based on direction of crossing.
- Track direction using the sign of movement relative to the line (e.g., top-to-bottom = entry, bottom-to-top = exit).
- **Common accuracy issues:** people walking side-by-side (occlusion undercounts), people lingering exactly at the line (double-counting) — mitigate with a debounce/cooldown per track ID.

### 1.4 Dwell Time
- Simple concept once tracking works: **dwell time = (last_seen_timestamp − first_seen_timestamp)** for a track ID within a defined zone (e.g., an aisle polygon).
- Define **zones as polygons** on the frame (or better, on a top-down floor-plan after homography transform — see below); check if a track's centroid is inside a zone each frame.

### 1.5 Heatmaps & Spatial Transformation
- **Homography / perspective transform:** camera view is angled, but a heatmap should represent the actual floor layout (bird's-eye view). Use OpenCV's `cv2.getPerspectiveTransform` with 4 known reference points (e.g., store corners) to map pixel coordinates → floor-plan coordinates.
- **Density heatmap construction:** accumulate every tracked centroid's floor-plan coordinate over time into a 2D histogram/grid, then apply Gaussian blur and a color map (e.g., using `matplotlib`/`seaborn` or OpenCV's `applyColorMap`) to visualize traffic density per aisle/zone.
- **Zonal traffic density:** count of unique track IDs per zone per time window (e.g., per 15 minutes).

### 1.6 Queue & Crowd Density Estimation
- Two approaches, pick based on time available:
  1. **Detection + counting in a zone (simpler, recommended):** define a "queue zone" polygon near checkout; count number of person-detections whose centroid falls inside it → that's the queue length. Combine with tracking to avoid double counting the same person across frames.
  2. **Crowd density estimation (advanced):** for very dense crowds where individual detection fails, use density-map regression models (e.g., CSRNet-style) that predict a density heatmap and integrate it to get a people count — mention as the "next step for very high-density scenarios" if you don't implement it.
- **Wait/service time estimation:** track how long each person's ID stays inside the queue zone before disappearing (approximates service time) — same dwell-time logic reused.
- **Queueing theory (mention conceptually, impresses judges):**
  - **Little's Law:** `L = λ × W` (average number in queue = arrival rate × average wait time). You can use this to *sanity check* your measured wait time against observed arrival rate and queue length.
  - **Predictive congestion alerting:** simple threshold-based rule first (e.g., "if queue length > 8 for more than 2 minutes, alert"), and if time permits, a lightweight moving-average/linear trend forecast to predict when a queue *will* cross threshold soon (proactive, not just reactive).

### 1.7 Privacy Techniques Specific to This Module
- Never store face crops or run any facial recognition/embedding model.
- Track IDs are **random session-scoped identifiers**, reset when a person leaves the frame for good — never linked across store visits or cameras.
- Only emit aggregate/structured events (counts, zone entries, coordinates) — never raw frames — to the event bus (per Master Plan schema).

---

## 2. Implementation Plan (Step-by-Step)

### Phase 1 — Setup (Hr 0–2)
1. Get a sample retail/store camera video (search for public retail surveillance sample datasets, or record your own walking test video in the hackathon venue).
2. Set up Python env: `ultralytics` (YOLOv8), `opencv-python`, and a tracker — easiest is `pip install ultralytics` which has ByteTrack built in (`model.track(source=..., tracker="bytetrack.yaml")`).

### Phase 2 — Detection + Tracking Baseline (Hr 2–8)
1. Run YOLOv8n person detection + ByteTrack on your sample video — confirm each person gets a stable ID across frames.
2. Visualize: draw bounding boxes + ID labels on output video to sanity-check tracking quality.
3. Note and log ID switch occurrences to later report an accuracy estimate.

### Phase 3 — Footfall Counting (Hr 8–12)
1. Define a virtual entry/exit line (hardcode pixel coordinates for your camera setup).
2. Implement line-crossing logic with direction detection + per-track cooldown to prevent double count.
3. Validate against manual count on a test clip — record precision (e.g., "counted 48/50 correctly = 96%").

### Phase 4 — Dwell Time + Zones (Hr 12–16)
1. Define 2–3 zone polygons (e.g., "Aisle 1," "Aisle 2," "Checkout Approach").
2. Track first-seen/last-seen timestamps per track ID per zone → compute dwell time.
3. Emit `dwell` events matching the shared schema.

### Phase 5 — Heatmap (Hr 16–20)
1. Pick 4 reference points in your camera frame and corresponding floor-plan coordinates; compute homography matrix.
2. Transform all tracked centroids over the session into floor-plan space, accumulate into a grid, render as a heatmap image (this can be a nice static/periodic image sent to the dashboard, doesn't need to be per-frame).

### Phase 6 — Queue Module (Hr 20–26)
1. Define a queue-zone polygon near a mock checkout counter.
2. Count person-detections/tracks inside the zone per frame → queue length signal.
3. Compute wait time using dwell-time logic reused from Phase 4, scoped to the queue zone.
4. Implement simple threshold-based congestion alert (`queue_length > N for > T seconds → alert event`).

### Phase 7 — Integration (Hr 26–30)
1. Package all outputs as JSON events matching the Master Plan schema; hand off model files to Hardware Duo for ONNX/TensorRT conversion.
2. Test on the edge device with Hardware Duo; verify FPS/latency doesn't break tracking continuity (if frame sampling is too sparse, tracking quality drops — coordinate on the sampling rate).

---

## 3. Suggested Tech Stack

| Task | Tool |
|---|---|
| Detection | YOLOv8n/YOLOv11n (Ultralytics) |
| Tracking | ByteTrack (built into Ultralytics `.track()`) or DeepSORT |
| Homography/heatmap | OpenCV (`getPerspectiveTransform`, `warpPerspective`), NumPy, Matplotlib |
| Zone/polygon checks | Shapely (`Polygon.contains(point)`) or OpenCV `pointPolygonTest` |
| Event publishing | Python `paho-mqtt` client → local broker (from Hardware Duo) |

---

## 4. Critical Questions You Might Be Asked

1. **"How is this different from facial recognition — how do you prove no identity is captured?"**
   → The pipeline only ever outputs bounding box coordinates and a randomly generated session ID; there is no facial embedding model anywhere in the pipeline, and IDs are discarded/reset once a person leaves frame — there's no data structure that could be reverse-mapped to a person's identity.

2. **"What happens when two people walk close together or cross paths — does your count break?"**
   → Explain ID switches honestly: occlusion can cause a brief tracking error; ByteTrack mitigates this better than plain SORT by re-associating low-confidence detections, but it's not perfect — state your measured error rate.

3. **"How do you calculate dwell time — what if someone briefly leaves and re-enters a zone?"**
   → Explain your first-seen/last-seen approach and mention the trade-off — if a track ID resets on re-entry (privacy-preserving design), that appears as two separate dwell events rather than one continuous one; state this as a known trade-off between privacy and metric perfection.

4. **"How does the heatmap actually get built — is it live or aggregated?"**
   → Explain homography transform (angled camera → floor plan) and that it's built from an accumulation of tracked positions over a time window (e.g., hourly), not literally rendered every frame.

5. **"Your queue detection — does it work in a genuinely crowded/dense queue where people fully overlap?"**
   → Be honest that simple detection-based counting degrades in very dense crowds due to occlusion; mention density-estimation models (e.g., CSRNet-style crowd counting) as the documented next step for high-density scenarios.

6. **"What's your actual precision number for footfall counting, and how did you validate it?"**
   → Have a real number from manual ground-truth comparison on a test clip — judges specifically listed >90% precision as a KPI, so this must be answerable with a real (even if small-sample) number.

7. **"Why not just use a dedicated crowd-counting/density model everywhere instead of tracking?"**
   → Tracking-based counting gives you accurate directional footfall (in/out) and per-person dwell time — density models only give you aggregate counts, they can't tell you "did this person just enter" or "how long did they stay," which are core requirements here.
