# Duo 1 — Hardware & Edge Deployment Team

**Members:** 2 Hardware engineers
**Mission:** Own everything from "camera sees pixels" to "optimized model runs on-device under 1s latency" to "events sync to cloud reliably even offline." You are the foundation both software duos build on top of.

---

## 1. Topics & Concepts You Must Understand

### 1.1 Edge Compute Hardware
- **What "edge AI" means:** running inference physically near the camera (on a Jetson, NPU-enabled gateway, or even a laptop GPU for demo) instead of sending video to the cloud.
- **Hardware options** (know trade-offs of whichever you use):
  - **NVIDIA Jetson (Nano/Orin/Xavier)** — GPU-accelerated, supports TensorRT, best CV ecosystem support. Most common hackathon choice.
  - **Google Coral / Edge TPU** — very low power, needs TFLite models, less flexible.
  - **Intel NPU / OpenVINO-based devices** — good CPU-class inference, ONNX-friendly.
  - **Raspberry Pi + USB accelerator** — budget option, weaker compute.
  - If no real edge board is available, simulate with a **laptop with GPU** and explicitly state "this represents Jetson-class edge compute" in the demo.
- **Camera setup:** USB webcam / IP camera (RTSP stream) / CSI camera module. Understand **field of view, mounting height/angle** (top-down for queue/footfall counting reduces occlusion; angled for shelf monitoring).

### 1.2 Model Optimization & Deployment Formats
This is your most important technical contribution — know it cold.
- **ONNX (Open Neural Network Exchange):** universal intermediate format; software team trains in PyTorch → export to `.onnx`.
- **TensorRT:** NVIDIA's inference optimizer; converts ONNX → `.engine` file, applies layer fusion, kernel auto-tuning, and precision calibration. Gives biggest speedup on Jetson.
- **TFLite:** Google's lightweight format for mobile/edge (Coral, Android). Needs post-training quantization.
- **Quantization** (know this well, it WILL be asked):
  - FP32 (full precision, default training) → FP16 (half precision, ~2x speedup, minimal accuracy loss) → INT8 (4x smaller/faster, needs calibration dataset, some accuracy loss).
  - **Post-training quantization (PTQ)** vs **Quantization-aware training (QAT)** — PTQ is what you'll realistically do in a hackathon.
- **Pruning:** removing redundant weights/channels to shrink model size (mention conceptually, likely won't implement in hackathon timeframe).
- **Model choice matters:** prefer lightweight architectures — YOLOv8n/YOLOv11n (nano variants), MobileNet-SSD, or NanoDet over heavy models like YOLOv8x — for real edge feasibility.

### 1.3 Real-Time Inference Pipeline
- **Frame sampling:** you don't need to run inference on every single frame (e.g., process 1 in every 3-5 frames for tracking tasks) to hit latency targets.
- **Batching vs streaming inference** — for real-time camera feeds you use streaming (1 frame at a time), not batch.
- **Latency budget:** target <1s end-to-end (capture → preprocess → inference → postprocess → event emit). Measure and log each stage separately so you can show a breakdown.
- **Multi-model pipelines on one device:** if running person-detector + shelf-detector + queue-model concurrently, consider:
  - Running them on **separate time-sliced zones/cameras** rather than all models on all frames.
  - Using a **shared backbone** if models are similar architectures (advanced — mention as future work if not implemented).

### 1.4 Local Event Bus & Offline-First Sync
- **MQTT** (lightweight pub/sub, ideal for IoT/edge) or **Redis Streams** — local device publishes structured JSON events (see Master Plan schema) to a local broker.
- **Local buffering:** SQLite or a local file queue stores events when internet is down.
- **Sync agent design:**
  - Periodically checks connectivity.
  - Pushes buffered events to cloud ingestion API in batches when online.
  - Exponential backoff retry logic.
  - Idempotent writes (use event UUIDs) so re-sending doesn't duplicate data.
- **Bandwidth resilience:** you're only ever sending small JSON events (a few KB), never video — this is the core architectural reason the system works on poor connectivity.

### 1.5 Privacy-by-Design at the Hardware Layer
- No raw video frame or image crop should ever be written to disk persistently or transmitted — only numeric outputs (bounding boxes, counts, embeddings-free track IDs).
- If frames must be buffered briefly in memory for tracking continuity, ensure they're **never persisted to disk** and are discarded immediately after inference.
- Session-based anonymous IDs: generate a random ID per detected track that has no link to identity and resets between sessions/store visits.

---

## 2. Implementation Plan (Step-by-Step)

### Phase 1 — Setup (Hr 0–2)
1. Flash/prepare your edge device OS (JetPack for Jetson, or set up laptop CUDA environment as fallback).
2. Verify camera feed works: `v4l2-ctl --list-devices` or test RTSP stream with OpenCV/ffmpeg.
3. Install base inference runtime: PyTorch (for initial testing), then TensorRT/ONNX Runtime.
4. Agree on the **event JSON schema** with both software duos (see Master Plan §4) — do this FIRST, before anyone writes integration code.

### Phase 2 — Get a Baseline Model Running (Hr 2–8)
1. Take a pretrained lightweight detector (YOLOv8n) and get it running raw (PyTorch) on the edge device — just to confirm the hardware pipeline works end-to-end (camera → model → console output).
2. Measure baseline FPS/latency. This is your reference point to show improvement later.

### Phase 3 — Receive Models from Software Duos & Optimize (Hr 8–16)
1. Software Duo A hands you their trained/fine-tuned person-tracking model (PyTorch/.pt).
2. Software Duo B hands you their shelf-detection model.
3. For each:
   - Export to ONNX: `torch.onnx.export(...)`.
   - Validate ONNX output matches PyTorch output (sanity check on a few frames).
   - Convert ONNX → TensorRT engine (`trtexec` tool) with FP16 (and INT8 if time permits with a small calibration set).
   - Benchmark: record latency per stage (preprocess/inference/postprocess), FPS, and accuracy delta vs. original model.

### Phase 4 — Wire the Local Event Bus (Hr 14–20)
1. Set up local MQTT broker (Mosquitto) on the edge device.
2. Each model's postprocessing script publishes structured events to relevant MQTT topics (`store/zone/footfall`, `store/zone/shelf`, `store/zone/queue`).
3. Build the sync agent: subscribes locally, buffers to SQLite, pushes to cloud API when online, handles retry/backoff.

### Phase 5 — Integration Testing (Hr 20–26)
1. Run all models concurrently (or time-sliced) on live/recorded camera feed.
2. Kill network mid-run — confirm events buffer locally and don't get lost.
3. Restore network — confirm buffered events flush to cloud and dashboard updates.
4. Record final latency numbers for the presentation (per-module and end-to-end).

### Phase 6 — Demo Readiness (Hr 26–30)
1. Prepare a **live demo** (camera pointed at a mock aisle/queue) AND a **backup recorded video** in case live hardware fails during presentation — always have both.
2. Prepare a simple architecture diagram slide showing: Camera → Edge Device (models + optimization) → Local Bus → Sync Agent → Cloud → Dashboard.

---

## 3. Suggested Tech Stack

| Layer | Tool |
|---|---|
| Edge board | NVIDIA Jetson (or GPU laptop as fallback) |
| Camera | USB webcam or RTSP IP camera |
| Base inference | PyTorch → ONNX Runtime → TensorRT |
| Alternative lightweight path | TFLite (if using Coral/CPU-only device) |
| Local message bus | Mosquitto (MQTT) or Redis |
| Local buffer/storage | SQLite |
| Sync agent | Python script with `paho-mqtt` + `requests`, retry via `tenacity` |
| Monitoring | Simple Python logging + timestamps for latency benchmarking |

---

## 4. Critical Questions You Might Be Asked

1. **"What's your actual measured end-to-end latency, and how did you measure it?"**
   → Have real numbers: timestamp at frame capture vs. timestamp at event published. Show a breakdown table (capture, preprocess, inference, postprocess).

2. **"Why TensorRT/ONNX instead of just running PyTorch directly?"**
   → PyTorch models are unoptimized for inference — TensorRT applies layer fusion, precision calibration (FP16/INT8), and kernel auto-tuning, giving typically 2–5x speedup, which is necessary to hit the <1s latency target on edge hardware with limited compute vs. a cloud GPU.

3. **"What accuracy did you lose by quantizing to FP16/INT8?"**
   → Be honest with a number (e.g., "we saw ~1-2% mAP drop with FP16, acceptable for our 90% precision target"). If you didn't measure this, say what you'd measure with more time.

4. **"How do you handle multiple models running on one edge device without exceeding compute budget?"**
   → Explain your approach: time-slicing across zones/cameras, running lighter nano-architectures, or dedicating separate compute resources if using multi-camera setup.

5. **"What happens if the camera or edge device physically fails?"**
   → Store operations continue on POS/manual processes; system should fail gracefully (alert on the dashboard "device offline," not crash silently). Mention this as a resilience consideration even if not fully implemented.

6. **"Could someone intercept the local events and reconstruct identity from them?"**
   → No — events contain only bounding box coordinates, counts, and ephemeral session IDs, never facial features, images, or any biometric embedding. There is nothing to reverse-engineer identity from.

7. **"How would you scale this to multiple cameras per store or multiple stores?"**
   → Each device is autonomous and only transmits lightweight JSON, so bandwidth cost stays flat regardless of number of stores; multiple cameras per store just means multiple edge devices (or a multi-stream pipeline) publishing to the same local/central event schema.

8. **"Why not just use the cloud since your internet is stable during this hackathon?"**
   → Real retail stores have inconsistent connectivity (basements, remote locations, ISP outages) and pushing continuous video to cloud is bandwidth-expensive and privacy-risky at scale — edge-first is a deliberate architecture choice, not a hackathon shortcut.
