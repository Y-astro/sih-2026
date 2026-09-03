import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# Auto-detect and switch to .venv if running with system python that lacks dependencies
venv_python = os.path.join(
    PROJECT_ROOT,
    ".venv",
    "Scripts" if sys.platform.startswith("win") else "bin",
    "python.exe" if sys.platform.startswith("win") else "python"
)
if os.path.exists(venv_python) and os.path.abspath(sys.executable) != os.path.abspath(venv_python):
    try:
        import cv2
        import ultralytics
    except ImportError:
        # Re-execute process inside the virtual environment seamlessly
        if sys.platform.startswith("win"):
            import subprocess
            res = subprocess.call([venv_python] + sys.argv)
            sys.exit(res)
        else:
            os.execv(venv_python, [venv_python] + sys.argv)

import argparse
import time
import threading
import json

# Ensure Qt compatibility on Linux Wayland/X11
if not sys.platform.startswith("win"):
    os.environ["QT_QPA_PLATFORM"] = "xcb"

try:
    import cv2
    import requests
    import uvicorn
except ImportError as e:
    print(f"\n[Environment Error] Missing required package: {e}")
    print("Please activate your virtual environment before running:")
    print("  Linux/macOS: source .venv/bin/activate")
    print("  Windows:     .venv\\Scripts\\activate\n")
    sys.exit(1)

from team2_shopper.tracker import ShopperTrackerPipeline
from team3_shelf.shelf_engine import ShelfEngine
from team3_backend.app import app
from dashboard.preview_hud import draw_hud_overlay
from dashboard.cli_dashboard import run_cli_dashboard
from core.geometry import scale_config_to_frame

def get_default_config_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "store_config.json")

def load_config(config_path: str = None) -> dict:
    if config_path is None:
        config_path = get_default_config_path()
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)

def run_backend():
    print("[Backend] Starting FastAPI Server on http://127.0.0.1:8000 ...")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")

def open_capture_device(source):
    if str(source).isdigit():
        idx = int(source)
        if sys.platform.startswith("win"):
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(idx)
        else:
            cap = cv2.VideoCapture(idx)
    else:
        cap = cv2.VideoCapture(source)
    return cap

def run_vision_loop(config: dict, source, show_hud: bool = True, auto_shelf: bool = False):
    backend_url = config.get("backend_url", "http://127.0.0.1:8000")
    print(f"[Vision Engine] Initializing Camera / Video Source ({source})...")

    cap = open_capture_device(source)
    if not cap.isOpened():
        print(f"[Vision Engine] Error: Could not open video source {source}. Please verify camera index.")
        return

    shopper_pipeline = None
    shelf_engine = None
    initialized = False

    print("[Vision Engine] Processing loop active. Press q on video window to stop.")

    stats = {"in": 0, "out": 0, "fps": 30.0}
    show_templates = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            if not str(source).isdigit():
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                time.sleep(0.1)
                continue

        # Dynamic Resolution Scaling & Initialization on first valid frame
        if not initialized:
            h, w, _ = frame.shape
            print(f"[Vision Engine] Camera Resolution detected: {w}x{h}. Auto-scaling geometry...")
            config = scale_config_to_frame(config, w, h)
            shopper_pipeline = ShopperTrackerPipeline(config)
            shelf_engine = ShelfEngine(config)
            if auto_shelf or len(config.get("shelves", [])) == 0:
                shelf_engine.auto_detect_shelves(frame)
            initialized = True

        t_now = time.time()

        # Run Shopper MOT & Footfall
        tracks, shopper_events = shopper_pipeline.process_frame(frame, t_now)
        
        # Run Shelf & Planogram
        detected_shelves, shelf_events = shelf_engine.process_frame(frame, t_now)

        # Ingest to local backend
        all_events = shopper_events + shelf_events
        for ev in all_events:
            try:
                requests.post(f"{backend_url}/api/v1/events", json=ev.model_dump(), timeout=0.2)
                if ev.event_type == "footfall":
                    stats["in"] = ev.payload.get("running_total_in", stats["in"])
                    stats["out"] = ev.payload.get("running_total_out", stats["out"])
            except Exception:
                pass

        if show_hud:
            dynamic_queues = shopper_pipeline.queue_engine.dynamic_queues if shopper_pipeline else []
        if show_hud:
            hud_frame = draw_hud_overlay(
                frame, tracks, detected_shelves, config, stats,
                dynamic_queues=dynamic_queues, show_templates=show_templates
            )
            if hud_frame is not None:
                cv2.imshow("Retail Intelligence Platform - Live Edge Preview", hud_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    cap.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="Edge AI Retail Intelligence Platform")
    parser.add_argument("--mode", choices=["live", "mock", "dashboard-only", "server-only", "calibrate"], default="live",
                        help="Execution mode: live, mock, dashboard-only, server-only, calibrate (visual zone setup)")
    parser.add_argument("--camera", default="0", help="Camera index or path to video file (default: 0)")
    parser.add_argument("--config", default=None, help="Path to store_config.json")
    parser.add_argument("--no-gui", action="store_true", help="Disable OpenCV video preview window")
    parser.add_argument("--auto-shelf", action="store_true", help="Automatically discover shelf tiers and products from live camera")
    parser.add_argument("--clean", action="store_true", help="Start with clean blank slate (zero dummy template boxes)")
    args = parser.parse_args()

    cfg_path = args.config or get_default_config_path()

    if args.mode == "calibrate":
        from tools.calibrate_store import VisualStoreCalibrator
        calib = VisualStoreCalibrator(source=args.camera, config_path=cfg_path, clean=args.clean)
        calib.run()
        return

    if args.mode == "server-only":
        run_backend()
        return

    if args.mode == "dashboard-only":
        run_cli_dashboard()
        return

    config = load_config(cfg_path)
    if args.clean:
        config["virtual_lines"] = []
        config["zones"] = []
        config["shelves"] = []
        print("[Launcher] Clean mode: all dummy template boxes cleared. Running pure AI detection.")

    # Start FastAPI server in a background daemon thread
    server_thread = threading.Thread(target=run_backend, daemon=True)
    server_thread.start()
    time.sleep(1.5) # Wait for server startup

    if args.mode == "mock":
        from scripts.mock_event_stream import generate_mock_stream
        mock_thread = threading.Thread(target=generate_mock_stream, daemon=True)
        mock_thread.start()
        print("[Launcher] Mock stream active. Launching CLI Dashboard...")
        run_cli_dashboard()

    elif args.mode == "live":
        # Launch Rich Live CLI Dashboard in daemon thread
        dash_thread = threading.Thread(target=run_cli_dashboard, daemon=True)
        dash_thread.start()

        # Run Vision Loop in main thread for stable OpenCV GUI & event loop
        run_vision_loop(config, args.camera, not args.no_gui, args.auto_shelf)

if __name__ == "__main__":
    main()
