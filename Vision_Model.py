# SENTRY - Smart Edge Network for Tracking Retail Yield
# Core CV module for SIH26179: detects people, tracks them across frames,
# and reports zone-based analytics (footfall count + live queue length).
#
# Works on three kinds of input:
#   - a webcam                 : python retail_zone_analytics.py --source 0
#   - a video file              : python retail_zone_analytics.py --source demo.mp4
#   - a single image (for slides/testing): python retail_zone_analytics.py --source photo.jpg --image
#
# Model: YOLOv8n (nano) - pretrained on COCO, detects "person" (class 0)
# out of the box, no custom training needed for this MVP.

import argparse
import platform
import subprocess
import time
import cv2
import numpy as np
import requests
from ultralytics import YOLO

WINDOW_NAME = "VYAPAR BUDDHI - Zone Analytics"

# ---------------------------------------------------------------
# Zone definition - a polygon marking the "checkout / queue" area.
# Coordinates are (x, y) points, normalised 0-1 so the same config
# works regardless of the camera's actual resolution.
# ---------------------------------------------------------------
CHECKOUT_ZONE_NORM = [(0.05, 0.55), (0.95, 0.55), (0.95, 1.0), (0.05, 1.0)]
QUEUE_ALERT_THRESHOLD = 4  # trigger "open another counter" past this many people

# Preferred capture resolutions to try, highest first. Most webcams will
# silently clamp to the nearest mode they actually support, so we just ask
# for the best one and let the driver pick the closest match.
CAMERA_RES_CANDIDATES = [(1920, 1080), (1280, 720), (640, 480)]


def zone_to_pixels(zone_norm, width, height):
    return np.array([[int(x * width), int(y * height)] for x, y in zone_norm], dtype=np.int32)


def point_in_zone(cx, cy, zone_px):
    return cv2.pointPolygonTest(zone_px, (float(cx), float(cy)), False) >= 0


def draw_overlay(frame, boxes_ids, zone_px, footfall_total, queue_count):
    # Draw the checkout zone
    overlay = frame.copy()
    cv2.fillPoly(overlay, [zone_px], (15, 124, 130))
    frame = cv2.addWeighted(overlay, 0.18, frame, 0.82, 0)
    cv2.polylines(frame, [zone_px], isClosed=True,
                  color=(15, 124, 130), thickness=2)

    # Draw each tracked person
    for (x1, y1, x2, y2, track_id, in_zone) in boxes_ids:
        color = (0, 0, 220) if in_zone else (40, 180, 40)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"ID {track_id}" if track_id is not None else "person"
        cv2.putText(frame, label, (x1, max(y1 - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    # KPI banner
    banner_h = 70
    cv2.rectangle(frame, (0, 0), (frame.shape[1], banner_h), (27, 42, 74), -1)
    cv2.putText(frame, f"Footfall (this frame): {footfall_total}", (15, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(frame, f"Checkout queue: {queue_count}", (15, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    if queue_count > QUEUE_ALERT_THRESHOLD:
        cv2.putText(frame, "ALERT: Open another counter", (350, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
    return frame


def send_queue_to_esp32(esp32_ip, queue_count, timeout=0.5):
    """Push the live queue count to the ESP32's /queue endpoint.
    Fire-and-forget: a slow/offline ESP32 should never stall the video loop,
    so failures are swallowed and just logged."""
    try:
        requests.post(f"http://{esp32_ip}/queue",
                      json={"queue": queue_count}, timeout=timeout)
    except requests.exceptions.RequestException as e:
        print(f"[ESP32] could not reach {esp32_ip}: {e}")


def fetch_esp32_status(esp32_ip, timeout=0.5):
    """Pull the ESP32's own entrance-counter numbers (occupancy, footfall,
    store_full) so they can be overlaid on the same dashboard/frame."""
    try:
        r = requests.get(f"http://{esp32_ip}/status", timeout=timeout)
        if r.ok:
            return r.json()
    except requests.exceptions.RequestException:
        pass
    return None


def process_frame(model, frame, use_tracking=True):
    h, w = frame.shape[:2]
    zone_px = zone_to_pixels(CHECKOUT_ZONE_NORM, w, h)

    if use_tracking:
        results = model.track(frame, classes=[0], persist=True, verbose=False)
    else:
        results = model.predict(frame, classes=[0], verbose=False)

    boxes_ids = []
    queue_count = 0
    r = results[0]
    if r.boxes is not None:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            track_id = int(box.id[0]) if (
                use_tracking and box.id is not None) else None
            # use feet position, not centre, for zone check
            cx, cy = (x1 + x2) // 2, y2
            in_zone = point_in_zone(cx, cy, zone_px)
            if in_zone:
                queue_count += 1
            boxes_ids.append((x1, y1, x2, y2, track_id, in_zone))

    footfall_total = len(boxes_ids)
    annotated = draw_overlay(frame, boxes_ids, zone_px,
                             footfall_total, queue_count)
    return annotated, footfall_total, queue_count


def configure_camera(cap, requested_width=None, requested_height=None):
    """Ask the camera driver for the highest resolution it supports so
    the feed doesn't look soft/pixelated once it's stretched to fullscreen.

    Also nudges the backend to use MJPG where available, since the default
    YUYV mode on many USB webcams caps out at low frame rates at higher
    resolutions, which would otherwise force the driver back down to a
    smaller frame size.
    """
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    candidates = CAMERA_RES_CANDIDATES
    if requested_width and requested_height:
        candidates = [(requested_width, requested_height)] + candidates

    for w, h in candidates:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # Close enough to what we asked for -> good, stop here.
        if actual_w >= w * 0.9 and actual_h >= h * 0.9:
            print(f"[Camera] capturing at {actual_w}x{actual_h}")
            return actual_w, actual_h

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[Camera] falling back to driver default {actual_w}x{actual_h}")
    return actual_w, actual_h


def minimize_window(window_name):
    """OpenCV's highgui has no built-in minimize call, so we reach into the
    OS window manager directly. Best-effort: if the platform/tooling isn't
    available, we just print instructions instead of crashing the app."""
    system = platform.system()
    try:
        if system == "Windows":
            import ctypes
            SW_MINIMIZE = 6
            hwnd = ctypes.windll.user32.FindWindowW(None, window_name)
            if hwnd:
                ctypes.windll.user32.ShowWindow(hwnd, SW_MINIMIZE)
                return True
        elif system == "Linux":
            # Requires wmctrl (sudo apt install wmctrl). Silently no-ops if missing.
            result = subprocess.run(
                ["wmctrl", "-r", window_name, "-b", "add,hidden"],
                capture_output=True, timeout=1)
            return result.returncode == 0
    except Exception as e:
        print(f"[Window] could not minimize: {e}")
    return False


def get_screen_resolution(default=(1920, 1080)):
    """Best-effort lookup of the actual monitor resolution so we can scale
    the frame to fill the screen without guessing (guessing is what causes
    the 'blurry stretched fullscreen' look)."""
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        w, h = root.winfo_screenwidth(), root.winfo_screenheight()
        root.destroy()
        if w > 0 and h > 0:
            return w, h
    except Exception:
        pass
    return default


def fit_to_screen(frame, screen_w, screen_h):
    """Letterbox the frame to fill the screen while preserving aspect ratio,
    using a quality-appropriate interpolation method (avoids the blocky look
    of the default nearest-neighbour resize when scaling up)."""
    h, w = frame.shape[:2]
    scale = min(screen_w / w, screen_h / h)
    new_w, new_h = int(w * scale), int(h * scale)

    interp = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
    resized = cv2.resize(frame, (new_w, new_h), interpolation=interp)

    canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
    x_off = (screen_w - new_w) // 2
    y_off = (screen_h - new_h) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas


def main():
    parser = argparse.ArgumentParser(
        epilog="While the window is open: 'q' quits, 'f' toggles fullscreen, "
               "'m' drops out of fullscreen and minimizes the window.")
    parser.add_argument("--source", default="0",
                        help="0 for webcam, or a path to a video/image file")
    parser.add_argument("--image", action="store_true",
                        help="treat --source as a single static image")
    parser.add_argument("--out", default="output.jpg",
                        help="output path for image mode")
    parser.add_argument("--esp32-ip", default=None,
                        help="IP address of the ESP32 node (shown on its LCD on boot), "
                        "e.g. 192.168.1.42. If omitted, runs standalone with no ESP32.")
    parser.add_argument("--fullscreen", dest="fullscreen", action="store_true",
                        default=True, help="show the live window fullscreen (default: on)")
    parser.add_argument("--windowed", dest="fullscreen", action="store_false",
                        help="show the live window in a normal, resizable window instead")
    parser.add_argument("--cam-width", type=int, default=None,
                        help="request a specific camera capture width, e.g. 1920")
    parser.add_argument("--cam-height", type=int, default=None,
                        help="request a specific camera capture height, e.g. 1080")
    args = parser.parse_args()

    # auto-downloads pretrained weights on first run
    model = YOLO("yolov8n.pt")

    if args.image:
        frame = cv2.imread(args.source)
        annotated, footfall, queue = process_frame(
            model, frame, use_tracking=True)
        cv2.imwrite(args.out, annotated)
        print(
            f"Footfall: {footfall} | Queue count: {queue} | Saved to {args.out}")
        return

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)

    # Only worth tuning capture resolution for a live camera - a video file
    # already has whatever resolution it was recorded at.
    if isinstance(source, int):
        configure_camera(cap, args.cam_width, args.cam_height)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    screen_w, screen_h = get_screen_resolution()
    # Give windowed mode a sane default size/position (roughly centred,
    # 70% of the screen) so toggling out of fullscreen doesn't leave you
    # with a window that's the wrong size or half off-screen.
    windowed_w, windowed_h = int(screen_w * 0.7), int(screen_h * 0.7)
    windowed_x, windowed_y = (
        screen_w - windowed_w) // 2, (screen_h - windowed_h) // 2
    cv2.resizeWindow(WINDOW_NAME, windowed_w, windowed_h)
    cv2.moveWindow(WINDOW_NAME, windowed_x, windowed_y)
    if args.fullscreen:
        cv2.setWindowProperty(
            WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    last_esp32_push = 0.0
    last_esp32_status = None
    ESP32_PUSH_INTERVAL_S = 1.0  # don't flood the ESP32's web server every frame

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        annotated, footfall, queue = process_frame(
            model, frame, use_tracking=True)

        if args.esp32_ip:
            now = time.time()
            if now - last_esp32_push > ESP32_PUSH_INTERVAL_S:
                send_queue_to_esp32(args.esp32_ip, queue)
                last_esp32_status = fetch_esp32_status(args.esp32_ip)
                last_esp32_push = now
            if last_esp32_status:
                occ = last_esp32_status.get("occupancy", "?")
                today = last_esp32_status.get("footfall_today", "?")
                cv2.putText(annotated, f"Entrance node -> in-store: {occ} | today: {today}",
                            (15, annotated.shape[0] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        display_frame = annotated
        if args.fullscreen:
            display_frame = fit_to_screen(annotated, screen_w, screen_h)

        cv2.imshow(WINDOW_NAME, display_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("f"):
            # Fullscreen has no title bar, so drop to windowed first -
            # that's what makes the window draggable/minimizable again.
            args.fullscreen = not args.fullscreen
            cv2.setWindowProperty(
                WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN if args.fullscreen else cv2.WINDOW_NORMAL)
            if not args.fullscreen:
                cv2.resizeWindow(WINDOW_NAME, windowed_w, windowed_h)
                cv2.moveWindow(WINDOW_NAME, windowed_x, windowed_y)
        if key == ord("m"):
            # Minimizing a fullscreen window doesn't work on most window
            # managers, so exit fullscreen first, then minimize.
            if args.fullscreen:
                args.fullscreen = False
                cv2.setWindowProperty(
                    WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            minimize_window(WINDOW_NAME)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
