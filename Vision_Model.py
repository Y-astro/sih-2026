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
import time
import cv2
import numpy as np
import requests
from ultralytics import YOLO

# ---------------------------------------------------------------
# Zone definition - a polygon marking the "checkout / queue" area.
# Coordinates are (x, y) points, normalised 0-1 so the same config
# works regardless of the camera's actual resolution.
# ---------------------------------------------------------------
CHECKOUT_ZONE_NORM = [(0.05, 0.55), (0.95, 0.55), (0.95, 1.0), (0.05, 1.0)]
QUEUE_ALERT_THRESHOLD = 4  # trigger "open another counter" past this many people


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0",
                        help="0 for webcam, or a path to a video/image file")
    parser.add_argument("--image", action="store_true",
                        help="treat --source as a single static image")
    parser.add_argument("--out", default="output.jpg",
                        help="output path for image mode")
    parser.add_argument("--esp32-ip", default=None,
                        help="IP address of the ESP32 node (shown on its LCD on boot), "
                        "e.g. 192.168.1.42. If omitted, runs standalone with no ESP32.")
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

        cv2.imshow("VYAPAR BUDDHI - Zone Analytics", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
