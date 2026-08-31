import cv2
import numpy as np

def draw_hud_overlay(frame: np.ndarray, tracks: list, detected_shelves: list, config: dict, stats: dict) -> np.ndarray:
    if frame is None:
        return None

    vis = frame.copy()
    h, w, _ = vis.shape

    # 1. Draw Virtual Crossing Lines
    for line in config.get("virtual_lines", []):
        p1 = tuple(line["p1"])
        p2 = tuple(line["p2"])
        cv2.line(vis, p1, p2, (0, 255, 255), 3)
        cv2.putText(vis, f"ENTRY LINE: {line.get(id)}", (p1[0], p1[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # 2. Draw Zones
    for zone in config.get("zones", []):
        pts = np.array(zone["polygon"], np.int32).reshape((-1, 1, 2))
        color = (255, 100, 0) if zone.get("type") == "queue_zone" else (0, 200, 100)
        cv2.polylines(vis, [pts], True, color, 2)
        cv2.putText(vis, zone.get("name", ""), (pts[0][0][0], pts[0][0][1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    # 3. Draw Shelf Bins
    for shelf in config.get("shelves", []):
        x1, y1, x2, y2 = shelf["bounding_box"]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (200, 50, 255), 2)
        cv2.putText(vis, f"SHELF: {shelf.get(expected_sku)}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 50, 255), 1)

    # 4. Draw Person Tracks
    for t in tracks:
        bbox = t["bbox"]
        tid = t["track_id"]
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis, f"Track #{tid}", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # Centroid
        cx, cy = int(t["centroid"][0]), int(t["centroid"][1])
        cv2.circle(vis, (cx, cy), 4, (0, 0, 255), -1)

    # 5. Top Bar HUD Info
    cv2.rectangle(vis, (0, 0), (w, 40), (20, 20, 20), -1)
    hud_str = f"FPS: {stats.get(fps, 30):.1f} | In: {stats.get(in, 0)} | Out: {stats.get(out, 0)} | Tracks: {len(tracks)}"
    cv2.putText(vis, hud_str, (15, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return vis
