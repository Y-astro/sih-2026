import cv2
import numpy as np

def draw_hud_overlay(
    frame: np.ndarray,
    tracks: list,
    detected_objects: list,
    config: dict,
    stats: dict,
    dynamic_queues: list = None,
    show_templates: bool = False
) -> np.ndarray:
    """
    Intelligent Visual HUD Overlay:
    - Renders real-time AI detections (people, products, items) with bounding boxes and confidence.
    - Renders dynamic queues automatically discovered by people spatial clustering.
    - Only renders static template zones/shelves if explicitly enabled (show_templates=True)
      or if real objects are detected inside them.
    """
    if frame is None:
        return None

    vis = frame.copy()
    h, w, _ = vis.shape

    # 1. Draw Real Detected Retail Objects (Bottles, Cups, Cans, Boxes, Books, etc.)
    if detected_objects:
        for obj in detected_objects:
            bx = obj.get("bbox", [])
            if len(bx) == 4:
                x1, y1, x2, y2 = int(bx[0]), int(bx[1]), int(bx[2]), int(bx[3])
                cls_name = obj.get("class_name", "item")
                conf = obj.get("confidence", 0.0)
                
                # Draw neon cyan box around actual detected product
                cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 255, 0), 2)
                lbl = f"{cls_name.upper()} {conf*100:.0f}%"
                cv2.putText(vis, lbl, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
                
                # Dot on center
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                cv2.circle(vis, (cx, cy), 3, (0, 255, 255), -1)

    # 2. Draw Dynamic AI Queues (discovers groups of people automatically)
    if dynamic_queues:
        for dq in dynamic_queues:
            bx = dq["bbox"]
            count = dq["count"]
            status = dq.get("status", "normal")
            color = (0, 0, 255) if status == "congested" else ((0, 255, 255) if status == "warning" else (255, 165, 0))
            cv2.rectangle(vis, (bx[0], bx[1]), (bx[2], bx[3]), color, 2)
            cv2.putText(vis, f"AI QUEUE: {count} ppl ({status.upper()})", (bx[0], max(15, bx[1] - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 3. Draw Person Tracks
    for t in tracks:
        bbox = t["bbox"]
        tid = t["track_id"]
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis, f"Shopper #{tid}", (x1, max(15, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cx, cy = int(t["centroid"][0]), int(t["centroid"][1])
        cv2.circle(vis, (cx, cy), 4, (0, 0, 255), -1)

    # 4. Optional / Active Configured Zones (Virtual Lines, Shelf ROIs, Zones)
    # Only drawn if show_templates is True or if configured by user
    if show_templates:
        # Virtual Crossing Lines
        for line in config.get("virtual_lines", []):
            p1 = tuple(line["p1"])
            p2 = tuple(line["p2"])
            cv2.line(vis, p1, p2, (0, 255, 255), 2)
            cv2.putText(vis, f"GATE: {line.get('id', '')}", (p1[0], p1[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        # Configured Zones
        for zone in config.get("zones", []):
            pts = np.array(zone["polygon"], np.int32).reshape((-1, 1, 2))
            color = (255, 100, 0) if zone.get("type") == "queue_zone" else (0, 200, 100)
            cv2.polylines(vis, [pts], True, color, 1)
            cv2.putText(vis, zone.get("name", ""), (pts[0][0][0], max(15, pts[0][0][1] - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Configured Shelf Bins
        for shelf in config.get("shelves", []):
            x1, y1, x2, y2 = shelf["bounding_box"]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (200, 50, 255), 1)
            cv2.putText(vis, f"SHELF: {shelf.get('expected_sku', '')}", (x1, max(15, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 50, 255), 1)

    # 5. Top Bar HUD
    cv2.rectangle(vis, (0, 0), (w, 36), (20, 20, 20), -1)
    fps_val = stats.get("fps", 30.0)
    in_val = stats.get("in", 0)
    out_val = stats.get("out", 0)
    num_tracks = len(tracks)
    num_items = len(detected_objects) if detected_objects else 0
    hud_str = f"FPS: {fps_val:.1f} | In: {in_val} | Out: {out_val} | Shoppers: {num_tracks} | Products: {num_items}"
    cv2.putText(vis, hud_str, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    # Helper hint on bottom
    hint = "Press [T] to toggle template boxes | [Q] to quit"
    cv2.putText(vis, hint, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

    return vis
