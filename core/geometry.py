import copy
from typing import List, Tuple, Dict, Any

def normalize_point(pt: List[float], width: float, height: float) -> List[float]:
    return [round(pt[0] / width, 4), round(pt[1] / height, 4)]

def denormalize_point(pt: List[float], width: float, height: float) -> List[int]:
    return [int(round(pt[0] * width)), int(round(pt[1] * height))]

def scale_config_to_frame(config: dict, actual_w: int, actual_h: int) -> dict:
    """
    Dynamically scales all geometry (virtual lines, zones, shelves, homography)
    from the reference camera resolution in config to the actual camera resolution.
    Supports both absolute pixel coordinates and normalized [0.0 - 1.0] coordinates.
    """
    cfg = copy.deepcopy(config)
    ref_w = cfg.get("camera", {}).get("width", actual_w)
    ref_h = cfg.get("camera", {}).get("height", actual_h)

    if ref_w == actual_w and ref_h == actual_h:
        return cfg

    scale_x = actual_w / float(ref_w)
    scale_y = actual_h / float(ref_h)

    # Scale virtual lines
    for line in cfg.get("virtual_lines", []):
        p1 = line.get("p1", [0, 0])
        p2 = line.get("p2", [0, 0])
        # If normalized [0.0 - 1.0]
        if all(0.0 <= v <= 1.0 for v in p1 + p2):
            line["p1"] = [int(p1[0] * actual_w), int(p1[1] * actual_h)]
            line["p2"] = [int(p2[0] * actual_w), int(p2[1] * actual_h)]
        else:
            line["p1"] = [int(p1[0] * scale_x), int(p1[1] * scale_y)]
            line["p2"] = [int(p2[0] * scale_x), int(p2[1] * scale_y)]

    # Scale polygon zones
    for zone in cfg.get("zones", []):
        poly = zone.get("polygon", [])
        new_poly = []
        is_norm = all(0.0 <= pt[0] <= 1.0 and 0.0 <= pt[1] <= 1.0 for pt in poly)
        for pt in poly:
            if is_norm:
                new_poly.append([int(pt[0] * actual_w), int(pt[1] * actual_h)])
            else:
                new_poly.append([int(pt[0] * scale_x), int(pt[1] * scale_y)])
        zone["polygon"] = new_poly

    # Scale shelf bounding boxes
    for shelf in cfg.get("shelves", []):
        bbox = shelf.get("bounding_box", [0, 0, 0, 0])
        if all(0.0 <= v <= 1.0 for v in bbox):
            shelf["bounding_box"] = [
                int(bbox[0] * actual_w), int(bbox[1] * actual_h),
                int(bbox[2] * actual_w), int(bbox[3] * actual_h)
            ]
        else:
            shelf["bounding_box"] = [
                int(bbox[0] * scale_x), int(bbox[1] * scale_y),
                int(bbox[2] * scale_x), int(bbox[3] * scale_y)
            ]

    # Scale homography points
    if "homography" in cfg:
        src = cfg["homography"].get("src_points", [])
        new_src = []
        is_norm = all(0.0 <= pt[0] <= 1.0 and 0.0 <= pt[1] <= 1.0 for pt in src)
        for pt in src:
            if is_norm:
                new_src.append([int(pt[0] * actual_w), int(pt[1] * actual_h)])
            else:
                new_src.append([int(pt[0] * scale_x), int(pt[1] * scale_y)])
        cfg["homography"]["src_points"] = new_src

    # Update camera specs to actual
    cfg["camera"]["width"] = actual_w
    cfg["camera"]["height"] = actual_h

    return cfg
