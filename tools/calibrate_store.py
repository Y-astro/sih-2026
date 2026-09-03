import cv2
import numpy as np
import json
import os
import sys

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from team3_shelf.shelf_engine import ShelfEngine
from core.geometry import scale_config_to_frame

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "store_config.json")

class VisualStoreCalibrator:
    def __init__(self, source=0, config_path=CONFIG_PATH):
        self.source = source
        self.config_path = config_path
        self.config = self.load_config()
        self.shelf_engine = ShelfEngine(self.config)

        # Calibration state
        self.mode = "NAV" # NAV, LINE, AISLE, QUEUE, SHELF, HOMOGRAPHY
        self.current_points = []
        self.drag_start = None
        self.drag_end = None
        self.is_dragging = False
        self.status_msg = "Press [1-5] to calibrate geometry, [A] to Auto-Detect, [S] to Save"

    def load_config(self) -> dict:
        if os.path.exists(self.config_path):
            with open(self.config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"store_id": "store_001", "virtual_lines": [], "zones": [], "shelves": []}

    def save_config(self):
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=2)
        self.status_msg = f"SAVED CONFIG TO {os.path.basename(self.config_path)} SUCCESSFULLY!"
        print(f"[Calibrator] {self.status_msg}")

    def mouse_callback(self, event, x, y, flags, param):
        if self.mode == "LINE":
            if event == cv2.EVENT_LBUTTONDOWN:
                if len(self.current_points) == 0:
                    self.current_points.append([x, y])
                elif len(self.current_points) == 1:
                    self.current_points.append([x, y])
                    # Save line
                    line_def = {
                        "id": f"entrance_line_{len(self.config.get('virtual_lines', [])) + 1}",
                        "name": "Entrance Gate",
                        "p1": self.current_points[0],
                        "p2": self.current_points[1],
                        "direction_in": "down",
                        "debounce_seconds": 1.5
                    }
                    if "virtual_lines" not in self.config:
                        self.config["virtual_lines"] = []
                    self.config["virtual_lines"].append(line_def)
                    self.status_msg = f"Created Line ({line_def['id']}). Press [S] to Save."
                    self.current_points = []
                    self.mode = "NAV"

        elif self.mode in ["AISLE", "QUEUE"]:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.current_points.append([x, y])
                self.status_msg = f"Added vertex {len(self.current_points)}. Press [C] to complete polygon."

        elif self.mode == "SHELF":
            if event == cv2.EVENT_LBUTTONDOWN:
                self.drag_start = (x, y)
                self.drag_end = (x, y)
                self.is_dragging = True
            elif event == cv2.EVENT_MOUSEMOVE and self.is_dragging:
                self.drag_end = (x, y)
            elif event == cv2.EVENT_LBUTTONUP and self.is_dragging:
                self.is_dragging = False
                self.drag_end = (x, y)
                x1, y1 = min(self.drag_start[0], self.drag_end[0]), min(self.drag_start[1], self.drag_end[1])
                x2, y2 = max(self.drag_start[0], self.drag_end[0]), max(self.drag_start[1], self.drag_end[1])
                if (x2 - x1) > 20 and (y2 - y1) > 20:
                    shelf_def = {
                        "shelf_id": f"shelf_custom_{len(self.config.get('shelves', [])) + 1}",
                        "zone_id": "aisle_custom",
                        "bounding_box": [x1, y1, x2, y2],
                        "expected_sku": "bottle",
                        "sku_name": "Product Tier",
                        "target_count": 4,
                        "min_stock_alert": 1
                    }
                    if "shelves" not in self.config:
                        self.config["shelves"] = []
                    self.config["shelves"].append(shelf_def)
                    self.status_msg = f"Created Shelf Box ({shelf_def['shelf_id']}). Press [S] to Save."
                    self.mode = "NAV"

        elif self.mode == "HOMOGRAPHY":
            if event == cv2.EVENT_LBUTTONDOWN:
                self.current_points.append([x, y])
                self.status_msg = f"Homography Corner {len(self.current_points)}/4 clicked."
                if len(self.current_points) == 4:
                    if "homography" not in self.config:
                        self.config["homography"] = {"grid_rows": 20, "grid_cols": 20}
                    self.config["homography"]["src_points"] = list(self.current_points)
                    self.status_msg = "Homography 4-corners calibrated! Press [S] to Save."
                    self.current_points = []
                    self.mode = "NAV"

    def run(self):
        # Open camera
        if str(self.source).isdigit():
            idx = int(self.source)
            if sys.platform.startswith("win"):
                cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
                if not cap.isOpened():
                    cap = cv2.VideoCapture(idx)
            else:
                cap = cv2.VideoCapture(idx)
        else:
            cap = cv2.VideoCapture(self.source)

        if not cap.isOpened():
            print(f"[Calibrator] Error: Could not open source {self.source}")
            return

        window_name = "Store Visual Calibrator & Zone Builder"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.mouse_callback)

        print("\n=== STORE VISUAL CALIBRATOR ACTIVE ===")
        print("Keys:")
        print("  [1] - Calibrate Entrance Line (Click 2 points)")
        print("  [2] - Calibrate Aisle Zone Polygon (Click points, [C] to finish)")
        print("  [3] - Calibrate Queue Zone Polygon (Click points, [C] to finish)")
        print("  [4] - Calibrate Shelf Bounding Box (Click & drag box)")
        print("  [5] - Calibrate Homography (Click 4 floor corners: TL, TR, BR, BL)")
        print("  [A] - Auto-Detect Shelves from visible products (AI-driven)")
        print("  [S] - Save Configuration to JSON")
        print("  [R] - Reset current active drawing")
        print("  [Q] - Exit Calibrator\n")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            h, w, _ = frame.shape
            vis = frame.copy()

            # 1. Draw existing virtual lines
            for line in self.config.get("virtual_lines", []):
                p1, p2 = tuple(line["p1"]), tuple(line["p2"])
                cv2.line(vis, p1, p2, (0, 255, 255), 2)
                cv2.putText(vis, line.get("id", "line"), (p1[0], p1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

            # 2. Draw existing zones
            for zone in self.config.get("zones", []):
                poly = np.array(zone.get("polygon", []), np.int32).reshape((-1, 1, 2))
                color = (255, 120, 0) if zone.get("type") == "queue_zone" else (0, 220, 100)
                cv2.polylines(vis, [poly], True, color, 2)
                cv2.putText(vis, zone.get("name", ""), (poly[0][0][0], poly[0][0][1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

            # 3. Draw existing shelves
            for shelf in self.config.get("shelves", []):
                bx = shelf.get("bounding_box", [0, 0, 0, 0])
                cv2.rectangle(vis, (bx[0], bx[1]), (bx[2], bx[3]), (200, 50, 255), 2)
                cv2.putText(vis, f"Shelf: {shelf.get('expected_sku')}", (bx[0], bx[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 50, 255), 1)

            # 4. Draw in-progress points
            if self.mode in ["LINE", "HOMOGRAPHY"] and self.current_points:
                for pt in self.current_points:
                    cv2.circle(vis, tuple(pt), 5, (0, 0, 255), -1)

            if self.mode in ["AISLE", "QUEUE"] and len(self.current_points) > 0:
                pts_arr = np.array(self.current_points, np.int32).reshape((-1, 1, 2))
                cv2.polylines(vis, [pts_arr], False, (0, 255, 255), 2)
                for pt in self.current_points:
                    cv2.circle(vis, tuple(pt), 4, (0, 0, 255), -1)

            if self.mode == "SHELF" and self.is_dragging and self.drag_start and self.drag_end:
                cv2.rectangle(vis, self.drag_start, self.drag_end, (0, 255, 255), 2)

            # 5. Top Bar HUD
            cv2.rectangle(vis, (0, 0), (w, 50), (25, 25, 25), -1)
            mode_color = (0, 255, 0) if self.mode == "NAV" else (0, 255, 255)
            cv2.putText(vis, f"MODE: {self.mode}", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, mode_color, 2)
            cv2.putText(vis, self.status_msg, (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)

            # 6. Bottom Helper Keys Bar
            cv2.rectangle(vis, (0, h - 30), (w, h), (20, 20, 20), -1)
            keys_str = "[1] Line | [2] Aisle | [3] Queue | [4] Shelf | [5] Homography | [A] Auto-Detect | [S] Save | [Q] Exit"
            cv2.putText(vis, keys_str, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)

            cv2.imshow(window_name, vis)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == 27: # Q or ESC
                break
            elif key == ord('1'):
                self.mode = "LINE"
                self.current_points = []
                self.status_msg = "Click 2 points on screen to place Entrance Line."
            elif key == ord('2'):
                self.mode = "AISLE"
                self.current_points = []
                self.status_msg = "Click points to outline Aisle. Press [C] to complete."
            elif key == ord('3'):
                self.mode = "QUEUE"
                self.current_points = []
                self.status_msg = "Click points to outline Checkout Queue. Press [C] to complete."
            elif key == ord('4'):
                self.mode = "SHELF"
                self.current_points = []
                self.status_msg = "Click and drag rectangle over Shelf Tier."
            elif key == ord('5'):
                self.mode = "HOMOGRAPHY"
                self.current_points = []
                self.status_msg = "Click 4 floor corners (Top-Left, Top-Right, Bottom-Right, Bottom-Left)."
            elif key == ord('c'): # Close polygon
                if self.mode in ["AISLE", "QUEUE"] and len(self.current_points) >= 3:
                    zone_type = "shopping_aisle" if self.mode == "AISLE" else "queue_zone"
                    prefix = "aisle" if self.mode == "AISLE" else "checkout"
                    count = len([z for z in self.config.get("zones", []) if z.get("type") == zone_type]) + 1
                    zone_def = {
                        "id": f"{prefix}_zone_{count}",
                        "name": f"{prefix.title()} Zone {count}",
                        "polygon": list(self.current_points),
                        "type": zone_type
                    }
                    if "zones" not in self.config:
                        self.config["zones"] = []
                    self.config["zones"].append(zone_def)
                    self.status_msg = f"Created Zone ({zone_def['id']}). Press [S] to Save."
                    self.current_points = []
                    self.mode = "NAV"
            elif key == ord('a') or key == ord('A'):
                # Auto-Detect Shelves from current frame
                self.status_msg = "Running AI Auto-Discovery on visible products..."
                auto_shelves = self.shelf_engine.auto_detect_shelves(frame)
                self.config["shelves"] = auto_shelves
                self.status_msg = f"Auto-detected {len(auto_shelves)} shelf tier(s)! Press [S] to Save."
            elif key == ord('s') or key == ord('S'):
                self.save_config()
            elif key == ord('r') or key == ord('R'):
                self.mode = "NAV"
                self.current_points = []
                self.status_msg = "Reset draft drawing."

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "0"
    calibrator = VisualStoreCalibrator(source=src)
    calibrator.run()
