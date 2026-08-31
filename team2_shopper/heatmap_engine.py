import time
import numpy as np
import cv2
from typing import List, Optional
from schemas.events import BaseEvent, DensityMatrixPayload

class HeatmapEngine:
    def __init__(self, homography_config: dict, store_id: str = "store_001"):
        self.store_id = store_id
        self.rows = homography_config.get("grid_rows", 20)
        self.cols = homography_config.get("grid_cols", 20)
        self.emit_interval = 4.0 # Emit every 4 seconds for live demo
        self.last_emit_time = 0.0

        src = np.array(homography_config.get("src_points", [[0, 0], [640, 0], [640, 480], [0, 480]]), dtype=np.float32)
        dst = np.array([[0, 0], [self.cols - 1, 0], [self.cols - 1, self.rows - 1], [0, self.rows - 1]], dtype=np.float32)
        self.H = cv2.getPerspectiveTransform(src, dst)

        # 2D Accumulator grid with exponential decay
        self.grid = np.zeros((self.rows, self.cols), dtype=np.float32)

    def update(self, tracks: List[dict], current_time: float = None) -> Optional[BaseEvent]:
        if current_time is None:
            current_time = time.time()

        # Decay previous heat slightly to create a moving rolling density
        self.grid *= 0.98

        for t in tracks:
            cx, cy = t["centroid"]
            pt = np.array([[[cx, cy]]], dtype=np.float32)
            transformed = cv2.perspectiveTransform(pt, self.H)[0][0]
            gx = int(np.clip(round(transformed[0]), 0, self.cols - 1))
            gy = int(np.clip(round(transformed[1]), 0, self.rows - 1))
            
            # Splat heat with small Gaussian kernel
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    ny, nx = gy + dy, gx + dx
                    if 0 <= ny < self.rows and 0 <= nx < self.cols:
                        weight = 1.0 if (dy == 0 and dx == 0) else 0.5
                        self.grid[ny, nx] += weight

        if (current_time - self.last_emit_time) >= self.emit_interval:
            self.last_emit_time = current_time
            max_val = np.max(self.grid)
            norm_matrix = (self.grid / max_val) if max_val > 0 else self.grid
            norm_matrix = np.clip(norm_matrix, 0.0, 1.0)

            payload = DensityMatrixPayload(
                grid_rows=self.rows,
                grid_cols=self.cols,
                matrix=[[round(float(v), 3) for v in row] for row in norm_matrix],
                window_seconds=300,
                peak_zone="Aisle 1 / Entrance"
            )

            return BaseEvent(
                event_type="density_matrix",
                store_id=self.store_id,
                zone_id="store_floor_grid",
                confidence=0.95,
                payload=payload.model_dump()
            )

        return None
