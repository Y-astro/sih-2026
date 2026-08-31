import time
from typing import Dict, List, Tuple, Optional
from schemas.events import BaseEvent, FootfallPayload

class FootfallEngine:
    def __init__(self, lines_config: List[dict], store_id: str = "store_001"):
        self.lines_config = lines_config
        self.store_id = store_id
        self.total_in = 0
        self.total_out = 0
        self.track_history: Dict[int, List[Tuple[float, float, float]]] = {}
        self.cooldowns: Dict[Tuple[int, str], float] = {}

    def update(self, tracks: List[dict], current_time: float = None) -> List[BaseEvent]:
        if current_time is None:
            current_time = time.time()
            
        events = []
        active_ids = set()

        for t in tracks:
            track_id = t["track_id"]
            cx, cy = t["centroid"]
            active_ids.add(track_id)

            if track_id not in self.track_history:
                self.track_history[track_id] = []
            self.track_history[track_id].append((cx, cy, current_time))
            if len(self.track_history[track_id]) > 30:
                self.track_history[track_id].pop(0)

            hist = self.track_history[track_id]
            if len(hist) < 2:
                continue

            prev_x, prev_y, _ = hist[-2]
            curr_x, curr_y, _ = hist[-1]

            for line in self.lines_config:
                line_id = line["id"]
                cooldown_key = (track_id, line_id)
                if cooldown_key in self.cooldowns and (current_time - self.cooldowns[cooldown_key]) < line.get("debounce_seconds", 2.0):
                    continue

                p1 = line["p1"]
                p2 = line["p2"]
                
                # Check line intersection
                crossed, direction = self._check_crossing(prev_x, prev_y, curr_x, curr_y, p1, p2, line.get("direction_in", "down"))
                if crossed:
                    self.cooldowns[cooldown_key] = current_time
                    if direction == "in":
                        self.total_in += 1
                        payload = FootfallPayload(
                            direction="in",
                            in_delta=1,
                            out_delta=0,
                            running_total_in=self.total_in,
                            running_total_out=self.total_out,
                            line_id=line_id
                        )
                    else:
                        self.total_out += 1
                        payload = FootfallPayload(
                            direction="out",
                            in_delta=0,
                            out_delta=1,
                            running_total_in=self.total_in,
                            running_total_out=self.total_out,
                            line_id=line_id
                        )

                    events.append(BaseEvent(
                        event_type="footfall",
                        store_id=self.store_id,
                        zone_id=line.get("zone_id", "entrance_main"),
                        anonymous_track_id=f"sess_{track_id}",
                        confidence=t.get("confidence", 0.95),
                        payload=payload.model_dump()
                    ))

        # Cleanup lost tracks
        stale_ids = [tid for tid in self.track_history if tid not in active_ids and (current_time - self.track_history[tid][-1][2]) > 10.0]
        for tid in stale_ids:
            del self.track_history[tid]

        return events

    def _check_crossing(self, x1, y1, x2, y2, lp1, lp2, expected_in_dir: str):
        # Horizontal or near-horizontal line crossing check
        lx1, ly1 = lp1
        lx2, ly2 = lp2
        
        # Check if segment crosses line segment
        min_lx = min(lx1, lx2) - 20
        max_lx = max(lx1, lx2) + 20
        
        if not (min_lx <= x2 <= max_lx):
            return False, None

        line_y = (ly1 + ly2) / 2.0
        if y1 < line_y <= y2:
            direction = "in" if expected_in_dir == "down" else "out"
            return True, direction
        elif y1 > line_y >= y2:
            direction = "out" if expected_in_dir == "down" else "in"
            return True, direction
            
        return False, None
