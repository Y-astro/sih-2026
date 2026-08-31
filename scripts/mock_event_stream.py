import time
import random
import requests
import numpy as np

BACKEND_URL = "http://127.0.0.1:8000"

def generate_mock_stream():
    print(f"[Mock Stream] Starting simulated store activity pushing to {BACKEND_URL}...")
    
    total_in = 12
    total_out = 4
    matrix = np.zeros((20, 20), dtype=float)

    while True:
        # 1. Simulate Footfall Event (every ~3-5s)
        if random.random() < 0.6:
            direction = "in" if random.random() < 0.75 else "out"
            if direction == "in":
                total_in += 1
            else:
                total_out += 1
            
            ff_payload = {
                "event_type": "footfall",
                "store_id": "store_001",
                "zone_id": "entrance_main",
                "anonymous_track_id": f"sess_{random.randint(100, 999)}",
                "confidence": 0.96,
                "payload": {
                    "direction": direction,
                    "in_delta": 1 if direction == "in" else 0,
                    "out_delta": 1 if direction == "out" else 0,
                    "running_total_in": total_in,
                    "running_total_out": total_out,
                    "line_id": "entrance_line_1"
                }
            }
            try:
                requests.post(f"{BACKEND_URL}/api/v1/events", json=ff_payload, timeout=0.5)
            except Exception:
                pass

        # 2. Simulate Queue State (Checkout 1 & 2)
        q_len = random.randint(2, 6)
        status = "congested" if q_len >= 5 else ("warning" if q_len >= 3 else "normal")
        rec = "High congestion: Open Counter 2 immediately!" if status == "congested" else None
        q_payload = {
            "event_type": "queue_state",
            "store_id": "store_001",
            "zone_id": "checkout_zone_1",
            "confidence": 0.93,
            "payload": {
                "counter_id": "checkout_zone_1",
                "queue_length": q_len,
                "estimated_wait_seconds": q_len * 30,
                "congestion_status": status,
                "alert_triggered": (status == "congested"),
                "recommended_action": rec
            }
        }
        try:
            requests.post(f"{BACKEND_URL}/api/v1/events", json=q_payload, timeout=0.5)
        except Exception:
            pass

        # 3. Simulate Shelf Monitoring
        shelf_choices = [
            {"id": "shelf_A1_top", "sku": "Premium Sparkling Water", "stock": random.choice([0, 1, 4]), "max": 4},
            {"id": "shelf_A1_bottom", "sku": "Energy Drink Can 250ml", "stock": random.choice([1, 2, 4]), "max": 4},
            {"id": "shelf_B1_top", "sku": "Artisan Cereal & Granola", "stock": random.choice([0, 3]), "max": 3}
        ]
        chosen = random.choice(shelf_choices)
        if chosen["stock"] == 0:
            atype = "shelf_oos"
            sev = "critical"
        elif chosen["stock"] == 1:
            atype = "shelf_lowstock"
            sev = "high"
        else:
            atype = None

        if atype:
            shelf_payload = {
                "event_type": atype,
                "store_id": "store_001",
                "zone_id": "aisle_1_beverages",
                "confidence": 0.91,
                "payload": {
                    "shelf_id": chosen["id"],
                    "zone_id": "aisle_1_beverages",
                    "expected_sku": chosen["sku"],
                    "detected_sku": "Empty" if chosen["stock"] == 0 else chosen["sku"],
                    "stock_count": chosen["stock"],
                    "max_capacity": chosen["max"],
                    "fill_percentage": round((chosen["stock"] / chosen["max"]) * 100.0, 1),
                    "alert_type": atype,
                    "restock_ticket_id": f"WMS-TK-{random.randint(1000, 9999)}",
                    "severity": sev
                }
            }
            try:
                requests.post(f"{BACKEND_URL}/api/v1/events", json=shelf_payload, timeout=0.5)
            except Exception:
                pass

        # 4. Simulate Density Matrix Update
        matrix *= 0.92
        # Add random active shopper clusters
        for _ in range(random.randint(2, 5)):
            rx, ry = random.randint(2, 17), random.randint(2, 17)
            matrix[ry-1:ry+2, rx-1:rx+2] += random.uniform(0.3, 0.8)
        
        m_max = np.max(matrix)
        norm_m = (matrix / m_max) if m_max > 0 else matrix
        dm_payload = {
            "event_type": "density_matrix",
            "store_id": "store_001",
            "zone_id": "store_floor_grid",
            "confidence": 0.95,
            "payload": {
                "grid_rows": 20,
                "grid_cols": 20,
                "matrix": [[round(float(v), 2) for v in row] for row in norm_m],
                "window_seconds": 300,
                "peak_zone": "Aisle 1 / Entrance"
            }
        }
        try:
            requests.post(f"{BACKEND_URL}/api/v1/events", json=dm_payload, timeout=0.5)
        except Exception:
            pass

        time.sleep(2.5)

if __name__ == "__main__":
    generate_mock_stream()
