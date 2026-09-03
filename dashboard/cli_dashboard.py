import sys
import time
import requests

# Ensure UTF-8 output on Windows PowerShell / Command Prompt
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.live import Live
from rich.align import Align

console = Console()

def render_ascii_heatmap(matrix: list) -> Text:
    if not matrix or len(matrix) == 0:
        return Text("No Heatmap Data")
    
    chars = [" ", "░", "▒", "▓", "█"]
    t = Text()
    for row in matrix:
        for val in row:
            idx = min(int(val * len(chars)), len(chars) - 1)
            char = chars[idx]
            if val < 0.2:
                style = "blue"
            elif val < 0.5:
                style = "cyan"
            elif val < 0.7:
                style = "yellow"
            elif val < 0.9:
                style = "bold dark_orange"
            else:
                style = "bold red"
            t.append(f"{char}{char}", style=style)
        t.append("\n")
    return t

def build_dashboard_layout(data: dict) -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="main", ratio=1),
        Layout(name="footer", size=3)
    )
    layout["main"].split_row(
        Layout(name="left", ratio=1),
        Layout(name="right", ratio=1)
    )
    layout["left"].split_column(
        Layout(name="kpi", size=7),
        Layout(name="queues", size=9),
        Layout(name="events", ratio=1)
    )
    layout["right"].split_column(
        Layout(name="heatmap", size=15),
        Layout(name="shelves", ratio=1)
    )

    # Header
    sys_stat = data.get("system_status", {})
    fps = sys_stat.get("fps", 30.0)
    lat = sys_stat.get("latency_ms", 15.0)
    store_id = data.get("store_id", "store_001")
    hdr_text = Text.from_markup(
        f"[bold white on blue] EDGE AI RETAIL INTELLIGENCE PLATFORM [/] | Store: [bold green]{store_id}[/] | FPS: [bold yellow]{fps:.1f}[/] | Edge Latency: [bold yellow]{lat:.1f}ms[/] | Zero PII: [bold green]ENFORCED[/]"
    )
    layout["header"].update(Panel(Align.center(hdr_text), style="bold blue"))

    # Left: KPIs
    kpi_table = Table(show_header=True, header_style="bold magenta", expand=True)
    kpi_table.add_column("Live Occupancy", justify="center", style="bold green")
    kpi_table.add_column("Total Footfall In", justify="center", style="bold cyan")
    kpi_table.add_column("Total Footfall Out", justify="center", style="bold yellow")
    kpi_table.add_column("Active Shelf Alerts", justify="center", style="bold red")
    
    kpi_table.add_row(
        str(data.get("current_occupancy", 0)),
        str(data.get("total_footfall_in", 0)),
        str(data.get("total_footfall_out", 0)),
        str(len(data.get("shelf_alerts", {})))
    )
    layout["kpi"].update(Panel(kpi_table, title="[bold]Retail Traffic & Occupancy KPIs[/]"))

    # Left: Queues
    q_table = Table(show_header=True, header_style="bold cyan", expand=True)
    q_table.add_column("Counter ID", style="bold")
    q_table.add_column("Queue Length", justify="center")
    q_table.add_column("Est. Wait Time", justify="center")
    q_table.add_column("Status", justify="center")
    q_table.add_column("Action Recommendation", justify="left")

    queues = data.get("queues", {})
    if queues:
        for cid, q in queues.items():
            status = q.get("congestion_status", "normal")
            st_color = "red" if status == "congested" else ("yellow" if status == "warning" else "green")
            q_table.add_row(
                cid,
                f"{q.get('queue_length', 0)} ppl",
                f"{q.get('estimated_wait_seconds', 0)}s",
                f"[{st_color}]{status.upper()}[/{st_color}]",
                q.get("recommended_action") or "[dim]Flow optimal[/dim]"
            )
    else:
        q_table.add_row("checkout_1", "0 ppl", "0s", "[green]NORMAL[/green]", "Flow optimal")
    layout["queues"].update(Panel(q_table, title="[bold]Queue Intelligence & Checkout Congestion[/]"))

    # Left: Live Event Feed
    evt_table = Table(show_header=True, header_style="bold yellow", expand=True)
    evt_table.add_column("Time (UTC)", style="dim", width=12)
    evt_table.add_column("Type", style="bold", width=16)
    evt_table.add_column("Zone", width=18)
    evt_table.add_column("Details", justify="left")

    events = data.get("recent_events", [])
    for ev in events[:6]:
        ts = ev.get("timestamp", "").split("T")[-1][:8]
        etype = ev.get("event_type", "")
        zone = ev.get("zone_id", "")
        p = ev.get("payload", {})
        det = ""
        if etype == "footfall":
            det = f"Direction: {p.get('direction', '')} | Total: In {p.get('running_total_in', 0)} / Out {p.get('running_total_out', 0)}"
        elif etype == "queue_state":
            det = f"Length: {p.get('queue_length', 0)} | Status: {p.get('congestion_status', '')}"
        elif etype in ["shelf_oos", "shelf_lowstock"]:
            det = f"SKU: {p.get('expected_sku', '')} | Fill: {p.get('fill_percentage', 0)}% | Ticket: {p.get('restock_ticket_id', '')}"
        elif etype == "dwell":
            det = f"Dwell: {p.get('dwell_time_seconds', 0)}s in {p.get('zone_id', '')}"
        else:
            det = str(p)[:45]
        evt_table.add_row(ts, etype, zone, det)

    layout["events"].update(Panel(evt_table, title="[bold]Live Ingestion Feed (Audit Stream)[/]"))

    # Right: Heatmap
    heatmap_text = render_ascii_heatmap(data.get("density_matrix", []))
    layout["heatmap"].update(Panel(Align.center(heatmap_text), title="[bold]Top-Down Store Floor Heatmap (20x20 Grid)[/]"))

    # Right: Shelf Inventory & Planogram Audit
    shelf_table = Table(show_header=True, header_style="bold red", expand=True)
    shelf_table.add_column("Shelf ID", style="bold")
    shelf_table.add_column("Expected SKU")
    shelf_table.add_column("Stock", justify="center")
    shelf_table.add_column("Alert", justify="center")
    shelf_table.add_column("WMS Restock Ticket", justify="left")

    shelf_alerts = data.get("shelf_alerts", {})
    if shelf_alerts:
        for sid, alert in shelf_alerts.items():
            atype = alert.get("alert_type", "shelf_oos")
            st_color = "bold red" if atype == "shelf_oos" else "yellow"
            shelf_table.add_row(
                sid,
                alert.get("expected_sku", ""),
                f"{alert.get('stock_count', 0)}/{alert.get('max_capacity', 4)} ({alert.get('fill_percentage', 0)}%)",
                f"[{st_color}]{atype.upper()}[/{st_color}]",
                f"[bold cyan]{alert.get('restock_ticket_id', 'N/A')}[/]"
            )
    else:
        shelf_table.add_row("shelf_A1_top", "Water Bottles", "4/4 (100%)", "[green]COMPLIANT[/green]", "[dim]None[/dim]")
        shelf_table.add_row("shelf_A1_bottom", "Energy Cans", "4/4 (100%)", "[green]COMPLIANT[/green]", "[dim]None[/dim]")
    layout["shelves"].update(Panel(shelf_table, title="[bold]Shelf Monitoring & Planogram Audit (WMS Connected)[/]"))

    # Footer
    footer_text = Text.from_markup(
        "[dim]Press Ctrl+C to exit. System running offline-first edge compute. All PII discarded on device.[/dim]"
    )
    layout["footer"].update(Panel(Align.center(footer_text), style="dim"))

    return layout

def run_cli_dashboard(backend_url: str = "http://127.0.0.1:8000"):
    console.print("[bold green]Connecting to Retail Intelligence Backend...[/]")
    with Live(console=console, screen=False, refresh_per_second=4) as live:
        while True:
            try:
                resp = requests.get(f"{backend_url}/api/v1/store/snapshot", timeout=1.0)
                if resp.status_code == 200:
                    data = resp.json()
                    layout = build_dashboard_layout(data)
                    live.update(layout)
                else:
                    live.update(Panel(f"Backend HTTP {resp.status_code}"))
            except Exception as e:
                live.update(Panel(f"[bold red]Connecting to Backend at {backend_url}... ({e})[/]"))
            time.sleep(0.25)
