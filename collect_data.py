"""
collect_data.py
───────────────
Run this ONCE before training to collect real system data.

Usage:
    python collect_data.py --duration 300   # collect for 5 minutes (default)
    python collect_data.py --duration 3600  # collect for 1 hour (better training)

Output: data/system_data.csv

Each row = one second snapshot.
The CSV is used by train_models.py to train and save .pkl files.

Columns saved:
    timestamp, cpu_percent, ram_percent, ram_used_mb, ram_total_mb,
    ram_available_mb, disk_percent, swap_percent, cpu_freq_mhz,
    net_sent_mb, net_recv_mb,
    cpu_1min_avg, cpu_5min_avg, ram_1min_avg, ram_5min_avg,
    ram_slope_30s, cpu_slope_30s,
    hour_of_day, day_of_week,
    label_ram_critical (1 if RAM will exceed 85% in next 60s)
"""

import psutil
import csv
import time
import argparse
from datetime import datetime
from pathlib import Path
from collections import deque
import numpy as np

OUTPUT_PATH = Path("data/system_data.csv")
SAMPLE_INTERVAL = 1.0   # seconds between readings
WINDOW_30 = deque(maxlen=30)
WINDOW_60 = deque(maxlen=60)
WINDOW_300 = deque(maxlen=300)


def cpu_slope(series):
    if len(series) < 5:
        return 0.0
    xs = np.arange(len(series))
    slope, _ = np.polyfit(xs, list(series), 1)
    return round(float(slope), 4)


def collect(duration_seconds: int):
    OUTPUT_PATH.parent.mkdir(exist_ok=True)

    fieldnames = [
        "timestamp", "cpu_percent", "ram_percent",
        "ram_used_mb", "ram_total_mb", "ram_available_mb",
        "disk_percent", "swap_percent", "cpu_freq_mhz",
        "net_sent_mb", "net_recv_mb",
        "cpu_1min_avg", "cpu_5min_avg",
        "ram_1min_avg", "ram_5min_avg",
        "ram_slope_30s", "cpu_slope_30s",
        "hour_of_day", "day_of_week",
        "label_ram_critical",
    ]

    file_exists = OUTPUT_PATH.exists()
    with open(OUTPUT_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        print(f"Collecting data for {duration_seconds}s → {OUTPUT_PATH}")
        print("Press Ctrl+C to stop early.\n")

        # Buffer of future RAM values (for labeling)
        future_ram = deque(maxlen=61)

        start = time.time()
        rows_written = 0

        while time.time() - start < duration_seconds:
            t0 = time.time()
            now = datetime.now()

            cpu    = psutil.cpu_percent(interval=None)
            mem    = psutil.virtual_memory()
            disk   = psutil.disk_usage("/")
            swap   = psutil.swap_memory()
            freq   = psutil.cpu_freq()
            net    = psutil.net_io_counters()

            ram    = mem.percent
            WINDOW_30.append((cpu, ram))
            WINDOW_60.append((cpu, ram))
            WINDOW_300.append((cpu, ram))
            future_ram.append(ram)

            cpu_30 = [x[0] for x in WINDOW_30]
            cpu_300 = [x[0] for x in WINDOW_300]
            ram_60 = [x[1] for x in WINDOW_60]
            ram_300 = [x[1] for x in WINDOW_300]

            # Label: will RAM exceed 85% in the next 60 readings?
            # We use future_ram buffer filled from previous iterations
            label = 1 if (len(future_ram) == 61 and max(list(future_ram)[1:]) > 85.0) else 0

            row = {
                "timestamp":       now.isoformat(),
                "cpu_percent":     round(cpu, 2),
                "ram_percent":     round(ram, 2),
                "ram_used_mb":     round(mem.used / (1024**2), 1),
                "ram_total_mb":    round(mem.total / (1024**2), 1),
                "ram_available_mb": round(mem.available / (1024**2), 1),
                "disk_percent":    round(disk.percent, 2),
                "swap_percent":    round(swap.percent, 2),
                "cpu_freq_mhz":    round(freq.current, 1) if freq else 0.0,
                "net_sent_mb":     round(net.bytes_sent / (1024**2), 2),
                "net_recv_mb":     round(net.bytes_recv / (1024**2), 2),
                "cpu_1min_avg":    round(float(np.mean(cpu_30)), 2),
                "cpu_5min_avg":    round(float(np.mean(cpu_300)) if len(cpu_300) > 1 else cpu, 2),
                "ram_1min_avg":    round(float(np.mean(ram_60)), 2),
                "ram_5min_avg":    round(float(np.mean(ram_300)) if len(ram_300) > 1 else ram, 2),
                "ram_slope_30s":   cpu_slope([x[1] for x in WINDOW_30]),
                "cpu_slope_30s":   cpu_slope(cpu_30),
                "hour_of_day":     now.hour,
                "day_of_week":     now.weekday(),
                "label_ram_critical": label,
            }
            writer.writerow(row)
            rows_written += 1

            elapsed = time.time() - start
            pct = (elapsed / duration_seconds) * 100
            bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
            print(f"\r[{bar}] {pct:.0f}%  {rows_written} rows  CPU:{cpu:.1f}%  RAM:{ram:.1f}%", end="", flush=True)

            sleep_time = SAMPLE_INTERVAL - (time.time() - t0)
            if sleep_time > 0:
                time.sleep(sleep_time)

    print(f"\n\nDone! {rows_written} rows saved to {OUTPUT_PATH}")
    print("Next step: run  python train_models.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect system data for ML training")
    parser.add_argument("--duration", type=int, default=300, help="Collection duration in seconds")
    args = parser.parse_args()
    collect(args.duration)
