"""Resource monitoring utilities."""

import csv
import threading
import time
from datetime import datetime
from pathlib import Path

import psutil

try:
    import GPUtil
except ImportError:
    GPUtil = None  # GPU monitoring will be disabled if GPUtil is not available


class ResourceMonitor:
    """Logs CPU, RAM, disk, process memory, and GPU metrics to CSV."""

    def __init__(self, output_path: Path, interval: float = 2.0) -> None:
        self.output_path = output_path
        self.interval = interval
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._process = psutil.Process()

    def start(self) -> None:
        """Start monitoring in a background thread."""
        if self._thread is not None and self._thread.is_alive():
            return  # Already running
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop monitoring."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()

    @staticmethod
    def _get_gpu_metrics() -> dict:
        """Return GPU metrics if a supported GPU is available."""
        gpu_metrics = {
            "gpu_available": False,
            "gpu_name": "Not available",
            "gpu_load_percent": None,
            "gpu_memory_used_mb": None,
            "gpu_memory_total_mb": None,
        }

        if GPUtil is None:
            return gpu_metrics

        try:
            gpus = GPUtil.getGPUs()
        except Exception:
            return gpu_metrics
        if not gpus:
            return gpu_metrics

        gpu = gpus[0]  # Monitor the first GPU
        gpu_metrics.update(
            {
                "gpu_available": True,
                "gpu_name": gpu.name,
                "gpu_load_percent": round(gpu.load * 100, 2),
                "gpu_memory_used_mb": gpu.memoryUsed,
                "gpu_memory_total_mb": gpu.memoryTotal,
            }
        )
        return gpu_metrics

    def _run(self) -> None:
        """Write monitoring rows until stopped."""

        fieldnames = [
            "timestamp",
            "cpu_percent",
            "memory_percent",
            "memory_used_mb",
            "memory_available_mb",
            "process_memory_mb",
            "disk_usage_percent",
            "gpu_available",
            "gpu_name",
            "gpu_load_percent",
            "gpu_memory_used_mb",
            "gpu_memory_total_mb",
        ]

        with self.output_path.open("w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            while not self._stop_event.is_set():
                memory_info = self._process.memory_info()
                gpu_metrics = self._get_gpu_metrics()

                row = {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "cpu_percent": psutil.cpu_percent(interval=None),
                    "memory_percent": psutil.virtual_memory().percent,
                    "memory_used_mb": round(
                        psutil.virtual_memory().used / (1024 * 1024), 2
                    ),
                    "memory_available_mb": round(
                        psutil.virtual_memory().available / (1024 * 1024), 2
                    ),
                    "process_memory_mb": round(memory_info.rss / (1024 * 1024), 2),
                    "disk_usage_percent": psutil.disk_usage("/").percent,
                }

                row.update(gpu_metrics)
                writer.writerow(row)
                csvfile.flush()  # Ensure data is written to disk
                time.sleep(self.interval)
