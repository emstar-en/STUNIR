"""Resource monitoring for STUNIR."""

import os
import time
import threading
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class ResourceUsage:
    """Current resource usage snapshot."""
    memory_rss_mb: float
    memory_vms_mb: float
    cpu_percent: float
    open_files: int
    threads: int
    uptime_seconds: float


class ResourceMonitor:
    """Monitor system and process resources."""
    
    def __init__(self, sample_interval: float = 1.0):
        self.sample_interval = sample_interval
        self._pid = os.getpid()
        self._start_time = time.time()
        self._samples: List[ResourceUsage] = []
        self._lock = threading.Lock()
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
    
    def get_current_usage(self) -> ResourceUsage:
        """Get current resource usage."""
        memory_rss = 0.0
        memory_vms = 0.0
        cpu_percent = 0.0
        open_files = 0
        threads = 1
        
        # Memory from /proc
        try:
            with open(f'/proc/{self._pid}/statm', 'r') as f:
                parts = f.read().split()
                page_size = os.sysconf('SC_PAGE_SIZE')
                memory_vms = int(parts[0]) * page_size / (1024 * 1024)
                memory_rss = int(parts[1]) * page_size / (1024 * 1024)
        except (IOError, IndexError, ValueError):
            pass
        
        # Thread count
        try:
            with open(f'/proc/{self._pid}/stat', 'r') as f:
                parts = f.read().split()
                threads = int(parts[19])
        except (IOError, IndexError, ValueError):
            threads = threading.active_count()
        
        # Open files
        try:
            open_files = len(os.listdir(f'/proc/{self._pid}/fd'))
        except (OSError, IOError):
            pass
        
        return ResourceUsage(
            memory_rss_mb=memory_rss,
            memory_vms_mb=memory_vms,
            cpu_percent=cpu_percent,
            open_files=open_files,
            threads=threads,
            uptime_seconds=time.time() - self._start_time,
        )
    
    def start_monitoring(self) -> None:
        """Start background monitoring."""
        if self._monitor_thread is not None:
            return
        
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
    
    def stop_monitoring(self) -> None:
        """Stop background monitoring."""
        self._stop_event.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
            self._monitor_thread = None
    
    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while not self._stop_event.is_set():
            try:
                usage = self.get_current_usage()
                with self._lock:
                    self._samples.append(usage)
                    # Keep last 1000 samples
                    if len(self._samples) > 1000:
                        self._samples = self._samples[-1000:]
            except Exception:
                pass
            self._stop_event.wait(self.sample_interval)
    
    def get_samples(self, count: int = 100) -> List[ResourceUsage]:
        """Get recent samples."""
        with self._lock:
            return list(self._samples[-count:])
    
    def get_stats(self) -> Dict[str, float]:
        """Get statistics from samples."""
        with self._lock:
            if not self._samples:
                return {}
            
            return {
                'memory_rss_avg_mb': sum(s.memory_rss_mb for s in self._samples) / len(self._samples),
                'memory_rss_max_mb': max(s.memory_rss_mb for s in self._samples),
                'threads_avg': sum(s.threads for s in self._samples) / len(self._samples),
                'threads_max': max(s.threads for s in self._samples),
                'open_files_avg': sum(s.open_files for s in self._samples) / len(self._samples),
                'open_files_max': max(s.open_files for s in self._samples),
                'sample_count': len(self._samples),
            }


def get_resource_usage() -> ResourceUsage:
    """Get current resource usage (one-time snapshot)."""
    return ResourceMonitor().get_current_usage()
