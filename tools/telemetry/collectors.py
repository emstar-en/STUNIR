"""System metrics collectors for STUNIR telemetry."""

import os
import time
import platform
import threading
from typing import Dict, Optional

from .metrics import Gauge, get_registry


class SystemCollector:
    """Collect system-level metrics."""
    
    def __init__(self, registry=None):
        self.registry = registry or get_registry()
        self._setup_metrics()
    
    def _setup_metrics(self) -> None:
        self.cpu_usage = self.registry.gauge('system_cpu_usage_percent', 'CPU usage percentage')
        self.memory_used = self.registry.gauge('system_memory_used_bytes', 'Memory used in bytes')
        self.memory_available = self.registry.gauge('system_memory_available_bytes', 'Memory available')
        self.disk_used = self.registry.gauge('system_disk_used_bytes', 'Disk used in bytes')
        self.disk_free = self.registry.gauge('system_disk_free_bytes', 'Disk free in bytes')
        self.load_average = self.registry.gauge('system_load_average', 'System load average')
    
    def collect(self) -> Dict[str, float]:
        """Collect current system metrics."""
        metrics = {}
        
        # Load average (Unix only)
        try:
            load = os.getloadavg()
            self.load_average.set(load[0], period='1m')
            self.load_average.set(load[1], period='5m')
            self.load_average.set(load[2], period='15m')
            metrics['load_1m'] = load[0]
        except (OSError, AttributeError):
            pass
        
        # Memory info (Linux)
        try:
            with open('/proc/meminfo', 'r') as f:
                meminfo = {}
                for line in f:
                    parts = line.split(':')
                    if len(parts) == 2:
                        key = parts[0].strip()
                        value = parts[1].strip().split()[0]
                        meminfo[key] = int(value) * 1024  # Convert to bytes
                
                total = meminfo.get('MemTotal', 0)
                free = meminfo.get('MemFree', 0)
                available = meminfo.get('MemAvailable', free)
                used = total - available
                
                self.memory_used.set(used)
                self.memory_available.set(available)
                metrics['memory_used'] = used
                metrics['memory_available'] = available
        except (IOError, KeyError, ValueError):
            pass
        
        # Disk info
        try:
            stat = os.statvfs('/')
            total = stat.f_blocks * stat.f_frsize
            free = stat.f_bfree * stat.f_frsize
            used = total - free
            
            self.disk_used.set(used, mount='/')
            self.disk_free.set(free, mount='/')
            metrics['disk_used'] = used
            metrics['disk_free'] = free
        except (OSError, AttributeError):
            pass
        
        return metrics


class ProcessCollector:
    """Collect process-level metrics."""
    
    def __init__(self, registry=None):
        self.registry = registry or get_registry()
        self._pid = os.getpid()
        self._setup_metrics()
    
    def _setup_metrics(self) -> None:
        self.cpu_time = self.registry.gauge('process_cpu_seconds_total', 'Total CPU time')
        self.memory_rss = self.registry.gauge('process_memory_rss_bytes', 'Resident memory size')
        self.memory_vms = self.registry.gauge('process_memory_vms_bytes', 'Virtual memory size')
        self.open_fds = self.registry.gauge('process_open_fds', 'Open file descriptors')
        self.threads = self.registry.gauge('process_threads', 'Number of threads')
        self.start_time = self.registry.gauge('process_start_time_seconds', 'Process start time')
    
    def collect(self) -> Dict[str, float]:
        """Collect current process metrics."""
        metrics = {}
        
        # Process stats from /proc (Linux)
        try:
            with open(f'/proc/{self._pid}/stat', 'r') as f:
                stat = f.read().split()
                utime = int(stat[13]) / os.sysconf('SC_CLK_TCK')
                stime = int(stat[14]) / os.sysconf('SC_CLK_TCK')
                num_threads = int(stat[19])
                
                self.cpu_time.set(utime + stime)
                self.threads.set(num_threads)
                metrics['cpu_seconds'] = utime + stime
                metrics['threads'] = num_threads
        except (IOError, IndexError, ValueError):
            pass
        
        # Memory stats
        try:
            with open(f'/proc/{self._pid}/statm', 'r') as f:
                statm = f.read().split()
                page_size = os.sysconf('SC_PAGE_SIZE')
                vms = int(statm[0]) * page_size
                rss = int(statm[1]) * page_size
                
                self.memory_rss.set(rss)
                self.memory_vms.set(vms)
                metrics['memory_rss'] = rss
                metrics['memory_vms'] = vms
        except (IOError, IndexError, ValueError):
            pass
        
        # Open file descriptors
        try:
            fd_dir = f'/proc/{self._pid}/fd'
            num_fds = len(os.listdir(fd_dir))
            self.open_fds.set(num_fds)
            metrics['open_fds'] = num_fds
        except (OSError, IOError):
            pass
        
        return metrics


class MetricsCollectorThread(threading.Thread):
    """Background thread for periodic metrics collection."""
    
    def __init__(self, interval: float = 15.0, collectors=None):
        super().__init__(daemon=True)
        self.interval = interval
        self.collectors = collectors or [SystemCollector(), ProcessCollector()]
        self._stop_event = threading.Event()
    
    def run(self) -> None:
        while not self._stop_event.is_set():
            for collector in self.collectors:
                try:
                    collector.collect()
                except Exception:
                    pass
            self._stop_event.wait(self.interval)
    
    def stop(self) -> None:
        self._stop_event.set()


def collect_system_metrics() -> Dict[str, float]:
    """One-time collection of all system metrics."""
    metrics = {}
    metrics.update(SystemCollector().collect())
    metrics.update(ProcessCollector().collect())
    return metrics
