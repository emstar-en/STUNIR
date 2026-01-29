"""Metrics exporters for STUNIR telemetry."""

import json
import time
import socket
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pathlib import Path

from .metrics import MetricValue, MetricsRegistry, get_registry


class MetricsExporter(ABC):
    """Base class for metrics exporters."""
    
    @abstractmethod
    def export(self, metrics: List[MetricValue]) -> str:
        """Export metrics to string format."""
        pass
    
    def export_to_file(self, filepath: str, metrics: Optional[List[MetricValue]] = None) -> None:
        """Export metrics to file."""
        if metrics is None:
            metrics = get_registry().collect_all()
        content = self.export(metrics)
        Path(filepath).write_text(content)


class PrometheusExporter(MetricsExporter):
    """Export metrics in Prometheus text format."""
    
    def export(self, metrics: List[MetricValue]) -> str:
        lines = []
        seen_names = set()
        
        for m in metrics:
            base_name = m.name.split('_bucket')[0].split('_count')[0].split('_sum')[0]
            
            if base_name not in seen_names:
                lines.append(f"# TYPE {base_name} gauge")
                seen_names.add(base_name)
            
            if m.labels:
                labels_str = ','.join(f'{k}="{v}"' for k, v in sorted(m.labels.items()))
                lines.append(f"{m.name}{{{labels_str}}} {m.value}")
            else:
                lines.append(f"{m.name} {m.value}")
        
        return '\n'.join(lines) + '\n'
    
    def serve(self, port: int = 9090, path: str = '/metrics') -> None:
        """Start HTTP server for Prometheus scraping."""
        from http.server import HTTPServer, BaseHTTPRequestHandler
        
        exporter = self
        
        class MetricsHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == path:
                    metrics = get_registry().collect_all()
                    content = exporter.export(metrics).encode()
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/plain; version=0.0.4')
                    self.send_header('Content-Length', str(len(content)))
                    self.end_headers()
                    self.wfile.write(content)
                else:
                    self.send_response(404)
                    self.end_headers()
            
            def log_message(self, format, *args):
                pass  # Suppress logs
        
        server = HTTPServer(('0.0.0.0', port), MetricsHandler)
        server.serve_forever()


class JsonExporter(MetricsExporter):
    """Export metrics as JSON."""
    
    def __init__(self, pretty: bool = True):
        self.pretty = pretty
    
    def export(self, metrics: List[MetricValue]) -> str:
        data = {
            'timestamp': time.time(),
            'metrics': [
                {
                    'name': m.name,
                    'value': m.value,
                    'labels': m.labels,
                    'timestamp': m.timestamp,
                }
                for m in metrics
            ]
        }
        if self.pretty:
            return json.dumps(data, indent=2, sort_keys=True)
        return json.dumps(data, separators=(',', ':'))


class StatsdExporter(MetricsExporter):
    """Export metrics in StatsD format."""
    
    def __init__(self, host: str = 'localhost', port: int = 8125, prefix: str = 'stunir'):
        self.host = host
        self.port = port
        self.prefix = prefix
        self._socket: Optional[socket.socket] = None
    
    def _get_socket(self) -> socket.socket:
        if self._socket is None:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        return self._socket
    
    def export(self, metrics: List[MetricValue]) -> str:
        lines = []
        for m in metrics:
            name = f"{self.prefix}.{m.name}"
            if m.labels:
                tags = ','.join(f"{k}={v}" for k, v in sorted(m.labels.items()))
                lines.append(f"{name}:{m.value}|g|#{tags}")
            else:
                lines.append(f"{name}:{m.value}|g")
        return '\n'.join(lines)
    
    def send(self, metrics: Optional[List[MetricValue]] = None) -> None:
        """Send metrics to StatsD server."""
        if metrics is None:
            metrics = get_registry().collect_all()
        
        sock = self._get_socket()
        for m in metrics:
            name = f"{self.prefix}.{m.name}"
            data = f"{name}:{m.value}|g"
            try:
                sock.sendto(data.encode(), (self.host, self.port))
            except OSError:
                pass
    
    def close(self) -> None:
        if self._socket:
            self._socket.close()
            self._socket = None


class LogExporter(MetricsExporter):
    """Export metrics to log format."""
    
    def export(self, metrics: List[MetricValue]) -> str:
        lines = []
        for m in metrics:
            labels_str = ' '.join(f"{k}={v}" for k, v in sorted(m.labels.items()))
            lines.append(f"metric={m.name} value={m.value} {labels_str}".strip())
        return '\n'.join(lines)
