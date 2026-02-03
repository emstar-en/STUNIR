# STUNIR Production Deployment

> Part of `docs/deployment/1069`

## Overview

This guide covers production deployment of STUNIR for enterprise environments.

## Production Requirements

### Hardware
| Component | Minimum | Recommended |
|-----------|---------|------------|
| CPU | 2 cores | 4+ cores |
| RAM | 4 GB | 8+ GB |
| Storage | 10 GB | 50+ GB SSD |

### Software
- Linux (Ubuntu 20.04+ / RHEL 8+) or macOS 12+
- Python 3.8+
- Git 2.20+
- Docker (optional, for containerized deployment)

## Deployment Options

### Option 1: Direct Installation

```bash
# Clone repository
git clone https://github.com/emstar-en/STUNIR.git /opt/stunir
cd /opt/stunir

# Install dependencies
pip install -r docs/requirements.txt

# Build native tools
cd tools/native/haskell/stunir-native
cabal build
cabal install --installdir=/usr/local/bin

# Set permissions
chmod +x /opt/stunir/scripts/*.sh
```

### Option 2: Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . /app/

RUN pip install -r docs/requirements.txt
RUN chmod +x scripts/*.sh

ENTRYPOINT ["./scripts/build.sh"]
```

```bash
# Build image
docker build -t stunir:latest .

# Run build
docker run -v $(pwd)/output:/app/output stunir:latest
```

### Option 3: Kubernetes Deployment

```yaml
# stunir-job.yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: stunir-build
spec:
  template:
    spec:
      containers:
      - name: stunir
        image: stunir:latest
        volumeMounts:
        - name: output
          mountPath: /app/output
      volumes:
      - name: output
        persistentVolumeClaim:
          claimName: stunir-pvc
      restartPolicy: Never
```

## Configuration

### Environment Configuration

```bash
# /etc/stunir/stunir.env
export STUNIR_PROFILE=native
export STUNIR_STRICT=true
export STUNIR_OUTPUT=/var/lib/stunir/output
export STUNIR_LOG_LEVEL=INFO
```

### Systemd Service

```ini
# /etc/systemd/system/stunir.service
[Unit]
Description=STUNIR Build Service
After=network.target

[Service]
Type=oneshot
User=stunir
Group=stunir
EnvironmentFile=/etc/stunir/stunir.env
WorkingDirectory=/opt/stunir
ExecStart=/opt/stunir/scripts/build.sh

[Install]
WantedBy=multi-user.target
```

## Monitoring

### Health Check

```bash
#!/bin/bash
# /opt/stunir/scripts/healthcheck.sh

# Verify Python
python3 --version || exit 1

# Verify tools
python3 -c "import tools" || exit 1

# Verify manifests module
python3 -c "import manifests" || exit 1

echo "STUNIR health check: OK"
```

### Prometheus Metrics

```python
# metrics.py
from prometheus_client import Counter, Histogram

build_total = Counter('stunir_builds_total', 'Total STUNIR builds')
build_duration = Histogram('stunir_build_duration_seconds', 'Build duration')
verify_failures = Counter('stunir_verify_failures_total', 'Verification failures')
```

## Security

### File Permissions
```bash
# Restrict access to sensitive files
chown -R stunir:stunir /opt/stunir
chmod 750 /opt/stunir
chmod 640 /opt/stunir/contracts/*
```

### Network Isolation
STUNIR does not require network access during builds. Isolate with:
```bash
# Run without network
unshare --net ./scripts/build.sh

# Or in Docker
docker run --network=none stunir:latest
```

## Backup & Recovery

### Backup Script
```bash
#!/bin/bash
tar -czf /backup/stunir-$(date +%Y%m%d).tar.gz \
  /opt/stunir/receipts \
  /opt/stunir/asm/ir \
  /opt/stunir/contracts
```

### Recovery
```bash
tar -xzf /backup/stunir-YYYYMMDD.tar.gz -C /
./scripts/verify_strict.sh --strict
```

## Related
- [Deployment Overview](README.md)
- [Troubleshooting](../troubleshooting/README.md)

---
*STUNIR Production Deployment v1.0*
