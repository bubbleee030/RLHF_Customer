#!/usr/bin/env bash
set -euo pipefail

if [[ ${EUID} -ne 0 ]]; then
  echo "Please run with sudo: sudo bash scripts/fix_gpu_runtime.sh"
  exit 1
fi

if [[ -f /var/run/cdi/nvidia.yaml ]]; then
  cp /var/run/cdi/nvidia.yaml /var/run/cdi/nvidia.yaml.bak
  sed -i '/\/run\/nvidia-persistenced\/socket/d' /var/run/cdi/nvidia.yaml
fi

mkdir -p /run/nvidia-persistenced
if [[ ! -e /run/nvidia-persistenced/socket ]]; then
  touch /run/nvidia-persistenced/socket
fi
chmod 666 /run/nvidia-persistenced/socket

modprobe nvidia || true
modprobe nvidia_uvm || true
modprobe nvidia_drm || true

systemctl restart docker || true

echo "GPU runtime patch applied."
