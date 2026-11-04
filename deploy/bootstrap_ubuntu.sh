#!/usr/bin/env bash
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y ca-certificates curl git python3-venv
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
chmod a+r /etc/apt/keyrings/docker.asc
. /etc/os-release
printf 'deb [arch=%s signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu %s stable\n' "$(dpkg --print-architecture)" "$VERSION_CODENAME" > /etc/apt/sources.list.d/docker.list
apt-get update
apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
systemctl enable --now docker
install -d -o ubuntu -g ubuntu /opt/northstar
if [ ! -f /swapfile ]; then
  fallocate -l 8G /swapfile
  chmod 600 /swapfile
  mkswap /swapfile
  swapon /swapfile
  printf '/swapfile none swap sw 0 0\n' >> /etc/fstab
fi
printf 'Host preparation complete.\n'
