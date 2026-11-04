#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
test -f .models/intent/config.json
chmod -R a+rX .models
sudo docker compose up --build -d
sudo docker compose ps
