#!/bin/bash
# snapshot.sh — runs once during /warm to build the snapshot baseline.
# Output of this script is what gets baked into the persisted snapshot.
# Edit this file, then commit and push to keep changes.
# See also: .calude/boot.sh (runs after every snapshot restore).
set -euo pipefail

sudo apt-get update
sudo apt-get install -y ffmpeg libsm6 libxext6

if ! command -v pdm >/dev/null 2>&1; then
  pip install --user pdm
fi

export PATH="$HOME/.local/bin:$PATH"

changed pdm.lock pyproject.toml && pdm install

redis-server --daemonize yes