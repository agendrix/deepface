#!/bin/bash
# boot.sh — runs after every snapshot restore in /invoke.
# Boots services, reconciles drift via the `changed` helper, then runs
# long-lived (typically `bin/dev &` + `wait`).
# Edit this file, then commit and push to keep changes.
# See also: .calude/snapshot.sh (runs once when the snapshot is built).
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"

redis-server --daemonize yes

changed pdm.lock pyproject.toml && pdm install

cd deepface/api/src
pdm run gunicorn --workers=1 --timeout=3600 --bind=0.0.0.0:5000 'app:create_app()' &
wait