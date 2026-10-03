#!/usr/bin/env python3
"""Spawn the ministral pipeline as a fully detached process (survives session end)."""
import os, subprocess, sys
from pathlib import Path

repo = Path(__file__).parent.parent
log = Path("/tmp/ministral_pipeline_full.log")
pid_file = Path("/tmp/ministral_pipeline.pid")

p = subprocess.Popen(
    ["bash", str(repo / "scripts/run_ministral_pipeline.sh")],
    stdout=open(log, "w"),
    stderr=subprocess.STDOUT,
    cwd=str(repo),
    start_new_session=True,   # detach from parent process group
    close_fds=True,
)
pid_file.write_text(str(p.pid))
print(f"Pipeline started: PID={p.pid}")
print(f"Log: {log}")
print(f"PID file: {pid_file}")
