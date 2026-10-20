"""Keep the line-poller timeline alive across container resets.

    python tools/sync_lag_data.py save       # gzip data/lag/timeline.jsonl into snapshots/lag/ (commit it)
    python tools/sync_lag_data.py restore    # merge the committed snapshot back into data/lag/ if the live file is shorter
"""
from __future__ import annotations

import argparse
import gzip
import shutil
from pathlib import Path

LIVE = Path("data/lag/timeline.jsonl")
SNAP = Path("snapshots/lag/timeline.jsonl.gz")


def save() -> None:
    SNAP.parent.mkdir(parents=True, exist_ok=True)
    with LIVE.open("rb") as src, gzip.open(SNAP, "wb", compresslevel=9) as dst:
        shutil.copyfileobj(src, dst)
    print(f"saved {LIVE.stat().st_size:,} bytes as {SNAP.stat().st_size:,} bytes: {SNAP}")


def restore() -> None:
    if not SNAP.exists():
        raise SystemExit("no snapshot committed yet")
    LIVE.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(SNAP, "rt") as f:
        saved = f.read().splitlines()
    live = LIVE.read_text().splitlines() if LIVE.exists() else []
    merged = list(dict.fromkeys(saved + live))                       # keep order, drop exact duplicates
    merged.sort(key=lambda l: l.split('"ts": "', 1)[1][:19] if '"ts": "' in l else "")
    LIVE.write_text("\n".join(merged) + "\n")
    print(f"timeline now has {len(merged):,} lines ({len(saved):,} from the snapshot, {len(live):,} live)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["save", "restore"])
    {"save": save, "restore": restore}[ap.parse_args().cmd]()
