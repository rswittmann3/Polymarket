"""One tick of the 30-second dashboard loop: wait, rebuild the data files, say what changed, exit.

    python tools/dashboard_tick.py 22        # sleep 22 s, then write OUT/trades.json and OUT/account.json

Prints one JSON line. `trades_changed` tells the caller whether trades.json needs uploading (it only
changes when an order is decided, filled or settled), so most ticks cost a single small upload.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import export_dashboard as E

OUT = Path("/tmp/claude-0/dash")


def main() -> None:
    time.sleep(float(sys.argv[1]) if len(sys.argv) > 1 else 0)
    OUT.mkdir(parents=True, exist_ok=True)
    trades = E.build_trades(E.load_records())
    account = E.build_account()
    tj = json.dumps(trades)
    h = hashlib.md5(tj.encode()).hexdigest()
    hf = OUT / "trades.hash"
    changed = not hf.exists() or hf.read_text() != h
    (OUT / "trades.json").write_text(tj)
    (OUT / "account.json").write_text(json.dumps([account]))
    hf.write_text(h)
    filled = [t for t in trades if t["status"] == "Filled"]
    done = [t for t in filled if t["result"] in ("Win", "Loss")]
    halted = [x for x in account["halted"].split(", ") if x and x != "none"]
    print(json.dumps({
        "trades_changed": changed, "bot": account["bot"], "halted": halted, "all_halted": len(halted) >= 4,
        "filled": len(filled), "open": sum(t["result"] == "Open" for t in filled),
        "wins": sum(t["result"] == "Win" for t in done), "losses": sum(t["result"] == "Loss" for t in done),
        "pnl": round(sum(t["pnl"] for t in done), 3), "balance": account["balance"], "updated": account["updated"]}))


if __name__ == "__main__":
    main()
