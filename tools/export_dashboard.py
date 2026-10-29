"""Turn the late-favorite bot's logs into the two JSON files the dashboard reads.

    python tools/export_dashboard.py OUT_DIR      # writes OUT_DIR/trades.json and OUT_DIR/account.json

trades.json: one row per decision (filled, unfilled, skipped, error), with P&L recomputed from the
fill price (never from the bot's own log line: early runs mis-priced short fills) and an `updates` JSON
string holding that order's timeline. account.json: one row with the live balance (read-only call),
whether the bot process is running, and per-asset halt state.
"""
from __future__ import annotations

import glob
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

START_BALANCE = 52.017503          # account cash before the first order of this session
ASSET_RE = re.compile(r"([a-z]+)-updown-5m-(\d{4})-(\d\d)-(\d\d)-(\d\d)(\d\d)z")

# First manual test (BTC 17:30 window): the record predates order read-back. Known from the account:
# bought 1.03 Up at an average of 0.98, settled Up, paid out 1.03.
MANUAL = {"DD6891Q012TT": {"fill": 0.98, "qty": 1.03, "won": True}}
UNFILLED_IDS = {"DD67Q58G12TT"}   # 17:33 order that expired with 0 contracts


def time_now() -> float:
    return datetime.now(timezone.utc).timestamp()


def market_result(asset: str, start: float):
    """'Up'/'Down' from the exchange once resolved, else None. One public call; cached per window."""
    import httpx
    slug = f"{asset.lower()}-updown-5m-{datetime.fromtimestamp(start, timezone.utc):%Y-%m-%d-%H%Mz}"
    if slug in _RES:
        return _RES[slug]
    res = None
    for _ in range(3):
        r = httpx.get(f"https://gateway.polymarket.us/v1/events/slug/{slug}", timeout=15)
        if r.status_code == 429:
            import time
            time.sleep(int(r.headers.get("retry-after", "10")))
            continue
        if r.status_code == 200:
            m = r.json().get("event", r.json())["markets"][0]
            if m.get("status") == "MARKET_STATUS_RESOLVED":
                yes, no = (float(x) for x in json.loads(m["outcomePrices"]))
                res = "Up" if yes > no else "Down" if no > yes else None
        break
    if res is not None:      # never cache "not resolved yet": a long-running server must ask again
        _RES[slug] = res
    return res


_RES: dict = {}


def iso(t: float) -> str:
    return datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_slug(slug: str):
    m = ASSET_RE.search(slug)
    asset = m.group(1).upper()
    start = datetime(int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5)), int(m.group(6)),
                     tzinfo=timezone.utc).timestamp()
    return asset, start


def load_records() -> list[dict]:
    rows = []
    for f in sorted(glob.glob("data/pmus/late_favorite*.jsonl"), key=os.path.getmtime):
        for line in open(f):
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_trades(records: list[dict]) -> list[dict]:
    settled = {}
    for r in records:
        if "settled" in r:
            settled[(r["settled"], (r.get("asset") or "").upper())] = r
    out, seen = [], set()
    for r in records:
        if "settled" in r or "slug" not in r:
            continue
        asset, start = parse_slug(r["slug"])
        key = (r["slug"], r["t"], bool(r.get("skipped")))
        if key in seen:
            continue
        seen.add(key)
        side, quoted = r.get("side"), r.get("ask")
        bid, offer = r.get("bid"), r.get("offer")
        book = f"long bid {bid[0]:.2f} / offer {offer[0]:.2f}" if bid and offer else ""
        ups = [{"t": iso(r["t"]), "text": f"Decided {side} at {quoted:.2f}, {r.get('left', 0)}s left ({book})"}]
        row = {"time": iso(r["t"]), "asset": asset, "window": f"{datetime.fromtimestamp(start, timezone.utc):%H:%M}",
               "side": side, "quoted": quoted, "fill": None, "qty": None, "cost": None, "status": "",
               "result": "", "pnl": None, "note": ""}
        if r.get("skipped"):
            row.update(status="Skipped", note=r["skipped"])
            ups = [{"t": iso(r["t"]), "text": f"Skipped {side} at {quoted:.2f}: {r['skipped']}"}]
            row["updates"] = json.dumps(ups)
            out.append(row)
            continue
        blob = json.dumps(r)
        manual_id = next((m for m in MANUAL if m in blob), None)   # its 404 read-back left only an error text
        if r.get("error") and not manual_id:
            row.update(status="Error", note=r["error"][:200])
            ups.append({"t": iso(r["t"]), "text": "Order error: " + r["error"][:160]})
            row["updates"] = json.dumps(ups)
            out.append(row)
            continue
        order = r.get("order") or {}
        oid = manual_id or (r.get("result") or {}).get("id") or order.get("id") or ""
        if not r.get("live"):
            row.update(status="Dry run", qty=r.get("qty"))
            row["updates"] = json.dumps(ups)
            out.append(row)
            continue
        qty, fill, comm, won = 0.0, None, 0.0, None
        if oid in MANUAL:
            m = MANUAL[oid]
            qty, fill, won = m["qty"], m["fill"], m["won"]
        elif oid in UNFILLED_IDS or not order:
            qty = 0.0
        else:
            qty = float(order.get("cumQuantity") or 0)
            raw = float((order.get("avgPx") or {}).get("value") or 0)
            if qty > 0 and raw > 0:
                fill = raw if side == "Up" else 1 - raw     # exchange reports the Up (long) scale
            comm = float((order.get("commissionNotionalTotalCollected") or {}).get("value") or 0)
        if qty <= 0:
            row.update(status="Unfilled", note="Fill-or-kill expired with nothing filled", qty=0.0)
            ups.append({"t": iso(r["t"] + 1), "text": "Order expired unfilled (book moved); no money spent"})
            row["updates"] = json.dumps(ups)
            out.append(row)
            continue
        row.update(status="Filled", fill=round(fill, 4), qty=round(qty, 2), cost=round(qty * fill + comm, 4))
        ups.append({"t": iso(r["t"] + 1), "text": f"Filled {qty:g} at {fill:.2f} (quoted {quoted:.2f})"
                    + (f", fee {comm:.2f}" if comm else "")})
        s = settled.get((r["slug"], asset))
        if s is not None:
            won = bool(s["won"])
        elif time_now() > start + 300 + 90:          # closed a while ago but the bot never logged it
            res = market_result(asset, start)
            if res is not None:
                won = res == side
        if won is None:
            row.update(result="Open", note="Window not settled yet")
        else:
            pnl = qty * (1 - fill) - comm if won else -(qty * fill) - comm
            row.update(result="Win" if won else "Loss", pnl=round(pnl, 4))
            when = s.get("t") if s else None
            ups.append({"t": iso(when if when else start + 300),
                        "text": f"Settled {'in your favor' if won else 'against you'}: {'WIN' if won else 'LOSS'} {pnl:+.3f}"})
        row["updates"] = json.dumps(ups)
        out.append(row)
    out.sort(key=lambda x: x["time"])
    for row in out:   # a unique key and the latest update, for the table
        row["id"] = f"{row['asset']}-{row['time']}"
        row["last"] = json.loads(row["updates"])[-1]["text"]
    return out


def bot_running() -> bool:
    """Heartbeat file first (works on any OS); `ps` as a fallback for a bot started before heartbeats existed."""
    hb = Path("data/pmus/heartbeat")
    try:
        if hb.exists() and time_now() - float(hb.read_text()) < 90:
            return True
    except ValueError:
        pass
    try:
        ps = subprocess.run(["ps", "aux"], capture_output=True, text=True).stdout
    except (FileNotFoundError, OSError):
        return False
    return any("pmus_late_favorite" in ln and "python" in ln and "export_dashboard" not in ln for ln in ps.splitlines())


def build_account() -> dict:
    from dotenv import dotenv_values
    from pm_edges.pmus import Signer, Trader
    v = dotenv_values(".env")
    b = Trader(Signer(v["POLYMARKET_KEY_ID"], v["POLYMARKET_SECRET_KEY"])).balances()["balances"][0]
    state = json.loads(Path("data/pmus/state.json").read_text()) if Path("data/pmus/state.json").exists() else {}
    halted = state.get("halted", {})
    return {"updated": iso(datetime.now(timezone.utc).timestamp()),
            "bot": "Running" if bot_running() else "Stopped",
            "balance": round(b["currentBalance"], 4), "buying_power": round(b["buyingPower"], 4),
            "start_balance": START_BALANCE, "halted": ", ".join(f"{k.upper()}" for k in halted) or "none",
            "stop_all": bool(state.get("stop_all"))}


def snapshot() -> dict:
    """Everything the dashboard page needs, fresh."""
    trades = build_trades(load_records())
    try:
        account = build_account()
    except Exception as e:   # no credentials / exchange unreachable: the page still shows the orders
        account = {"error": str(e)[:200], "updated": iso(time_now()), "bot": "Running" if bot_running() else "Stopped",
                   "balance": None, "buying_power": None, "start_balance": START_BALANCE, "halted": "", "stop_all": False}
    return {"trades": trades, "account": account, "generated": iso(time_now())}


def render_html(snap: dict | None = None) -> str:
    tpl = (Path(__file__).parent / "dashboard_template.html").read_text()
    return tpl.replace("/*EMBED*/null", json.dumps(snap) if snap else "null")


def main() -> None:
    if len(sys.argv) >= 3 and sys.argv[1] == "--html":
        every = float(sys.argv[sys.argv.index("--every") + 1]) if "--every" in sys.argv else 0
        out = Path(sys.argv[2])
        while True:
            try:
                tmp = out.with_suffix(".tmp")
                tmp.write_text(render_html(snapshot()))
                tmp.replace(out)                      # atomic: a reader never sees half a file
                print("wrote", out, iso(time_now()), flush=True)
            except Exception as e:                    # keep the loop alive through a network blip
                print("export failed:", e, flush=True)
            if not every:
                return
            import time
            time.sleep(every)
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    trades = build_trades(load_records())
    (out / "trades.json").write_text(json.dumps(trades))
    (out / "account.json").write_text(json.dumps([build_account()]))
    filled = [t for t in trades if t["status"] == "Filled"]
    print(f"{len(trades)} rows, {len(filled)} filled, "
          f"pnl {sum(t['pnl'] or 0 for t in filled):+.3f} -> {out}")


if __name__ == "__main__":
    main()
