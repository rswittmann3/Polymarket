"""Serve the live dashboard next to the bot.

    python tools/dashboard_server.py            # then open http://localhost:8765
    python tools/dashboard_server.py --port 9000

Reads the bot's logs in data/pmus/, asks Polymarket US for your balance (read-only, using the key in .env),
and answers the page's Refresh button at /api/data. Binds to localhost only: the page shows balances and
nothing here accepts a write. Run it from the repo root, on the machine where the bot writes its logs.
"""
from __future__ import annotations

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from export_dashboard import render_html, snapshot

CACHE_SECONDS = 5.0   # the exchange's public endpoint rate-limits; many tabs/refreshes share one build
_cache = {"at": 0.0, "snap": None}


def current() -> dict:
    if time.time() - _cache["at"] > CACHE_SECONDS or _cache["snap"] is None:
        _cache.update(at=time.time(), snap=snapshot())
    return _cache["snap"]


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?")[0]
        try:
            if path in ("/", "/index.html"):
                self._send(200, render_html().encode(), "text/html; charset=utf-8")   # no embed: page asks /api/data
            elif path == "/api/data":
                self._send(200, json.dumps(current()).encode(), "application/json")
            else:
                self._send(404, b"not found", "text/plain")
        except Exception as e:  # keep serving; the page shows the error
            self._send(500, json.dumps({"error": str(e)[:300]}).encode(), "application/json")

    def log_message(self, *a) -> None:   # quiet
        pass


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"dashboard on http://localhost:{args.port}  (Ctrl-C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
