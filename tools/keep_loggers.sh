#!/usr/bin/env bash
# Keep the Novig quote recorder and the DraftKings/injury poller running, and commit their logs so a container reset cannot lose them.
# Run from the repo root (the scheduled "Keep loggers alive" task does).  Read-only: neither logger can place an order.
set -u
cd "$(dirname "$0")/.."
set -a; [ -f .env ] && . ./.env; set +a
mkdir -p data/lag data/novig snapshots/lag snapshots/novig
[ -f data/lag/timeline.jsonl ] || python tools/sync_lag_data.py restore 2>/dev/null || true
for f in snapshots/novig/*.gz; do [ -e "$f" ] && [ ! -e "data/novig/$(basename "$f")" ] && cp "$f" data/novig/; done
started=""
if ! pgrep -f "novig_mm record" >/dev/null; then
  setsid -f nohup python -m novig_mm record --leagues NBA,NHL --interval 30 --duration 129600 --window-hours 36 --out data/novig >> data/novig/recorder.log 2>&1 < /dev/null
  started="$started novig_mm"
fi
if ! pgrep -f "tools/line_poller.py" >/dev/null; then
  setsid -f nohup env -u ODDS_API_KEY python tools/line_poller.py --hours 36 --every 3 >> data/lag/poller.log 2>&1 < /dev/null
  started="$started line_poller"
fi
python tools/sync_lag_data.py save >/dev/null 2>&1
cp -u data/novig/touch_*.jsonl.gz data/novig/meta_*.jsonl.gz snapshots/novig/ 2>/dev/null
git add snapshots/lag/timeline.jsonl.gz snapshots/novig 2>/dev/null
if ! git diff --cached --quiet; then
  git commit -q -m "Save quote and injury logs ($(date -u +%Y-%m-%dT%H:%MZ))

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01TgNRZx8Ayx7nTABT3ctLSx"
  git pull -q --rebase origin claude/new-session-uikz12 2>&1 | tail -1
  git push -q origin claude/new-session-uikz12 2>&1 | tail -1
fi
echo "restarted:${started:- none} | novig_mm=$(pgrep -fc 'novig_mm record') poller=$(pgrep -fc 'tools/line_poller.py') | touch rows today: $(zcat data/novig/touch_$(date -u +%F).jsonl.gz 2>/dev/null | wc -l)"
