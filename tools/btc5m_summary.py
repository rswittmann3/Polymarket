"""Summarise data/btc5m/polls.jsonl: ask-sum distribution and how often the fee-adjusted arb cleared."""
import collections
import json
import sys
from pathlib import Path

rows = [json.loads(l) for l in Path(sys.argv[1] if len(sys.argv) > 1 else "data/btc5m/polls.jsonl").open() if '"sum"' in l]
if not rows:
    sys.exit("no polls")
print(f"{len(rows)} polls over {(rows[-1]['t'] - rows[0]['t']) / 60:.0f} min, {len({r['slug'] for r in rows})} windows")
c = collections.Counter(r["sum"] for r in rows)
for k in sorted(c):
    print(f"  {k:.2f}  {c[k]:6d}  {100 * c[k] / len(rows):5.1f}%")
print("fee-adjusted opportunities:", sum(r["opp"] for r in rows))
print("ask-sum < 1.00:", sum(r["sum"] < 1 for r in rows))
print("lowest:", min(rows, key=lambda r: r["sum"]))
