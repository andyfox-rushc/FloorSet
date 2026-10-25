"""Replace rows of a prime_evaluate.py result file by rows of a rerun (same ids).

  python scripts/merge_results.py base.json rerun.json out.json
"""
import json
import sys

base = {r["id"]: r for r in json.load(open(sys.argv[1]))}
for r in json.load(open(sys.argv[2])):
    base[r["id"]] = r
json.dump([base[i] for i in sorted(base)], open(sys.argv[3], "w"))
print(f"merged {sys.argv[2]} into {sys.argv[1]} -> {sys.argv[3]}")
