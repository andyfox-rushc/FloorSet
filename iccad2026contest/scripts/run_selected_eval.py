import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from iccad2026_evaluate import ContestEvaluator

p = argparse.ArgumentParser()
p.add_argument("--test-ids", type=int, nargs="+", required=True)
p.add_argument("--output", default="selected_results_now.json")
args = p.parse_args()

evaluator = ContestEvaluator("../", verbose=True)
result = evaluator.evaluate("my_optimizer.py", test_ids=args.test_ids)

out = {
    "submission_name": result.submission_name,
    "timestamp": result.timestamp,
    "total_score": result.total_score,
    "test_results": [
        {k: v for k, v in asdict(r).items() if k != "positions"}
        for r in result.test_results
    ],
    "summary": result.summary,
}

with open(args.output, "w") as f:
    json.dump(out, f, indent=2)

print(f"\nWrote {args.output}")
