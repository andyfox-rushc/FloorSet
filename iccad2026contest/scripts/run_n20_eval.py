import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from iccad2026_evaluate import ContestEvaluator

evaluator = ContestEvaluator("../", verbose=True)
result = evaluator.evaluate("my_optimizer.py", test_ids=list(range(20)))

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

with open("n20_results_now.json", "w") as f:
    json.dump(out, f, indent=2)

print("\nWrote n20_results_now.json")
