from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# Ensure local tabpfn (TabPFN/src) is importable, if not installed
THIS_DIR = Path(__file__).resolve().parent
TPFN_SRC = THIS_DIR / "src"
if TPFN_SRC.exists():
    sys.path.insert(0, str(TPFN_SRC))

from eval import evaluate_models_with_cv  # type: ignore  # noqa: E402


def main() -> None:
    data_path = THIS_DIR / "data" / "train.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"train.csv not found at {data_path}")

    df = pd.read_csv(data_path)

    results = evaluate_models_with_cv(
        df,
        date_col="date_id",
        target_col="market_forward_excess_returns",
        n_splits=5,
        embargo_days=5,
        include_tabpfn=True,
        random_state=42,
    )

    # Prepare artifacts directory
    exp_dir = THIS_DIR / "experiments"
    exp_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_dir = exp_dir / f"run_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write JSON summary
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Write CSV table
    rows = []
    for name, info in results.items():
        rows.append({
            "model": name,
            "mean_score": info.get("mean_score"),
            "fold_scores": info.get("fold_scores"),
        })
    pd.DataFrame(rows).to_csv(out_dir / "results.csv", index=False)

    # Console summary
    print("Model results (mean adjusted Sharpe):")
    for name, info in results.items():
        print(f"- {name}: {info.get('mean_score'):.6f}")
    print(f"\nArtifacts written to: {str(out_dir)}")


if __name__ == "__main__":
    main()


