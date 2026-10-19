#!/usr/bin/env python3
"""
Convert posterior Z samples to binary edge ensemble M (one binary vector per draw).

  M_edge = 1 if Z_edge > 0 else 0

From repo root:
  python script/run_z_to_m.py \\
    --merged script/results/cullin_wt_config_multi_seeds/merged/merged_chains.pkl \\
    --out updated_analysis/cullin_wt_binary

From script/:
  python run_z_to_m.py --merged results/.../merged_chains.pkl --out ../updated_analysis/my_run
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from binary_ensemble import run_pipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    p = argparse.ArgumentParser(description="Z -> binary M ensemble from MCMC pickle")
    p.add_argument(
        "--merged",
        type=Path,
        required=True,
        help="Path to merged_chains.pkl (or any MCMC results pkl with samples['z'])",
    )
    p.add_argument(
        "--model-data",
        type=Path,
        default=None,
        help="Path to *_model_data.pkl (optional if discoverable under dataset dir)",
    )
    p.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output directory (e.g. under updated_analysis/)",
    )
    p.add_argument(
        "--save-adjacency",
        action="store_true",
        help="Also save (T,N,N) binary_adjacency.npy (larger file)",
    )
    args = p.parse_args()

    out = run_pipeline(
        merged_or_results_pkl=args.merged,
        model_data_pkl=args.model_data,
        output_dir=args.out,
        save_adjacency=args.save_adjacency,
    )
    print(f"Done: {out}")


if __name__ == "__main__":
    main()
