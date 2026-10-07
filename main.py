"""Command-line entry: generate / train / test / infer / run_all."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# allow `python main.py` from PE_DiD_main/
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.generate_questionnaire import generate_and_save
from src.infer import run_infer
from src.test import run_test
from src.train import train_all
from src.utils import load_config


def main():
    parser = argparse.ArgumentParser(
        description="PE-DiD: time-varying DID + PSM-DID for college PE reform and mental health"
    )
    parser.add_argument(
        "command",
        choices=["generate", "train", "test", "infer", "run_all"],
        help="generate questionnaire data, train networks, evaluate (Ch.4), or infer",
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--student_id", type=str, default=None)
    parser.add_argument("--skip-placebo", action="store_true",
                        help="skip the 500-rep placebo test (faster debugging)")
    args = parser.parse_args()
    cfg = load_config(args.config)

    if args.command == "generate":
        paths = generate_and_save(cfg)
        print("Generated files:")
        for k, p in paths.items():
            print(f"  {k}: {p}")
    elif args.command == "train":
        train_all(cfg)
    elif args.command == "test":
        run_test(cfg, skip_placebo=args.skip_placebo)
    elif args.command == "infer":
        run_infer(cfg, student_id=args.student_id)
    elif args.command == "run_all":
        generate_and_save(cfg)
        train_all(cfg)
        run_test(cfg, skip_placebo=args.skip_placebo)
        run_infer(cfg, student_id=args.student_id)


if __name__ == "__main__":
    main()
