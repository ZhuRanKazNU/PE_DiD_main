"""Run generate → train → test → infer."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.generate_questionnaire import generate_and_save  # noqa: E402
from src.infer import run_infer  # noqa: E402
from src.test import run_test  # noqa: E402
from src.train import train_all  # noqa: E402
from src.utils import load_config  # noqa: E402


if __name__ == "__main__":
    skip_placebo = "--skip-placebo" in sys.argv
    cfg = load_config()
    generate_and_save(cfg)
    train_all(cfg)
    run_test(cfg, skip_placebo=skip_placebo)
    run_infer(cfg)
