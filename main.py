import os
import sys
import argparse
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Mukti: Multi-Crop Leaf Disease Classification Using Explainable AI (XAI)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available commands:
  split      Generate leakage-free group-stratified train/val/test splits
  train      Train model with two-phase transfer learning (frozen -> fine-tune)
  evaluate   Evaluate trained checkpoint on test split (Accuracy, Macro-F1, Confusion Matrix)
  xai        Generate Grad-CAM / Grad-CAM++ visualizations and quantitative localization metrics
  test       Run automated unit and integration tests (pytest)

Examples:
  python main.py split
  python main.py train --model efficientnet_b0
  python main.py evaluate --checkpoint models/best_model.pth
  python main.py xai --checkpoint models/best_model.pth --samples-per-class 2
  python main.py test
""",
    )
    parser.add_argument(
        "command",
        choices=["split", "train", "evaluate", "xai", "test"],
        help="Command to run",
    )

    args, remaining_args = parser.parse_known_args()

    python_bin = sys.executable

    if args.command == "split":
        cmd = [python_bin, "scripts/01_prepare_splits.py"] + remaining_args
    elif args.command == "train":
        cmd = [python_bin, "scripts/02_train.py"] + remaining_args
    elif args.command == "evaluate":
        cmd = [python_bin, "scripts/03_evaluate.py"] + remaining_args
    elif args.command == "xai":
        cmd = [python_bin, "scripts/04_generate_xai.py"] + remaining_args
    elif args.command == "test":
        cmd = [python_bin, "-m", "pytest", "tests/", "-v"] + remaining_args
    else:
        parser.print_help()
        sys.exit(1)

    project_root = str(Path(__file__).resolve().parent)
    env = os.environ.copy()
    env["PYTHONPATH"] = project_root + (f":{env['PYTHONPATH']}" if "PYTHONPATH" in env else "")

    result = subprocess.run(cmd, env=env)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
