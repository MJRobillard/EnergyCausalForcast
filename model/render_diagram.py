"""
Render the SCM probabilistic graphical model using pyro.render_model().

Usage:
    python model/render_diagram.py
    python model/render_diagram.py --out results/scm_model.pdf
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from model.scm import render_diagram


def main() -> None:
    parser = argparse.ArgumentParser(description="Render SCM Pyro model diagram")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).parent.parent / "results" / "scm_model.pdf",
        help="Output file (PDF, PNG, or SVG)",
    )
    parser.add_argument(
        "--generative",
        action="store_true",
        help="Render fully generative graph (no observed weather/demand)",
    )
    args = parser.parse_args()
    render_diagram(str(args.out), conditioned=not args.generative)
    print(f"Saved diagram to {args.out}")


if __name__ == "__main__":
    main()
