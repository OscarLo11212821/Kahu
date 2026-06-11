#!/usr/bin/env python3
"""Run an NNUE A/B tournament: same engine binary, two evaluation networks.

Examples:
  python3 nnue_tournament.py
  python3 nnue_tournament.py --fast
  python3 nnue_tournament.py --nnue-a kuba.nnue --nnue-b kuba_2.nnue --openings 8
"""
import argparse
import sys
from pathlib import Path

from tournament import OPENINGS, run_tournament

ROOT = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description="Kuba NNUE head-to-head tournament")
    parser.add_argument("--engine", default=str(ROOT / "build/bin/kuba_engine"),
                        help="Path to kuba_engine")
    parser.add_argument("--nnue-a", default=str(ROOT / "kuba.nnue"),
                        help="First NNUE weights file")
    parser.add_argument("--nnue-b", default=str(ROOT / "kuba_2.nnue"),
                        help="Second NNUE weights file")
    parser.add_argument("--name-a", default="kuba.nnue")
    parser.add_argument("--name-b", default="kuba_2.nnue")
    parser.add_argument("--openings", type=int, default=8,
                        help="Opening count (1-%d, default all)" % len(OPENINGS))
    parser.add_argument("--fast", action="store_true",
                        help="Quick check: 3 openings (6 games)")
    args = parser.parse_args()

    engine = Path(args.engine)
    nnue_a = Path(args.nnue_a)
    nnue_b = Path(args.nnue_b)
    for label, path in [("engine", engine), ("nnue-a", nnue_a), ("nnue-b", nnue_b)]:
        if not path.is_file():
            print(f"error: {label} not found: {path}", file=sys.stderr)
            return 1

    count = 3 if args.fast else max(1, min(args.openings, len(OPENINGS)))
    book = OPENINGS[:count]

    print("=" * 50)
    print(" KUBA NNUE TOURNAMENT ")
    print("=" * 50)
    print(f"Engine : {engine}")
    print(f"A      : {nnue_a}  ({args.name_a})")
    print(f"B      : {nnue_b}  ({args.name_b})")
    print(f"Games  : {len(book) * 2}  ({len(book)} openings, colors swapped)")
    print(f"Clock  : 2500ms + 50ms/move")
    print("=" * 50)

    run_tournament(
        str(engine), str(engine),
        name_e1=args.name_a, name_e2=args.name_b,
        nnue_e1=str(nnue_a), nnue_e2=str(nnue_b),
        openings=book,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
