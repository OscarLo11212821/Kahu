#!/usr/bin/env python3
"""Fast 3-opening tournament for quick strength checks."""
import sys
from tournament import run_tournament, OPENINGS
import tournament

tournament.OPENINGS = OPENINGS[:3]

if __name__ == "__main__":
    path1 = sys.argv[1] if len(sys.argv) > 1 else "./engine_o"
    path2 = sys.argv[2] if len(sys.argv) > 2 else "./engine_n"
    run_tournament(path1, path2)
