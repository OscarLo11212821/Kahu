import subprocess
import time
import sys

# Several distinct starting positions
OPENINGS = [
    "WW3BB/WW1R1BB/2RRR2/1RRRRR1/2RRR2/BB1R1WW/BB3WW w 0 0",  
    "WW3BB/WW1R2B/1W1RRR1/1RRRRR1/1B1RRR1/BB1R2W/BB3WW w 0 0",       
    "WWW1BBB/W2R2B/2RRR2/1RRRRR1/2RRR2/B2R2W/BBB1WWW w 0 0",        
    "W1W1B1B/1W1R1B1/W1RRR2/WRRRRRB/2RRR1B/1B1R1W1/B1B1W1W w 0 0",
    "1WW3B/1WWR1BB/2RRRB1/1RRRRR1/1BRRR2/BB1RWW1/B4WW b 0 0",
    "WW4B/2WWR1B/2RRRB1/1RRRRB1/1BRRRR1/1BBRWW1/B4WW w 0 0",
    "2WW1BB/WW1R1BB/2RRR2/1BRRRRR/B1RRR2/1B1RWW1/1B3WW w 0 0",
    "1W1RBB1/WW1R1BB/1WRBR2/2BWRRR/1BBRWRW/5W1/7 w 2 2"
]

class Engine:
    def __init__(self, path, name, nnue_path=None):
        self.name = name
        cmd = [path]
        if nnue_path:
            cmd.extend(["--nnue", nnue_path])
        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
    
    def send(self, cmd):
        self.process.stdin.write(cmd + "\n")
        self.process.stdin.flush()

    def drain(self):
        import select
        while select.select([self.process.stdout], [], [], 0)[0]:
            self.process.stdout.readline()

    def get_bestmove(self):
        while True:
            line = self.process.stdout.readline().strip()
            if not line:
                continue
            # Print engine info lines (depth, score, pv)
            if line.startswith("info"):
                print(f"[{self.name}] {line}")
            if line.startswith("bestmove"):
                return line.split()[1]

    def get_status(self):
        self.send("status")
        while True:
            line = self.process.stdout.readline().strip()
            if line in ["white_wins", "black_wins", "ongoing"]:
                return line

    def close(self):
        self.send("quit")
        self.process.wait()


def play_game(engine_w, engine_b, kfen, wtime, btime, winc, binc):
    print(f"\n--- MATCH: {engine_w.name} (White) vs {engine_b.name} (Black) ---")
    
    engine_w.send(f"setboard {kfen}")
    engine_b.send(f"setboard {kfen}")

    time_w, time_b = wtime, btime
    turn = 'w'
    moves_played = 0

    while moves_played < 200:  # Cap at 200 moves to prevent infinite loops
        active = engine_w if turn == 'w' else engine_b
        
        # Calculate time taken for move
        t_start = time.time()
        active.send(f"go wtime {time_w} btime {time_b} winc {winc} binc {binc}")
        bestmove = active.get_bestmove()
        elapsed_ms = int((time.time() - t_start) * 1000)

        # Update clocks and check flags
        if turn == 'w':
            time_w -= elapsed_ms
            if time_w <= 0: return "black_wins", "White flagged"
            time_w += winc
        else:
            time_b -= elapsed_ms
            if time_b <= 0: return "white_wins", "Black flagged"
            time_b += binc

        print(f"Move {moves_played+1:03}: {active.name.ljust(10)} plays {bestmove.ljust(5)} (Time left: {time_w if turn=='w' else time_b}ms)")

        if bestmove == "000":
            status = active.get_status()
            if status == "ongoing":
                status = "white_wins" if turn == "b" else "black_wins"
            return status, "No legal moves"

        # Sync both engines
        engine_w.send(f"move {bestmove}")
        engine_b.send(f"move {bestmove}")
        engine_w.drain()
        engine_b.drain()

        # Check for winner
        status = active.get_status()
        if status != "ongoing":
            return status, "Checkmate/Capture Limit"

        turn = 'b' if turn == 'w' else 'w'
        moves_played += 1

    return "draw", "Move limit reached (200)"

def run_tournament(path_e1, path_e2, name_e1="Engine_V1", name_e2="Engine_V2",
                   nnue_e1=None, nnue_e2=None, openings=None):
    e1_score, e2_score, draws = 0, 0, 0
    wtime, btime = 2500, 2500 # 2.5 seconds starting time
    winc, binc = 50, 50         # 0.05s increment
    book = openings if openings is not None else OPENINGS

    for i, kfen in enumerate(book):
        print(f"\n=== OPENING {i+1}/{len(book)} ===")
        
        # Match 1: E1 is White, E2 is Black
        e1 = Engine(path_e1, name_e1, nnue_e1)
        e2 = Engine(path_e2, name_e2, nnue_e2)
        
        result, reason = play_game(e1, e2, kfen, wtime, btime, winc, binc)
        if result == "white_wins": e1_score += 1
        elif result == "black_wins": e2_score += 1
        else: draws += 1
        print(f"Result: {result} ({reason})\n")
        
        e1.close()
        e2.close()

        # Match 2: E2 is White, E1 is Black
        e2 = Engine(path_e2, name_e2, nnue_e2)
        e1 = Engine(path_e1, name_e1, nnue_e1)
        
        result, reason = play_game(e2, e1, kfen, wtime, btime, winc, binc)
        if result == "white_wins": e2_score += 1
        elif result == "black_wins": e1_score += 1
        else: draws += 1
        print(f"Result: {result} ({reason})\n")
        
        e1.close()
        e2.close()

    print("\n" + "="*40)
    print(" TOURNAMENT RESULTS ")
    print("="*40)
    print(f"{name_e1} Wins : {e1_score}")
    print(f"{name_e2} Wins : {e2_score}")
    print(f"Draws          : {draws}")
    return e1_score, e2_score, draws

if __name__ == "__main__":
    import argparse
    from pathlib import Path

    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Kuba engine vs engine tournament")
    parser.add_argument("engine", nargs="?", default=str(root / "build/bin/kuba_engine"),
                        help="Path to kuba_engine binary")
    parser.add_argument("--nnue-a", default=str(root / "kuba.nnue"),
                        help="NNUE file for competitor A")
    parser.add_argument("--nnue-b", default=str(root / "kuba_2.nnue"),
                        help="NNUE file for competitor B")
    parser.add_argument("--name-a", default="kuba.nnue", help="Display name for competitor A")
    parser.add_argument("--name-b", default="kuba_2.nnue", help="Display name for competitor B")
    parser.add_argument("--openings", type=int, default=8,
                        help="Number of opening positions (max %d)" % len(OPENINGS))
    args = parser.parse_args()

    book = OPENINGS[:max(1, min(args.openings, len(OPENINGS)))]
    run_tournament(args.engine, args.engine,
                   name_e1=args.name_a, name_e2=args.name_b,
                   nnue_e1=args.nnue_a, nnue_e2=args.nnue_b,
                   openings=book)