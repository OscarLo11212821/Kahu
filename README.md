# Kahu

**Kahu** is a strong, open-source AI engine and Graphical User Interface (GUI) for the abstract strategy board game **Kuba**. 

Kahu uses Alpha-Beta search and a custom **NNUE** (Efficiently Updatable Neural Network) evaluation function to play Kuba at a high level.

## What is Kuba?
Kuba is a two-player strategy game designed by Serge Cahu played on a 7x7 board. Players push lines of marbles horizontally or vertically. The goal is to either push off and capture 7 neutral red marbles, or push off all of the opponent's marbles.

## Features

### Engine & Search
* **Search Framework:** Principal Variation Search (PVS) with Iterative Deepening.
* **Quiescence Search:** Custom Q-Search that evaluates push-off threats, tactical captures, and forced replies to mitigate the horizon effect.
* **Heuristics:**
  * Transposition Table (TT) with depth/age replacement.
  * History Heuristic & Continuation History.
  * Killer Moves & Countermove Heuristic.
  * Search Extensions based on push-threat gains and tactical captures.
  
### NNUE Evaluation (RedRacer Architecture)
Kahu uses a custom NNUE architecture specifically designed for the rules of Kuba:
* **Bucketing:** 17 Material Phase buckets (push-complexity) and 16 Output buckets (red-race margin).
* **Feature Set:** `BOARD_SQ * 3` pieces (Own, Opponent, Red) utilizing a Column-major gravity index (push-oriented squares).
* **Push-Clash Head:** Custom duel-bucket differentials measuring mobility and push-off threats.
* **Topology:** `(Features) -> 256 -> 32 -> 1` with clipped ReLUs and a linear skip-connection bypass.

### Built-in NNUE Trainer
* Written in C++ with no external dependencies.
* Multi-threaded batch processing.
* **SIMD Accelerated:** Utilizes AVX2 (x86) or NEON (ARM) intrinsics for massive speedups during forward/backward passes.
* Adam Optimizer with gradient clipping.

### GUI
* Built-in graphical interface allowing human vs. human, human vs. AI, or AI vs. AI matches.
* Adjustable time controls and depth limits.

---

## Compiling Kahu

Kahu requires a modern C++ compiler supporting **C++20**.

### Prerequisites
* **CMake** 3.16+
* **GCC**, **Clang**, or **MSVC**
* **Qt 6** (Core, Gui, Widgets) — only needed for the GUI

### Build everything

```bash
git clone https://github.com/OscarLo11212821/Kahu.git
cd Kahu
make
```

Binaries are written to `build/bin/` (`kuba_engine`, `nnue_trainer`, and `KubaGUI` when Qt is available). `kuba.nnue` is copied next to the engine automatically.

### Build individual targets

```bash
make kuba_engine    # engine only
make nnue_trainer   # NNUE trainer only
make gui            # GUI only (requires Qt 6)
```

### Portable / release binaries

For distributable builds (no `-march=native`, suitable for GitHub Releases):

```bash
make release
```

Cross-compile for a specific platform:

```bash
./scripts/cross-build.sh macos-arm64
./scripts/cross-build.sh linux-x86_64
# or: make cross-macos-arm64
```

### CMake directly

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

---

## Usage

### Playing the Game (GUI)
Run the GUI from the build output directory (`kuba.nnue` and `kuba_engine` are placed alongside it automatically):

```bash
./build/bin/KubaGUI
```

### Using the Engine CLI

```bash
./build/bin/kuba_engine
```

### KPP v1 — Kahu Pipe Protocol

Kahu's headless mode speaks **KPP** (*Kahu Pipe Protocol*) **version 1** over stdin/stdout. KPP is a line-oriented text protocol for GUIs, referee scripts, and engine-vs-engine arenas. It is **not** UCI; command names and responses are Kuba-specific.

Launch the engine and send one command per line:

```bash
./build/bin/kuba_engine                  # loads kuba.nnue by default
./build/bin/kuba_engine --nnue other.nnue
```

#### Handshake and liveness

| Command | Response |
|---------|----------|
| `hello` or `identify` | Multi-line identification block (see below), ending with `ok` |
| `ping` | `pong` |
| `sync` | `synced` — engine is idle and accepting commands |

`hello` output:

```text
protocol kpp 1
engine Kahu 19
author OscarLo11212821
capabilities ping sync hello halt search go board move eval status legal nnue tt
nnue kuba.nnue v3 hidden=256 hidden2=32 cp_scale=3500
ok
```

Use `ping` as a lightweight heartbeat. Use `sync` after operations that may block internally (for example, reloading an NNUE file).

#### Board and game state

| Command | Description |
|---------|-------------|
| `reset` | Standard starting position (aborts any active search) |
| `board` | ASCII diagram to stdout |
| `getboard` | Current position as a KFEN string |
| `setboard <KFEN>` | Load a position (e.g. `setboard WW3BB/WW1R1BB/... w 0 0`) |
| `move <sq><dir>` | Apply a push (`a7S`, `d2N`, …). Invalid moves reply `illegal move` |
| `status` | `white_wins`, `black_wins`, or `ongoing` |
| `eval` | `score <cp>` from the loaded NNUE |
| `legal` or `moves` | `legal count N <move> …` listing all legal pushes |

Move notation: square (`a1`–`g7`) plus direction (`N`/`S`/`E`/`W`).

#### Search

Search runs on a background thread. While searching, the engine still accepts input; send `halt` (or `abort`) to stop and receive `bestmove` with the current best line.

| Command | Behaviour |
|---------|-----------|
| `search depth <n>` | Fixed-depth search |
| `search millis <ms>` | Search for `ms` milliseconds |
| `search clock wtime <ms> btime <ms> [wbonus <ms> bbonus <ms>]` | Clock-aware allocation |
| `search endless` or `search continuous` | Infinite analysis until `halt` |
| `search` | Defaults to 1000 ms |

During search the engine emits `info` lines (`depth`, `score`, `time`, `nodes`, `pv`). When finished (or halted) it prints `bestmove <move>`. No legal moves yields `bestmove 000`.

**Legacy alias:** `go` accepts the same modes but uses older token names (`time` instead of `millis`, `winc`/`binc` instead of `wbonus`/`bbonus`). Existing clients (including the bundled GUI) may keep using `go`.

#### NNUE weights

| Command | Description |
|---------|-------------|
| `nnue path <file>` | Load a network (`nnue file` and `nnue load` are aliases). Replies `nnue ok <file>` plus a status line, or `nnue fail <file>` |
| `nnue show` | Current network path and architecture (`nnue status` is an alias) |

Networks can also be selected at startup with `--nnue <file>`.

#### Engine maintenance

| Command | Description |
|---------|-------------|
| `tt clear` | Clear the transposition table (`tt reset` is an alias) |
| `halt` / `abort` | Stop the current search |
| `quit` / `exit` | Stop search and exit |

Unknown commands reply `error unknown_command <name>`.

#### Example session

```text
hello
protocol kpp 1
engine Kahu 19
author OscarLo11212821
capabilities ping sync hello halt search go board move eval status legal nnue tt
nnue kuba.nnue v3 hidden=256 hidden2=32 cp_scale=3500
ok
ping
pong
reset
move a7S
status
ongoing
search millis 1000
info depth 1 score -68 time 1 nodes 25 pv a1E
info depth 2 score -121 time 1 nodes 86 pv a1E a6S
...
bestmove g7W
nnue path kuba_2.nnue
nnue ok kuba_2.nnue
nnue kuba_2.nnue v3 hidden=256 hidden2=32 cp_scale=3500
search endless
info depth 1 score 42 time 0 nodes 18 pv a1E
...
halt
bestmove a1E
quit
```

#### Tournaments and testing

`nnue_tournament.py` runs head-to-head matches between two NNUE files using the same engine binary:

```bash
python3 nnue_tournament.py                      # kuba.nnue vs kuba_2.nnue, 8 openings
python3 nnue_tournament.py --fast                 # 3 openings, quicker smoke test
```
