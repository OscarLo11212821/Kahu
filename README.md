# Kahu

**Kahu** is a strong, open-source AI engine and Graphical User Interface (GUI) for the abstract strategy board game **Kuba**.

Kahu uses Principal Variation Search with a custom **NNUE** (Efficiently Updatable Neural Network) evaluation to play Kuba at a high level.

## What is Kuba?

Kuba is a two-player strategy game designed by Serge Cahu, played on a 7×7 board. Players push lines of marbles horizontally or vertically. You win by capturing 7 neutral red marbles, by pushing all of the opponent’s marbles off the board, or if the opponent has no legal push. Repeating the previous position is illegal (Ko).

---

## Compiling Kahu

Kahu requires a modern C++ compiler supporting **C++20**.

### Prerequisites

* **CMake** 3.16+
* **GCC**, **Clang**, or **MSVC**

### Build everything

```bash
git clone https://github.com/OscarLo11212821/Kahu.git
cd Kahu
make
```

## Usage

### Using the Engine CLI

```bash
./build/bin/kuba_engine
```

The engine loads `clashtower_v4.nnue` from the current working directory unless `--nnue` is given.

### KPP v1: Kahu Pipe Protocol

Kahu’s headless mode speaks **KPP** (*Kahu Pipe Protocol*) **version 1** over stdin/stdout. KPP is a line-oriented text protocol for GUIs, referee scripts, and engine-vs-engine arenas.

#### Handshake and liveness

| Command | Response |
|---------|----------|
| `hello` or `identify` | Multi-line identification block (see below), ending with `ok` |
| `ping` | `pong` |
| `sync` | `synced`: engine is idle and accepting commands |

`hello` output:

```text
protocol kpp 1
engine Kahu v1
author OscarLo11212821
capabilities ping sync hello halt search go board move eval status legal nnue tt threads
nnue clashtower_v4.nnue v4 hidden=256 hidden2=32 cp_scale=3500
threads 1
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
| `threads [n]` | Set search threads (1–64, default 1). Omit `n` to query. Lazy SMP: helpers share the TT. |

During search the engine emits `info` lines (`depth`, `score`, `time`, `nodes`, `nps`, `pv`). When finished (or halted) it prints `bestmove <move>`. No legal moves yields `bestmove 000`.

**Legacy alias:** `go` accepts the same modes but uses older token names (`time` instead of `millis`, `winc`/`binc` instead of `wbonus`/`bbonus`).

#### NNUE weights

| Command | Description |
|---------|-------------|
| `nnue path <file>` | Load a network (`nnue file` and `nnue load` are aliases). Replies `nnue ok <file>` plus a status line, or `nnue fail <file>` |
| `nnue show` | Current network path and architecture (`nnue status` is an alias) |

Networks can also be selected at startup with `--nnue <file>`. After loading a different net, `tt clear` is recommended (the TT caches evals).

#### Engine maintenance

| Command | Description |
|---------|-------------|
| `tt clear` | Clear the transposition table (`tt reset` is an alias) |
| `threads [n]` | Search thread count (replies `threads <n>`) |
| `halt` / `abort` | Stop the current search |
| `quit` / `exit` | Stop search and exit |

Unknown commands reply `error unknown_command <name>`.

## License

This project is licensed under the **GNU General Public License v3.0** - see the [LICENSE](LICENSE) file for details.