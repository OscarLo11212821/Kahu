#include "datagen.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <mutex>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>

namespace {

uint32_t xorshift32(uint32_t& state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

float rand_unit(uint32_t& state) {
    return static_cast<float>(xorshift32(state)) / static_cast<float>(UINT32_MAX);
}

uint32_t derive_game_seed(uint32_t base_seed, int thread_id, int game_index) {
    uint32_t s = base_seed;
    s ^= static_cast<uint32_t>(game_index + 1) * 0x9E3779B9u;
    s ^= static_cast<uint32_t>(thread_id + 1) * 0x85EBCA6Bu;
    if (s == 0) s = 1;
    return s;
}

std::string trim(const std::string& s) {
    size_t begin = 0;
    while (begin < s.size() && std::isspace(static_cast<unsigned char>(s[begin]))) ++begin;
    size_t end = s.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(s[end - 1]))) --end;
    return s.substr(begin, end - begin);
}

int parse_int(const std::string& value, const char* flag) {
    try {
        return std::stoi(value);
    } catch (...) {
        throw std::runtime_error(std::string("Invalid integer for ") + flag + ": " + value);
    }
}

float parse_float(const std::string& value, const char* flag) {
    try {
        return std::stof(value);
    } catch (...) {
        throw std::runtime_error(std::string("Invalid float for ") + flag + ": " + value);
    }
}

[[noreturn]] void usage(const char* argv0) {
    std::cerr
        << "Kuba self-play data generation\n\n"
        << "Usage:\n"
        << "  " << argv0 << " datagen [options]\n\n"
        << "Search (one required):\n"
        << "  --depth <n>                  Fixed search depth per move (default: 6)\n"
        << "  --movetime <ms>              Search time per move instead of depth\n\n"
        << "Games / output:\n"
        << "  --games <n>                  Number of self-play games (default: 1)\n"
        << "  --out <path>                 Output dataset path (default: kuba_data.txt)\n"
        << "  --append                     Append to output instead of overwriting\n"
        << "  --max-plies <n>              Ply limit per game (default: 200)\n"
        << "  --threads <n>                Worker threads (0=auto, default: 0)\n"
        << "  --log-every <n>              Progress log interval in plies (default: 25)\n\n"
        << "Starting positions:\n"
        << "  --start-kfen \"<KFEN>\"        Single starting position\n"
        << "  --starts-file <path>         File with one KFEN per line (# comments ok)\n"
        << "  --cycle-starts               Cycle starts in file order (default: random)\n\n"
        << "Diversity:\n"
        << "  --random-move-prob <p>       Play a random legal move with probability p (0..1)\n"
        << "  --random-plies <n>           Play random moves for first n plies\n"
        << "  --random-plies <min> <max>   Random opening length in [min, max] plies\n\n"
        << "Misc:\n"
        << "  --seed <n>                   RNG seed (default: random)\n"
        << "  --quiet                      Only startup + periodic summary\n";
    std::exit(2);
}

DataGenConfig parse_datagen_args(int argc, char** argv) {
    DataGenConfig cfg;

    auto need_value = [&](int& i) -> std::string {
        if (i + 1 >= argc) usage(argv[0]);
        return std::string(argv[++i]);
    };

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--games") cfg.games = parse_int(need_value(i), "--games");
        else if (arg == "--depth") cfg.depth = parse_int(need_value(i), "--depth");
        else if (arg == "--movetime") {
            cfg.movetime_ms = parse_int(need_value(i), "--movetime");
            cfg.depth = 0;
        }
        else if (arg == "--out") cfg.output = need_value(i);
        else if (arg == "--append") cfg.append = true;
        else if (arg == "--max-plies") cfg.max_plies = parse_int(need_value(i), "--max-plies");
        else if (arg == "--threads") cfg.threads = parse_int(need_value(i), "--threads");
        else if (arg == "--log-every") cfg.log_every = parse_int(need_value(i), "--log-every");
        else if (arg == "--start-kfen") cfg.start_kfen = need_value(i);
        else if (arg == "--starts-file") cfg.starts_file = need_value(i);
        else if (arg == "--cycle-starts") cfg.cycle_starts = true;
        else if (arg == "--random-move-prob") {
            cfg.random_move_prob = parse_float(need_value(i), "--random-move-prob");
        } else if (arg == "--random-plies") {
            cfg.random_plies_min = parse_int(need_value(i), "--random-plies");
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                cfg.random_plies_max = parse_int(need_value(i), "--random-plies");
            } else {
                cfg.random_plies_max = cfg.random_plies_min;
            }
        } else if (arg == "--seed") cfg.seed = static_cast<uint32_t>(std::stoul(need_value(i)));
        else if (arg == "--quiet") cfg.quiet = true;
        else if (arg == "--help" || arg == "-h") usage(argv[0]);
        else usage(argv[0]);
    }

    if (cfg.games <= 0) throw std::runtime_error("--games must be positive");
    if (cfg.depth <= 0 && cfg.movetime_ms <= 0) throw std::runtime_error("Set --depth or --movetime");
    if (cfg.depth > 0 && cfg.movetime_ms > 0) throw std::runtime_error("Use either --depth or --movetime, not both");
    if (cfg.max_plies <= 0) throw std::runtime_error("--max-plies must be positive");
    if (cfg.log_every <= 0) throw std::runtime_error("--log-every must be positive");
    if (cfg.threads < 0) throw std::runtime_error("--threads must be >= 0");
    if (cfg.random_move_prob < 0.0f || cfg.random_move_prob > 1.0f) {
        throw std::runtime_error("--random-move-prob must be in [0, 1]");
    }
    if (cfg.random_plies_min < 0 || cfg.random_plies_max < cfg.random_plies_min) {
        throw std::runtime_error("--random-plies range is invalid");
    }
    return cfg;
}

int resolve_thread_count(const DataGenConfig& cfg) {
    int n = cfg.threads;
    if (n <= 0) {
        const unsigned hw = std::thread::hardware_concurrency();
        n = static_cast<int>(hw > 0 ? hw : 4);
    }
    return std::max(1, std::min(n, cfg.games));
}

SearchResult search_position(Engine& engine, const Position& pos, const DataGenConfig& cfg) {
    if (cfg.movetime_ms > 0) return engine.search_time_ex(pos, cfg.movetime_ms, true);
    return engine.search_depth_ex(pos, cfg.depth, true);
}

Move pick_random_move(const std::vector<Move>& moves, uint32_t& rng_state) {
    const size_t idx = static_cast<size_t>(xorshift32(rng_state)) % moves.size();
    return moves[idx];
}

int random_plies_for_game(const DataGenConfig& cfg, uint32_t& rng_state) {
    if (cfg.random_plies_max <= 0) return 0;
    if (cfg.random_plies_min == cfg.random_plies_max) return cfg.random_plies_min;
    const int span = cfg.random_plies_max - cfg.random_plies_min + 1;
    return cfg.random_plies_min + static_cast<int>(xorshift32(rng_state) % static_cast<uint32_t>(span));
}

struct DatagenShared {
    std::mutex out_mu;
    std::mutex log_mu;
    std::ofstream& out;
    const DataGenConfig& cfg;
    const std::vector<std::string>& starts;
    uint32_t base_seed;
    int total_games;
    int num_threads;
    std::atomic<int> next_game{0};
    std::atomic<int> games_finished{0};
    std::atomic<int> total_positions{0};

    void emit_line(const std::string& line) {
        std::lock_guard<std::mutex> lock(out_mu);
        out << line << '\n';
        out.flush();
    }

    void log_line(const std::string& msg) {
        std::lock_guard<std::mutex> lock(log_mu);
        std::cerr << msg << std::flush;
    }
};

void log_startup(const DataGenConfig& cfg, size_t num_starts, uint32_t seed, int threads) {
    std::cerr << "datagen: games=" << cfg.games
              << " threads=" << threads
              << " output=" << cfg.output
              << (cfg.append ? " (append)" : " (truncate)")
              << '\n';
    if (cfg.movetime_ms > 0) {
        std::cerr << "datagen: search=movetime " << cfg.movetime_ms << "ms\n";
    } else {
        std::cerr << "datagen: search=depth " << cfg.depth << '\n';
    }
    std::cerr << "datagen: starts=" << num_starts
              << (cfg.cycle_starts ? " (cycle)" : " (random)")
              << " max_plies=" << cfg.max_plies
              << " seed=" << seed;
    if (cfg.random_move_prob > 0.0f) std::cerr << " random_move_prob=" << cfg.random_move_prob;
    if (cfg.random_plies_max > 0) {
        std::cerr << " random_plies=" << cfg.random_plies_min << '-' << cfg.random_plies_max;
    }
    std::cerr << '\n';
    if (!cfg.quiet) {
        std::cerr << "datagen: streaming KFEN|score lines to " << cfg.output
                  << " (flush every position, log every " << cfg.log_every << " plies)\n";
    }
    std::cerr << std::flush;
}

int play_datagen_game(Position pos, Engine& engine, DatagenShared& shared,
                      uint32_t& rng_state, int game_index, int thread_id) {
    const DataGenConfig& cfg = shared.cfg;
    const int random_plies = random_plies_for_game(cfg, rng_state);
    int positions_written = 0;
    const auto game_start = std::chrono::steady_clock::now();
    const bool concurrent = shared.num_threads > 1;

    if (!cfg.quiet) {
        std::ostringstream msg;
        msg << "datagen: ";
        if (concurrent) msg << "t" << thread_id << ' ';
        msg << "game " << (game_index + 1) << '/' << shared.total_games
            << " start " << pos.to_kfen();
        if (random_plies > 0) msg << " (random_plies=" << random_plies << ')';
        msg << '\n';
        shared.log_line(msg.str());
    }

    for (int ply = 0; ply < cfg.max_plies; ++ply) {
        if (pos.get_winner() != EMPTY) break;

        std::vector<Move> moves = pos.generate_legal_moves();
        if (moves.empty()) break;

        const SearchResult search = search_position(engine, pos, cfg);
        {
            std::ostringstream line;
            line << pos.to_kfen() << '|' << search.score;
            shared.emit_line(line.str());
        }
        ++positions_written;
        shared.total_positions.fetch_add(1, std::memory_order_relaxed);

        if (!cfg.quiet && (ply == 0 || (ply + 1) % cfg.log_every == 0)) {
            const auto now = std::chrono::steady_clock::now();
            const auto elapsed_ms =
                std::chrono::duration_cast<std::chrono::milliseconds>(now - game_start).count();
            const int total_pos = shared.total_positions.load(std::memory_order_relaxed);

            std::ostringstream msg;
            if (concurrent) {
                msg << "datagen: t" << thread_id << " game " << (game_index + 1) << '/'
                    << shared.total_games << " ply " << (ply + 1)
                    << " game_pos=" << positions_written
                    << " total_pos=" << total_pos
                    << " score=" << search.score
                    << " depth=" << search.depth
                    << " elapsed_ms=" << elapsed_ms << '\n';
                shared.log_line(msg.str());
            } else {
                msg << "\rdatagen: game " << (game_index + 1) << '/' << shared.total_games
                    << " ply " << (ply + 1)
                    << " game_pos=" << positions_written
                    << " total_pos=" << total_pos
                    << " score=" << search.score
                    << " depth=" << search.depth
                    << " elapsed_ms=" << elapsed_ms << "   ";
                shared.log_line(msg.str());
            }
        }

        Move play = search.move;
        if (play.sq == 255 && !moves.empty()) play = moves[0];

        bool use_random = ply < random_plies;
        if (!use_random && cfg.random_move_prob > 0.0f && rand_unit(rng_state) < cfg.random_move_prob) {
            use_random = true;
        }
        if (use_random) play = pick_random_move(moves, rng_state);

        if (!pos.do_move(play.sq, play.dir)) break;
    }

    const int finished = shared.games_finished.fetch_add(1, std::memory_order_relaxed) + 1;
    const int total_pos = shared.total_positions.load(std::memory_order_relaxed);

    if (!cfg.quiet) {
        const auto game_end = std::chrono::steady_clock::now();
        const auto elapsed_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(game_end - game_start).count();
        std::ostringstream msg;
        if (concurrent) {
            msg << "datagen: t" << thread_id << " game " << (game_index + 1) << '/'
                << shared.total_games << " done positions=" << positions_written
                << " total_pos=" << total_pos
                << " games_done=" << finished
                << " elapsed_ms=" << elapsed_ms << '\n';
        } else {
            msg << "\rdatagen: game " << (game_index + 1) << '/' << shared.total_games
                << " done positions=" << positions_written
                << " total_pos=" << total_pos
                << " elapsed_ms=" << elapsed_ms << "                    \n";
        }
        shared.log_line(msg.str());
    } else if (finished % 10 == 0 || finished == shared.total_games) {
        std::ostringstream msg;
        msg << "datagen: " << finished << '/' << shared.total_games
            << " games, " << total_pos << " positions\n";
        shared.log_line(msg.str());
    }

    return positions_written;
}

void datagen_worker(int thread_id, DatagenShared& shared) {
    Engine engine;

    while (true) {
        const int game_index = shared.next_game.fetch_add(1, std::memory_order_relaxed);
        if (game_index >= shared.total_games) break;

        uint32_t rng_state = derive_game_seed(shared.base_seed, thread_id, game_index);

        Position pos;
        const std::string kfen = pick_start_kfen(shared.cfg, shared.starts, game_index, rng_state);
        pos.set_from_kfen(kfen);

        play_datagen_game(pos, engine, shared, rng_state, game_index, thread_id);
    }
}

} // namespace

bool load_start_kfens(const DataGenConfig& cfg, std::vector<std::string>& out) {
    out.clear();

    if (!cfg.start_kfen.empty()) {
        out.push_back(cfg.start_kfen);
    }

    if (!cfg.starts_file.empty()) {
        std::ifstream in(cfg.starts_file);
        if (!in) throw std::runtime_error("Failed to open starts file: " + cfg.starts_file);

        std::string line;
        while (std::getline(in, line)) {
            line = trim(line);
            if (line.empty() || line[0] == '#') continue;
            out.push_back(line);
        }
    }

    if (out.empty()) {
        Position pos;
        pos.set_initial_state();
        out.push_back(pos.to_kfen());
    }
    return !out.empty();
}

std::string pick_start_kfen(const DataGenConfig& cfg, const std::vector<std::string>& starts,
                            int game_index, uint32_t& rng_state) {
    if (starts.size() == 1) return starts[0];
    if (cfg.cycle_starts) return starts[static_cast<size_t>(game_index) % starts.size()];
    const size_t idx = static_cast<size_t>(xorshift32(rng_state)) % starts.size();
    return starts[idx];
}

int run_datagen(int argc, char** argv) {
    if (argc < 2) usage(argv[0]);

    try {
        DataGenConfig cfg = parse_datagen_args(argc, argv);

        std::vector<std::string> starts;
        load_start_kfens(cfg, starts);

        uint32_t base_seed = cfg.seed;
        if (base_seed == 0) {
            std::random_device rd;
            base_seed = rd();
        }

        const int threads = resolve_thread_count(cfg);
        log_startup(cfg, starts.size(), base_seed, threads);

        std::ios::openmode mode = cfg.append ? std::ios::app : std::ios::trunc;
        std::ofstream output(cfg.output, mode);
        if (!output) throw std::runtime_error("Failed to open output: " + cfg.output);

        DatagenShared shared{ {}, {}, output, cfg, starts, base_seed, cfg.games, threads };
        const auto run_start = std::chrono::steady_clock::now();

        std::vector<std::thread> workers;
        workers.reserve(static_cast<size_t>(threads));
        for (int t = 0; t < threads; ++t) {
            workers.emplace_back([&, t]() { datagen_worker(t, shared); });
        }
        for (auto& w : workers) w.join();

        const auto run_end = std::chrono::steady_clock::now();
        const auto elapsed_s =
            std::chrono::duration_cast<std::chrono::milliseconds>(run_end - run_start).count() / 1000.0;
        const int total_positions = shared.total_positions.load(std::memory_order_relaxed);

        std::cerr << "datagen: finished " << cfg.games << " games, "
                  << total_positions << " positions -> " << cfg.output
                  << " (" << elapsed_s << "s, " << threads << " threads)\n" << std::flush;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "datagen error: " << e.what() << '\n';
        return 1;
    }
}
