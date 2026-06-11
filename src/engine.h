#pragma once
#include "position.h"
#include <vector>
#include <chrono>
#include <atomic>
#include <cstring>
#include <cstdint>

enum TTFlag : uint8_t { TT_EXACT, TT_ALPHA, TT_BETA };

struct TTEntry {
    uint64_t hash = 0;
    Move best_move = {255, 255};
    int16_t score = 0;
    int8_t depth = -1;
    TTFlag flag = TT_EXACT;
    uint8_t age = 0;
};

struct TTBucket {
    TTEntry entries[4];
};

struct SearchResult {
    Move move = {255, 255};
    int score = 0;
    int depth = 0;
    uint64_t nodes = 0;
    bool aborted = false;
};

class Engine {
public:
    Engine();

    Move search_time(const Position& pos, int time_ms);
    Move search_depth(const Position& pos, int max_depth);
    Move search_clock(const Position& pos, int wtime, int btime, int winc, int binc);
    Move search_endless(const Position& pos);

    SearchResult search_time_ex(const Position& pos, int time_ms, bool quiet = false);
    SearchResult search_depth_ex(const Position& pos, int max_depth, bool quiet = false);
    SearchResult search_endless_ex(const Position& pos, bool quiet = false);

    void stop();
    void clear_tt();

private:
    static const int INF = 30000;
    static const int TT_SIZE = 1 << 20;
    uint8_t current_age = 0;

    std::vector<TTBucket> tt;
    std::atomic<bool> time_over;
    std::atomic<bool> soft_time_over;
    uint64_t nodes;
    std::chrono::time_point<std::chrono::steady_clock> start_time;
    int max_time_ms;

    Move killer_moves[128][2];
    Move countermoves[64][4];
    int history[64][256];
    int continuation[256][256];

    static int move_pack(Move m) { return m.sq * 4 + m.dir; }

    int evaluate(const Position& pos);
    int pvs(Position pos, int depth, int alpha, int beta, int ply, Move prev = {255, 255}, int extensions = 0);
    int qsearch(Position pos, int alpha, int beta, int ply, int qdepth = 0);

    void check_time();
    void clear_heuristics();
    Move best_from_tt(const Position& pos) const;

    std::vector<Move> extract_pv(Position pos, int depth);
    void print_info(int depth, int score, int elapsed, const std::vector<Move>& pv);
};
