#pragma once

#include "position.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

enum TTFlag : uint8_t {
    TT_EXACT,
    TT_ALPHA,
    TT_BETA
};

struct TTEntry {
    uint32_t key = 0;
    Move best_move = {255, 255};
    int16_t score = 0;
    int16_t eval = 0;
    int8_t depth = -1;
    TTFlag flag = TT_EXACT;
    uint8_t age = 0;
};

static_assert(sizeof(TTEntry) == 16, "TTEntry should be 16 bytes");

struct alignas(64) TTBucket {
    TTEntry entries[4];
};

static_assert(sizeof(TTBucket) == 64, "TTBucket should be 64 bytes");

struct SearchResult {
    Move move = {255, 255};
    int score = 0;
    int depth = 0;
    uint64_t nodes = 0;
    bool aborted = false;
};

struct SearchParams {
    bool use_nmp      = true;
    bool use_rfp      = true;
    bool use_razoring = true;
    bool use_futility = true;
    bool use_lmp      = true;
    bool use_lmr      = true;
    bool use_iir      = true;
    bool use_qtt      = true;
    bool use_singular = true;

    int rfp_margin = 170;
    int fut_margin = 120;
    int fut_base = 80;
    int lmp_base = 3;

    int q_threat_plies = 2;
    int q_max_depth = 16;

    int nmp_min_depth = 3;
    int ext_budget = 2;

    int singular_min_depth = 6;
    int singular_margin = 24;

    double lmr_base = 0.75;
    double lmr_div = 2.25;
};

class Engine {
public:
    static constexpr int MAX_THREADS = 64;

    Engine();

    Move search_time(const Position& pos, int time_ms);
    Move search_depth(const Position& pos, int max_depth);
    Move search_clock(
        const Position& pos,
        int wtime,
        int btime,
        int winc,
        int binc
    );
    Move search_endless(const Position& pos);

    SearchResult search_time_ex(
        const Position& pos,
        int time_ms,
        bool quiet = false
    );

    SearchResult search_depth_ex(
        const Position& pos,
        int max_depth,
        bool quiet = false
    );

    SearchResult search_endless_ex(
        const Position& pos,
        bool quiet = false
    );

    void search_begin();

    SearchResult search_one_depth(
        const Position& pos,
        int depth,
        bool quiet = false
    );

    void stop();

    // Configuration methods require that no search is running.
    void clear_tt();
    void set_threads(int n);

    int threads() const {
        return num_threads_;
    }

    const SearchParams& params() const {
        return params_;
    }

    void set_params(const SearchParams& p);

private:
    static constexpr int INF = 30000;
    static constexpr int MAX_PLY = 128;

    // 1M buckets * 64 bytes = 64 MiB.
    static constexpr int TT_SIZE = 1 << 20;

    // A correctness-first shared TT implementation.
    // Locks protect entire entry snapshots and replacements.
    static constexpr int TT_LOCK_COUNT = 1 << 12;

    struct StackEntry {
        Move move;
        int static_eval;
        bool null_move;
    };

    struct Worker {
        Engine* eng = nullptr;
        int id = 0;

        // Only the owning worker modifies nodes.
        uint64_t nodes = 0;

        // Other threads read this periodically published snapshot.
        std::atomic<uint64_t> published_nodes{0};

        StackEntry ss_[MAX_PLY + 4];

        Move killer_moves[MAX_PLY + 2][2];
        Move countermoves[64][4];

        int history[2][64][4];
        int capture_history[2][64][4];
        int threat_history[2][64][4];
        int cont_hist[2][256][256];

        Move root_move_ = {255, 255};
        Move iter_best_ = {255, 255};

        void init();
        void age_heuristics();
        void publish_nodes();

        // Count this node, periodically check the clock, and test stop.
        bool visit_node();

        void update_quiet_stats(
            int side,
            Move m,
            Move prev,
            Move prev2,
            int bonus
        );

        int pvs(
            const Position& pos,
            int depth,
            int alpha,
            int beta,
            int ply,
            int extensions,
            Move excluded = Move{255, 255},
            bool allow_null = true
        );

        int qsearch(
            const Position& pos,
            int alpha,
            int beta,
            int ply,
            int qdepth
        );
    };

    SearchParams params_;
    int lmr_table_[64][64];

    int num_threads_ = 0;
    std::vector<std::unique_ptr<Worker>> workers_;

    std::vector<TTBucket> tt_;
    std::unique_ptr<std::mutex[]> tt_locks_;

    uint8_t current_age_ = 0;

    std::atomic<bool> time_over_{false};

    std::chrono::time_point<std::chrono::steady_clock> start_time_;
    int soft_ms_ = 0;
    int hard_ms_ = 0;

    // Main-search-thread state; helpers do not access these directly.
    Move root_move_ = {255, 255};
    Move id_best_ = {255, 255};
    int id_score_ = 0;
    int id_stable_ = 0;
    int id_completed_ = 0;

    // Explicitly published main-thread score for helper aspiration.
    std::atomic<int> helper_score_{0};

    void init_lmr();
    int evaluate(const Position& pos);

    SearchResult search_internal(
        const Position& pos,
        int max_depth,
        int soft_ms,
        int hard_ms,
        bool quiet,
        bool endless = false
    );

    void helper_loop(int id, Position pos);

    // Probe returns a coherent local snapshot, never a live TT pointer.
    bool probe_tt(uint64_t hash, TTEntry& out) const;

    void store_tt(
        uint64_t hash,
        Move best,
        int score,
        int eval,
        int depth,
        TTFlag flag
    );

    void prefetch_tt(uint64_t hash) const;

    int elapsed_ms() const;
    void check_time();

    Move best_from_tt(const Position& pos) const;
    uint64_t total_nodes() const;

    std::vector<Move> extract_pv(Position pos, int depth);

    void print_info(
        int depth,
        int score,
        int elapsed,
        const std::vector<Move>& pv
    );
};

using EngineInfoCallback = void (*)(
    int depth,
    int score,
    int elapsed,
    uint64_t nodes,
    const char* pv
);

// Install/change only while no engine search is running.
void engine_set_info_callback(EngineInfoCallback cb);