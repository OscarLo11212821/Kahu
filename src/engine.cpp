#include "engine.h"
#include "nnue.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <optional>
#include <string>

#ifndef __EMSCRIPTEN__
#include <thread>
#endif

namespace {

constexpr int MAX_MOVES = 32;
constexpr int UNKNOWN = -1000000;
constexpr int SCORE_INF = 30000;
constexpr int MATE_THRESHOLD = SCORE_INF - 1000;

constexpr Move NO_MOVE = {255, 255};

inline bool same_move(Move a, Move b) {
    return a.sq == b.sq && a.dir == b.dir;
}

inline bool valid(Move m) {
    return m.sq < 64 && m.dir < 4;
}

inline int pack(Move m) {
    return m.sq * 4 + m.dir;
}

inline int side_index(Color c) {
    return c == WHITE ? 0 : 1;
}

inline int stat_bonus(int depth) {
    return std::min(300 * depth - 250, 2000);
}

inline void update_stat(int& h, int bonus) {
    h += bonus - h * std::abs(bonus) / 16384;
}

inline uint32_t key32(uint64_t hash) {
    return static_cast<uint32_t>(hash >> 32);
}

// optional does not default-construct Position.
// It also correctly destroys every successfully constructed child.
struct GenList {
    Move moves[MAX_MOVES];
    std::optional<Position> next[MAX_MOVES];
    int n = 0;
};

inline Color fast_winner(const Position& pos) {
    if (pos.captured_red[WHITE] >= 7 || pos.bitboards[BLACK] == 0)
        return WHITE;

    if (pos.captured_red[BLACK] >= 7 || pos.bitboards[WHITE] == 0)
        return BLACK;

    return EMPTY;
}

void gen_moves(const Position& pos, GenList& ml) {
    ml.n = 0;

    const uint64_t occ = pos.occupied;
    uint64_t pieces = pos.bitboards[pos.side_to_move];

    while (pieces) {
        const int sq = std::countr_zero(pieces);
        pieces &= pieces - 1;

        for (int dir = 0; dir < 4; ++dir) {
            const int behind = Position::get_adj(sq, dir ^ 1);

            if (behind != -1 && ((occ >> behind) & 1ULL))
                continue;

            if (ml.n >= MAX_MOVES)
                return;

            auto& child = ml.next[ml.n];
            child.emplace(pos);

            if (child->do_move(sq, dir)) {
                ml.moves[ml.n] = {
                    static_cast<uint8_t>(sq),
                    static_cast<uint8_t>(dir)
                };
                ++ml.n;
            } else {
                child.reset();
            }
        }
    }
}

inline bool is_tactical(
    const Position& pos,
    const Position& next
) {
    return std::popcount(pos.occupied)
        != std::popcount(next.occupied);
}

int tactical_value(
    const Position& pos,
    const Position& next
) {
    const Color me = pos.side_to_move;
    const Color opp = me == WHITE ? BLACK : WHITE;

    int value = 0;

    value += (
        next.captured_red[me] - pos.captured_red[me]
    ) * 1200;

    value += (
        std::popcount(pos.bitboards[opp])
        - std::popcount(next.bitboards[opp])
    ) * 400;

    value -= (
        std::popcount(pos.bitboards[me])
        - std::popcount(next.bitboards[me])
    ) * 400;

    return value;
}

inline int tt_score_from(int score, int ply) {
    if (score > MATE_THRESHOLD)
        return score - ply;

    if (score < -MATE_THRESHOLD)
        return score + ply;

    return score;
}

inline int tt_score_to(int score, int ply) {
    if (score > MATE_THRESHOLD)
        return score + ply;

    if (score < -MATE_THRESHOLD)
        return score - ply;

    return score;
}

inline int refine_eval(
    int eval,
    const TTEntry* entry,
    int ply
) {
    if (!entry)
        return eval;

    const int score = tt_score_from(entry->score, ply);

    if (entry->flag == TT_EXACT)
        return score;

    if (entry->flag == TT_BETA && score > eval)
        return score;

    if (entry->flag == TT_ALPHA && score < eval)
        return score;

    return eval;
}

inline bool tt_cutoff(
    const TTEntry& entry,
    int score,
    int alpha,
    int beta
) {
    if (entry.flag == TT_EXACT)
        return true;

    if (entry.flag == TT_ALPHA && score <= alpha)
        return true;

    if (entry.flag == TT_BETA && score >= beta)
        return true;

    return false;
}

} // namespace

static EngineInfoCallback g_info_cb = nullptr;

void engine_set_info_callback(EngineInfoCallback cb) {
    g_info_cb = cb;
}

// -----------------------------------------------------------------------------
// Engine setup
// -----------------------------------------------------------------------------

Engine::Engine()
    : tt_(TT_SIZE),
      tt_locks_(std::make_unique<std::mutex[]>(TT_LOCK_COUNT)) {
    init_lmr();
    set_threads(1);
}

void Engine::set_params(const SearchParams& p) {
    params_ = p;

    params_.rfp_margin = std::clamp(params_.rfp_margin, 0, 10000);
    params_.fut_margin = std::clamp(params_.fut_margin, 0, 10000);
    params_.fut_base = std::clamp(params_.fut_base, 0, 10000);
    params_.lmp_base = std::clamp(params_.lmp_base, 0, 1000);

    params_.q_max_depth = std::clamp(
        params_.q_max_depth, 0, MAX_PLY - 2
    );

    params_.q_threat_plies = std::clamp(
        params_.q_threat_plies, 0, params_.q_max_depth
    );

    params_.nmp_min_depth = std::clamp(
        params_.nmp_min_depth, 2, MAX_PLY - 8
    );

    params_.ext_budget = std::clamp(
        params_.ext_budget, 0, MAX_PLY - 8
    );

    // Keep excluded-move verification above qsearch entry depth.
    params_.singular_min_depth = std::clamp(
        params_.singular_min_depth, 4, MAX_PLY - 8
    );

    params_.singular_margin = std::clamp(
        params_.singular_margin, 0, 1000
    );

    if (!std::isfinite(params_.lmr_base))
        params_.lmr_base = 0.75;

    if (!std::isfinite(params_.lmr_div))
        params_.lmr_div = 2.25;

    params_.lmr_base = std::clamp(params_.lmr_base, 0.0, 8.0);
    params_.lmr_div = std::clamp(params_.lmr_div, 0.25, 16.0);

    init_lmr();

    // Selective-search parameters affect the meaning of cached results.
    clear_tt();
}

void Engine::set_threads(int n) {
#ifdef __EMSCRIPTEN__
    n = 1;
#else
    n = std::clamp(n, 1, MAX_THREADS);
#endif

    if (n == num_threads_ && !workers_.empty())
        return;

    workers_.clear();
    workers_.reserve(static_cast<size_t>(n));

    for (int i = 0; i < n; ++i) {
        auto worker = std::make_unique<Worker>();
        worker->eng = this;
        worker->id = i;
        worker->init();
        workers_.push_back(std::move(worker));
    }

    num_threads_ = n;
}

void Engine::init_lmr() {
    for (int depth = 0; depth < 64; ++depth) {
        for (int move = 0; move < 64; ++move) {
            if (depth == 0 || move == 0) {
                lmr_table_[depth][move] = 0;
            } else {
                lmr_table_[depth][move] = static_cast<int>(
                    params_.lmr_base
                    + std::log(static_cast<double>(depth))
                    * std::log(static_cast<double>(move))
                    / params_.lmr_div
                );
            }
        }
    }
}

int Engine::evaluate(const Position& pos) {
    // Reserve the mate-score band for actual search terminal scores.
    return std::clamp(
        nnue_evaluate(pos),
        -MATE_THRESHOLD + 1,
        MATE_THRESHOLD - 1
    );
}

// -----------------------------------------------------------------------------
// Timing and node reporting
// -----------------------------------------------------------------------------

int Engine::elapsed_ms() const {
    const auto now = std::chrono::steady_clock::now();

    return static_cast<int>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            now - start_time_
        ).count()
    );
}

void Engine::stop() {
    time_over_.store(true, std::memory_order_relaxed);
}

void Engine::check_time() {
    if (hard_ms_ > 0 && elapsed_ms() >= hard_ms_)
        time_over_.store(true, std::memory_order_relaxed);
}

uint64_t Engine::total_nodes() const {
    uint64_t total = 0;

    for (const auto& worker : workers_) {
        total += worker->published_nodes.load(
            std::memory_order_relaxed
        );
    }

    return total;
}

void Engine::Worker::publish_nodes() {
    published_nodes.store(nodes, std::memory_order_relaxed);
}

bool Engine::Worker::visit_node() {
    ++nodes;

    if ((nodes & 2047ULL) == 0) {
        publish_nodes();
        eng->check_time();
    }

    return eng->time_over_.load(std::memory_order_relaxed);
}

// -----------------------------------------------------------------------------
// Worker heuristics
// -----------------------------------------------------------------------------

void Engine::Worker::init() {
    nodes = 0;
    published_nodes.store(0, std::memory_order_relaxed);

    std::memset(history, 0, sizeof(history));
    std::memset(capture_history, 0, sizeof(capture_history));
    std::memset(threat_history, 0, sizeof(threat_history));
    std::memset(cont_hist, 0, sizeof(cont_hist));

    for (auto& killers : killer_moves)
        killers[0] = killers[1] = NO_MOVE;

    for (auto& row : countermoves)
        for (auto& move : row)
            move = NO_MOVE;

    for (auto& entry : ss_)
        entry = {NO_MOVE, UNKNOWN, false};

    root_move_ = NO_MOVE;
    iter_best_ = NO_MOVE;
}

void Engine::Worker::age_heuristics() {
    for (auto& side : history)
        for (auto& row : side)
            for (int& value : row)
                value /= 2;

    for (auto& side : capture_history)
        for (auto& row : side)
            for (int& value : row)
                value /= 2;

    for (auto& side : threat_history)
        for (auto& row : side)
            for (int& value : row)
                value /= 2;

    for (auto& table : cont_hist)
        for (auto& row : table)
            for (int& value : row)
                value /= 2;
}

void Engine::Worker::update_quiet_stats(
    int side,
    Move move,
    Move prev,
    Move prev2,
    int bonus
) {
    update_stat(history[side][move.sq][move.dir], bonus);

    if (valid(prev))
        update_stat(cont_hist[0][pack(prev)][pack(move)], bonus);

    if (valid(prev2))
        update_stat(cont_hist[1][pack(prev2)][pack(move)], bonus);
}

// -----------------------------------------------------------------------------
// Thread-safe transposition table
// -----------------------------------------------------------------------------

bool Engine::probe_tt(uint64_t hash, TTEntry& out) const {
    const size_t bucket_index =
        static_cast<size_t>(hash & (TT_SIZE - 1));

    const size_t lock_index =
        bucket_index & (TT_LOCK_COUNT - 1);

    std::lock_guard<std::mutex> guard(tt_locks_[lock_index]);

    const TTBucket& bucket = tt_[bucket_index];
    const uint32_t key = key32(hash);

    bool found = false;

    for (const TTEntry& entry : bucket.entries) {
        if (entry.key != key || entry.depth < 0)
            continue;

        if (!found || entry.depth > out.depth) {
            out = entry;
            found = true;
        }
    }

    return found;
}

void Engine::store_tt(
    uint64_t hash,
    Move best,
    int score,
    int eval,
    int depth,
    TTFlag flag
) {
    const size_t bucket_index =
        static_cast<size_t>(hash & (TT_SIZE - 1));

    const size_t lock_index =
        bucket_index & (TT_LOCK_COUNT - 1);

    std::lock_guard<std::mutex> guard(tt_locks_[lock_index]);

    TTBucket& bucket = tt_[bucket_index];
    const uint32_t key = key32(hash);

    depth = std::clamp(depth, 0, 127);

    int replacement = -1;

    for (int i = 0; i < 4; ++i) {
        TTEntry& entry = bucket.entries[i];

        if (entry.key != key || entry.depth < 0)
            continue;

        const bool replace =
            entry.age != current_age_
            || depth >= entry.depth
            || (
                flag == TT_EXACT
                && entry.flag != TT_EXACT
                && depth + 2 >= entry.depth
            );

        if (!replace) {
            if (!valid(entry.best_move) && valid(best))
                entry.best_move = best;

            return;
        }

        // A stand-pat entry need not erase a useful existing move.
        if (!valid(best) && valid(entry.best_move))
            best = entry.best_move;

        replacement = i;
        break;
    }

    if (replacement == -1) {
        int worst_value = 1 << 30;

        for (int i = 0; i < 4; ++i) {
            const TTEntry& entry = bucket.entries[i];

            if (entry.depth < 0) {
                replacement = i;
                break;
            }

            const int age_distance = static_cast<uint8_t>(
                current_age_ - entry.age
            );

            const int value =
                entry.depth - 8 * std::min(age_distance, 32);

            if (value < worst_value) {
                worst_value = value;
                replacement = i;
            }
        }
    }

    bucket.entries[replacement] = TTEntry{
        key,
        best,
        static_cast<int16_t>(std::clamp(score, -32767, 32767)),
        static_cast<int16_t>(std::clamp(eval, -32767, 32767)),
        static_cast<int8_t>(depth),
        flag,
        current_age_
    };
}

void Engine::prefetch_tt(uint64_t hash) const {
#if defined(__GNUC__) || defined(__clang__)
    __builtin_prefetch(
        &tt_[static_cast<size_t>(hash & (TT_SIZE - 1))],
        0,
        1
    );
#else
    (void)hash;
#endif
}

void Engine::clear_tt() {
    // Caller must ensure no search is running.
    for (TTBucket& bucket : tt_)
        for (TTEntry& entry : bucket.entries)
            entry = TTEntry{};
}

// -----------------------------------------------------------------------------
// Root fallback and PV reporting
// -----------------------------------------------------------------------------

Move Engine::best_from_tt(const Position& pos) const {
    if (fast_winner(pos) != EMPTY)
        return NO_MOVE;

    TTEntry entry;
    const bool hit = probe_tt(pos.hash, entry);

    GenList moves;
    gen_moves(pos, moves);

    if (hit && valid(entry.best_move)) {
        for (int i = 0; i < moves.n; ++i) {
            if (same_move(moves.moves[i], entry.best_move))
                return moves.moves[i];
        }
    }

    return moves.n > 0 ? moves.moves[0] : NO_MOVE;
}

std::vector<Move> Engine::extract_pv(Position pos, int depth) {
    std::vector<Move> pv;
    std::vector<uint64_t> visited;

    pv.reserve(static_cast<size_t>(std::max(depth, 0)));
    visited.reserve(static_cast<size_t>(std::max(depth, 0)));

    for (int i = 0; i < depth; ++i) {
        if (fast_winner(pos) != EMPTY)
            break;

        if (
            std::find(visited.begin(), visited.end(), pos.hash)
            != visited.end()
        ) {
            break;
        }

        TTEntry entry;

        if (!probe_tt(pos.hash, entry) || !valid(entry.best_move))
            break;

        const uint64_t old_hash = pos.hash;
        const Move move = entry.best_move;

        // Append only after validating the move.
        if (!pos.do_move(move.sq, move.dir))
            break;

        visited.push_back(old_hash);
        pv.push_back(move);
    }

    return pv;
}

void Engine::print_info(
    int depth,
    int score,
    int elapsed,
    const std::vector<Move>& pv
) {
    const uint64_t nodes = total_nodes();

    const uint64_t nps = elapsed > 0
        ? nodes * 1000ULL / static_cast<uint64_t>(elapsed)
        : 0;

    std::string pv_string;

    for (Move move : pv) {
        if (!pv_string.empty())
            pv_string += ' ';

        pv_string += move_to_string(move);
    }

    if (g_info_cb) {
        g_info_cb(
            depth,
            score,
            elapsed,
            nodes,
            pv_string.c_str()
        );
    }

#ifndef __EMSCRIPTEN__
    std::cout
        << "info depth " << depth
        << " score " << score
        << " time " << elapsed
        << " nodes " << nodes
        << " nps " << nps
        << " pv";

    for (Move move : pv)
        std::cout << ' ' << move_to_string(move);

    std::cout << std::endl;
#else
    (void)nps;
#endif
}

// -----------------------------------------------------------------------------
// Quiescence
// -----------------------------------------------------------------------------

int Engine::Worker::qsearch(
    const Position& pos,
    int alpha,
    int beta,
    int ply,
    int qdepth
) {
    if (visit_node())
        return alpha;

    const Color me = pos.side_to_move;
    const Color opp = me == WHITE ? BLACK : WHITE;

    const Color winner = fast_winner(pos);

    if (winner == me)
        return Engine::INF - ply;

    if (winner == opp)
        return -Engine::INF + ply;

    // No-legal-move loss must be detected before stand pat, the depth
    // cap, or a TT cutoff assumes this is a nonterminal position.
    GenList moves;
    gen_moves(pos, moves);

    if (moves.n == 0)
        return -Engine::INF + ply;

    if (ply >= Engine::MAX_PLY - 2)
        return eng->evaluate(pos);

    const SearchParams& params = eng->params_;
    const bool pv_node = beta - alpha > 1;
    const int alpha_orig = alpha;

    TTEntry snapshot;
    const bool hit =
        params.use_qtt && eng->probe_tt(pos.hash, snapshot);

    const Move tt_move = hit ? snapshot.best_move : NO_MOVE;

    // Quiet-threat coverage and remaining qsearch depth depend on
    // qdepth. Only qdepth == 0 uses/stores searched TT bounds.
    // Other qdepths may still reuse raw static evaluation and ordering.
    const bool use_bounds = hit && qdepth == 0;

    if (use_bounds && !pv_node) {
        const int score = tt_score_from(snapshot.score, ply);

        if (tt_cutoff(snapshot, score, alpha, beta))
            return score;
    }

    const int static_eval = hit
        ? snapshot.eval
        : eng->evaluate(pos);

    const int stand_pat = use_bounds
        ? refine_eval(static_eval, &snapshot, ply)
        : static_eval;

    int best_score = stand_pat;

    if (best_score >= beta)
        return best_score;

    if (best_score > alpha)
        alpha = best_score;

    if (qdepth >= params.q_max_depth)
        return best_score;

    const int ci = side_index(me);
    const bool generate_threats = qdepth < params.q_threat_plies;

    const int base_threats = generate_threats
        ? pos.push_off_threats(me)
        : 0;

    struct QMove {
        int index;
        int score;
    };

    QMove ordered[MAX_MOVES];
    int count = 0;

    for (int i = 0; i < moves.n; ++i) {
        const Move move = moves.moves[i];
        const Position& next = *moves.next[i];

        const bool immediate_win = fast_winner(next) == me;
        int ordering_score = 0;

        if (immediate_win) {
            // Never delta-prune a proven terminal win.
            ordering_score = 2000000;
        } else if (is_tactical(pos, next)) {
            const int gain = tactical_value(pos, next);
            const bool red =
                next.captured_red[me] > pos.captured_red[me];

            if (!red && stand_pat + gain + 150 <= alpha)
                continue;

            ordering_score =
                100000
                + gain * 64
                + capture_history[ci][move.sq][move.dir];
        } else {
            if (!generate_threats)
                continue;

            if (stand_pat + 200 <= alpha)
                continue;

            const int threat_gain =
                next.push_off_threats(me) - base_threats;

            if (threat_gain <= 0)
                continue;

            ordering_score =
                threat_gain * 80
                + threat_history[ci][move.sq][move.dir];
        }

        if (same_move(move, tt_move))
            ordering_score += 1000000;

        ordered[count++] = {i, ordering_score};
    }

    Move best_move = NO_MOVE;

    for (int i = 0; i < count; ++i) {
        int best_index = i;

        for (int j = i + 1; j < count; ++j) {
            if (ordered[j].score > ordered[best_index].score)
                best_index = j;
        }

        std::swap(ordered[i], ordered[best_index]);

        const int index = ordered[i].index;
        const Move move = moves.moves[index];
        const Position& next = *moves.next[index];

        eng->prefetch_tt(next.hash);

        const int score = -qsearch(
            next,
            -beta,
            -alpha,
            ply + 1,
            qdepth + 1
        );

        if (eng->time_over_.load(std::memory_order_relaxed))
            return best_score;

        if (score > best_score) {
            best_score = score;
            best_move = move;

            if (score > alpha) {
                alpha = score;

                if (alpha >= beta)
                    break;
            }
        }
    }

    if (params.use_qtt && qdepth == 0) {
        const TTFlag flag = best_score >= beta
            ? TT_BETA
            : best_score > alpha_orig
                ? TT_EXACT
                : TT_ALPHA;

        eng->store_tt(
            pos.hash,
            best_move,
            tt_score_to(best_score, ply),
            static_eval,
            0,
            flag
        );
    }

    return best_score;
}

// -----------------------------------------------------------------------------
// Principal variation search
// -----------------------------------------------------------------------------

int Engine::Worker::pvs(
    const Position& pos,
    int depth,
    int alpha,
    int beta,
    int ply,
    int extensions,
    Move excluded,
    bool allow_null
) {
    if (visit_node())
        return alpha;

    const bool root = ply == 0;
    const bool singular_probe = valid(excluded);

    const Color me = pos.side_to_move;
    const Color opp = me == WHITE ? BLACK : WHITE;

    const Color winner = fast_winner(pos);

    // Check terminal positions before mate-distance window tightening.
    if (winner == me)
        return Engine::INF - ply;

    if (winner == opp)
        return -Engine::INF + ply;

    if (!root) {
        alpha = std::max(alpha, -Engine::INF + ply);
        beta = std::min(beta, Engine::INF - ply - 1);

        if (alpha >= beta)
            return alpha;
    }

    if (depth <= 0)
        return qsearch(pos, alpha, beta, ply, 0);

    if (ply >= Engine::MAX_PLY - 2) {
        GenList moves;
        gen_moves(pos, moves);

        if (moves.n == 0)
            return -Engine::INF + ply;

        return eng->evaluate(pos);
    }

    const bool pv_node = beta - alpha > 1;
    const int alpha_orig = alpha;

    const SearchParams& params = eng->params_;
    const int ci = side_index(me);

    const Move prev = ply >= 1
        ? ss_[ply - 1].move
        : NO_MOVE;

    const Move prev2 = ply >= 2
        ? ss_[ply - 2].move
        : NO_MOVE;

    // Snapshot is local and remains valid across recursive TT writes.
    TTEntry snapshot;

    const bool tt_hit =
        !singular_probe && eng->probe_tt(pos.hash, snapshot);

    const TTEntry* entry = tt_hit ? &snapshot : nullptr;
    const Move tt_move = tt_hit ? snapshot.best_move : NO_MOVE;

    if (tt_hit && !pv_node && snapshot.depth >= depth) {
        const int score = tt_score_from(snapshot.score, ply);

        if (tt_cutoff(snapshot, score, alpha, beta))
            return score;
    }

    GenList moves;
    gen_moves(pos, moves);

    if (moves.n == 0)
        return -Engine::INF + ply;

    const int static_eval = tt_hit
        ? snapshot.eval
        : eng->evaluate(pos);

    const int eval = refine_eval(static_eval, entry, ply);

    ss_[ply].static_eval = static_eval;

    const bool improving =
        ply < 2
        || ss_[ply - 2].static_eval == UNKNOWN
        || static_eval > ss_[ply - 2].static_eval;

    // An excluded-move node cannot use shortcuts that assume the
    // excluded move remains available.
    const bool whole_node_pruning =
        !root
        && !pv_node
        && !singular_probe
        && std::abs(beta) < MATE_THRESHOLD;

    if (whole_node_pruning) {
        if (
            params.use_rfp
            && depth <= 4
            && eval
                - params.rfp_margin * depth
                + (improving ? 60 : 0)
                >= beta
        ) {
            return beta;
        }

        if (
            params.use_razoring
            && depth <= 3
            && eval + 200 * depth <= alpha
        ) {
            const int score = qsearch(
                pos, alpha, beta, ply, 0
            );

            if (eng->time_over_.load(std::memory_order_relaxed))
                return alpha;

            // Only a confirmed fail-low justifies razoring.
            if (score <= alpha)
                return score;
        }
    }

    int base_threats = UNKNOWN;
    int base_opp_threats = UNKNOWN;

    auto get_base_threats = [&]() {
        if (base_threats == UNKNOWN)
            base_threats = pos.push_off_threats(me);

        return base_threats;
    };

    auto get_base_opp_threats = [&]() {
        if (base_opp_threats == UNKNOWN)
            base_opp_threats = pos.push_off_threats(opp);

        return base_opp_threats;
    };

    // Null-move pruning.
    if (
        whole_node_pruning
        && allow_null
        && params.use_nmp
        && depth >= params.nmp_min_depth
        && eval >= beta
        && !ss_[ply - 1].null_move
        && get_base_opp_threats() == 0
        && std::popcount(pos.bitboards[me]) >= 3
    ) {
        Position null_pos = pos;
        null_pos.do_null_move();

        const StackEntry saved_stack = ss_[ply];

        ss_[ply].move = NO_MOVE;
        ss_[ply].null_move = true;

        const int reduction =
            3 + depth / 4
            + std::min(2, (eval - beta) / 200);

        int score = -pvs(
            null_pos,
            depth - 1 - reduction,
            -beta,
            -beta + 1,
            ply + 1,
            extensions,
            NO_MOVE,
            allow_null
        );

        ss_[ply] = saved_stack;

        if (eng->time_over_.load(std::memory_order_relaxed))
            return alpha;

        if (score >= beta) {
            if (score > MATE_THRESHOLD)
                score = beta;

            if (depth >= 8) {
                // Explicitly disable NMP in the verification subtree.
                const int verified = pvs(
                    pos,
                    depth - 1 - reduction,
                    beta - 1,
                    beta,
                    ply,
                    extensions,
                    NO_MOVE,
                    false
                );

                ss_[ply] = saved_stack;

                if (eng->time_over_.load(std::memory_order_relaxed))
                    return alpha;

                if (verified >= beta)
                    return score;
            } else {
                return score;
            }
        }
    }

    // Do not add IIR to a singular probe simply because its TT access
    // was deliberately disabled.
    if (
        params.use_iir
        && !root
        && !singular_probe
        && !valid(tt_move)
        && depth >= 4
    ) {
        --depth;
    }

    // Singular extension verification.
    bool tt_move_singular = false;

    if (
        params.use_singular
        && !root
        && !singular_probe
        && tt_hit
        && valid(tt_move)
        && depth >= params.singular_min_depth
        && snapshot.depth >= depth - 3
        && snapshot.flag != TT_ALPHA
        && std::abs(snapshot.score) < MATE_THRESHOLD
    ) {
        // Validate the TT move against the generated legal list.
        bool tt_move_legal = false;

        for (int i = 0; i < moves.n; ++i) {
            if (same_move(moves.moves[i], tt_move)) {
                tt_move_legal = true;
                break;
            }
        }

        if (tt_move_legal) {
            const int tt_score = tt_score_from(
                snapshot.score, ply
            );

            const int singular_beta =
                tt_score - params.singular_margin * depth / 16;

            const int singular_depth = (depth - 1) / 2;

            const StackEntry saved_stack = ss_[ply];

            const int score = pvs(
                pos,
                singular_depth,
                singular_beta - 1,
                singular_beta,
                ply,
                extensions,
                tt_move,
                false
            );

            ss_[ply] = saved_stack;

            if (eng->time_over_.load(std::memory_order_relaxed))
                return alpha;

            if (score < singular_beta) {
                tt_move_singular = true;
            } else if (singular_beta >= beta) {
                return singular_beta;
            }
        }
    }

    struct ScoredMove {
        int index;
        int score;
        int history_score;
        bool tactical;
    };

    ScoredMove ordered[MAX_MOVES];
    int count = 0;

    for (int i = 0; i < moves.n; ++i) {
        const Move move = moves.moves[i];

        if (singular_probe && same_move(move, excluded))
            continue;

        const Position& next = *moves.next[i];
        const bool tactical = is_tactical(pos, next);

        const int gain = tactical
            ? tactical_value(pos, next)
            : 0;

        // Compute quiet history even for TT/killer/counter moves,
        // so their LMR history adjustment is not silently zero.
        int history_score = 0;

        if (!tactical) {
            history_score = history[ci][move.sq][move.dir];

            if (valid(prev)) {
                history_score += cont_hist[0][pack(prev)][pack(move)];
            }

            if (valid(prev2)) {
                history_score += cont_hist[1][pack(prev2)][pack(move)];
            }
        }

        int score;

        if (fast_winner(next) == me) {
            score = 2000000;
        } else if (root && same_move(move, root_move_)) {
            score = 1100000;
        } else if (same_move(move, tt_move)) {
            score = 1000000;
        } else if (tactical && gain > 0) {
            score =
                600000
                + gain * 64
                + capture_history[ci][move.sq][move.dir];
        } else if (same_move(move, killer_moves[ply][0])) {
            score = 500000;
        } else if (same_move(move, killer_moves[ply][1])) {
            score = 490000;
        } else if (
            valid(prev)
            && same_move(move, countermoves[prev.sq][prev.dir])
        ) {
            score = 480000;
        } else if (tactical) {
            score =
                -100000
                + gain * 64
                + capture_history[ci][move.sq][move.dir];
        } else {
            score = history_score;
        }

        ordered[count++] = {
            i,
            score,
            history_score,
            tactical
        };
    }

    // The only legal move was excluded.
    if (count == 0)
        return alpha;

    int best_score = -Engine::INF - 1;
    Move best_move = NO_MOVE;
    int moves_searched = 0;

    Move quiets_tried[MAX_MOVES];
    Move tacticals_tried[MAX_MOVES];
    int quiet_count = 0;
    int tactical_count = 0;

    const int reds_left =
        13
        - pos.captured_red[WHITE]
        - pos.captured_red[BLACK];

    const int lmp_limit =
        (params.lmp_base + depth * depth)
        / (improving ? 1 : 2);

    for (int i = 0; i < count; ++i) {
        int best_index = i;

        for (int j = i + 1; j < count; ++j) {
            if (ordered[j].score > ordered[best_index].score)
                best_index = j;
        }

        std::swap(ordered[i], ordered[best_index]);

        const ScoredMove& scored = ordered[i];
        const Move move = moves.moves[scored.index];
        const Position& next = *moves.next[scored.index];

        const bool quiet = !scored.tactical;

        const int threat_gain =
            next.push_off_threats(me) - get_base_threats();

        int opponent_threat_reduction = UNKNOWN;

        auto get_opponent_threat_reduction = [&]() {
            if (opponent_threat_reduction == UNKNOWN) {
                opponent_threat_reduction =
                    get_base_opp_threats()
                    - next.push_off_threats(opp);
            }

            return opponent_threat_reduction;
        };

        const bool killer =
            same_move(move, killer_moves[ply][0])
            || same_move(move, killer_moves[ply][1]);

        const bool counter =
            valid(prev)
            && same_move(move, countermoves[prev.sq][prev.dir]);

        const bool immediate_win = fast_winner(next) == me;

        // Keep excluded-move verification less speculative:
        // no LMP/futility pruning at the exclusion node itself.
        if (
            !root
            && !singular_probe
            && !immediate_win
            && moves_searched > 0
            && quiet
            && threat_gain < 1
            && best_score > -MATE_THRESHOLD
        ) {
            const bool prune =
                (
                    params.use_lmp
                    && depth <= 7
                    && moves_searched >= lmp_limit
                )
                || (
                    params.use_futility
                    && depth <= 5
                    && eval
                        + params.fut_margin * depth
                        + params.fut_base
                        <= alpha
                );

            if (prune && get_opponent_threat_reduction() < 1)
                continue;
        }

        int extension = 0;

        if (extensions < params.ext_budget && !immediate_win) {
            if (next.captured_red[me] > pos.captured_red[me]) {
                extension = 1;
            } else if (
                threat_gain >= 2 || get_base_opp_threats() >= 3
            ) {
                extension = 1;
            } else if (get_opponent_threat_reduction() >= 2) {
                extension = 1;
            } else if (reds_left <= 1) {
                extension = 1;
            } else if (
                tt_move_singular && same_move(move, tt_move)
            ) {
                extension = 1;
            }
        }

        const int new_depth = depth - 1 + extension;

        ss_[ply].move = move;
        ss_[ply].null_move = false;

        eng->prefetch_tt(next.hash);

        int score;

        if (moves_searched == 0) {
            score = -pvs(
                next,
                new_depth,
                -beta,
                -alpha,
                ply + 1,
                extensions + extension,
                NO_MOVE,
                allow_null
            );
        } else {
            int reduction = 0;

            if (
                params.use_lmr
                && !singular_probe
                && !immediate_win
                && depth >= 3
                && quiet
                && moves_searched >= (pv_node ? 3 : 2)
            ) {
                reduction = eng->lmr_table_[
                    std::min(depth, 63)
                ][
                    std::min(moves_searched, 63)
                ];

                if (pv_node)
                    --reduction;

                if (killer || counter)
                    --reduction;

                if (
                    threat_gain >= 1
                    || get_opponent_threat_reduction() >= 1
                ) {
                    --reduction;
                }

                if (!improving)
                    ++reduction;

                reduction -= std::clamp(
                    scored.history_score / 12000,
                    -1,
                    1
                );

                reduction = std::clamp(
                    reduction,
                    0,
                    std::max(0, new_depth - 1)
                );
            }

            score = -pvs(
                next,
                new_depth - reduction,
                -alpha - 1,
                -alpha,
                ply + 1,
                extensions + extension,
                NO_MOVE,
                allow_null
            );

            if (
                reduction > 0
                && score > alpha
                && !eng->time_over_.load(std::memory_order_relaxed)
            ) {
                score = -pvs(
                    next,
                    new_depth,
                    -alpha - 1,
                    -alpha,
                    ply + 1,
                    extensions + extension,
                    NO_MOVE,
                    allow_null
                );
            }

            if (
                score > alpha
                && score < beta
                && !eng->time_over_.load(std::memory_order_relaxed)
            ) {
                score = -pvs(
                    next,
                    new_depth,
                    -beta,
                    -alpha,
                    ply + 1,
                    extensions + extension,
                    NO_MOVE,
                    allow_null
                );
            }
        }

        if (eng->time_over_.load(std::memory_order_relaxed)) {
            return best_score > -Engine::INF - 1
                ? best_score
                : alpha;
        }

        ++moves_searched;

        if (score > best_score) {
            best_score = score;
            best_move = move;

            if (root)
                iter_best_ = move;
        }

        if (score > alpha) {
            alpha = score;

            if (alpha >= beta) {
                // Avoid training this node's ordinary heuristics from
                // a deliberately restricted alternative-move search.
                if (!singular_probe) {
                    const int bonus = stat_bonus(depth);

                    if (quiet) {
                        if (!same_move(killer_moves[ply][0], move)) {
                            killer_moves[ply][1] = killer_moves[ply][0];
                            killer_moves[ply][0] = move;
                        }

                        if (valid(prev))
                            countermoves[prev.sq][prev.dir] = move;

                        update_quiet_stats(
                            ci, move, prev, prev2, bonus
                        );

                        for (int q = 0; q < quiet_count; ++q) {
                            update_quiet_stats(
                                ci,
                                quiets_tried[q],
                                prev,
                                prev2,
                                -bonus
                            );
                        }

                        if (threat_gain >= 2) {
                            update_stat(
                                threat_history[ci][move.sq][move.dir],
                                bonus
                            );
                        }
                    } else {
                        update_stat(
                            capture_history[ci][move.sq][move.dir],
                            bonus
                        );

                        for (int t = 0; t < tactical_count; ++t) {
                            const Move tried = tacticals_tried[t];

                            update_stat(
                                capture_history[ci][tried.sq][tried.dir],
                                -bonus
                            );
                        }
                    }
                }

                break;
            }
        }

        if (quiet)
            quiets_tried[quiet_count++] = move;
        else
            tacticals_tried[tactical_count++] = move;
    }

    if (moves_searched == 0)
        return alpha;

    if (!singular_probe) {
        const TTFlag flag = best_score <= alpha_orig
            ? TT_ALPHA
            : best_score >= beta
                ? TT_BETA
                : TT_EXACT;

        eng->store_tt(
            pos.hash,
            best_move,
            tt_score_to(best_score, ply),
            static_eval,
            depth,
            flag
        );
    }

    return best_score;
}

// -----------------------------------------------------------------------------
// Iterative deepening
// -----------------------------------------------------------------------------

void Engine::search_begin() {
    start_time_ = std::chrono::steady_clock::now();

    soft_ms_ = 0;
    hard_ms_ = 0;

    time_over_.store(false, std::memory_order_relaxed);
    ++current_age_;

    root_move_ = NO_MOVE;
    id_best_ = NO_MOVE;
    id_score_ = 0;
    id_stable_ = 0;
    id_completed_ = 0;

    helper_score_.store(0, std::memory_order_relaxed);

    for (auto& worker : workers_) {
        worker->nodes = 0;
        worker->published_nodes.store(0, std::memory_order_relaxed);

        worker->age_heuristics();

        worker->root_move_ = NO_MOVE;
        worker->iter_best_ = NO_MOVE;

        for (auto& entry : worker->ss_)
            entry = {NO_MOVE, UNKNOWN, false};

        for (auto& killers : worker->killer_moves)
            killers[0] = killers[1] = NO_MOVE;
    }
}

void Engine::helper_loop(int id, Position pos) {
    Worker& worker = *workers_[static_cast<size_t>(id)];

    // Generate root choices once for safe ordering diversification.
    // Helpers still search every root move unless a normal cutoff occurs.
    Move root_choices[MAX_MOVES];
    int root_count = 0;

    {
        GenList moves;
        gen_moves(pos, moves);
        root_count = moves.n;

        for (int i = 0; i < moves.n; ++i)
            root_choices[i] = moves.moves[i];
    }

    if (root_count == 0) {
        worker.publish_nodes();
        return;
    }

    static constexpr int DEPTH_OFFSET[8] = {
        0, 1, 1, 2, 2, 3, 3, 4
    };

    const int offset = DEPTH_OFFSET[id % 8];
    int iteration = 1 + offset;

    while (!time_over_.load(std::memory_order_relaxed)) {
        const int depth = std::min(iteration, MAX_PLY - 8);

        worker.root_move_ = root_choices[
            (depth + id - 1) % root_count
        ];

        worker.iter_best_ = NO_MOVE;

        int alpha = -INF - 1;
        int beta = INF + 1;

        const int guess = helper_score_.load(
            std::memory_order_relaxed
        );

        if (depth >= 5 && std::abs(guess) < INF - 2000) {
            const int window = 40 + 20 * (id % 3);

            alpha = std::max(-INF - 1, guess - window);
            beta = std::min(INF + 1, guess + window);
        }

        const int score = worker.pvs(
            pos, depth, alpha, beta, 0, 0
        );

        worker.publish_nodes();

        if (
            !time_over_.load(std::memory_order_relaxed)
            && (score <= alpha || score >= beta)
        ) {
            // The initial bound was already valid. Re-search to obtain
            // a more useful full-window result before advancing depth.
            worker.pvs(
                pos, depth, -INF - 1, INF + 1, 0, 0
            );

            worker.publish_nodes();
        }

        ++iteration;

        if (iteration > MAX_PLY - 8)
            iteration = 1 + offset;
    }

    worker.publish_nodes();
}

SearchResult Engine::search_one_depth(
    const Position& pos,
    int depth,
    bool quiet
) {
    SearchResult result;
    result.move = id_best_;
    result.score = id_score_;
    result.depth = id_completed_;
    result.nodes = total_nodes();
    result.aborted = time_over_.load(std::memory_order_relaxed);

    if (result.aborted)
        return result;

    Worker& worker = *workers_[0];
    worker.root_move_ = root_move_;

    const int target_depth = std::clamp(
        depth, 1, MAX_PLY - 8
    );

    int alpha = -INF - 1;
    int beta = INF + 1;
    int delta = 30;
    int score = id_score_;

    if (
        target_depth >= 4
        && std::abs(score) < INF - 2000
    ) {
        alpha = std::max(-INF - 1, score - delta);
        beta = std::min(INF + 1, score + delta);
    }

    worker.iter_best_ = NO_MOVE;

    while (true) {
        score = worker.pvs(
            pos, target_depth, alpha, beta, 0, 0
        );

        worker.publish_nodes();

        if (time_over_.load(std::memory_order_relaxed))
            break;

        if (score <= alpha) {
            beta = (alpha + beta) / 2;
            delta *= 2;
            alpha = std::max(-INF - 1, score - delta);
        } else if (score >= beta) {
            delta *= 2;
            beta = std::min(INF + 1, score + delta);
        } else {
            break;
        }

        if (delta > 2000) {
            alpha = -INF - 1;
            beta = INF + 1;
        }
    }

    result.nodes = total_nodes();

    if (time_over_.load(std::memory_order_relaxed)) {
        result.aborted = true;

        if (!valid(result.move)) {
            result.move = valid(worker.iter_best_)
                ? worker.iter_best_
                : best_from_tt(pos);
        }

        return result;
    }

    const Move best = valid(worker.iter_best_)
        ? worker.iter_best_
        : best_from_tt(pos);

    if (same_move(best, id_best_))
        ++id_stable_;
    else
        id_stable_ = 0;

    id_best_ = best;
    root_move_ = best;
    id_score_ = score;
    id_completed_ = target_depth;

    helper_score_.store(score, std::memory_order_relaxed);

    if (!quiet) {
        print_info(
            target_depth,
            score,
            elapsed_ms(),
            extract_pv(pos, target_depth)
        );
    }

    result.move = best;
    result.score = score;
    result.depth = target_depth;
    result.aborted = false;

    return result;
}

SearchResult Engine::search_internal(
    const Position& pos,
    int max_depth,
    int soft_ms,
    int hard_ms,
    bool quiet,
    bool endless
) {
    search_begin();

    soft_ms_ = std::max(0, soft_ms);
    hard_ms_ = std::max(0, hard_ms);

    max_depth = std::clamp(max_depth, 1, MAX_PLY - 8);

    // Resolve terminal roots before launching helpers.
    const Color winner = fast_winner(pos);

    if (winner != EMPTY) {
        SearchResult terminal;
        terminal.score = winner == pos.side_to_move ? INF : -INF;
        return terminal;
    }

    {
        GenList moves;
        gen_moves(pos, moves);

        if (moves.n == 0) {
            SearchResult terminal;
            terminal.score = -INF;
            return terminal;
        }
    }

#ifndef __EMSCRIPTEN__
    std::vector<std::thread> helpers;

    if (num_threads_ > 1) {
        helpers.reserve(static_cast<size_t>(num_threads_ - 1));

        for (int i = 1; i < num_threads_; ++i) {
            helpers.emplace_back(
                [this, i, pos]() {
                    helper_loop(i, pos);
                }
            );
        }
    }
#endif

    SearchResult result;
    int depth = 1;

    while (!time_over_.load(std::memory_order_relaxed)) {
        if (soft_ms_ > 0 && depth > 1) {
            const double scale = id_stable_ >= 4
                ? 0.6
                : id_stable_ >= 2
                    ? 0.8
                    : 1.0;

            if (elapsed_ms() >= soft_ms_ * scale)
                break;
        }

        result = search_one_depth(pos, depth, quiet);

        if (time_over_.load(std::memory_order_relaxed))
            break;

        if (
            soft_ms_ > 0
            && std::abs(result.score) >= INF - MAX_PLY
            && depth >= 6
        ) {
            break;
        }

        if (depth >= max_depth) {
            if (!endless)
                break;

            // Stay responsive to stop() even after the supported
            // maximum depth has been reached.
        } else {
            ++depth;
        }
    }

    const bool aborted = time_over_.load(
        std::memory_order_relaxed
    );

    time_over_.store(true, std::memory_order_relaxed);

#ifndef __EMSCRIPTEN__
    for (auto& helper : helpers)
        helper.join();
#endif

    workers_[0]->publish_nodes();

    if (!valid(result.move)) {
        result.move = valid(id_best_)
            ? id_best_
            : best_from_tt(pos);
    }

    result.depth = id_completed_;
    result.nodes = total_nodes();
    result.aborted = aborted;

    if (id_completed_ == 0)
        result.score = id_score_;

    return result;
}

// -----------------------------------------------------------------------------
// Public search entry points
// -----------------------------------------------------------------------------

SearchResult Engine::search_depth_ex(
    const Position& pos,
    int max_depth,
    bool quiet
) {
    return search_internal(
        pos, max_depth, 0, 0, quiet
    );
}

Move Engine::search_depth(
    const Position& pos,
    int max_depth
) {
    return search_depth_ex(pos, max_depth, false).move;
}

SearchResult Engine::search_time_ex(
    const Position& pos,
    int time_ms,
    bool quiet
) {
    const int hard = std::max(1, time_ms);

    const int soft = static_cast<int>(
        std::max<int64_t>(
            1,
            static_cast<int64_t>(hard) * 3 / 4
        )
    );

    return search_internal(
        pos, MAX_PLY - 8, soft, hard, quiet
    );
}

Move Engine::search_time(
    const Position& pos,
    int time_ms
) {
    return search_time_ex(pos, time_ms, false).move;
}

Move Engine::search_clock(
    const Position& pos,
    int wtime,
    int btime,
    int winc,
    int binc
) {
    const int64_t time_left = std::max(
        0,
        pos.side_to_move == WHITE ? wtime : btime
    );

    const int64_t increment = std::max(
        0,
        pos.side_to_move == WHITE ? winc : binc
    );

    constexpr int64_t overhead = 30;

    int64_t soft = time_left / 25 + increment * 3 / 4;
    int64_t hard = std::min(soft * 3, time_left / 3);

    hard = std::max<int64_t>(
        1,
        std::min(hard, time_left - overhead)
    );

    soft = std::max<int64_t>(
        1,
        std::min(soft, hard)
    );

    return search_internal(
        pos,
        MAX_PLY - 8,
        static_cast<int>(soft),
        static_cast<int>(hard),
        false
    ).move;
}

SearchResult Engine::search_endless_ex(
    const Position& pos,
    bool quiet
) {
    return search_internal(
        pos, MAX_PLY - 8, 0, 0, quiet, true
    );
}

Move Engine::search_endless(const Position& pos) {
    return search_endless_ex(pos, false).move;
}