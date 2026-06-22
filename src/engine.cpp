#include "engine.h"
#include "nnue.h"
#include <algorithm>
#include <bit>
#include <iostream>

struct MoveList {
    Move buf[128];
    int n = 0;
};

static int adj_sq(int sq, int dir) {
    if (dir == NORTH) return (sq >= 7) ? sq - 7 : -1;
    if (dir == SOUTH) return (sq <= 41) ? sq + 7 : -1;
    if (dir == EAST) return (sq % 7 != 6) ? sq + 1 : -1;
    if (dir == WEST) return (sq % 7 != 0) ? sq - 1 : -1;
    return -1;
}

static Color color_at(const Position& pos, int sq) {
    if (pos.bitboards[WHITE] & (1ULL << sq)) return WHITE;
    if (pos.bitboards[BLACK] & (1ULL << sq)) return BLACK;
    if (pos.bitboards[RED] & (1ULL << sq)) return RED;
    return EMPTY;
}

static void gen_moves(const Position& pos, MoveList& ml) {
    ml.n = 0;
    uint64_t my_pieces = pos.bitboards[pos.side_to_move];
    while (my_pieces) {
        int sq = std::countr_zero(my_pieces);
        my_pieces &= my_pieces - 1;
        for (int dir = 0; dir < 4; dir++) {
            Position next = pos;
            if (next.do_move(sq, dir)) ml.buf[ml.n++] = {(uint8_t)sq, (uint8_t)dir};
        }
    }
}

static int push_off_threats(const Position& pos, Color side) {
    uint64_t occ = pos.bitboards[WHITE] | pos.bitboards[BLACK] | pos.bitboards[RED];
    uint64_t my_bb = pos.bitboards[side];
    int threats = 0;

    while (my_bb) {
        int sq = std::countr_zero(my_bb);
        my_bb &= my_bb - 1;
        for (int dir = 0; dir < 4; dir++) {
            int behind = adj_sq(sq, dir ^ 1);
            if (behind != -1 && (occ & (1ULL << behind))) continue;

            int curr = adj_sq(sq, dir);
            if (curr == -1 || !(occ & (1ULL << curr))) continue;

            Color falling = EMPTY;
            while (curr != -1 && (occ & (1ULL << curr))) {
                falling = color_at(pos, curr);
                curr = adj_sq(curr, dir);
            }
            if (curr == -1 && falling != side && falling != EMPTY)
                threats += (falling == RED) ? 3 : 1;
        }
    }
    return threats;
}

static bool is_tactical(const Position& pos, const Position& next) {
    return std::popcount(pos.bitboards[WHITE]) != std::popcount(next.bitboards[WHITE])
        || std::popcount(pos.bitboards[BLACK]) != std::popcount(next.bitboards[BLACK])
        || std::popcount(pos.bitboards[RED]) != std::popcount(next.bitboards[RED])
        || pos.captured_red[WHITE] != next.captured_red[WHITE]
        || pos.captured_red[BLACK] != next.captured_red[BLACK];
}

static int tactical_value(const Position& pos, const Position& next) {
    Color me = pos.side_to_move;
    Color opp = (me == WHITE) ? BLACK : WHITE;
    int val = 0;
    val += (next.captured_red[me] - pos.captured_red[me]) * 1200;
    val += (std::popcount(pos.bitboards[opp]) - std::popcount(next.bitboards[opp])) * 400;
    val -= (std::popcount(pos.bitboards[me]) - std::popcount(next.bitboards[me])) * 400;
    return val;
}

static int push_see(const Position& pos, Move m) {
    Position next = pos;
    next.do_move(m.sq, m.dir);
    return tactical_value(pos, next);
}

Engine::Engine() : tt(TT_SIZE) {
    std::memset(killer_moves, 0, sizeof(killer_moves));
    std::memset(countermoves, 0, sizeof(countermoves));
    std::memset(history, 0, sizeof(history));
    std::memset(continuation, 0, sizeof(continuation));
    std::memset(capture_history, 0, sizeof(capture_history));
    std::memset(threat_history, 0, sizeof(threat_history));
}

Move Engine::best_from_tt(const Position& pos) const {
    const TTBucket& bucket = tt[pos.hash % TT_SIZE];
    Move best = {255, 255};
    int best_depth = -1;

    for (int i = 0; i < 4; i++) {
        const TTEntry& e = bucket.entries[i];
        if (e.hash == pos.hash && e.best_move.sq != 255 && e.depth > best_depth) {
            best = e.best_move;
            best_depth = e.depth;
        }
    }
    if (best.sq != 255) return best;

    MoveList ml;
    gen_moves(pos, ml);
    if (ml.n > 0) return ml.buf[0];
    return {255, 255};
}

void Engine::stop() { time_over = true; }

void Engine::check_time() {
    if (max_time_ms > 0) {
        auto now = std::chrono::steady_clock::now();
        int elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();
        if (elapsed >= max_time_ms * 90 / 100) soft_time_over = true;
        if (elapsed >= max_time_ms) time_over = true;
    }
}

void Engine::clear_heuristics() {
    for (int i = 0; i < 64; i++) {
        for (int j = 0; j < 256; j++) {
            history[i][j] /= 2;
            capture_history[i][j] /= 2;
            threat_history[i][j] /= 2;
        }
    }
    for (int i = 0; i < 256; i++) {
        for (int j = 0; j < 256; j++) {
            continuation[i][j] /= 2;
        }
    }
}

int Engine::evaluate(const Position& pos) {
    return nnue_evaluate(pos);
}

std::vector<Move> Engine::extract_pv(Position pos, int depth) {
    std::vector<Move> pv;
    std::vector<uint64_t> visited;

    for (int i = 0; i < depth; i++) {
        const TTBucket& bucket = tt[pos.hash % TT_SIZE];
        Move best = {255, 255};

        int best_depth = -1;
        for (int j = 0; j < 4; j++) {
            const TTEntry& e = bucket.entries[j];
            if (e.hash == pos.hash && e.depth > best_depth && e.best_move.sq != 255) {
                best = e.best_move;
                best_depth = e.depth;
            }
        }

        if (best.sq != 255) {
            if (std::find(visited.begin(), visited.end(), pos.hash) != visited.end()) break;
            visited.push_back(pos.hash);
            pv.push_back(best);
            pos.do_move(best.sq, best.dir);
        } else break;
    }
    return pv;
}

void Engine::print_info(int depth, int score, int elapsed, const std::vector<Move>& pv) {
    std::cout << "info depth " << depth << " score " << score
              << " time " << elapsed << " nodes " << nodes << " pv";
    for (Move m : pv) std::cout << " " << move_to_string(m);
    std::cout << std::endl;
}

int Engine::qsearch(Position pos, int alpha, int beta, int ply, int qdepth) {
    if ((nodes++ & 2047) == 0) check_time();
    if (time_over) return alpha;

    int stand_pat = evaluate(pos);
    if (stand_pat >= beta) return beta;
    if (alpha < stand_pat) alpha = stand_pat;

    if (qdepth >= 16) return alpha;

    MoveList ml;
    gen_moves(pos, ml);

    struct ScoredMove { Move m; int score; };
    ScoredMove scored[128];
    int count = 0;

    Color me = pos.side_to_move;
    int my_threats = push_off_threats(pos, me);

    for (int i = 0; i < ml.n; i++) {
        Move m = ml.buf[i];
        Position next = pos;
        next.do_move(m.sq, m.dir);
        bool tact = is_tactical(pos, next);
        int threat_gain = push_off_threats(next, me) - my_threats;
        if (!tact && threat_gain <= 0) continue;

        int see = push_see(pos, m);
        int prio = see + threat_gain * 80 + capture_history[m.sq][m.dir] + threat_history[m.sq][m.dir];
        scored[count++] = {m, prio};
    }

    for (int i = 0; i < count; ++i) {
        int best = i;
        for (int j = i + 1; j < count; ++j)
            if (scored[j].score > scored[best].score) best = j;
        std::swap(scored[i], scored[best]);

        Position next = pos;
        next.do_move(scored[i].m.sq, scored[i].m.dir);
        int tg = push_off_threats(next, me) - my_threats;
        int ext = 0;
        if (next.captured_red[me] > pos.captured_red[me]) ext = 1;
        if (tg >= 2) ext = 1;
        int score = -qsearch(next, -beta, -alpha, ply + 1, qdepth + 1 + ext);
        if (score >= beta) return beta;
        if (score > alpha) alpha = score;
    }

    return alpha;
}

SearchResult Engine::search_depth_ex(const Position& pos, int max_depth, bool quiet) {
    max_time_ms = 0;
    time_over = false;
    soft_time_over = false;
    nodes = 0;
    start_time = std::chrono::steady_clock::now();
    clear_heuristics();
    nnue_clear_eval_cache();
    current_age++;
    root_move_ = {255, 255};

    SearchResult result;
    int score = 0;
    int completed_depth = 0;

    for (int d = 1; d <= max_depth; d++) {
        int alpha = -INF - 1;
        int beta = INF + 1;
        int delta = 25;

        if (d >= 8 && std::abs(score) < INF - 2000) {
            alpha = std::max(-INF - 1, score - delta);
            beta = std::min(INF + 1, score + delta);
        }

        if (d >= 5 && root_move_.sq == 255) {
            pvs(pos, std::max(1, d - 3), alpha, beta, 0);
        }

        while (true) {
            score = pvs(pos, d, alpha, beta, 0);
            if (time_over) break;

            if (score <= alpha) {
                delta *= 2;
                alpha = std::max(-INF - 1, score - delta);
            } else if (score >= beta) {
                delta *= 2;
                beta = std::min(INF + 1, score + delta);
            } else break;
        }

        if (time_over) break;

        result.move = best_from_tt(pos);
        root_move_ = result.move;
        result.score = score;
        completed_depth = d;

        if (!quiet) {
            auto now = std::chrono::steady_clock::now();
            int elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();
            print_info(d, score, elapsed, extract_pv(pos, d));
        }
    }

    if (result.move.sq == 255) result.move = best_from_tt(pos);
    result.depth = completed_depth;
    result.nodes = nodes;
    result.aborted = time_over;
    if (completed_depth == 0) result.score = score;
    return result;
}

Move Engine::search_depth(const Position& pos, int max_depth) {
    return search_depth_ex(pos, max_depth, false).move;
}

SearchResult Engine::search_time_ex(const Position& pos, int time_ms, bool quiet) {
    start_time = std::chrono::steady_clock::now();
    max_time_ms = time_ms;
    time_over = false;
    soft_time_over = false;
    nodes = 0;
    nnue_clear_eval_cache();
    current_age++;
    root_move_ = {255, 255};

    SearchResult result;
    int score = 0;
    int completed_depth = 0;

    for (int d = 1; d <= 64; d++) {
        if (soft_time_over && d > 1) break;

        int alpha = -INF - 1;
        int beta = INF + 1;
        int delta = 40;

        if (d >= 3 && std::abs(score) < INF - 2000) {
            alpha = std::max(-INF - 1, score - delta);
            beta = std::min(INF + 1, score + delta);
        }

        if (d >= 5 && root_move_.sq == 255) {
            pvs(pos, std::max(1, d - 3), alpha, beta, 0);
        }

        while (true) {
            score = pvs(pos, d, alpha, beta, 0);
            if (time_over) break;

            if (score <= alpha) {
                delta *= 2;
                alpha = std::max(-INF - 1, score - delta);
            } else if (score >= beta) {
                delta *= 2;
                beta = std::min(INF + 1, score + delta);
            } else break;
        }

        if (time_over) {
            if (completed_depth == 0) result.move = best_from_tt(pos);
            break;
        }

        result.move = best_from_tt(pos);
        root_move_ = result.move;
        result.score = score;
        completed_depth = d;

        if (!quiet) {
            auto now = std::chrono::steady_clock::now();
            int elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();
            print_info(d, score, elapsed, extract_pv(pos, d));
        }
    }

    if (result.move.sq == 255) result.move = best_from_tt(pos);
    result.depth = completed_depth;
    result.nodes = nodes;
    result.aborted = time_over;
    if (completed_depth == 0) result.score = score;
    return result;
}

Move Engine::search_time(const Position& pos, int time_ms) {
    return search_time_ex(pos, time_ms, false).move;
}

int Engine::pvs(Position pos, int depth, int alpha, int beta, int ply, Move prev, int extensions) {
    if ((nodes++ & 2047) == 0) check_time();
    if (time_over) return alpha;

    int mate_value = INF - ply;
    if (alpha < -mate_value) alpha = -mate_value;
    if (beta > mate_value - 1) beta = mate_value - 1;
    if (alpha >= beta) return alpha;

    Color winner = pos.get_winner();
    if (winner != EMPTY) {
        if (winner == pos.side_to_move) return INF - ply;
        if (winner == ((pos.side_to_move == WHITE) ? BLACK : WHITE)) return -INF + ply;
    }

    if (depth <= 0) return qsearch(pos, alpha, beta, ply);

    int alpha_orig = alpha;
    int k_ply = std::min(ply, 127);

    TTEntry* tte = nullptr;
    int tte_depth = -1;
    TTBucket& bucket = tt[pos.hash % TT_SIZE];
    for (int i = 0; i < 4; i++) {
        if (bucket.entries[i].hash == pos.hash && bucket.entries[i].depth > tte_depth) {
            tte = &bucket.entries[i];
            tte_depth = bucket.entries[i].depth;
        }
    }

    bool tt_cutoff = tte && tte->depth >= depth && ply > 0;

    if (tt_cutoff) {
        int tt_score = tte->score;
        if (tt_score > INF - 1000) tt_score -= ply;
        else if (tt_score < -INF + 1000) tt_score += ply;

        if (tte->flag == TT_EXACT) return tt_score;
        if (tte->flag == TT_ALPHA && tt_score <= alpha) return tt_score;
        if (tte->flag == TT_BETA && tt_score >= beta) return tt_score;
    }

    MoveList ml;
    gen_moves(pos, ml);
    int move_count = ml.n;

    if (move_count == 0) return -INF + ply;

    int static_eval = evaluate(pos);
    if (depth <= 3 && ply > 0) {
        if (static_eval + 200 * depth <= alpha)
            return qsearch(pos, alpha, beta, ply);
        if (depth <= 2 && static_eval - 180 * depth >= beta)
            return beta;
    }

    struct ScoredMove { Move m; int score; int see; bool tact; int threat_gain; int opp_threat_drop; };
    ScoredMove scored_moves[128];

    Color me = pos.side_to_move;
    Color opp = (me == WHITE) ? BLACK : WHITE;
    const int base_threats = push_off_threats(pos, me);
    const int base_opp_threats = push_off_threats(pos, opp);

    for (int i = 0; i < move_count; i++) {
        Move m = ml.buf[i];

        Position next = pos;
        next.do_move(m.sq, m.dir);
        const int threat_gain = push_off_threats(next, me) - base_threats;
        int m_score = history[m.sq][m.dir];
        if (ply == 0 && m == root_move_) m_score = 1100000;
        else if (tte && m == tte->best_move) m_score = 1000000;
        else if (m == killer_moves[k_ply][0]) m_score = 900000;
        else if (m == killer_moves[k_ply][1]) m_score = 800000;
        else if (prev.sq != 255 && m == countermoves[prev.sq][prev.dir]) m_score = 750000;
        else if (is_tactical(pos, next)) {
            int see = push_see(pos, m);
            m_score = 500000 + see * 64 + tactical_value(pos, next) + capture_history[m.sq][m.dir];
        }
        else if (threat_gain >= 1) m_score += threat_history[m.sq][m.dir];
        else if (prev.sq != 255) m_score += continuation[move_pack(prev)][move_pack(m)];
        const int opp_threat_drop = base_opp_threats - push_off_threats(next, opp);
        m_score += std::max(0, threat_gain * 48);
        scored_moves[i] = {m, m_score, push_see(pos, m), is_tactical(pos, next), threat_gain, opp_threat_drop};
    }

    int best_score = -INF - 1;
    Move best_move = {255, 255};
    int moves_searched = 0;

    for (int i = 0; i < move_count; ++i) {
        int best_idx = i;
        for (int j = i + 1; j < move_count; ++j) {
            if (scored_moves[j].score > scored_moves[best_idx].score) best_idx = j;
        }
        std::swap(scored_moves[i], scored_moves[best_idx]);

        Move m = scored_moves[i].m;
        Position next = pos;
        next.do_move(m.sq, m.dir);

        bool is_tact = scored_moves[i].tact;
        bool is_killer = (m == killer_moves[k_ply][0] || m == killer_moves[k_ply][1]);
        int threat_gain = scored_moves[i].threat_gain;
        int opp_threat_drop = scored_moves[i].opp_threat_drop;
        int ext = 0;
        if (extensions < 2 && (is_tact || threat_gain >= 2)) ext = 1;
        if (extensions < 2 && opp_threat_drop >= 2) ext = 1;
        if (extensions < 2 && base_opp_threats >= 3) ext = 1;
        if (extensions < 2 && next.captured_red[me] > pos.captured_red[me]) ext = 1;
        int new_depth = depth - 1 + ext;
        int score;

        if (depth <= 4 && moves_searched > 0 && !is_tact && threat_gain < 1 && opp_threat_drop < 1
            && scored_moves[i].see <= 0
            && static_eval + 110 * depth + scored_moves[i].see < alpha) {
            continue;
        }
        if (depth <= 7 && moves_searched >= 3 + depth * depth && !is_tact && threat_gain < 1
            && opp_threat_drop < 1 && scored_moves[i].see <= 0) {
            continue;
        }

        if (moves_searched == 0) {
            score = -pvs(next, new_depth, -beta, -alpha, ply + 1, m, extensions + ext);
        } else {
            int R = 0;
            bool is_counter = prev.sq != 255 && m == countermoves[prev.sq][prev.dir];
            if (depth >= 5 && ply > 0 && moves_searched >= 4 && !is_tact && !is_killer && !is_counter
                && threat_gain < 1 && opp_threat_drop < 1 && scored_moves[i].see <= 0) {
                R = 1;
            }

            if (R > 0) {
                score = -pvs(next, new_depth - R, -alpha - 1, -alpha, ply + 1, m, extensions);
                if (score > alpha)
                    score = -pvs(next, new_depth, -beta, -alpha, ply + 1, m, extensions + ext);
            } else {
                score = -pvs(next, new_depth, -alpha - 1, -alpha, ply + 1, m, extensions + ext);
                if (score > alpha && score < beta)
                    score = -pvs(next, new_depth, -beta, -alpha, ply + 1, m, extensions + ext);
            }
        }

        if (time_over) return (best_score > -INF - 1) ? best_score : alpha;

        moves_searched++;

        if (score > best_score) {
            best_score = score;
            best_move = m;
        }
        if (score > alpha) alpha = score;
        if (alpha >= beta) {
            if (!is_tact) {
                killer_moves[k_ply][1] = killer_moves[k_ply][0];
                killer_moves[k_ply][0] = m;
                if (prev.sq != 255) {
                    countermoves[prev.sq][prev.dir] = m;
                    continuation[move_pack(prev)][move_pack(m)] += depth * depth;
                }
                history[m.sq][m.dir] += depth * depth;
                if (history[m.sq][m.dir] > 16384) {
                    for (int r = 0; r < 64; r++)
                        for (int c = 0; c < 256; c++)
                            history[r][c] >>= 1;
                }
            } else {
                capture_history[m.sq][m.dir] += depth * depth;
                if (capture_history[m.sq][m.dir] > 16384) {
                    for (int r = 0; r < 64; r++)
                        for (int c = 0; c < 256; c++)
                            capture_history[r][c] >>= 1;
                }
            }
            if (threat_gain >= 2) {
                threat_history[m.sq][m.dir] += depth * depth;
                if (threat_history[m.sq][m.dir] > 16384) {
                    for (int r = 0; r < 64; r++)
                        for (int c = 0; c < 256; c++)
                            threat_history[r][c] >>= 1;
                }
            }
            break;
        }
    }

    int tt_score = best_score;
    if (tt_score > INF - 1000) tt_score += ply;
    else if (tt_score < -INF + 1000) tt_score -= ply;

    TTFlag new_flag = TT_EXACT;
    if (best_score <= alpha_orig) new_flag = TT_ALPHA;
    else if (best_score >= beta) new_flag = TT_BETA;

    TTEntry new_entry = {pos.hash, best_move, (int16_t)tt_score, (int8_t)depth, new_flag, current_age};

    int replace_idx = -1;
    for (int i = 0; i < 4; i++) {
        if (bucket.entries[i].hash == pos.hash) {
            if (depth >= bucket.entries[i].depth ||
                bucket.entries[i].age != current_age ||
                new_flag == TT_EXACT) {
                replace_idx = i;
            } else {
                replace_idx = -2;
            }
            break;
        }
    }

    if (replace_idx == -2) {
        int min_depth = 999;
        for (int i = 0; i < 4; i++) {
            if (bucket.entries[i].hash != pos.hash && bucket.entries[i].depth < min_depth) {
                min_depth = bucket.entries[i].depth;
                replace_idx = i;
            }
        }
    }
    if (replace_idx == -1) {
        for (int i = 0; i < 4; i++) if (bucket.entries[i].age != current_age) { replace_idx = i; break; }
    }
    if (replace_idx == -1) {
        int min_depth = 999;
        for (int i = 0; i < 4; i++) {
            if (bucket.entries[i].depth < min_depth) {
                min_depth = bucket.entries[i].depth;
                replace_idx = i;
            }
        }
    }
    if (replace_idx >= 0) bucket.entries[replace_idx] = new_entry;

    return best_score;
}

Move Engine::search_clock(const Position& pos, int wtime, int btime, int winc, int binc) {
    int time_left = (pos.side_to_move == WHITE) ? wtime : btime;
    int inc = (pos.side_to_move == WHITE) ? winc : binc;

    int allocated_time = (time_left / 20) + (inc / 2);
    if (allocated_time > time_left - 50) allocated_time = std::max(1, time_left - 50);

    return search_time(pos, allocated_time);
}

SearchResult Engine::search_endless_ex(const Position& pos, bool quiet) {
    max_time_ms = 0;
    time_over = false;
    soft_time_over = false;
    nodes = 0;
    start_time = std::chrono::steady_clock::now();
    clear_heuristics();
    nnue_clear_eval_cache();
    current_age++;
    root_move_ = {255, 255};

    SearchResult result;
    int score = 0;
    int completed_depth = 0;

    for (int d = 1; d <= 512; d++) {
        if (time_over) break;

        int alpha = -INF - 1;
        int beta = INF + 1;
        int delta = 25;

        if (d >= 8 && std::abs(score) < INF - 2000) {
            alpha = std::max(-INF - 1, score - delta);
            beta = std::min(INF + 1, score + delta);
        }

        while (true) {
            score = pvs(pos, d, alpha, beta, 0);
            if (time_over) break;

            if (score <= alpha) {
                delta *= 2;
                alpha = std::max(-INF - 1, score - delta);
            } else if (score >= beta) {
                delta *= 2;
                beta = std::min(INF + 1, score + delta);
            } else break;
        }

        if (time_over) break;

        result.move = best_from_tt(pos);
        root_move_ = result.move;
        result.score = score;
        completed_depth = d;

        if (!quiet) {
            auto now = std::chrono::steady_clock::now();
            int elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();
            print_info(d, score, elapsed, extract_pv(pos, d));
        }
    }

    if (result.move.sq == 255) result.move = best_from_tt(pos);
    result.depth = completed_depth;
    result.nodes = nodes;
    result.aborted = time_over;
    if (completed_depth == 0) result.score = score;
    return result;
}

Move Engine::search_endless(const Position& pos) {
    return search_endless_ex(pos, false).move;
}

void Engine::clear_tt() {
    for (TTBucket& bucket : tt) {
        for (int i = 0; i < 4; i++) {
            bucket.entries[i] = TTEntry{};
        }
    }
}
