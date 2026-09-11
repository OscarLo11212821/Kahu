#pragma once
#include "types.h"
#include "zobrist.h"
#include <bit>
#include <string>
#include <vector>

namespace kuba_adj {
struct AdjTable {
    int8_t a[49][4];
    constexpr AdjTable() : a{} {
        for (int sq = 0; sq < 49; sq++) {
            a[sq][NORTH] = (sq >= 7) ? static_cast<int8_t>(sq - 7) : -1;
            a[sq][SOUTH] = (sq <= 41) ? static_cast<int8_t>(sq + 7) : -1;
            a[sq][EAST]  = (sq % 7 != 6) ? static_cast<int8_t>(sq + 1) : -1;
            a[sq][WEST]  = (sq % 7 != 0) ? static_cast<int8_t>(sq - 1) : -1;
        }
    }
};
inline constexpr AdjTable table{};
} // namespace kuba_adj

class Position {
public:
    uint64_t bitboards[3];
    uint8_t captured_red[2];
    Color side_to_move;

    uint64_t hash;
    uint64_t hash_prev;
    uint64_t occupied;   // WHITE | BLACK | RED

    Position();
    void set_initial_state();

    void set_from_kfen(const std::string& kfen);
    std::string to_kfen() const;

    bool do_move(int sq, int dir);
    void do_null_move();

    std::vector<Move> generate_legal_moves() const;
    bool has_legal_moves() const;
    Color get_winner() const;
    void print() const;

    static int get_adj(int sq, int dir) { return kuba_adj::table.a[sq][dir]; }

    Color get_color_at(int sq) const {
        const uint64_t m = 1ULL << sq;
        if (bitboards[WHITE] & m) return WHITE;
        if (bitboards[BLACK] & m) return BLACK;
        if (bitboards[RED]   & m) return RED;
        return EMPTY;
    }

    bool can_push(int sq, int dir) const;
    int push_off_threats(Color side) const;
    int push_mobility(Color side) const;

private:
    void clear_piece(int sq, Color c) { bitboards[c] &= ~(1ULL << sq); }
    void set_piece(int sq, Color c) { bitboards[c] |= (1ULL << sq); }
    uint64_t compute_hash() const;
    void refresh_occupied() { occupied = bitboards[WHITE] | bitboards[BLACK] | bitboards[RED]; }
};

inline bool Position::do_move(int sq, int dir) {
    const int behind = get_adj(sq, dir ^ 1);
    if (behind != -1 && ((occupied >> behind) & 1)) return false;

    int line[7];
    Color col[7];
    int n = 0;
    int curr = sq;
    while (curr != -1 && ((occupied >> curr) & 1)) {
        line[n] = curr;
        col[n] = get_color_at(curr);
        n++;
        curr = get_adj(curr, dir);
    }
    if (n == 0) return false;

    const Color me = side_to_move;
    uint64_t h = hash;

    Color falling = EMPTY;
    int falling_sq = -1;
    if (curr == -1) {
        falling_sq = line[n - 1];
        falling = col[n - 1];
        if (falling == me) return false;
        h ^= Zobrist::pieces[falling][falling_sq];
        if (falling == RED) {
            h ^= Zobrist::cap[captured_red[WHITE]][captured_red[BLACK]];
            h ^= Zobrist::cap[captured_red[WHITE] + (me == WHITE)]
                             [captured_red[BLACK] + (me == BLACK)];
        }
        n--;
    }
    for (int i = 0; i < n; i++)
        h ^= Zobrist::pieces[col[i]][line[i]] ^ Zobrist::pieces[col[i]][get_adj(line[i], dir)];
    h ^= Zobrist::black_to_move;

    if (h == hash_prev) return false;

    if (falling_sq != -1) {
        clear_piece(falling_sq, falling);
        occupied &= ~(1ULL << falling_sq);
        if (falling == RED) captured_red[me]++;
    }
    for (int i = n - 1; i >= 0; i--) {
        const int from = line[i];
        const int to = get_adj(from, dir);
        clear_piece(from, col[i]);
        set_piece(to, col[i]);
        occupied ^= (1ULL << from);
        occupied ^= (1ULL << to);
    }
    side_to_move = (me == WHITE) ? BLACK : WHITE;
    hash_prev = hash;
    hash = h;
    return true;
}

inline void Position::do_null_move() {
    side_to_move = (side_to_move == WHITE) ? BLACK : WHITE;
    hash ^= Zobrist::black_to_move;
    hash_prev = 0;
}

inline bool Position::can_push(int sq, int dir) const {
    const int behind = get_adj(sq, dir ^ 1);
    if (behind != -1 && ((occupied >> behind) & 1)) return false;
    int curr = sq;
    int last = sq;
    int n = 0;
    while (curr != -1 && ((occupied >> curr) & 1)) {
        last = curr;
        n++;
        curr = get_adj(curr, dir);
    }
    if (n == 0) return false;
    if (curr == -1 && get_color_at(last) == side_to_move) return false;
    return true;
}

inline int Position::push_off_threats(Color side) const {
    const uint64_t occ = occupied;
    const uint64_t red = bitboards[RED];
    const uint64_t mine = bitboards[side];
    uint8_t rank[7], file[7];
    for (int r = 0; r < 7; r++)
        rank[r] = static_cast<uint8_t>((occ >> (7 * r)) & 127u);
    for (int f = 0; f < 7; f++) {
        uint8_t c = 0;
        uint64_t bit = 1ULL << f;
        for (int r = 0; r < 7; r++, bit <<= 7)
            if (occ & bit) c = static_cast<uint8_t>(c | (1u << r));
        file[f] = c;
    }

    auto weight = [&](int last) {
        const uint64_t m = 1ULL << last;
        if (red & m) return 3;
        if (!(mine & m)) return 1;
        return 0;
    };

    int threats = 0;
    uint64_t my = mine;
    while (my) {
        const int sq = std::countr_zero(my);
        my &= my - 1;
        const int r = sq / 7;
        const int f = sq % 7;
        const uint8_t row = rank[r];
        const uint8_t col = file[f];

        // NORTH (decreasing rank). Behind = SOUTH.
        if ((r == 6 || !(col & (1u << (r + 1)))) && r > 0 && (col & (1u << (r - 1)))) {
            int last_r = r - 1;
            while (last_r > 0 && (col & (1u << (last_r - 1)))) last_r--;
            if (last_r == 0) threats += weight(f);
        }
        // SOUTH (increasing rank). Behind = NORTH.
        if ((r == 0 || !(col & (1u << (r - 1)))) && r < 6 && (col & (1u << (r + 1)))) {
            int last_r = r + 1;
            while (last_r < 6 && (col & (1u << (last_r + 1)))) last_r++;
            if (last_r == 6) threats += weight(last_r * 7 + f);
        }
        // EAST (increasing file). Behind = WEST.
        if ((f == 0 || !(row & (1u << (f - 1)))) && f < 6 && (row & (1u << (f + 1)))) {
            int last_f = f + 1;
            while (last_f < 6 && (row & (1u << (last_f + 1)))) last_f++;
            if (last_f == 6) threats += weight(r * 7 + last_f);
        }
        // WEST (decreasing file). Behind = EAST.
        if ((f == 6 || !(row & (1u << (f + 1)))) && f > 0 && (row & (1u << (f - 1)))) {
            int last_f = f - 1;
            while (last_f > 0 && (row & (1u << (last_f - 1)))) last_f--;
            if (last_f == 0) threats += weight(r * 7);
        }
    }
    return threats;
}

inline int Position::push_mobility(Color side) const {
    const uint64_t occ = occupied;
    uint8_t rank[7], file[7];
    for (int r = 0; r < 7; r++)
        rank[r] = static_cast<uint8_t>((occ >> (7 * r)) & 127u);
    for (int f = 0; f < 7; f++) {
        uint8_t c = 0;
        uint64_t bit = 1ULL << f;
        for (int r = 0; r < 7; r++, bit <<= 7)
            if (occ & bit) c = static_cast<uint8_t>(c | (1u << r));
        file[f] = c;
    }

    int mob = 0;
    uint64_t my = bitboards[side];
    while (my) {
        const int sq = std::countr_zero(my);
        my &= my - 1;
        const int r = sq / 7;
        const int f = sq % 7;
        const uint8_t row = rank[r];
        const uint8_t col = file[f];
        if ((r == 6 || !(col & (1u << (r + 1)))) && r > 0 && (col & (1u << (r - 1)))) mob++;
        if ((r == 0 || !(col & (1u << (r - 1)))) && r < 6 && (col & (1u << (r + 1)))) mob++;
        if ((f == 0 || !(row & (1u << (f - 1)))) && f < 6 && (row & (1u << (f + 1)))) mob++;
        if ((f == 6 || !(row & (1u << (f + 1)))) && f > 0 && (row & (1u << (f - 1)))) mob++;
    }
    return mob;
}
