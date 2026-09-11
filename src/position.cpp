#include "position.h"
#include "zobrist.h"
#include <iostream>
#include <sstream>
#include <bit>

Position::Position() { set_initial_state(); }

void Position::set_initial_state() {
    set_from_kfen("WW3BB/WW1R1BB/2RRR2/1RRRRR1/2RRR2/BB1R1WW/BB3WW w 0 0");
}

void Position::set_from_kfen(const std::string& kfen) {
    bitboards[WHITE] = bitboards[BLACK] = bitboards[RED] = 0;
    std::istringstream iss(kfen);
    std::string board_str, turn_str;
    int w_cap = 0, b_cap = 0;

    iss >> board_str >> turn_str >> w_cap >> b_cap;

    int sq = 0;
    for (char c : board_str) {
        if (c == '/') continue;
        if (isdigit(c)) { sq += (c - '0'); }
        else if (c == 'W') { set_piece(sq++, WHITE); }
        else if (c == 'B') { set_piece(sq++, BLACK); }
        else if (c == 'R') { set_piece(sq++, RED); }
    }

    side_to_move = (turn_str == "b") ? BLACK : WHITE;
    captured_red[WHITE] = (uint8_t)w_cap;
    captured_red[BLACK] = (uint8_t)b_cap;
    refresh_occupied();
    hash = compute_hash();
    hash_prev = 0;
}

std::string Position::to_kfen() const {
    std::string kfen = "";
    for (int r = 0; r < 7; ++r) {
        int empty = 0;
        for (int c = 0; c < 7; ++c) {
            Color col = get_color_at(r * 7 + c);
            if (col == EMPTY) { empty++; }
            else {
                if (empty > 0) { kfen += std::to_string(empty); empty = 0; }
                if (col == WHITE) kfen += "W";
                else if (col == BLACK) kfen += "B";
                else if (col == RED) kfen += "R";
            }
        }
        if (empty > 0) kfen += std::to_string(empty);
        if (r < 6) kfen += "/";
    }
    kfen += (side_to_move == WHITE) ? " w " : " b ";
    kfen += std::to_string(captured_red[WHITE]) + " " + std::to_string(captured_red[BLACK]);
    return kfen;
}

std::vector<Move> Position::generate_legal_moves() const {
    std::vector<Move> moves;
    moves.reserve(32);
    uint64_t my_pieces = bitboards[side_to_move];

    while (my_pieces) {
        int sq = std::countr_zero(my_pieces);
        my_pieces &= my_pieces - 1;
        for (int dir = 0; dir < 4; ++dir) {
            if (!can_push(sq, dir)) continue;
            Position next = *this;
            if (next.do_move(sq, dir)) moves.push_back({ (uint8_t)sq, (uint8_t)dir });
        }
    }
    return moves;
}

bool Position::has_legal_moves() const {
    uint64_t my_pieces = bitboards[side_to_move];
    while (my_pieces) {
        int sq = std::countr_zero(my_pieces);
        my_pieces &= my_pieces - 1;
        for (int dir = 0; dir < 4; ++dir) {
            if (!can_push(sq, dir)) continue;
            Position next = *this;
            if (next.do_move(sq, dir)) return true;
        }
    }
    return false;
}

Color Position::get_winner() const {
    if (captured_red[WHITE] >= 7) return WHITE;
    if (captured_red[BLACK] >= 7) return BLACK;
    if (bitboards[WHITE] == 0) return BLACK;
    if (bitboards[BLACK] == 0) return WHITE;
    if (!has_legal_moves()) return (side_to_move == WHITE) ? BLACK : WHITE;
    return EMPTY;
}

uint64_t Position::compute_hash() const {
    uint64_t h = 0;
    for (int c = 0; c < 3; c++) {
        uint64_t bb = bitboards[c];
        while (bb) {
            h ^= Zobrist::pieces[c][std::countr_zero(bb)];
            bb &= bb - 1;
        }
    }
    if (side_to_move == BLACK) h ^= Zobrist::black_to_move;
    h ^= Zobrist::cap[captured_red[WHITE]][captured_red[BLACK]];
    return h;
}

void Position::print() const {
    std::cout << "\n  a b c d e f g\n";
    for (int r = 0; r < 7; ++r) {
        std::cout << (7 - r) << " ";
        for (int c = 0; c < 7; ++c) {
            Color col = get_color_at(r * 7 + c);
            if (col == WHITE) std::cout << "W ";
            else if (col == BLACK) std::cout << "B ";
            else if (col == RED) std::cout << "R ";
            else std::cout << ". ";
        }
        std::cout << "\n";
    }
    std::cout << "Captured Red - W: " << (int)captured_red[WHITE]
              << " | B: " << (int)captured_red[BLACK] << "\n\n";
}
