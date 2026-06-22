#include "position.h"
#include "zobrist.h"
#include <iostream>
#include <sstream>
#include <bit> 

Position::Position() { set_initial_state(); }

void Position::set_initial_state() {
    // Standard Kuba Setup string
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
    captured_red[WHITE] = w_cap;
    captured_red[BLACK] = b_cap;
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

int Position::get_adj(int sq, int dir) const {
    if (dir == NORTH) return (sq >= 7) ? sq - 7 : -1;
    if (dir == SOUTH) return (sq <= 41) ? sq + 7 : -1;
    if (dir == EAST) return (sq % 7 != 6) ? sq + 1 : -1;
    if (dir == WEST) return (sq % 7 != 0) ? sq - 1 : -1;
    return -1;
}

bool Position::do_move(int sq, int dir) {
    uint64_t occupied = bitboards[WHITE] | bitboards[BLACK] | bitboards[RED];
    int opposite_dir = dir ^ 1; 
    int behind = get_adj(sq, opposite_dir);
    
    if (behind != -1 && (occupied & (1ULL << behind))) return false; 

    int curr = sq;
    int pieces[7];
    int num_pieces = 0;

    while (curr != -1 && (occupied & (1ULL << curr))) {
        pieces[num_pieces++] = curr;
        curr = get_adj(curr, dir);
    }

    uint64_t old_hash = this->hash;

    if (curr == -1) {
        int falling_sq = pieces[num_pieces - 1];
        Color falling_color = get_color_at(falling_sq);
        if (falling_color == side_to_move) return false; 
        if (falling_color == RED) captured_red[side_to_move]++;
        clear_piece(falling_sq, falling_color);
        num_pieces--; 
    }

    for (int i = num_pieces - 1; i >= 0; i--) {
        int from = pieces[i];
        int to = get_adj(from, dir);
        Color c = get_color_at(from);
        clear_piece(from, c);
        set_piece(to, c);
    }

    side_to_move = (side_to_move == WHITE) ? BLACK : WHITE;
    this->hash = compute_hash();
    if (this->hash == this->hash_prev) return false; 

    this->hash_prev = old_hash;
    return true;
}

std::vector<Move> Position::generate_legal_moves() const {
    std::vector<Move> moves;
    moves.reserve(64); 
    uint64_t my_pieces = bitboards[side_to_move];
    
    while (my_pieces) {
        int sq = std::countr_zero(my_pieces);
        my_pieces &= my_pieces - 1; 
        for (int dir = 0; dir < 4; ++dir) {
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

void Position::clear_piece(int sq, Color c) { bitboards[c] &= ~(1ULL << sq); }
void Position::set_piece(int sq, Color c) { bitboards[c] |= (1ULL << sq); }

Color Position::get_color_at(int sq) const {
    if (bitboards[WHITE] & (1ULL << sq)) return WHITE;
    if (bitboards[BLACK] & (1ULL << sq)) return BLACK;
    if (bitboards[RED]   & (1ULL << sq)) return RED;
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