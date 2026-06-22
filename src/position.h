#pragma once
#include "types.h"
#include <vector>
#include <string>

class Position {
public:
    uint64_t bitboards[3];
    uint8_t captured_red[2];
    Color side_to_move;
    
    uint64_t hash;
    uint64_t hash_prev;

    Position();
    void set_initial_state();
    
    // KFEN Support
    void set_from_kfen(const std::string& kfen);
    std::string to_kfen() const;

    bool do_move(int sq, int dir);
    std::vector<Move> generate_legal_moves() const;
    bool has_legal_moves() const;
    Color get_winner() const; 
    void print() const;

private:
    void clear_piece(int sq, Color c);
    void set_piece(int sq, Color c);
    Color get_color_at(int sq) const;
    int get_adj(int sq, int dir) const;
    uint64_t compute_hash() const;
};