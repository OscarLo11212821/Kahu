#pragma once
#include <cstdint>
#include <string>

enum Color { WHITE = 0, BLACK = 1, RED = 2, EMPTY = 3 };
enum Direction { NORTH = 0, SOUTH = 1, EAST = 2, WEST = 3 };

struct Move {
    uint8_t sq;
    uint8_t dir;

    bool operator==(const Move& other) const {
        return sq == other.sq && dir == other.dir;
    }
};

// Standard algebraic for 7x7: a1-g7 (a7 is top-left, g1 is bottom-right)
inline std::string move_to_string(Move m) {
    if (m.sq == 255) return "000"; // Null move
    char file = 'a' + (m.sq % 7);
    char rank = '1' + (6 - (m.sq / 7));
    char dir = (m.dir == NORTH) ? 'N' : (m.dir == SOUTH) ? 'S' : (m.dir == EAST) ? 'E' : 'W';
    return std::string{file, rank, dir};
}

inline Move string_to_move(const std::string& s) {
    if (s == "000") return {255, 255};
    int file = s[0] - 'a';
    int rank = s[1] - '1';
    int sq = (6 - rank) * 7 + file;
    int dir = NORTH;
    if (s[2] == 'S') dir = SOUTH;
    else if (s[2] == 'E') dir = EAST;
    else if (s[2] == 'W') dir = WEST;
    return {(uint8_t)sq, (uint8_t)dir};
}