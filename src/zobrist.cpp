#include "zobrist.h"
#include <random>

namespace Zobrist {
    uint64_t pieces[3][49];
    uint64_t black_to_move;
    uint64_t cap[16][16];

    void init() {
        std::mt19937_64 rng(123456789ULL); // Fixed seed for reproducibility
        for (int c = 0; c < 3; c++) {
            for (int s = 0; s < 49; s++) {
                pieces[c][s] = rng();
            }
        }
        black_to_move = rng();
        for (int w = 0; w < 16; w++) {
            for (int b = 0; b < 16; b++) {
                uint64_t z = (uint64_t)(w * 16 + b + 1) * 0x9E3779B97F4A7C15ULL;
                z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
                z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
                cap[w][b] = z ^ (z >> 31);
            }
        }
    }
}
