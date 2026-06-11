#include "zobrist.h"
#include <random>

namespace Zobrist {
    uint64_t pieces[3][49];
    uint64_t black_to_move;

    void init() {
        std::mt19937_64 rng(123456789ULL); // Fixed seed for reproducibility
        for (int c = 0; c < 3; c++) {
            for (int s = 0; s < 49; s++) {
                pieces[c][s] = rng();
            }
        }
        black_to_move = rng();
    }
}