#pragma once
#include <cstdint>

namespace Zobrist {
    extern uint64_t pieces[3][49];
    extern uint64_t black_to_move;
    void init();
}