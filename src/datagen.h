#pragma once
#include "engine.h"
#include <cstdint>
#include <string>
#include <vector>

struct DataGenConfig {
    int games = 1;
    int depth = 6;
    int movetime_ms = 0;
    int threads = 0;

    std::string output = "kuba_data.txt";
    bool append = false;

    std::string start_kfen;
    std::string starts_file;
    bool cycle_starts = false;

    float random_move_prob = 0.0f;
    int random_plies_min = 0;
    int random_plies_max = 0;

    int max_plies = 200;
    int log_every = 25;
    uint32_t seed = 0;
    bool quiet = false;
};

int run_datagen(int argc, char** argv);

bool load_start_kfens(const DataGenConfig& cfg, std::vector<std::string>& out);
std::string pick_start_kfen(const DataGenConfig& cfg, const std::vector<std::string>& starts,
                            int game_index, uint32_t& rng_state);
