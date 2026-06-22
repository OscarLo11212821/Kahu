#include "datagen.h"
#include "nnue.h"
#include "protocol.h"
#include "zobrist.h"
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    Zobrist::init();

    std::string net_path = "clashtower.nnue";
    bool datagen_mode = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "datagen") {
            datagen_mode = true;
        } else if (arg == "--nnue" && i + 1 < argc) {
            net_path = argv[++i];
        }
    }

    if (!nnue_load(net_path)) {
        std::cerr << "Failed to load NNUE network: " << net_path << '\n';
        return 1;
    }

    if (datagen_mode) {
        return run_datagen(argc, argv);
    }

    Protocol ckp(net_path);
    ckp.loop();
    return 0;
}
