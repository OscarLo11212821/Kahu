#include "nnue.h"
#include "zobrist.h"
#include "protocol.h"
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

Protocol::Protocol(const std::string& nnue_path) : nnue_path_(nnue_path) {
    Zobrist::init();
    board.set_initial_state();
}

void Protocol::reap_search() {
    if (search_thread_.joinable() && !search_active_) {
        search_thread_.join();
    }
}

void Protocol::send_nnue_info() {
    const NnueNetwork* net = nnue_network();
    if (!net || !net->loaded()) {
        std::cout << "nnue none" << std::endl;
        return;
    }
    std::cout << "nnue " << nnue_path_
              << " v" << net->format_version()
              << " hidden=" << net->hidden_size()
              << " hidden2=" << net->hidden2_size()
              << " cp_scale=" << net->cp_scale()
              << std::endl;
}

void Protocol::send_hello() {
    std::cout << "protocol " << PROTOCOL_ID << " " << PROTOCOL_VERSION << std::endl;
    std::cout << "engine " << ENGINE_NAME << " " << ENGINE_VERSION << std::endl;
    std::cout << "author " << ENGINE_AUTHOR << std::endl;
    std::cout << "capabilities ping sync hello halt search go board move eval status legal nnue tt"
              << std::endl;
    send_nnue_info();
    std::cout << "ok" << std::endl;
}

void Protocol::send_sync() {
    std::cout << "synced" << std::endl;
}

bool Protocol::load_nnue(const std::string& path) {
    if (!nnue_load(path)) {
        std::cout << "nnue fail " << path << std::endl;
        return false;
    }
    nnue_path_ = path;
    std::cout << "nnue ok " << path << std::endl;
    send_nnue_info();
    return true;
}

void Protocol::halt_search() {
    if (!search_active_) return;
    engine.stop();
    if (search_thread_.joinable()) {
        search_thread_.join();
    }
    search_active_ = false;
}

bool Protocol::parse_search_params(std::istringstream& iss, SearchParams& out, bool legacy_go) {
    std::string token;
    bool is_clock = false;

    while (iss >> token) {
        if (token == "depth") {
            iss >> out.depth;
            out.mode = SearchParams::Mode::Depth;
        } else if (token == "endless" || token == "continuous") {
            out.mode = SearchParams::Mode::Endless;
        } else if (legacy_go && token == "time") {
            iss >> out.millis;
            out.mode = SearchParams::Mode::Millis;
        } else if (!legacy_go && token == "millis") {
            iss >> out.millis;
            out.mode = SearchParams::Mode::Millis;
        } else if (token == "wtime") {
            iss >> out.wtime;
            is_clock = true;
            out.mode = SearchParams::Mode::Clock;
        } else if (token == "btime") {
            iss >> out.btime;
            is_clock = true;
            out.mode = SearchParams::Mode::Clock;
        } else if (legacy_go && token == "winc") {
            iss >> out.wbonus;
            is_clock = true;
            out.mode = SearchParams::Mode::Clock;
        } else if (legacy_go && token == "binc") {
            iss >> out.bbonus;
            is_clock = true;
            out.mode = SearchParams::Mode::Clock;
        } else if (!legacy_go && (token == "wbonus" || token == "wextra")) {
            iss >> out.wbonus;
            is_clock = true;
            out.mode = SearchParams::Mode::Clock;
        } else if (!legacy_go && (token == "bbonus" || token == "bextra")) {
            iss >> out.bbonus;
            is_clock = true;
            out.mode = SearchParams::Mode::Clock;
        } else {
            std::cout << "error unknown_search_option " << token << std::endl;
            return false;
        }
    }

    if (is_clock) out.mode = SearchParams::Mode::Clock;
    return true;
}

void Protocol::start_search(const SearchParams& params) {
    reap_search();

    if (search_active_) {
        std::cout << "error search_busy" << std::endl;
        return;
    }

    Position search_pos = board;
    if (search_pos.generate_legal_moves().empty()) {
        std::cout << "bestmove 000" << std::endl;
        return;
    }

    search_active_ = true;
    engine.stop();

    search_thread_ = std::thread([this, search_pos, params]() {
        Move best = {255, 255};

        switch (params.mode) {
            case SearchParams::Mode::Depth:
                best = engine.search_depth(search_pos, params.depth);
                break;
            case SearchParams::Mode::Millis:
                best = engine.search_time(search_pos, params.millis);
                break;
            case SearchParams::Mode::Clock:
                best = engine.search_clock(search_pos, params.wtime, params.btime,
                                           params.wbonus, params.bbonus);
                break;
            case SearchParams::Mode::Endless:
                best = engine.search_endless(search_pos);
                break;
            case SearchParams::Mode::Default:
                best = engine.search_time(search_pos, 1000);
                break;
        }

        std::cout << "bestmove " << move_to_string(best) << std::endl;
        search_active_ = false;
    });
}

void Protocol::handle_line(const std::string& line) {
    if (line.empty()) return;

    std::istringstream iss(line);
    std::string cmd;
    iss >> cmd;

    if (cmd == "ping") {
        std::cout << "pong" << std::endl;
    }
    else if (cmd == "hello" || cmd == "identify") {
        send_hello();
    }
    else if (cmd == "sync") {
        send_sync();
    }
    else if (cmd == "halt" || cmd == "abort") {
        halt_search();
    }
    else if (cmd == "reset") {
        halt_search();
        board.set_initial_state();
    }
    else if (cmd == "board") {
        board.print();
    }
    else if (cmd == "getboard") {
        std::cout << board.to_kfen() << std::endl;
    }
    else if (cmd == "setboard") {
        std::string kfen = line.substr(9);
        board.set_from_kfen(kfen);
    }
    else if (cmd == "eval") {
        std::cout << "score " << nnue_evaluate(board) << std::endl;
    }
    else if (cmd == "status") {
        Color w = board.get_winner();
        if (w == WHITE) std::cout << "white_wins" << std::endl;
        else if (w == BLACK) std::cout << "black_wins" << std::endl;
        else std::cout << "ongoing" << std::endl;
    }
    else if (cmd == "legal" || cmd == "moves") {
        std::vector<Move> moves = board.generate_legal_moves();
        std::cout << "legal count " << moves.size();
        for (const Move& m : moves) {
            std::cout << " " << move_to_string(m);
        }
        std::cout << std::endl;
    }
    else if (cmd == "nnue") {
        std::string sub;
        iss >> sub;
        if (sub == "path" || sub == "file" || sub == "load") {
            std::string path;
            std::getline(iss, path);
            size_t start = path.find_first_not_of(" \t");
            if (start != std::string::npos) path = path.substr(start);
            else path.clear();
            if (path.empty()) {
                std::cout << "error nnue_path_missing" << std::endl;
            } else {
                load_nnue(path);
            }
        } else if (sub == "show" || sub == "status" || sub.empty()) {
            send_nnue_info();
        } else {
            std::cout << "error unknown_nnue_command " << sub << std::endl;
        }
    }
    else if (cmd == "tt") {
        std::string sub;
        iss >> sub;
        if (sub == "clear" || sub == "reset") {
            engine.clear_tt();
            std::cout << "tt cleared" << std::endl;
        } else {
            std::cout << "error unknown_tt_command " << sub << std::endl;
        }
    }
    else if (cmd == "move") {
        std::string m_str;
        iss >> m_str;
        Move m = string_to_move(m_str);
        if (m.sq == 255) return;
        if (!board.do_move(m.sq, m.dir)) {
            std::cout << "illegal move" << std::endl;
        }
    }
    else if (cmd == "search" || cmd == "go") {
        SearchParams params;
        if (!parse_search_params(iss, params, cmd == "go")) return;
        start_search(params);
    }
    else if (cmd == "quit" || cmd == "exit") {
        halt_search();
        throw std::runtime_error("quit");
    }
    else {
        std::cout << "error unknown_command " << cmd << std::endl;
    }
}

void Protocol::loop() {
    std::string line;
    try {
        while (std::getline(std::cin, line)) {
            reap_search();
            handle_line(line);
        }
    } catch (const std::runtime_error&) {
        // quit requested
    }
    halt_search();
}
