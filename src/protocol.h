#pragma once
#include "position.h"
#include "engine.h"
#include <atomic>
#include <mutex>
#include <string>
#include <thread>

class Protocol {
public:
    static constexpr const char* PROTOCOL_ID = "kpp";
    static constexpr int PROTOCOL_VERSION = 1;
    static constexpr const char* ENGINE_NAME = "Kahu";
    static constexpr const char* ENGINE_VERSION = "20";
    static constexpr const char* ENGINE_AUTHOR = "OscarLo11212821";

    explicit Protocol(const std::string& nnue_path = "clashtower.nnue");
    void loop();

private:
    struct SearchParams {
        enum class Mode { Default, Depth, Millis, Clock, Endless } mode = Mode::Default;
        int depth = 0;
        int millis = 0;
        int wtime = 0;
        int btime = 0;
        int wbonus = 0;
        int bbonus = 0;
    };

    Position board;
    Engine engine;
    std::string nnue_path_;

    std::thread search_thread_;
    std::atomic<bool> search_active_{false};
    std::mutex search_mutex_;

    void reap_search();
    void send_hello();
    void send_sync();
    void send_nnue_info();
    bool load_nnue(const std::string& path);
    void halt_search();
    bool parse_search_params(std::istringstream& iss, SearchParams& out, bool legacy_go);
    void start_search(const SearchParams& params);
    void handle_line(const std::string& line);
};
