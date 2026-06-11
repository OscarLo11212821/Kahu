#pragma once
#include "position.h"
#include <cstdint>
#include <string>
#include <vector>

class NnueNetwork {
public:
    bool load(const std::string& path);
    bool loaded() const { return !w1q.empty(); }
    int format_version() const { return networkVersion; }

    int evaluate(const Position& pos) const;

    int hidden_size() const { return hidden; }
    int hidden2_size() const { return hidden2; }
    float cp_scale() const { return cpScale; }

private:
    static constexpr int BOARD_SQ = 49;
    static constexpr int MAX_HIDDEN = 512;
    static constexpr int MAX_HIDDEN2 = 64;
    static constexpr int MATERIAL_BUCKETS = 17;
    static constexpr int OUTPUT_BUCKETS = 16;
    static constexpr int PIECE_FEATURES = BOARD_SQ * 3;

    // V1 (legacy): capture-bucket aux features
    static constexpr int CAP_FEATURES = 8;
    static constexpr int FEATURES_PER_MATERIAL_V1 = PIECE_FEATURES + CAP_FEATURES + CAP_FEATURES + 1;
    static constexpr int BIAS_FEATURE_V1 = PIECE_FEATURES + 2 * CAP_FEATURES;

    // V3 (RedRacer): push-duel aux features + push-clash head
    static constexpr int DUEL_BUCKETS = 8;
    static constexpr int FEATURES_PER_MATERIAL_V3 = PIECE_FEATURES + DUEL_BUCKETS + DUEL_BUCKETS + 1;
    static constexpr int PUSH_THREAT_OFF = PIECE_FEATURES;
    static constexpr int MOBILITY_OFF = PIECE_FEATURES + DUEL_BUCKETS;
    static constexpr int BIAS_FEATURE_V3 = PIECE_FEATURES + 2 * DUEL_BUCKETS;
    static constexpr int FEATURE_COUNT_V3 = MATERIAL_BUCKETS * FEATURES_PER_MATERIAL_V3;

    int hidden = 0;
    int hidden2 = 0;
    int networkVersion = 0;
    float reluCap = 1.0f;
    float cpScale = 3500.0f;
    float w1Scale = 1.0f;

    std::vector<int16_t> w1q;
    std::vector<float> w2;
    std::vector<float> b2;
    std::vector<float> w3;
    std::vector<float> b3;
    std::vector<float> wSkip;
    std::vector<float> cpLut;

    void quantize_w1(const std::vector<float>& w1f);
    void build_cp_lut();
    void accumulate_perspective_v1(const Position& pos, int perspective, int base_material,
                                   float* act) const;
    void accumulate_perspective_v3(const Position& pos, int perspective, int base_material,
                                 float* act) const;
    float forward_head_v1(const float* us_act, const float* them_act, float* z2, float* h2,
                          int bucket) const;
    float forward_head_v3(const float* us_act, const float* them_act, float* z2, float* h2,
                          float* duel_adv, float* duel_field, int bucket) const;
    int cp_from_pred(float pred) const;
};

bool nnue_load(const std::string& path);
const NnueNetwork* nnue_network();
void nnue_clear_eval_cache();
int nnue_evaluate(const Position& pos);
