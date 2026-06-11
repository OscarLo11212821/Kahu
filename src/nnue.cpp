#include "nnue.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#define KUBA_NNUE_SIMD 1
#elif defined(__AVX2__)
#include <immintrin.h>
#define KUBA_NNUE_SIMD 1
#endif

namespace {

static NnueNetwork g_network;

constexpr int REL_OWN = 0;
constexpr int REL_OPP = 1;
constexpr int REL_RED = 2;
constexpr int MAX_HIDDEN = 512;
constexpr int MAX_HIDDEN2 = 64;

constexpr int EVAL_CACHE_BITS = 18;
constexpr size_t EVAL_CACHE_SIZE = static_cast<size_t>(1) << EVAL_CACHE_BITS;

struct EvalCacheEntry {
    uint64_t key = 0;
    int score = 0;
};

EvalCacheEntry g_eval_cache[EVAL_CACHE_SIZE];

inline uint64_t eval_key(const Position& pos) {
    return pos.hash
         ^ (static_cast<uint64_t>(pos.captured_red[WHITE]) * 0x9E3779B185EBCA87ULL)
         ^ (static_cast<uint64_t>(pos.captured_red[BLACK]) * 0xC2B2AE3D27D4EB4FULL);
}

uint8_t g_orient_black[49];

struct Tables {
    Tables() {
        for (int sq = 0; sq < 49; ++sq) {
            const int rank = sq / 7;
            const int file = sq % 7;
            g_orient_black[sq] = static_cast<uint8_t>((6 - rank) * 7 + file);
        }
    }
};
const Tables g_tables;

inline int adj_sq(int sq, int dir) {
    if (dir == NORTH) return (sq >= 7) ? sq - 7 : -1;
    if (dir == SOUTH) return (sq <= 41) ? sq + 7 : -1;
    if (dir == EAST) return (sq % 7 != 6) ? sq + 1 : -1;
    if (dir == WEST) return (sq % 7 != 0) ? sq - 1 : -1;
    return -1;
}

inline Color color_at(const Position& pos, int sq) {
    const uint64_t mask = 1ULL << sq;
    if (pos.bitboards[WHITE] & mask) return WHITE;
    if (pos.bitboards[BLACK] & mask) return BLACK;
    if (pos.bitboards[RED] & mask) return RED;
    return EMPTY;
}

inline int push_column_square(int perspective, int sq) {
    int rank = sq / 7;
    const int file = sq % 7;
    if (perspective == BLACK) rank = 6 - rank;
    return file * 7 + rank;
}

inline int duel_bucket(int delta) {
    return std::clamp(delta, -4, 3) + 4;
}

constexpr int kPieceFeatures = 49 * 3;
constexpr int kMaterialBuckets = 17;
constexpr int kOutputBuckets = 16;

inline int feature_rel_color_v3(int local_feat) {
    if (local_feat >= kPieceFeatures) return -1;
    return local_feat % 3;
}

inline int count_playable_pieces(const Position& pos) {
    return std::popcount(pos.bitboards[WHITE]) + std::popcount(pos.bitboards[BLACK]);
}

inline int material_bucket_v1(const Position& pos) {
    return std::clamp(count_playable_pieces(pos), 0, kMaterialBuckets - 1);
}

inline int material_bucket_v3(const Position& pos) {
    int reds = static_cast<int>(pos.captured_red[WHITE]) + static_cast<int>(pos.captured_red[BLACK]);
    reds += std::popcount(pos.bitboards[RED]);
    const int pieces = count_playable_pieces(pos);
    return std::clamp(reds + pieces / 4, 0, kMaterialBuckets - 1);
}

inline int output_bucket_v1(const Position& pos) {
    return std::clamp(
        static_cast<int>(pos.captured_red[WHITE]) + static_cast<int>(pos.captured_red[BLACK]),
        0, kOutputBuckets - 1);
}

inline int output_bucket_v3(const Position& pos) {
    const int stm = static_cast<int>(pos.side_to_move);
    const int margin = static_cast<int>(pos.captured_red[stm])
                     - static_cast<int>(pos.captured_red[stm ^ 1]);
    return std::clamp(margin + 8, 0, kOutputBuckets - 1);
}

inline int push_mobility(const Position& pos, Color side) {
    int mob = 0;
    const uint64_t occupied = pos.bitboards[WHITE] | pos.bitboards[BLACK] | pos.bitboards[RED];
    uint64_t bb = pos.bitboards[side];
    while (bb) {
        const int sq = static_cast<int>(std::countr_zero(bb));
        bb &= bb - 1ULL;
        for (int dir = 0; dir < 4; ++dir) {
            const int behind = adj_sq(sq, dir ^ 1);
            if (behind != -1 && (occupied & (1ULL << behind))) continue;
            const int curr = adj_sq(sq, dir);
            if (curr != -1 && (occupied & (1ULL << curr))) ++mob;
        }
    }
    return mob;
}

inline int push_off_threats(const Position& pos, Color side) {
    int threats = 0;
    const uint64_t occupied = pos.bitboards[WHITE] | pos.bitboards[BLACK] | pos.bitboards[RED];
    uint64_t bb = pos.bitboards[side];
    while (bb) {
        const int sq = static_cast<int>(std::countr_zero(bb));
        bb &= bb - 1ULL;
        for (int dir = 0; dir < 4; ++dir) {
            const int behind = adj_sq(sq, dir ^ 1);
            if (behind != -1 && (occupied & (1ULL << behind))) continue;
            int curr = adj_sq(sq, dir);
            if (curr == -1 || !(occupied & (1ULL << curr))) continue;

            Color falling = EMPTY;
            while (curr != -1 && (occupied & (1ULL << curr))) {
                falling = color_at(pos, curr);
                curr = adj_sq(curr, dir);
            }
            if (curr == -1 && falling != EMPTY && falling != side) {
                threats += (falling == RED) ? 3 : 1;
            }
        }
    }
    return threats;
}

inline void push_clash_combine(const float* us_act, const float* them_act,
                               float* duel_adv, float* duel_field, int n) {
    for (int i = 0; i < n; ++i) {
        duel_adv[i] = us_act[i] - them_act[i];
        duel_field[i] = us_act[i] * them_act[i];
    }
}

inline float fast_tanh(float x) {
    if (x > 4.0f) return 1.0f;
    if (x < -4.0f) return -1.0f;
    const float x2 = x * x;
    return x * (27.0f + x2) / (27.0f + 9.0f * x2);
}

#if defined(KUBA_NNUE_SIMD) && (defined(__ARM_NEON) || defined(__aarch64__))

inline void simd_zero_i32(int32_t* p, int n) {
    int k = 0;
    const int32x4_t z = vdupq_n_s32(0);
    for (; k + 4 <= n; k += 4) vst1q_s32(p + k, z);
    for (; k < n; ++k) p[k] = 0;
}

inline void simd_axpy_i16_i32(int32_t* acc, const int16_t* row, int n) {
    int k = 0;
    for (; k + 8 <= n; k += 8) {
        const int16x8_t r = vld1q_s16(row + k);
        const int32x4_t lo = vmovl_s16(vget_low_s16(r));
        const int32x4_t hi = vmovl_s16(vget_high_s16(r));
        vst1q_s32(acc + k, vaddq_s32(vld1q_s32(acc + k), lo));
        vst1q_s32(acc + k + 4, vaddq_s32(vld1q_s32(acc + k + 4), hi));
    }
    for (; k < n; ++k) acc[k] += row[k];
}

inline void simd_accum_feature_row_v3(int32_t* acc, const int16_t* row, int rel_color, int n) {
    if (rel_color == REL_RED) {
        const int half = n / 2;
        for (int k = 0; k < half; ++k) acc[k] -= row[k];
        for (int k = half; k < n; ++k) acc[k] += row[k];
    } else {
        simd_axpy_i16_i32(acc, row, n);
    }
}

inline void simd_i32_to_f32_scale(float* dst, const int32_t* src, float scale, int n) {
    const float32x4_t vs = vdupq_n_f32(scale);
    int k = 0;
    for (; k + 4 <= n; k += 4) {
        const int32x4_t iv = vld1q_s32(src + k);
        const float32x4_t fv = vmulq_f32(vs, vcvtq_f32_s32(iv));
        vst1q_f32(dst + k, fv);
    }
    for (; k < n; ++k) dst[k] = static_cast<float>(src[k]) * scale;
}

inline void simd_clip_relu_sq(float* act, const float* acc, float cap, int n) {
    const float32x4_t vcap = vdupq_n_f32(cap);
    const float32x4_t v0 = vdupq_n_f32(0.0f);
    int k = 0;
    for (; k + 4 <= n; k += 4) {
        float32x4_t x = vld1q_f32(acc + k);
        x = vmaxq_f32(v0, vminq_f32(vcap, x));
        vst1q_f32(act + k, vmulq_f32(x, x));
    }
    for (; k < n; ++k) {
        float x = acc[k];
        if (x < 0.0f) x = 0.0f;
        else if (x > cap) x = cap;
        act[k] = x * x;
    }
}

inline void simd_w2_accum(float* z2, float x, const float* row, int h2) {
    if (x == 0.0f) return;
    const float32x4_t vx = vdupq_n_f32(x);
    int j = 0;
    for (; j + 4 <= h2; j += 4) {
        vst1q_f32(z2 + j, vmlaq_f32(vld1q_f32(z2 + j), vx, vld1q_f32(row + j)));
    }
    for (; j < h2; ++j) z2[j] += x * row[j];
}

inline float simd_dot(const float* a, const float* b, int n) {
    float32x4_t sum = vdupq_n_f32(0.0f);
    int k = 0;
    for (; k + 4 <= n; k += 4) {
        sum = vmlaq_f32(sum, vld1q_f32(a + k), vld1q_f32(b + k));
    }
    float s = vaddvq_f32(sum);
    for (; k < n; ++k) s += a[k] * b[k];
    return s;
}

#elif defined(KUBA_NNUE_SIMD) && defined(__AVX2__)

inline void simd_zero_i32(int32_t* p, int n) {
    const __m256i z = _mm256_setzero_si256();
    int k = 0;
    for (; k + 8 <= n; k += 8) _mm256_storeu_si256(reinterpret_cast<__m256i*>(p + k), z);
    for (; k < n; ++k) p[k] = 0;
}

inline void simd_axpy_i16_i32(int32_t* acc, const int16_t* row, int n) {
    int k = 0;
    for (; k + 8 <= n; k += 8) {
        const __m128i r16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(row + k));
        const __m256i r32 = _mm256_cvtepi16_epi32(r16);
        __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc + k));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc + k), _mm256_add_epi32(a, r32));
    }
    for (; k < n; ++k) acc[k] += row[k];
}

inline void simd_accum_feature_row_v3(int32_t* acc, const int16_t* row, int rel_color, int n) {
    if (rel_color == REL_RED) {
        const int half = n / 2;
        for (int k = 0; k < half; ++k) acc[k] -= row[k];
        for (int k = half; k < n; ++k) acc[k] += row[k];
    } else {
        simd_axpy_i16_i32(acc, row, n);
    }
}

inline void simd_i32_to_f32_scale(float* dst, const int32_t* src, float scale, int n) {
    const __m256 vs = _mm256_set1_ps(scale);
    int k = 0;
    for (; k + 8 <= n; k += 8) {
        const __m256i iv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + k));
        const __m256 fv = _mm256_mul_ps(_mm256_cvtepi32_ps(iv), vs);
        _mm256_storeu_ps(dst + k, fv);
    }
    for (; k < n; ++k) dst[k] = static_cast<float>(src[k]) * scale;
}

inline void simd_clip_relu_sq(float* act, const float* acc, float cap, int n) {
    const __m256 vcap = _mm256_set1_ps(cap);
    const __m256 v0 = _mm256_setzero_ps();
    int k = 0;
    for (; k + 8 <= n; k += 8) {
        __m256 x = _mm256_loadu_ps(acc + k);
        x = _mm256_max_ps(v0, _mm256_min_ps(vcap, x));
        _mm256_storeu_ps(act + k, _mm256_mul_ps(x, x));
    }
    for (; k < n; ++k) {
        float x = acc[k];
        if (x < 0.0f) x = 0.0f;
        else if (x > cap) x = cap;
        act[k] = x * x;
    }
}

inline void simd_w2_accum(float* z2, float x, const float* row, int h2) {
    if (x == 0.0f) return;
    const __m256 vx = _mm256_set1_ps(x);
    int j = 0;
    for (; j + 8 <= h2; j += 8) {
        _mm256_storeu_ps(z2 + j, _mm256_fmadd_ps(vx, _mm256_loadu_ps(row + j), _mm256_loadu_ps(z2 + j)));
    }
    for (; j < h2; ++j) z2[j] += x * row[j];
}

inline float simd_dot(const float* a, const float* b, int n) {
    __m256 sum = _mm256_setzero_ps();
    int k = 0;
    for (; k + 8 <= n; k += 8) {
        sum = _mm256_fmadd_ps(_mm256_loadu_ps(a + k), _mm256_loadu_ps(b + k), sum);
    }
    __m128 hi = _mm256_extractf128_ps(sum, 1);
    __m128 lo = _mm256_castps256_ps128(sum);
    __m128 s4 = _mm_add_ps(lo, hi);
    s4 = _mm_hadd_ps(s4, s4);
    s4 = _mm_hadd_ps(s4, s4);
    float s = _mm_cvtss_f32(s4);
    for (; k < n; ++k) s += a[k] * b[k];
    return s;
}

#else

inline void simd_zero_i32(int32_t* p, int n) {
    std::fill(p, p + n, 0);
}
inline void simd_axpy_i16_i32(int32_t* acc, const int16_t* row, int n) {
    for (int k = 0; k < n; ++k) acc[k] += row[k];
}
inline void simd_accum_feature_row_v3(int32_t* acc, const int16_t* row, int rel_color, int n) {
    if (rel_color == REL_RED) {
        const int half = n / 2;
        for (int k = 0; k < half; ++k) acc[k] -= row[k];
        for (int k = half; k < n; ++k) acc[k] += row[k];
    } else {
        simd_axpy_i16_i32(acc, row, n);
    }
}
inline void simd_i32_to_f32_scale(float* dst, const int32_t* src, float scale, int n) {
    for (int k = 0; k < n; ++k) dst[k] = static_cast<float>(src[k]) * scale;
}
inline void simd_clip_relu_sq(float* act, const float* acc, float cap, int n) {
    for (int k = 0; k < n; ++k) {
        float x = acc[k];
        if (x < 0.0f) x = 0.0f;
        else if (x > cap) x = cap;
        act[k] = x * x;
    }
}
inline void simd_w2_accum(float* z2, float x, const float* row, int h2) {
    if (x == 0.0f) return;
    for (int j = 0; j < h2; ++j) z2[j] += x * row[j];
}
inline float simd_dot(const float* a, const float* b, int n) {
    float s = 0.0f;
    for (int k = 0; k < n; ++k) s += a[k] * b[k];
    return s;
}

#endif

struct alignas(64) EvalScratch {
    int32_t acc_i[MAX_HIDDEN];
    float acc_f[MAX_HIDDEN];
    float act_w[MAX_HIDDEN];
    float act_b[MAX_HIDDEN];
    float z2[MAX_HIDDEN2];
    float h2[MAX_HIDDEN2];
    float duel_adv[MAX_HIDDEN];
    float duel_field[MAX_HIDDEN];
};

EvalScratch& thread_scratch() {
    thread_local EvalScratch scratch;
    return scratch;
}

} // namespace

void nnue_clear_eval_cache() {
    std::memset(g_eval_cache, 0, sizeof(g_eval_cache));
}

bool nnue_load(const std::string& path) {
    return g_network.load(path);
}

const NnueNetwork* nnue_network() {
    return g_network.loaded() ? &g_network : nullptr;
}

int nnue_evaluate(const Position& pos) {
    if (!g_network.loaded()) return 0;

    const uint64_t key = eval_key(pos);
    const size_t slot = key & (EVAL_CACHE_SIZE - 1);
    const EvalCacheEntry& e = g_eval_cache[slot];
    if (e.key == key) return e.score;

    const int score = g_network.evaluate(pos);
    g_eval_cache[slot] = {key, score};
    return score;
}

void NnueNetwork::quantize_w1(const std::vector<float>& w1f) {
    float max_abs = 1e-8f;
    for (float v : w1f) max_abs = std::max(max_abs, std::abs(v));

    w1Scale = max_abs / 32767.0f;
    w1q.resize(w1f.size());
    for (size_t i = 0; i < w1f.size(); ++i) {
        const float scaled = w1f[i] / w1Scale;
        w1q[i] = static_cast<int16_t>(std::clamp(static_cast<int>(std::lround(scaled)), -32767, 32767));
    }
}

void NnueNetwork::build_cp_lut() {
    cpLut.resize(4096);
    for (int i = 0; i < 4096; ++i) {
        const float p = (static_cast<float>(i) / 2047.5f) - 1.0f;
        const float clamped = std::clamp(p, -0.999999f, 0.999999f);
        cpLut[static_cast<size_t>(i)] = std::atanh(clamped) * cpScale;
    }
}

bool NnueNetwork::load(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        std::cerr << "nnue: failed to open " << path << '\n';
        return false;
    }

    struct SaveHeader {
        char magic[16];
        std::uint32_t version;
        std::uint32_t hidden_u;
        std::uint32_t hidden2_u;
        std::uint32_t features_per_material;
        std::uint32_t feature_count;
        float relu_cap;
        float cp_scale;
    } h{};

    if (!in.read(reinterpret_cast<char*>(&h), sizeof(h))) return false;

    int expectedVersion = 0;
    std::uint32_t expectedFeaturesPerMaterial = 0;
    std::uint32_t expectedFeatureCount = 0;
    if (std::memcmp(h.magic, "KUBA_NNUE_V3\0", 13) == 0) {
        expectedVersion = 3;
        expectedFeaturesPerMaterial = FEATURES_PER_MATERIAL_V3;
        expectedFeatureCount = FEATURE_COUNT_V3;
    } else if (std::memcmp(h.magic, "KUBA_NNUE_V1\0", 13) == 0) {
        expectedVersion = 1;
        expectedFeaturesPerMaterial = FEATURES_PER_MATERIAL_V1;
        expectedFeatureCount = MATERIAL_BUCKETS * FEATURES_PER_MATERIAL_V1;
    } else {
        std::cerr << "nnue: bad magic in " << path << " (expected KUBA_NNUE_V1 or KUBA_NNUE_V3)\n";
        return false;
    }
    if (h.version != static_cast<std::uint32_t>(expectedVersion)
        || h.features_per_material != expectedFeaturesPerMaterial
        || h.feature_count != expectedFeatureCount) {
        std::cerr << "nnue: incompatible feature layout in " << path << '\n';
        return false;
    }

    networkVersion = expectedVersion;
    hidden = static_cast<int>(h.hidden_u);
    hidden2 = static_cast<int>(h.hidden2_u);
    reluCap = h.relu_cap;
    cpScale = h.cp_scale;

    if (hidden > MAX_HIDDEN || hidden2 > MAX_HIDDEN2) {
        std::cerr << "nnue: network too large (hidden=" << hidden << " hidden2=" << hidden2 << ")\n";
        return false;
    }

    const size_t w1_n = static_cast<size_t>(expectedFeatureCount) * static_cast<size_t>(hidden);
    const size_t w2_n = static_cast<size_t>(OUTPUT_BUCKETS * 2 * hidden) * static_cast<size_t>(hidden2);
    const size_t b2_n = static_cast<size_t>(OUTPUT_BUCKETS * hidden2);
    const size_t w3_n = static_cast<size_t>(OUTPUT_BUCKETS * hidden2);
    const size_t b3_n = static_cast<size_t>(OUTPUT_BUCKETS);
    const size_t skip_n = static_cast<size_t>(OUTPUT_BUCKETS * 2 * hidden);

    std::vector<float> w1f(w1_n);
    w2.resize(w2_n);
    b2.resize(b2_n);
    w3.resize(w3_n);
    b3.resize(b3_n);
    wSkip.resize(skip_n);

    auto must_read = [&](void* dst, size_t bytes, const char* what) -> bool {
        if (!in.read(reinterpret_cast<char*>(dst), static_cast<std::streamsize>(bytes))) {
            std::cerr << "nnue: truncated " << what << " in " << path << '\n';
            return false;
        }
        return true;
    };

    if (!must_read(w1f.data(), w1_n * sizeof(float), "w1")) return false;
    if (!must_read(w2.data(), w2_n * sizeof(float), "w2")) return false;
    if (!must_read(b2.data(), b2_n * sizeof(float), "b2")) return false;
    if (!must_read(w3.data(), w3_n * sizeof(float), "w3")) return false;
    if (!must_read(b3.data(), b3_n * sizeof(float), "b3")) return false;
    if (!must_read(wSkip.data(), skip_n * sizeof(float), "wSkip")) return false;

    quantize_w1(w1f);
    build_cp_lut();
    nnue_clear_eval_cache();

#if defined(KUBA_NNUE_SIMD) && defined(__AVX2__)
    const char* simd = "AVX2+i16";
#elif defined(KUBA_NNUE_SIMD)
    const char* simd = "NEON+i16";
#else
    const char* simd = "scalar+i16";
#endif
    std::cerr << "nnue: loaded " << path
              << " v" << networkVersion
              << " hidden=" << hidden << " hidden2=" << hidden2
              << " cp_scale=" << cpScale << " simd=" << simd
              << " eval_cache=" << (1 << EVAL_CACHE_BITS) << '\n';
    return true;
}

void NnueNetwork::accumulate_perspective_v1(const Position& pos, int perspective, int base_material,
                                            float* act) const {
    const int H = hidden;
    const size_t stride = static_cast<size_t>(H);
    const int16_t* w1_base = w1q.data();

    EvalScratch& scratch = thread_scratch();
    int32_t* acc_i = scratch.acc_i;
    simd_zero_i32(acc_i, H);

    auto add_feat = [&](int feat) {
        const int16_t* row = w1_base + static_cast<size_t>(feat) * stride;
        simd_axpy_i16_i32(acc_i, row, H);
    };

    add_feat(base_material + BIAS_FEATURE_V1);

    auto add_bb = [&](uint64_t bb, int rel) {
        while (bb) {
            const int sq = std::countr_zero(bb);
            bb &= bb - 1ULL;
            const int oriented = (perspective == BLACK) ? g_orient_black[sq] : sq;
            add_feat(base_material + oriented * 3 + rel);
        }
    };

    add_bb(pos.bitboards[perspective], REL_OWN);
    add_bb(pos.bitboards[perspective ^ 1], REL_OPP);
    add_bb(pos.bitboards[RED], REL_RED);

    const int our_cap = static_cast<int>(pos.captured_red[perspective]);
    const int their_cap = static_cast<int>(pos.captured_red[perspective ^ 1]);
    add_feat(base_material + PIECE_FEATURES + our_cap);
    add_feat(base_material + PIECE_FEATURES + CAP_FEATURES + their_cap);

    float* acc_f = scratch.acc_f;
    simd_i32_to_f32_scale(acc_f, acc_i, w1Scale, H);
    simd_clip_relu_sq(act, acc_f, reluCap, H);
}

void NnueNetwork::accumulate_perspective_v3(const Position& pos, int perspective, int base_material,
                                              float* act) const {
    const int H = hidden;
    const size_t stride = static_cast<size_t>(H);
    const int16_t* w1_base = w1q.data();

    EvalScratch& scratch = thread_scratch();
    int32_t* acc_i = scratch.acc_i;
    simd_zero_i32(acc_i, H);

    auto add_feat = [&](int feat) {
        const int local = feat % FEATURES_PER_MATERIAL_V3;
        const int16_t* row = w1_base + static_cast<size_t>(feat) * stride;
        simd_accum_feature_row_v3(acc_i, row, feature_rel_color_v3(local), H);
    };

    add_feat(base_material + BIAS_FEATURE_V3);

    for (int sq = 0; sq < BOARD_SQ; ++sq) {
        const Color cell = color_at(pos, sq);
        if (cell == EMPTY) continue;

        int rel = REL_RED;
        if (cell != RED) rel = static_cast<int>(cell) ^ perspective;

        const int sq_rel = push_column_square(perspective, sq);
        add_feat(base_material + sq_rel * 3 + rel);
    }

    const Color side = static_cast<Color>(perspective);
    const Color opp = static_cast<Color>(perspective ^ 1);
    const int threat_delta = push_off_threats(pos, side) - push_off_threats(pos, opp);
    const int mob_delta = push_mobility(pos, side) - push_mobility(pos, opp);
    add_feat(base_material + PUSH_THREAT_OFF + duel_bucket(threat_delta));
    add_feat(base_material + MOBILITY_OFF + duel_bucket(mob_delta));

    float* acc_f = scratch.acc_f;
    simd_i32_to_f32_scale(acc_f, acc_i, w1Scale, H);
    simd_clip_relu_sq(act, acc_f, reluCap, H);
}

float NnueNetwork::forward_head_v1(const float* us_act, const float* them_act, float* z2, float* h2,
                                     int bucket) const {
    const int H = hidden;
    const int H2 = hidden2;

    const size_t off2 = static_cast<size_t>(bucket) * static_cast<size_t>(2 * H * H2);
    const size_t off_b2 = static_cast<size_t>(bucket) * static_cast<size_t>(H2);
    const size_t off3 = static_cast<size_t>(bucket) * static_cast<size_t>(H2);
    const size_t off_skip = static_cast<size_t>(bucket) * static_cast<size_t>(2 * H);

    std::memcpy(z2, b2.data() + off_b2, static_cast<size_t>(H2) * sizeof(float));
    const float* w2_b = w2.data() + off2;

    for (int i = 0; i < H; ++i) {
        simd_w2_accum(z2, us_act[i], w2_b + static_cast<size_t>(i) * static_cast<size_t>(H2), H2);
    }
    for (int i = 0; i < H; ++i) {
        simd_w2_accum(z2, them_act[i], w2_b + static_cast<size_t>(H + i) * static_cast<size_t>(H2), H2);
    }

    simd_clip_relu_sq(h2, z2, reluCap, H2);

    const float* w3_b = w3.data() + off3;
    const float* skip_b = wSkip.data() + off_skip;
    return b3[static_cast<size_t>(bucket)]
         + simd_dot(h2, w3_b, H2)
         + simd_dot(us_act, skip_b, H)
         + simd_dot(them_act, skip_b + H, H);
}

float NnueNetwork::forward_head_v3(const float* us_act, const float* them_act, float* z2, float* h2,
                                   float* duel_adv, float* duel_field, int bucket) const {
    const int H = hidden;
    const int H2 = hidden2;

    push_clash_combine(us_act, them_act, duel_adv, duel_field, H);

    const size_t off2 = static_cast<size_t>(bucket) * static_cast<size_t>(2 * H * H2);
    const size_t off_b2 = static_cast<size_t>(bucket) * static_cast<size_t>(H2);
    const size_t off3 = static_cast<size_t>(bucket) * static_cast<size_t>(H2);
    const size_t off_skip = static_cast<size_t>(bucket) * static_cast<size_t>(2 * H);

    std::memcpy(z2, b2.data() + off_b2, static_cast<size_t>(H2) * sizeof(float));
    const float* w2_b = w2.data() + off2;

    for (int i = 0; i < H; ++i) {
        simd_w2_accum(z2, duel_adv[i], w2_b + static_cast<size_t>(i) * static_cast<size_t>(H2), H2);
    }
    for (int i = 0; i < H; ++i) {
        simd_w2_accum(z2, duel_field[i], w2_b + static_cast<size_t>(H + i) * static_cast<size_t>(H2), H2);
    }

    simd_clip_relu_sq(h2, z2, reluCap, H2);

    const float* w3_b = w3.data() + off3;
    const float* skip_b = wSkip.data() + off_skip;
    return b3[static_cast<size_t>(bucket)]
         + simd_dot(h2, w3_b, H2)
         + simd_dot(duel_adv, skip_b, H)
         + simd_dot(duel_field, skip_b + H, H);
}

int NnueNetwork::cp_from_pred(float pred) const {
    const float clamped = std::clamp(pred, -1.0f, 1.0f);
    const int idx = static_cast<int>((clamped + 1.0f) * 2047.5f + 0.5f);
    return static_cast<int>(std::lround(cpLut[static_cast<size_t>(idx)]));
}

int NnueNetwork::evaluate(const Position& pos) const {
    if (!loaded()) return 0;

    const Color me = pos.side_to_move;
    const Color opp = (me == WHITE) ? BLACK : WHITE;
    const int my_pieces = std::popcount(pos.bitboards[me]);
    const int opp_pieces = std::popcount(pos.bitboards[opp]);
    const int my_reds = pos.captured_red[me];
    const int opp_reds = pos.captured_red[opp];

    constexpr int INF = 30000;
    if (opp_pieces == 0 && my_pieces > 0) return INF - 200;
    if (my_pieces == 0 && opp_pieces > 0) return -INF + 200;
    if (my_reds >= 7) return INF;
    if (opp_reds >= 7) return -INF;

    EvalScratch& scratch = thread_scratch();
    const float* us_act = nullptr;
    const float* them_act = nullptr;
    float z3 = 0.0f;

    if (networkVersion >= 3) {
        const int material_bucket = material_bucket_v3(pos);
        const int output_bucket = output_bucket_v3(pos);
        const int base = material_bucket * FEATURES_PER_MATERIAL_V3;

        accumulate_perspective_v3(pos, WHITE, base, scratch.act_w);
        accumulate_perspective_v3(pos, BLACK, base, scratch.act_b);
        us_act = (me == WHITE) ? scratch.act_w : scratch.act_b;
        them_act = (me == WHITE) ? scratch.act_b : scratch.act_w;
        z3 = forward_head_v3(us_act, them_act, scratch.z2, scratch.h2,
                             scratch.duel_adv, scratch.duel_field, output_bucket);
    } else {
        const int material_bucket = material_bucket_v1(pos);
        const int output_bucket = output_bucket_v1(pos);
        const int base = material_bucket * FEATURES_PER_MATERIAL_V1;

        accumulate_perspective_v1(pos, WHITE, base, scratch.act_w);
        accumulate_perspective_v1(pos, BLACK, base, scratch.act_b);
        us_act = (me == WHITE) ? scratch.act_w : scratch.act_b;
        them_act = (me == WHITE) ? scratch.act_b : scratch.act_w;
        z3 = forward_head_v1(us_act, them_act, scratch.z2, scratch.h2, output_bucket);
    }

    const float pred = fast_tanh(z3);
    return cp_from_pred(pred);
}
