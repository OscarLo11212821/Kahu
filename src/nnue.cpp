#include "nnue.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <istream>
#include <streambuf>

#if defined(__EMSCRIPTEN__)
#elif defined(__ARM_NEON) || defined(__aarch64__)
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
    return pos.hash;
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

// V4: numerically stable sigmoid for the gated clash skip.
inline float sigmoid_f(float x) {
    if (x >= 0.0f) {
        const float e = std::exp(-x);
        return 1.0f / (1.0f + e);
    }
    const float e = std::exp(x);
    return e / (1.0f + e);
}

// V4: RMSNorm with per-bucket FiLM: y = gamma * (x / rms) + beta.
// Matches nnue_trainer.cpp rmsNormForward (double sum, eps 1e-5).
constexpr float kRmsNormEps = 1e-5f;
inline void rms_norm_forward(const float* x, const float* gamma, const float* beta,
                             float* y, int n) {
    double sum_sq = 0.0;
    for (int i = 0; i < n; ++i) sum_sq += static_cast<double>(x[i]) * x[i];
    const float rms = std::sqrt(static_cast<float>(sum_sq / n) + kRmsNormEps);
    const float inv = 1.0f / rms;
    for (int i = 0; i < n; ++i) y[i] = gamma[i] * (x[i] * inv) + beta[i];
}

// V4: explicit line-conveyor dense features (THEORY Sec.4). 28 dims: 7 ranks +
// 7 files x us/them (STM-relative). Value 0/1/1.5 (none/piece/red payload),
// max if several conveyors share a line. Same push-from-square / falling-piece
// / idx rules as the trainer scan; bitboard rank/file tables (cf.
// Position::push_off_threats) instead of per-square walks.
constexpr int kLineFeats = 28;
inline void compute_line_conveyors(const Position& pos, int stm, float* out28) {
    for (int i = 0; i < kLineFeats; ++i) out28[i] = 0.0f;
    const uint64_t occ = pos.occupied;
    const uint64_t red = pos.bitboards[RED];
    uint8_t rank[7], file[7];
    for (int r = 0; r < 7; r++)
        rank[r] = static_cast<uint8_t>((occ >> (7 * r)) & 127u);
    for (int f = 0; f < 7; f++) {
        uint8_t c = 0;
        uint64_t bit = 1ULL << f;
        for (int r = 0; r < 7; r++, bit <<= 7)
            if (occ & bit) c = static_cast<uint8_t>(c | (1u << r));
        file[f] = c;
    }

    for (int side = 0; side < 2; ++side) {
        const int us_them = (side == stm) ? 0 : 1;
        const uint64_t mine = pos.bitboards[side];
        uint64_t my = mine;
        while (my) {
            const int sq = std::countr_zero(my);
            my &= my - 1;
            const int r = sq / 7;
            const int f = sq % 7;
            const uint8_t row = rank[r];
            const uint8_t col = file[f];

            // NORTH push: conveyor line is file f.
            if ((r == 6 || !(col & (1u << (r + 1)))) && r > 0 && (col & (1u << (r - 1)))) {
                int last_r = r - 1;
                while (last_r > 0 && (col & (1u << (last_r - 1)))) last_r--;
                if (last_r == 0) {
                    const uint64_t m = 1ULL << f;
                    float v = 0.0f;
                    if (red & m) v = 1.5f;
                    else if (!(mine & m)) v = 1.0f;
                    const int idx = 14 + f * 2 + us_them;
                    if (v > out28[idx]) out28[idx] = v;
                }
            }
            // SOUTH push: conveyor line is file f.
            if ((r == 0 || !(col & (1u << (r - 1)))) && r < 6 && (col & (1u << (r + 1)))) {
                int last_r = r + 1;
                while (last_r < 6 && (col & (1u << (last_r + 1)))) last_r++;
                if (last_r == 6) {
                    const uint64_t m = 1ULL << (6 * 7 + f);
                    float v = 0.0f;
                    if (red & m) v = 1.5f;
                    else if (!(mine & m)) v = 1.0f;
                    const int idx = 14 + f * 2 + us_them;
                    if (v > out28[idx]) out28[idx] = v;
                }
            }
            // EAST push: conveyor line is rank r.
            if ((f == 0 || !(row & (1u << (f - 1)))) && f < 6 && (row & (1u << (f + 1)))) {
                int last_f = f + 1;
                while (last_f < 6 && (row & (1u << (last_f + 1)))) last_f++;
                if (last_f == 6) {
                    const uint64_t m = 1ULL << (r * 7 + last_f);
                    float v = 0.0f;
                    if (red & m) v = 1.5f;
                    else if (!(mine & m)) v = 1.0f;
                    const int idx = r * 2 + us_them;
                    if (v > out28[idx]) out28[idx] = v;
                }
            }
            // WEST push: conveyor line is rank r.
            if ((f == 6 || !(row & (1u << (f + 1)))) && f > 0 && (row & (1u << (f - 1)))) {
                int last_f = f - 1;
                while (last_f > 0 && (row & (1u << (last_f - 1)))) last_f--;
                if (last_f == 0) {
                    const uint64_t m = 1ULL << (r * 7);
                    float v = 0.0f;
                    if (red & m) v = 1.5f;
                    else if (!(mine & m)) v = 1.0f;
                    const int idx = r * 2 + us_them;
                    if (v > out28[idx]) out28[idx] = v;
                }
            }
        }
    }
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
        int k = 0;
        for (; k + 8 <= half; k += 8) {
            const int16x8_t r = vld1q_s16(row + k);
            const int32x4_t lo = vmovl_s16(vget_low_s16(r));
            const int32x4_t hi = vmovl_s16(vget_high_s16(r));
            vst1q_s32(acc + k, vsubq_s32(vld1q_s32(acc + k), lo));
            vst1q_s32(acc + k + 4, vsubq_s32(vld1q_s32(acc + k + 4), hi));
        }
        for (; k < half; ++k) acc[k] -= row[k];
        for (; k + 8 <= n; k += 8) {
            const int16x8_t r = vld1q_s16(row + k);
            const int32x4_t lo = vmovl_s16(vget_low_s16(r));
            const int32x4_t hi = vmovl_s16(vget_high_s16(r));
            vst1q_s32(acc + k, vaddq_s32(vld1q_s32(acc + k), lo));
            vst1q_s32(acc + k + 4, vaddq_s32(vld1q_s32(acc + k + 4), hi));
        }
        for (; k < n; ++k) acc[k] += row[k];
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

// Blocked variant for h2 == 32: z[32] += sum_i x[i] * W[i][32] with the whole
// z tile held in 8 NEON registers (one load + one store total). Exact zeros
// are still skipped, so sparse inputs stay cheap while dense inputs avoid
// re-streaming z once per row (the V4 head's bottleneck at ~35% density).
inline void simd_w2_blocked32(float* z, const float* x, int n, const float* w) {
    float32x4_t r0 = vld1q_f32(z), r1 = vld1q_f32(z + 4);
    float32x4_t r2 = vld1q_f32(z + 8), r3 = vld1q_f32(z + 12);
    float32x4_t r4 = vld1q_f32(z + 16), r5 = vld1q_f32(z + 20);
    float32x4_t r6 = vld1q_f32(z + 24), r7 = vld1q_f32(z + 28);
    for (int i = 0; i < n; ++i) {
        const float xi = x[i];
        if (xi == 0.0f) continue;
        const float32x4_t vx = vdupq_n_f32(xi);
        const float* row = w + static_cast<size_t>(i) * 32;
        r0 = vmlaq_f32(r0, vx, vld1q_f32(row));
        r1 = vmlaq_f32(r1, vx, vld1q_f32(row + 4));
        r2 = vmlaq_f32(r2, vx, vld1q_f32(row + 8));
        r3 = vmlaq_f32(r3, vx, vld1q_f32(row + 12));
        r4 = vmlaq_f32(r4, vx, vld1q_f32(row + 16));
        r5 = vmlaq_f32(r5, vx, vld1q_f32(row + 20));
        r6 = vmlaq_f32(r6, vx, vld1q_f32(row + 24));
        r7 = vmlaq_f32(r7, vx, vld1q_f32(row + 28));
    }
    vst1q_f32(z, r0); vst1q_f32(z + 4, r1);
    vst1q_f32(z + 8, r2); vst1q_f32(z + 12, r3);
    vst1q_f32(z + 16, r4); vst1q_f32(z + 20, r5);
    vst1q_f32(z + 24, r6); vst1q_f32(z + 28, r7);
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
        int k = 0;
        for (; k + 8 <= half; k += 8) {
            const __m128i r16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(row + k));
            const __m256i r32 = _mm256_cvtepi16_epi32(r16);
            __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc + k));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc + k), _mm256_sub_epi32(a, r32));
        }
        for (; k < half; ++k) acc[k] -= row[k];
        for (; k + 8 <= n; k += 8) {
            const __m128i r16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(row + k));
            const __m256i r32 = _mm256_cvtepi16_epi32(r16);
            __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc + k));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc + k), _mm256_add_epi32(a, r32));
        }
        for (; k < n; ++k) acc[k] += row[k];
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

// Blocked variant for h2 == 32: whole z tile in 4 AVX registers. See NEON note.
inline void simd_w2_blocked32(float* z, const float* x, int n, const float* w) {
    __m256 r0 = _mm256_loadu_ps(z), r1 = _mm256_loadu_ps(z + 8);
    __m256 r2 = _mm256_loadu_ps(z + 16), r3 = _mm256_loadu_ps(z + 24);
    for (int i = 0; i < n; ++i) {
        const float xi = x[i];
        if (xi == 0.0f) continue;
        const __m256 vx = _mm256_set1_ps(xi);
        const float* row = w + static_cast<size_t>(i) * 32;
        r0 = _mm256_fmadd_ps(vx, _mm256_loadu_ps(row), r0);
        r1 = _mm256_fmadd_ps(vx, _mm256_loadu_ps(row + 8), r1);
        r2 = _mm256_fmadd_ps(vx, _mm256_loadu_ps(row + 16), r2);
        r3 = _mm256_fmadd_ps(vx, _mm256_loadu_ps(row + 24), r3);
    }
    _mm256_storeu_ps(z, r0); _mm256_storeu_ps(z + 8, r1);
    _mm256_storeu_ps(z + 16, r2); _mm256_storeu_ps(z + 24, r3);
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
// Scalar fallback of the blocked h2 == 32 accumulator (same zero-skip).
inline void simd_w2_blocked32(float* z, const float* x, int n, const float* w) {
    for (int i = 0; i < n; ++i) {
        const float xi = x[i];
        if (xi == 0.0f) continue;
        const float* row = w + static_cast<size_t>(i) * 32;
        for (int j = 0; j < 32; ++j) z[j] += xi * row[j];
    }
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
    // V4 residual head buffers + line-conveyor features.
    float z2a[MAX_HIDDEN2];
    float h2a_pre[MAX_HIDDEN2];
    float h2a[MAX_HIDDEN2];
    float z2b[MAX_HIDDEN2];
    float res[MAX_HIDDEN2];
    float h2b_pre[MAX_HIDDEN2];
    float h2b[MAX_HIDDEN2];
    float line_feats[kLineFeats];
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

bool nnue_load_memory(const void* data, size_t size) {
    if (!data || size == 0) return false;
    struct MemBuf : std::streambuf {
        MemBuf(const char* p, size_t n) {
            auto* s = const_cast<char*>(p);
            setg(s, s, s + n);
        }
    };
    MemBuf buf(static_cast<const char*>(data), size);
    std::istream in(&buf);
    return g_network.load_from_stream(in, "memory");
}

const NnueNetwork* nnue_network() {
    return g_network.loaded() ? &g_network : nullptr;
}

int nnue_evaluate(const Position& pos) {
    if (!g_network.loaded()) return 0;

    const uint64_t key = eval_key(pos);
    const size_t slot = key & (EVAL_CACHE_SIZE - 1);
    EvalCacheEntry& e = g_eval_cache[slot];
    if (e.key == key) return e.score;

    const int score = g_network.evaluate(pos);
    e.key = key;
    e.score = score;
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
    return load_from_stream(in, path.c_str());
}

bool NnueNetwork::load_from_stream(std::istream& in, const char* name) {

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
    if (std::memcmp(h.magic, "KUBA_NNUE_V4\0", 13) == 0) {
        expectedVersion = 4;
        expectedFeaturesPerMaterial = FEATURES_PER_MATERIAL_V3;
        expectedFeatureCount = FEATURE_COUNT_V3;
    } else if (std::memcmp(h.magic, "KUBA_NNUE_V3\0", 13) == 0) {
        expectedVersion = 3;
        expectedFeaturesPerMaterial = FEATURES_PER_MATERIAL_V3;
        expectedFeatureCount = FEATURE_COUNT_V3;
    } else if (std::memcmp(h.magic, "KUBA_NNUE_V1\0", 13) == 0) {
        expectedVersion = 1;
        expectedFeaturesPerMaterial = FEATURES_PER_MATERIAL_V1;
        expectedFeatureCount = MATERIAL_BUCKETS * FEATURES_PER_MATERIAL_V1;
    } else {
        std::cerr << "nnue: bad magic in " << name << " (expected KUBA_NNUE_V1/V3/V4)\n";
        return false;
    }
    if (h.version != static_cast<std::uint32_t>(expectedVersion)
        || h.features_per_material != expectedFeaturesPerMaterial
        || h.feature_count != expectedFeatureCount) {
        std::cerr << "nnue: incompatible feature layout in " << name << '\n';
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
    // V4 extra tensors (w2=w2a, b2=b2a, w3=w3b above).
    const size_t w2b_n = static_cast<size_t>(OUTPUT_BUCKETS * hidden2) * static_cast<size_t>(hidden2);
    const size_t wline_n = static_cast<size_t>(OUTPUT_BUCKETS * LINE_FEATS) * static_cast<size_t>(hidden2);

    std::vector<float> w1f(w1_n);
    w2.resize(w2_n);
    b2.resize(b2_n);
    w3.resize(w3_n);
    b3.resize(b3_n);
    wSkip.resize(skip_n);
    if (expectedVersion >= 4) {
        lnG1.resize(b2_n);
        lnB1.resize(b2_n);
        w2b.resize(w2b_n);
        b2b.resize(b2_n);
        lnG2.resize(b2_n);
        lnB2.resize(b2_n);
        w3a.resize(w3_n);
        gateLogit.resize(b3_n);
        wLine.resize(wline_n);
    } else {
        // Clear V4 tensors so a reload of an older net cannot reuse stale V4 weights.
        lnG1.clear(); lnB1.clear();
        w2b.clear(); b2b.clear();
        lnG2.clear(); lnB2.clear();
        w3a.clear(); gateLogit.clear(); wLine.clear();
    }

    auto must_read = [&](void* dst, size_t bytes, const char* what) -> bool {
        if (!in.read(reinterpret_cast<char*>(dst), static_cast<std::streamsize>(bytes))) {
            std::cerr << "nnue: truncated " << what << " in " << name << '\n';
            return false;
        }
        return true;
    };

    if (!must_read(w1f.data(), w1_n * sizeof(float), "w1")) return false;
    if (expectedVersion >= 4) {
        // Order must match Model::save in nnue_trainer.cpp.
        if (!must_read(w2.data(), w2_n * sizeof(float), "w2a")) return false;
        if (!must_read(b2.data(), b2_n * sizeof(float), "b2a")) return false;
        if (!must_read(lnG1.data(), b2_n * sizeof(float), "lnG1")) return false;
        if (!must_read(lnB1.data(), b2_n * sizeof(float), "lnB1")) return false;
        if (!must_read(w2b.data(), w2b_n * sizeof(float), "w2b")) return false;
        if (!must_read(b2b.data(), b2_n * sizeof(float), "b2b")) return false;
        if (!must_read(lnG2.data(), b2_n * sizeof(float), "lnG2")) return false;
        if (!must_read(lnB2.data(), b2_n * sizeof(float), "lnB2")) return false;
        if (!must_read(w3.data(), w3_n * sizeof(float), "w3b")) return false;
        if (!must_read(w3a.data(), w3_n * sizeof(float), "w3a")) return false;
        if (!must_read(b3.data(), b3_n * sizeof(float), "b3")) return false;
        if (!must_read(wSkip.data(), skip_n * sizeof(float), "wSkip")) return false;
        if (!must_read(gateLogit.data(), b3_n * sizeof(float), "gateLogit")) return false;
        if (!must_read(wLine.data(), wline_n * sizeof(float), "wLine")) return false;
    } else {
        if (!must_read(w2.data(), w2_n * sizeof(float), "w2")) return false;
        if (!must_read(b2.data(), b2_n * sizeof(float), "b2")) return false;
        if (!must_read(w3.data(), w3_n * sizeof(float), "w3")) return false;
        if (!must_read(b3.data(), b3_n * sizeof(float), "b3")) return false;
        if (!must_read(wSkip.data(), skip_n * sizeof(float), "wSkip")) return false;
    }

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
    std::cerr << "nnue: loaded " << name
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
                                              float* act, int threat_delta, int mob_delta) const {
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

    auto add_bb = [&](uint64_t bb, int rel) {
        while (bb) {
            const int sq = std::countr_zero(bb);
            bb &= bb - 1ULL;
            const int sq_rel = push_column_square(perspective, sq);
            add_feat(base_material + sq_rel * 3 + rel);
        }
    };

    add_bb(pos.bitboards[perspective], REL_OWN);
    add_bb(pos.bitboards[perspective ^ 1], REL_OPP);
    add_bb(pos.bitboards[RED], REL_RED);

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
         + simd_dot(duel_adv, skip_b, H)
         + simd_dot(duel_field, skip_b + H, H);
}

// V4 deep residual head. Mirrors Model::forwardHead in nnue_trainer.cpp
// (returns raw z3; the caller applies fast_tanh). Here w2=w2a, b2=b2a, w3=w3b.
float NnueNetwork::forward_head_v4(const float* us_act, const float* them_act, const float* line_feats,
                                   float* z2a, float* h2a_pre, float* h2a,
                                   float* z2b, float* res, float* h2b_pre, float* h2b,
                                   float* duel_adv, float* duel_field, int bucket) const {
    const int H = hidden;
    const int H2 = hidden2;

    push_clash_combine(us_act, them_act, duel_adv, duel_field, H);

    const size_t off2a = static_cast<size_t>(bucket) * static_cast<size_t>(2 * H * H2);
    const size_t off_h2 = static_cast<size_t>(bucket) * static_cast<size_t>(H2);
    const size_t off2b = static_cast<size_t>(bucket) * static_cast<size_t>(H2 * H2);
    const size_t off_skip = static_cast<size_t>(bucket) * static_cast<size_t>(2 * H);
    const size_t off_line = static_cast<size_t>(bucket) * static_cast<size_t>(LINE_FEATS * H2);

    std::memcpy(z2a, b2.data() + off_h2, static_cast<size_t>(H2) * sizeof(float));
    const float* w2a_b = w2.data() + off2a;

    if (H2 == 32) {
        // Register-blocked: keeps the z2a tile in SIMD registers across rows.
        simd_w2_blocked32(z2a, us_act, H, w2a_b);
        simd_w2_blocked32(z2a, them_act, H, w2a_b + static_cast<size_t>(H) * 32);
    } else {
        for (int i = 0; i < H; ++i) {
            simd_w2_accum(z2a, us_act[i], w2a_b + static_cast<size_t>(i) * static_cast<size_t>(H2), H2);
        }
        for (int i = 0; i < H; ++i) {
            simd_w2_accum(z2a, them_act[i], w2a_b + static_cast<size_t>(H + i) * static_cast<size_t>(H2), H2);
        }
    }
    // Dense conveyor bypass (sparse: skip zeros; typical 0-4 active of 28).
    if (line_feats) {
        const float* wline_b = wLine.data() + off_line;
        for (int k = 0; k < LINE_FEATS; ++k) {
            const float x = line_feats[k];
            if (x != 0.0f) simd_w2_accum(z2a, x, wline_b + static_cast<size_t>(k) * static_cast<size_t>(H2), H2);
        }
    }

    rms_norm_forward(z2a, lnG1.data() + off_h2, lnB1.data() + off_h2, h2a_pre, H2);
    simd_clip_relu_sq(h2a, h2a_pre, reluCap, H2);

    std::memcpy(z2b, b2b.data() + off_h2, static_cast<size_t>(H2) * sizeof(float));
    const float* w2b_b = w2b.data() + off2b;
    if (H2 == 32) {
        simd_w2_blocked32(z2b, h2a, H2, w2b_b);
    } else {
        for (int i = 0; i < H2; ++i) {
            simd_w2_accum(z2b, h2a[i], w2b_b + static_cast<size_t>(i) * static_cast<size_t>(H2), H2);
        }
    }
    for (int j = 0; j < H2; ++j) res[j] = z2b[j] + h2a[j];

    rms_norm_forward(res, lnG2.data() + off_h2, lnB2.data() + off_h2, h2b_pre, H2);
    simd_clip_relu_sq(h2b, h2b_pre, reluCap, H2);

    const float* w3b_b = w3.data() + off_h2;
    const float* w3a_b = w3a.data() + off_h2;
    const float* skip_b = wSkip.data() + off_skip;
    const float gate = sigmoid_f(gateLogit[static_cast<size_t>(bucket)]);
    const float skip = simd_dot(duel_adv, skip_b, H) + simd_dot(duel_field, skip_b + H, H);
    return b3[static_cast<size_t>(bucket)]
         + simd_dot(h2b, w3b_b, H2)
         + simd_dot(h2a, w3a_b, H2)
         + gate * skip;
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

    if (networkVersion >= 4) {
        const int material_bucket = material_bucket_v3(pos);
        const int output_bucket = output_bucket_v3(pos);
        const int base = material_bucket * FEATURES_PER_MATERIAL_V3;
        const int tW = pos.push_off_threats(WHITE);
        const int tB = pos.push_off_threats(BLACK);
        const int mW = pos.push_mobility(WHITE);
        const int mB = pos.push_mobility(BLACK);

        accumulate_perspective_v3(pos, WHITE, base, scratch.act_w, tW - tB, mW - mB);
        accumulate_perspective_v3(pos, BLACK, base, scratch.act_b, tB - tW, mB - mW);
        us_act = (me == WHITE) ? scratch.act_w : scratch.act_b;
        them_act = (me == WHITE) ? scratch.act_b : scratch.act_w;
        compute_line_conveyors(pos, static_cast<int>(me), scratch.line_feats);
        z3 = forward_head_v4(us_act, them_act, scratch.line_feats,
                             scratch.z2a, scratch.h2a_pre, scratch.h2a,
                             scratch.z2b, scratch.res, scratch.h2b_pre, scratch.h2b,
                             scratch.duel_adv, scratch.duel_field, output_bucket);
    } else if (networkVersion >= 3) {
        const int material_bucket = material_bucket_v3(pos);
        const int output_bucket = output_bucket_v3(pos);
        const int base = material_bucket * FEATURES_PER_MATERIAL_V3;
        const int tW = pos.push_off_threats(WHITE);
        const int tB = pos.push_off_threats(BLACK);
        const int mW = pos.push_mobility(WHITE);
        const int mB = pos.push_mobility(BLACK);

        accumulate_perspective_v3(pos, WHITE, base, scratch.act_w, tW - tB, mW - mB);
        accumulate_perspective_v3(pos, BLACK, base, scratch.act_b, tB - tW, mB - mW);
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
