#ifdef __CUDACC_RTC__
#define COLORSPACE_FUNC __device__ __inline__
#else
#define COLORSPACE_FUNC static
#pragma once
#include <cmath>
#include <cfloat>
#pragma warning (push)
#pragma warning (disable: 4819)
#include <cuda_runtime.h>
#pragma warning (pop)
#endif

//以下zimgのsrc\zimg\colorspace\gamma.cppより拝借、一部改変

const float REC709_ALPHA = 1.09929682680944f;
const float REC709_BETA = 0.018053968510807f;

const float SMPTE_240M_ALPHA = 1.111572195921731f;
const float SMPTE_240M_BETA  = 0.022821585529445f;

// Adjusted for continuity of first derivative.
const float SRGB_ALPHA = 1.055010718947587f;
const float SRGB_BETA = 0.003041282560128f;

const float ST2084_M1 = 0.1593017578125f;
const float ST2084_M2 = 78.84375f;
const float ST2084_C1 = 0.8359375f;
const float ST2084_C2 = 18.8515625f;
const float ST2084_C3 = 18.6875f;

const float ARIB_B67_A = 0.17883277f;
const float ARIB_B67_B = 0.28466892f;
const float ARIB_B67_C = 0.55991073f;

const float FLOAT_EPS = 1.175494351e-38f;


// Chosen for compatibility with higher precision REC709_ALPHA/REC709_BETA.
// See: ITU-R BT.2390-2 5.3.1
const float ST2084_OOTF_SCALE = 59.49080238715383f;

COLORSPACE_FUNC float rec_709_oetf(float x) {
    if (x < REC709_BETA)
        x = x * 4.5f;
    else
        x = REC709_ALPHA * powf(x, 0.45f) - (REC709_ALPHA - 1.0f);

    return x;
}

COLORSPACE_FUNC float rec_709_inverse_oetf(float x) {
    if (x < 4.5f * REC709_BETA)
        x = x / 4.5f;
    else
        x = powf((x + (REC709_ALPHA - 1.0f)) / REC709_ALPHA, 1.0f / 0.45f);

    return x;
}

// Ignore the BT.1886 provisions for limited contrast and assume an ideal CRT.
COLORSPACE_FUNC float rec_1886_eotf(float x) {
    return x < 0.0f ? 0.0f : powf(x, 2.4f);
}

COLORSPACE_FUNC float rec_1886_inverse_eotf(float x) {
    return x < 0.0f ? 0.0f : powf(x, 1.0f / 2.4f);
}

COLORSPACE_FUNC float ootf_1_2(float x) {
    return x < 0.0f ? x : powf(x, 1.2f);
}

COLORSPACE_FUNC float inverse_ootf_1_2(float x) {
    return x < 0.0f ? x : powf(x, 1.0f / 1.2f);
}

COLORSPACE_FUNC float ootf_st2084(float x) {
    return rec_1886_eotf(rec_709_oetf(x * ST2084_OOTF_SCALE)) / 100.0f;
}

COLORSPACE_FUNC float inverse_ootf_st2084(float x) {
    return rec_709_inverse_oetf(rec_1886_inverse_eotf(x * 100.0f)) / ST2084_OOTF_SCALE;
}

COLORSPACE_FUNC float log100_oetf(float x) {
    return x <= 0.01f ? 0.0f : 1.0f + log10f(x) * (1.0f / 2.0f);
}

COLORSPACE_FUNC float log100_inverse_oetf(float x) {
    return x <= 0.0f ? 0.01f : powf(10.0f, 2 * (x - 1.0f));
}

COLORSPACE_FUNC float log316_oetf(float x) {
    return x <= 0.00316227766f ? 0.0f : 1.0f + log10f(x) * (1.0f / 2.5f);
}

COLORSPACE_FUNC float log316_inverse_oetf(float x) {
    return x <= 0.0f ? 0.00316227766f : powf(10.0f, 2.5f * (x - 1.0f));
}

COLORSPACE_FUNC float rec_470m_oetf(float x) {
    return x < 0.0f ? 0.0f : powf(x, 2.2f);
}

COLORSPACE_FUNC float rec_470m_inverse_oetf(float x) {
    return x < 0.0f ? 0.0f : powf(x, 1.0f / 2.2f);
}

COLORSPACE_FUNC float rec_470bg_oetf(float x) {
    return x < 0.0f ? 0.0f : powf(x, 2.8f);
}

COLORSPACE_FUNC float rec_470bg_inverse_oetf(float x) {
    return x < 0.0f ? 0.0f : powf(x, 1.0f / 2.8f);
}

COLORSPACE_FUNC float smpte_240m_oetf(float x) {
    if (x < 4.0f * SMPTE_240M_BETA)
        x = x * (1.0f / 4.0f);
    else
        x = powf((x + (SMPTE_240M_ALPHA - 1.0f)) / SMPTE_240M_ALPHA, 1.0f / 0.45f);

    return x;
}

COLORSPACE_FUNC float smpte_240m_inverse_oetf(float x) {
    if (x < SMPTE_240M_BETA)
        x = x * 4.0f;
    else
        x = SMPTE_240M_ALPHA * powf(x, 0.45f) - (SMPTE_240M_ALPHA - 1.0f);

    return x;
}

COLORSPACE_FUNC float xvycc_oetf(float x) {
    return copysignf(rec_709_oetf(fabsf(x)), x);
}

float xvycc_inverse_oetf(float x) {
    return copysignf(rec_709_inverse_oetf(fabsf(x)), x);
}

COLORSPACE_FUNC float arib_b67_oetf(float x) {
    // Prevent negative pixels from yielding NAN.
    x = fmaxf(x, 0.0f);

    if (x <= (1.0f / 12.0f))
        x = sqrtf(3.0f * x);
    else
        x = ARIB_B67_A * logf(12.0f * x - ARIB_B67_B) + ARIB_B67_C;

    return x;
}

COLORSPACE_FUNC float arib_b67_inverse_oetf(float x) {
    // Prevent negative pixels expanding into positive values.
    x = fmaxf(x, 0.0f);

    if (x <= 0.5f)
        x = (x * x) * (1.0f / 3.0f);
    else
        x = (expf((x - ARIB_B67_C) / ARIB_B67_A) + ARIB_B67_B) * (1.0f / 12.0f);

    return x;
}

COLORSPACE_FUNC float srgb_eotf(float x) {
    if (x < 12.92f * SRGB_BETA)
        x *= (1.0f / 12.92f);
    else
        x = powf((x + (SRGB_ALPHA - 1.0f)) * (1.0f / SRGB_ALPHA), 2.4f);

    return x;
}

COLORSPACE_FUNC float srgb_inverse_eotf(float x) {
    if (x < SRGB_BETA)
        x = x * 12.92f;
    else
        x = SRGB_ALPHA * powf(x, 1.0f / 2.4f) - (SRGB_ALPHA - 1.0f);

    return x;
}

// Handle values in the range [0.0-1.0] such that they match a legacy CRT.
COLORSPACE_FUNC float xvycc_eotf(float x) {
    if (x < 0.0f || x > 1.0f)
        return copysignf(rec_709_inverse_oetf(fabsf(x)), x);
    else
        return copysignf(rec_1886_eotf(fabsf(x)), x);
}

COLORSPACE_FUNC float xvycc_inverse_eotf(float x) {
    if (x < 0.0f || x > 1.0f)
        return copysignf(rec_709_oetf(fabsf(x)), x);
    else
        return copysignf(rec_1886_inverse_eotf(fabsf(x)), x);
}

COLORSPACE_FUNC float st_2084_eotf(float x) {
    // Filter negative values to avoid NAN.
    if (x > 0.0f) {
        float xpow = powf(x, 1.0f / ST2084_M2);
        float num = fmaxf(xpow - ST2084_C1, 0.0f);
        float den = fmaxf(ST2084_C2 - ST2084_C3 * xpow, FLOAT_EPS);
        x = powf(num / den, 1.0f / ST2084_M1);
    } else {
        x = 0.0f;
    }

    return x;
}

COLORSPACE_FUNC float st_2084_inverse_eotf(float x) {
    // Filter negative values to avoid NAN, and also special-case 0 so that (f(g(0)) == 0).
    if (x > 0.0f) {
        float xpow = powf(x, ST2084_M1);
#if 0
        // Original formulation from SMPTE ST 2084:2014 publication.
        float num = ST2084_C1 + ST2084_C2 * xpow;
        float den = 1.0f + ST2084_C3 * xpow;
        x = powf(num / den, ST2084_M2);
#else
        // More stable arrangement that avoids some cancellation error.
        float num = (ST2084_C1 - 1.0f) + (ST2084_C2 - ST2084_C3) * xpow;
        float den = 1.0f + ST2084_C3 * xpow;
        x = powf(1.0f + num / den, ST2084_M2);
#endif
    } else {
        x = 0.0f;
    }

    return x;
}

// Applies a per-channel correction instead of the iterative method specified in Rec.2100.
COLORSPACE_FUNC float arib_b67_eotf(float x) {
    return ootf_1_2(arib_b67_inverse_oetf(x));
}

COLORSPACE_FUNC float arib_b67_inverse_eotf(float x) {
    return arib_b67_oetf(inverse_ootf_1_2(x));
}

COLORSPACE_FUNC float st_2084_oetf(float x) {
    return st_2084_inverse_eotf(ootf_st2084(x));
}

COLORSPACE_FUNC float st_2084_inverse_oetf(float x) {
    return inverse_ootf_st2084(st_2084_eotf(x));
}

COLORSPACE_FUNC float3 aribB67Ops(float3 v, float kr, float kg, float kb, float scale) {
    const float gamma = 1.2f;
    float r = v.x * scale;
    float g = v.y * scale;
    float b = v.z * scale;

    float yd = fmaxf(kr * r + kg * g + kb * b, FLOAT_EPS);
    float ys_inv = powf(yd, (1.0f - gamma) / gamma);

    v.x = arib_b67_oetf(r * ys_inv);
    v.y = arib_b67_oetf(g * ys_inv);
    v.z = arib_b67_oetf(b * ys_inv);
    return v;
}

COLORSPACE_FUNC float3 aribB67InvOps(float3 v, float kr, float kg, float kb, float scale) {
    const float gamma = 1.2f;
    float r = v.x;
    float g = v.y;
    float b = v.z;

    float ys = fmaxf(kr * r + kg * g + kb * b, FLOAT_EPS);
    ys = powf(ys, gamma - 1.0f);

    v.x = arib_b67_inverse_oetf(r * ys) * scale;
    v.y = arib_b67_inverse_oetf(g * ys) * scale;
    v.z = arib_b67_inverse_oetf(b * ys) * scale;
    return v;
}

COLORSPACE_FUNC float3 matrix_mul(float m[3][3], float3 v) {
    float3 ret;
    ret.x = m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z;
    ret.y = m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z;
    ret.z = m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z;
    return ret;
}

//参考: https://gist.github.com/4re/34ccbb95732c1bef47c3d2975ac62395
COLORSPACE_FUNC float hable(float x, float A, float B, float C, float D, float E, float F) {
    return ((x*(A*x+C*B)+D*E) / (x*(A*x+B)+D*F)) - E/F;
}

COLORSPACE_FUNC float hdr2sdr_hable(float x, float source_peak, float ldr_nits, float A, float B, float C, float D, float E, float F, float W) {
    const float eb = source_peak / ldr_nits;
    const float t0 = hable(eb * x, A, B, C, D, E, F);
    const float t1 = hable(W, A, B, C, D, E, F);
    return t0 / t1;
}

COLORSPACE_FUNC float hdr2sdr_mobius(float x, float source_peak, float ldr_nits, float t, float peak) {
    const float eb = source_peak / ldr_nits;
    x *= eb;
    peak *= eb;
    if (x <= t) {
        return x;
    }

    float a = -t * t * (peak - 1.0f) / (t * t - 2.0f * t + peak);
    float b = (t * t - 2.0f * t * peak + peak) / fmaxf(peak - 1.0f, 1e-6f);
    return (b * b + 2.0f * b * t + t * t) / (b - a) * (x + a) / (x + b);
}

COLORSPACE_FUNC float hdr2sdr_reinhard(float x, float source_peak, float ldr_nits, float offset, float peak) {
    const float eb = source_peak / ldr_nits;
    peak *= eb;
    x *= eb;
    return x / (x + offset) * (peak + offset) / peak;
}
