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

typedef float4 LUTVEC;

#ifndef clamp
#define clamp(x, low, high) (((x) <= (high)) ? (((x) >= (low)) ? (x) : (low)) : (high))
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

const float MP_REF_WHITE = 203.0f;
const float MP_REF_WHITE_HLG = 3.17955f;

// Common constants for SMPTE ST.2084 (HDR)
const float PQ_M1 = 2610.0f / 4096.0f * 1.0f / 4.0f;
const float PQ_M2 = 2523.0f / 4096.0f * 128.0f;
const float PQ_C1 = 3424.0f / 4096.0f;
const float PQ_C2 = 2413.0f / 4096.0f * 32.0f;
const float PQ_C3 = 2392.0f / 4096.0f * 32.0f;

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

//pq_space_to_linear
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

//linear_to_pq_space
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

COLORSPACE_FUNC float hdr2sdr_hable(float x, float source_peak, float ldr_nits, float A, float B, float C, float D, float E, float F) {
    const float eb = source_peak / ldr_nits;
    const float t0 = hable(x, A, B, C, D, E, F);
    const float t1 = hable(eb, A, B, C, D, E, F);
    return t0 / t1;
}

COLORSPACE_FUNC float hdr2sdr_mobius(float x, float source_peak, float ldr_nits, float t, float peak) {
    const float eb = source_peak / ldr_nits;
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
    return x / (x + offset) * (peak + offset) / peak;
}

COLORSPACE_FUNC float linear_to_pq_space(float x) {
    if (x > 0.0f) {
        x *= MP_REF_WHITE / 10000.0f;
        x = powf(x, PQ_M1);
        x = (PQ_C1 + PQ_C2 * x) / (1.0f + PQ_C3 * x);
        x = powf(x, PQ_M2);
        return x;
    } else {
        return 0.0f;
    }
}

COLORSPACE_FUNC float pq_space_to_linear(float x) {
    if (x > 0.0f) {
        x = powf(x, 1.0f / PQ_M2);
        x = fmaxf(x - PQ_C1, 0.0f) / (PQ_C2 - PQ_C3 * x);
        x = powf(x, 1.0f / PQ_M1);
        x *= 10000.0f / MP_REF_WHITE;
        return x;
    } else {
        return 0.0f;
    }
}

COLORSPACE_FUNC float apply_bt2390(float x, const float maxLum) {
    const float ks = 1.5f * maxLum - 0.5f;
    float tb = (x - ks) / (1.0f - ks);
    float tb2 = tb * tb;
    float tb3 = tb2 * tb;
    float pb = (2.0f * tb3 - 3.0f * tb2 + 1.0f) * ks +
        (tb3 - 2.0f * tb2 + tb) * (1.0f - ks) +
        (-2.0f * tb3 + 3.0f * tb2) * maxLum;
    //x = mix(pb, x, lessThan(x, ks));
    x = (x < ks) ? x : pb;
    return x;
}

COLORSPACE_FUNC float mix(float x, float y, float a) {
    a = (a < 0.0f) ? 0.0f : a;
    a = (a > 1.0f) ? 1.0f : a;
    return (x) * (1.0f - (a)) + (y) * (a);
}

COLORSPACE_FUNC float lut3d_linear_interp(float v0, float v1, float a) {
    return v0 + (v1 - v0) * a;
}

COLORSPACE_FUNC float3 lut3d_linear_interp(float3 v0, float3 v1, float a) {
    float3 r;
    r.x = lut3d_linear_interp(v0.x, v1.x, a);
    r.y = lut3d_linear_interp(v0.y, v1.y, a);
    r.z = lut3d_linear_interp(v0.z, v1.z, a);
    return r;
}

COLORSPACE_FUNC int lut3d_prev_idx(float x) {
    return (int)x;
}

COLORSPACE_FUNC int lut3d_near_idx(float x) {
    return (int)(x + 0.5f);
}

COLORSPACE_FUNC int lut3d_next_idx(float x, int size) {
    int next = lut3d_prev_idx(x) + 1;
    return (next >= size) ? size - 1 : next;
}

COLORSPACE_FUNC float lut3d_prelut(const float s, const int idx, const int size,
    const float prelutmin[3], const float prelutscale[3], const float *__restrict__ prelut) {
    const float x = clamp((s - prelutmin[idx]) * prelutscale[idx], 0.0f, (float)(size - 1));
    const float c0 = prelut[idx * size + lut3d_prev_idx(x)];
    const float c1 = prelut[idx * size + lut3d_next_idx(x, size)];
    return lut3d_linear_interp(c0, c1, x - lut3d_prev_idx(x));
}

COLORSPACE_FUNC float3 lut3d_prelut(const float3 in, const int size,
    const float prelutmin[3], const float prelutscale[3], const float *__restrict__ prelut) {
    float3 out;
    out.x = lut3d_prelut(in.x, 0, size, prelutmin, prelutscale, prelut);
    out.y = lut3d_prelut(in.y, 1, size, prelutmin, prelutscale, prelut);
    out.z = lut3d_prelut(in.z, 2, size, prelutmin, prelutscale, prelut);
    return out;
}

COLORSPACE_FUNC float3 lut3d_get_table(const LUTVEC *__restrict__ lut, const int x, const int y, const int z, const int lutSize0, const int lutSize01) {
    LUTVEC val = lut[x * lutSize01 + y * lutSize0 + z];
    float3 out;
    out.x = val.x;
    out.y = val.y;
    out.z = val.z;
    return out;
}

COLORSPACE_FUNC float3 lut3d_interp_nearest(float3 in, const LUTVEC *__restrict__ lut, const int lutSize0, const int lutSize01) {
    return lut3d_get_table(lut, lut3d_near_idx(in.x), lut3d_near_idx(in.y), lut3d_near_idx(in.z), lutSize0, lutSize01);
}

//参考: https://en.wikipedia.org/wiki/Trilinear_interpolation
COLORSPACE_FUNC float3 lut3d_interp_trilinear(float3 in, const LUTVEC *__restrict__ lut, const int lutSize0, const int lutSize01) {
    const int x0 = lut3d_prev_idx(in.x);
    const int x1 = lut3d_next_idx(in.x, lutSize0);
    const int y0 = lut3d_prev_idx(in.y);
    const int y1 = lut3d_next_idx(in.y, lutSize0);
    const int z0 = lut3d_prev_idx(in.z);
    const int z1 = lut3d_next_idx(in.z, lutSize0);
    const float scalex = in.x - x0;
    const float scaley = in.y - y0;
    const float scalez = in.z - z0;
    const float3 c000  = lut3d_get_table(lut, x0, y0, z0, lutSize0, lutSize01);
    const float3 c001  = lut3d_get_table(lut, x0, y0, z1, lutSize0, lutSize01);
    const float3 c010  = lut3d_get_table(lut, x0, y1, z0, lutSize0, lutSize01);
    const float3 c011  = lut3d_get_table(lut, x0, y1, z1, lutSize0, lutSize01);
    const float3 c100  = lut3d_get_table(lut, x1, y0, z0, lutSize0, lutSize01);
    const float3 c101  = lut3d_get_table(lut, x1, y0, z1, lutSize0, lutSize01);
    const float3 c110  = lut3d_get_table(lut, x1, y1, z0, lutSize0, lutSize01);
    const float3 c111  = lut3d_get_table(lut, x1, y1, z1, lutSize0, lutSize01);
    const float3 c00   = lut3d_linear_interp(c000, c100, scalex);
    const float3 c10   = lut3d_linear_interp(c010, c110, scalex);
    const float3 c01   = lut3d_linear_interp(c001, c101, scalex);
    const float3 c11   = lut3d_linear_interp(c011, c111, scalex);
    const float3 c0    = lut3d_linear_interp(c00,  c10,  scaley);
    const float3 c1    = lut3d_linear_interp(c01,  c11,  scaley);
    const float3 c     = lut3d_linear_interp(c0,   c1,   scalez);
    return c;
}

//参考: http://www.filmlight.ltd.uk/pdf/whitepapers/FL-TL-TN-0057-SoftwareLib.pdf
COLORSPACE_FUNC float3 lut3d_interp_tetrahedral(float3 in, const LUTVEC *__restrict__ lut, const int lutSize0, const int lutSize01) {
    const int x0 = lut3d_prev_idx(in.x);
    const int x1 = lut3d_next_idx(in.x, lutSize0);
    const int y0 = lut3d_prev_idx(in.y);
    const int y1 = lut3d_next_idx(in.y, lutSize0);
    const int z0 = lut3d_prev_idx(in.z);
    const int z1 = lut3d_next_idx(in.z, lutSize0);
    const float scalex = in.x - x0;
    const float scaley = in.y - y0;
    const float scalez = in.z - z0;
    float scale0, scale1, scale2;
    int xA, yA, zA, xB, yB, zB;
    if (scalex > scaley) {
        if (scaley > scalez) {
            scale0 = scalex;
            scale1 = scaley;
            scale2 = scalez;
            xA = x1; yA = y0; zA = z0;
            xB = x1; yB = y1; zB = z0;
        } else if (scalex > scalez) {
            scale0 = scalex;
            scale1 = scalez;
            scale2 = scaley;
            xA = x1; yA = y0; zA = z0;
            xB = x1; yB = y0; zB = z1;
        } else {
            scale0 = scalez;
            scale1 = scalex;
            scale2 = scaley;
            xA = x0; yA = y0; zA = z1;
            xB = x1; yB = y0; zB = z1;
        }
    } else {
        if (scalez > scaley) {
            scale0 = scalez;
            scale1 = scaley;
            scale2 = scalex;
            xA = x0; yA = y0; zA = z1;
            xB = x0; yB = y1; zB = z1;
        } else if (scalez > scalex) {
            scale0 = scaley;
            scale1 = scalez;
            scale2 = scalex;
            xA = x0; yA = y1; zA = z0;
            xB = x0; yB = y1; zB = z1;
        } else {
            scale0 = scaley;
            scale1 = scalez;
            scale2 = scalex;
            xA = x0; yA = y1; zA = z0;
            xB = x1; yB = y1; zB = z0;
        }
    }
    const float3 c000 = lut3d_get_table(lut, x0, y0, z0, lutSize0, lutSize01);
    const float3 c111 = lut3d_get_table(lut, x1, y1, z1, lutSize0, lutSize01);
    const float3 cA   = lut3d_get_table(lut, xA, yA, zA, lutSize0, lutSize01);
    const float3 cB   = lut3d_get_table(lut, xB, yB, zB, lutSize0, lutSize01);
    const float  s0   = 1.0f   - scale0;
    const float  s1   = scale0 - scale1;
    const float  s2   = scale1 - scale2;
    const float  s3   = scale2;
    float3 c;
    c.x = s0 * c000.x + s1 * cA.x + s2 * cB.x + s3 * c111.x;
    c.y = s0 * c000.y + s1 * cA.y + s2 * cB.y + s3 * c111.y;
    c.z = s0 * c000.z + s1 * cA.z + s2 * cB.z + s3 * c111.z;
    return c;
}

struct RGYColorspaceDevParams {
    int lut_offset;
    int prelut_offset;
    // 以降はoffsetの示す位置にデータ
    // データの取り出し方法は下記の関数を参照
};

float *getDevParamsPrelut(void *__restrict__ ptr) {
    return (float *)((char *)ptr + ((RGYColorspaceDevParams *)ptr)->prelut_offset);
}

const float *getDevParamsPrelut(const void *__restrict__ ptr) {
    return (const float *)((const char *)ptr + ((RGYColorspaceDevParams *)ptr)->prelut_offset);
}

LUTVEC *getDevParamsLut(void *__restrict__ ptr) {
    return (LUTVEC *)((char *)ptr + ((RGYColorspaceDevParams *)ptr)->lut_offset);
}

const LUTVEC *getDevParamsLut(const void *__restrict__ ptr) {
    return (const LUTVEC *)((const char *)ptr + ((RGYColorspaceDevParams *)ptr)->lut_offset);
}
