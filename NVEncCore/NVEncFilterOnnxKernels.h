#pragma once

#include "NVEncFilter.h"

struct OnnxColorCoeffs {
    float yoff, yscale, yrange, coff, cscale, crange;
    float vr, ug, vg, ub;
    float ry, gy, by, ru, gu, bu, rv, gv, bv;
};

RGY_ERR run_onnx_pack(float *dst, const RGYFrameInfo *src, int tensorW, int tensorH,
    int padL, int padT, bool ycbcr, const OnnxColorCoeffs& coeffs, cudaStream_t stream);
RGY_ERR run_onnx_unpack(RGYFrameInfo *dst, const float *src, bool ycbcr,
    const OnnxColorCoeffs& coeffs, cudaStream_t stream);
RGY_ERR run_onnx_pack_luma_padded(float *dst, const RGYFrameInfo *src, int width, int height,
    int padL, int padT, cudaStream_t stream);
RGY_ERR run_onnx_fill(float *dst, size_t count, float value, cudaStream_t stream);
RGY_ERR run_onnx_pad_planes(float *dst, const float *src, int sw, int sh, int dw, int dh,
    int padL, int padT, cudaStream_t stream);
RGY_ERR run_onnx_mask_resize(const float *src, float *dst, int sw, int sh, int dw, int dh, cudaStream_t stream);
RGY_ERR run_onnx_mask_detect(const float *src, int count, int *large, cudaStream_t stream);
RGY_ERR run_onnx_mask_composite(const float *frame, const float *model, const float *mask,
    const int *large, float *dst, int fw, int fh, int mw, int mh, cudaStream_t stream);
