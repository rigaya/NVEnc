// -----------------------------------------------------------------------------------------
// QSVEnc/NVEnc/VCEEnc by rigaya
// -----------------------------------------------------------------------------------------
//
// The MIT License
//
// Copyright (c) 2026 rigaya
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//
// ------------------------------------------------------------------------------------------

#pragma once

#include <array>
#include <cstdint>

#include "rgy_err.h"
#include "rgy_prm.h"

enum class RGYNnediField : int {
    Top = 0,
    Bottom = 1,
};

enum class RGYNnediPlane : int {
    Y = 0,
    U = 1,
    V = 2,
    A = 3,
};

struct RGYNnediFieldParam {
    VppNnediField field;
    VppNnediNSize nsize;
    int nns;
    VppNnediQuality quality;
    int prescreen;
    VppNnediErrorType errortype;
    int clamp;
    std::array<bool, 4> processPlane;

    RGYNnediFieldParam();
};

struct RGYNnediNetworkShape {
    int xdia;
    int ydia;
    int neurons;
};

struct RGYNnediTopology {
    int field;
    bool doubleRate;
    int frameMultiplier;
    int fpsMultiplier;
};

struct RGYNnediFrameMap {
    int sourceFrame;
    RGYNnediField generateField;
    RGYNnediField copyField;
    int sourceFieldOffset;
    int evalRefOffsetY;
    bool doubleRate;
};

struct RGYNnediPlanePadding {
    int width;
    int height;
    int hpad;
    int vpad;
    int refBaseOffsetX;
    int refBaseOffsetY;
};

struct RGYNnediMirrorIndex {
    int index;
    bool padded;
};

struct RGYNnediVec4MirrorIndex {
    int index4;
    bool reverseLanes;
    bool padded;
};

struct RGYNnediMirrorPixelIndex {
    int index;
    int index4;
    int lane;
    bool reverseLanes;
    bool padded;
};

static constexpr int RGY_NNEDI_HPAD = 32;
static constexpr int RGY_NNEDI_VPAD = 3;
static constexpr VppNnediField RGY_NNEDI_DEFAULT_FIELD = VPP_NNEDI_FIELD_BOB;
static constexpr VppNnediQuality RGY_NNEDI_DEFAULT_QUALITY = VPP_NNEDI_QUALITY_FAST;
static constexpr int RGY_NNEDI_DEFAULT_PRESCREEN = 2;
static constexpr VppNnediErrorType RGY_NNEDI_DEFAULT_ERRORTYPE = VPP_NNEDI_ETYPE_ABS;
static constexpr int RGY_NNEDI_DEFAULT_CLAMP = 1;

extern const std::array<int, 7> RGY_NNEDI_XDIA;
extern const std::array<int, 7> RGY_NNEDI_YDIA;
extern const std::array<int, 5> RGY_NNEDI_NNS;

RGY_ERR rgy_nnedi_validate_field_param(const RGYNnediFieldParam& param);
RGYNnediNetworkShape rgy_nnedi_network_shape(int nsize, int nns);
RGY_ERR rgy_nnedi_resolve_topology(RGYNnediTopology *topology, int field, bool inputTff);
RGY_ERR rgy_nnedi_map_output_frame(RGYNnediFrameMap *frameMap, const RGYNnediTopology& topology, int outputFrame);
RGYNnediPlanePadding rgy_nnedi_plane_padding(int srcWidth, int srcHeight, int xsub, int ysub);
RGYNnediMirrorIndex rgy_nnedi_mirror_index(int pos, int length);
RGYNnediVec4MirrorIndex rgy_nnedi_mirror_index4(int x4, int width4);
int rgy_nnedi_mirror_lane(int lane, bool reverseLanes);
RGYNnediMirrorPixelIndex rgy_nnedi_mirror_pixel_index(int x, int width);
bool rgy_nnedi_is_copied_field(int y, RGYNnediField copyField);
bool rgy_nnedi_is_generated_field(int y, RGYNnediField generateField);
