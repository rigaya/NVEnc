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

#include "rgy_filter_nnedi_field.h"

#include <algorithm>

namespace {

int rgy_nnedi_floor_div4(int x) {
    return (x >= 0) ? (x >> 2) : -(((-x) + 3) >> 2);
}

} // namespace

const std::array<int, 7> RGY_NNEDI_XDIA = { 8, 16, 32, 48, 8, 16, 32 };
const std::array<int, 7> RGY_NNEDI_YDIA = { 6,  6,  6,  6, 4,  4,  4 };
const std::array<int, 5> RGY_NNEDI_NNS  = { 16, 32, 64, 128, 256 };

RGYNnediFieldParam::RGYNnediFieldParam() :
    field(RGY_NNEDI_DEFAULT_FIELD),
    nsize(VPP_NNEDI_NSIZE_16x6),
    nns(32),
    quality(RGY_NNEDI_DEFAULT_QUALITY),
    prescreen(RGY_NNEDI_DEFAULT_PRESCREEN),
    errortype(RGY_NNEDI_DEFAULT_ERRORTYPE),
    processPlane({ true, true, true, false }) {
    clamp = RGY_NNEDI_DEFAULT_CLAMP;
}

RGY_ERR rgy_nnedi_validate_field_param(const RGYNnediFieldParam& param) {
    if ((int)param.field < -2 || (int)param.field > 3) {
        return RGY_ERR_INVALID_PARAM;
    }
    if ((int)param.nsize < 0 || (int)param.nsize >= (int)RGY_NNEDI_XDIA.size()) {
        return RGY_ERR_INVALID_PARAM;
    }
    if (std::find(RGY_NNEDI_NNS.begin(), RGY_NNEDI_NNS.end(), param.nns) == RGY_NNEDI_NNS.end()) {
        return RGY_ERR_INVALID_PARAM;
    }
    if (param.quality != VPP_NNEDI_QUALITY_FAST && param.quality != VPP_NNEDI_QUALITY_SLOW) {
        return RGY_ERR_INVALID_PARAM;
    }
    if (param.prescreen != RGY_NNEDI_DEFAULT_PRESCREEN
        || param.errortype != RGY_NNEDI_DEFAULT_ERRORTYPE
        || param.clamp != RGY_NNEDI_DEFAULT_CLAMP) {
        return RGY_ERR_UNSUPPORTED;
    }
    if (!param.processPlane[0] && !param.processPlane[1] && !param.processPlane[2] && !param.processPlane[3]) {
        return RGY_ERR_INVALID_PARAM;
    }
    return RGY_ERR_NONE;
}

RGYNnediNetworkShape rgy_nnedi_network_shape(int nsize, int nns) {
    if (nsize < 0 || nsize >= (int)RGY_NNEDI_XDIA.size()) {
        nsize = 0;
    }
    if (nns < 0 || nns >= (int)RGY_NNEDI_NNS.size()) {
        nns = 0;
    }
    RGYNnediNetworkShape shape;
    shape.xdia = RGY_NNEDI_XDIA[nsize];
    shape.ydia = RGY_NNEDI_YDIA[nsize];
    shape.neurons = RGY_NNEDI_NNS[nns];
    return shape;
}

RGY_ERR rgy_nnedi_resolve_topology(RGYNnediTopology *topology, int field, bool inputTff) {
    if (topology == nullptr || field < -2 || field > 3) {
        return RGY_ERR_INVALID_PARAM;
    }

    int resolved = field;
    if (resolved == -2) {
        resolved = inputTff ? 3 : 2;
    } else if (resolved == -1) {
        resolved = inputTff ? 1 : 0;
    }

    topology->field = resolved;
    topology->doubleRate = resolved > 1;
    topology->frameMultiplier = topology->doubleRate ? 2 : 1;
    topology->fpsMultiplier = topology->doubleRate ? 2 : 1;
    return RGY_ERR_NONE;
}

RGY_ERR rgy_nnedi_map_output_frame(RGYNnediFrameMap *frameMap, const RGYNnediTopology& topology, int outputFrame) {
    if (frameMap == nullptr || topology.field < 0 || topology.field > 3 || outputFrame < 0) {
        return RGY_ERR_INVALID_PARAM;
    }

    int fn = topology.field;
    if (topology.doubleRate) {
        if (outputFrame & 1) {
            fn = topology.field == 3 ? 0 : 1;
        } else {
            fn = topology.field == 3 ? 1 : 0;
        }
    }

    if (fn < 0 || fn > 1) {
        return RGY_ERR_INVALID_PARAM;
    }

    frameMap->sourceFrame = topology.doubleRate ? (outputFrame >> 1) : outputFrame;
    frameMap->generateField = (fn == 0) ? RGYNnediField::Top : RGYNnediField::Bottom;
    frameMap->copyField = (fn == 0) ? RGYNnediField::Bottom : RGYNnediField::Top;
    frameMap->sourceFieldOffset = (int)frameMap->copyField;
    frameMap->evalRefOffsetY = -frameMap->sourceFieldOffset;
    frameMap->doubleRate = topology.doubleRate;
    return RGY_ERR_NONE;
}

RGYNnediPlanePadding rgy_nnedi_plane_padding(int srcWidth, int srcHeight, int xsub, int ysub) {
    xsub = std::max(0, xsub);
    ysub = std::max(0, ysub);

    RGYNnediPlanePadding padding;
    padding.width = std::max(0, srcWidth >> xsub);
    padding.height = std::max(0, (srcHeight >> 1) >> ysub);
    // mirrors +/-32 pixels and +/-3 field lines in
    // the current plane. The reference frame base is intentionally offset by
    // twice that luma padding before the kernel writes negative indices.
    padding.hpad = RGY_NNEDI_HPAD;
    padding.vpad = RGY_NNEDI_VPAD;
    padding.refBaseOffsetX = (RGY_NNEDI_HPAD * 2) >> xsub;
    padding.refBaseOffsetY = (RGY_NNEDI_VPAD * 2) >> ysub;
    return padding;
}

RGYNnediMirrorIndex rgy_nnedi_mirror_index(int pos, int length) {
    RGYNnediMirrorIndex index;
    index.index = pos;
    index.padded = true;

    if (length <= 0) {
        index.index = 0;
    } else if (pos < 0) {
        index.index = -pos - 1;
    } else if (pos >= length) {
        index.index = length - (pos - length) - 1;
    } else {
        index.padded = false;
    }
    return index;
}

RGYNnediVec4MirrorIndex rgy_nnedi_mirror_index4(int x4, int width4) {
    const auto mirror = rgy_nnedi_mirror_index(x4, width4);
    RGYNnediVec4MirrorIndex index4;
    index4.index4 = mirror.index;
    // pads uchar4/ushort4 vectors. When a vector is mirrored at either
    // horizontal edge it swaps x<->w and y<->z, so scalar probes must reverse
    // lanes as well as mirror the vector index.
    index4.reverseLanes = mirror.padded;
    index4.padded = mirror.padded;
    return index4;
}

int rgy_nnedi_mirror_lane(int lane, bool reverseLanes) {
    lane &= 3;
    return reverseLanes ? (3 - lane) : lane;
}

RGYNnediMirrorPixelIndex rgy_nnedi_mirror_pixel_index(int x, int width) {
    const int x4 = rgy_nnedi_floor_div4(x);
    const int width4 = width >> 2;
    const auto mirror4 = rgy_nnedi_mirror_index4(x4, width4);
    const int lane = rgy_nnedi_mirror_lane(x - (x4 << 2), mirror4.reverseLanes);

    RGYNnediMirrorPixelIndex index;
    index.index4 = mirror4.index4;
    index.lane = lane;
    index.index = (width4 <= 0) ? 0 : (mirror4.index4 << 2) + lane;
    index.reverseLanes = mirror4.reverseLanes;
    index.padded = mirror4.padded;
    return index;
}

bool rgy_nnedi_is_copied_field(int y, RGYNnediField copyField) {
    return (y & 1) == (int)copyField;
}

bool rgy_nnedi_is_generated_field(int y, RGYNnediField generateField) {
    return (y & 1) == (int)generateField;
}
