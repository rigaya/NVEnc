// -----------------------------------------------------------------------------------------
// NVEnc by rigaya
// -----------------------------------------------------------------------------------------
// The MIT License
//
// Copyright (c) 2014-2016 rigaya
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

#include "NVEncParam.h"

using std::vector;

VppKnn::VppKnn() :
    enable(false),
    radius(FILTER_DEFAULT_KNN_RADIUS),
    strength(FILTER_DEFAULT_KNN_STRENGTH),
    lerpC(FILTER_DEFAULT_KNN_LERPC),
    weight_threshold(FILTER_DEFAULT_KNN_WEIGHT_THRESHOLD),
    lerp_threshold(FILTER_DEFAULT_KNN_LERPC_THRESHOLD) {
}

VppPmd::VppPmd() :
    enable(false),
    strength(FILTER_DEFAULT_PMD_STRENGTH),
    threshold(FILTER_DEFAULT_PMD_THRESHOLD),
    applyCount(FILTER_DEFAULT_PMD_APPLY_COUNT),
    useExp(FILTER_DEFAULT_PMD_USE_EXP) {

}

VppParam::VppParam() :
    bCheckPerformance(false),
    deinterlace(cudaVideoDeinterlaceMode_Weave),
    resizeInterp(NPPI_INTER_UNDEFINED),
    gaussMaskSize((NppiMaskSize)0),
    unsharp(),
    delogo(),
    knn(),
    pmd() {
    unsharp.bEnable = false;
    delogo.pFilePath = nullptr;
    delogo.pSelect = nullptr;
    delogo.nPosOffsetX = 0;
    delogo.nPosOffsetY = 0;
    delogo.nDepth = FILTER_DEFAULT_DELOGO_DEPTH;
    delogo.nYOffset = 0;
    delogo.nCbOffset = 0;
    delogo.nCrOffset = 0;
    delogo.nMode = DELOGO_MODE_REMOVE;
}
