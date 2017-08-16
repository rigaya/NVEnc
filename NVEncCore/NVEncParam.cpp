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

VppDeband::VppDeband() :
    enable(false),
    range(FILTER_DEFAULT_DEBAND_RANGE),
    threY(FILTER_DEFAULT_DEBAND_THRE_Y),
    threCb(FILTER_DEFAULT_DEBAND_THRE_CB),
    threCr(FILTER_DEFAULT_DEBAND_THRE_CR),
    ditherY(FILTER_DEFAULT_DEBAND_DITHER_Y),
    ditherC(FILTER_DEFAULT_DEBAND_DITHER_C),
    sample(FILTER_DEFAULT_DEBAND_MODE),
    seed(FILTER_DEFAULT_DEBAND_SEED),
    blurFirst(FILTER_DEFAULT_DEBAND_BLUR_FIRST),
    randEachFrame(FILTER_DEFAULT_DEBAND_RAND_EACH_FRAME) {

}


VppParam::VppParam() :
    bCheckPerformance(false),
    deinterlace(cudaVideoDeinterlaceMode_Weave),
    resizeInterp(NPPI_INTER_UNDEFINED),
    gaussMaskSize((NppiMaskSize)0),
    unsharp(),
    delogo(),
    knn(),
    pmd(),
    deband(),
    afs() {
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

VppAfs::VppAfs() :
    enable(false),
    tb_order(FILTER_DEFAULT_AFS_TB_ORDER),
    clip(scan_clip(FILTER_DEFAULT_AFS_CLIP_TB, FILTER_DEFAULT_AFS_CLIP_TB, FILTER_DEFAULT_AFS_CLIP_LR, FILTER_DEFAULT_AFS_CLIP_LR)),
    method_switch(FILTER_DEFAULT_AFS_METHOD_SWITCH),
    coeff_shift(FILTER_DEFAULT_AFS_COEFF_SHIFT),
    thre_shift(FILTER_DEFAULT_AFS_THRE_SHIFT),
    thre_deint(FILTER_DEFAULT_AFS_THRE_DEINT),
    thre_Ymotion(FILTER_DEFAULT_AFS_THRE_YMOTION),
    thre_Cmotion(FILTER_DEFAULT_AFS_THRE_CMOTION),
    analyze(FILTER_DEFAULT_AFS_ANALYZE),
    shift(FILTER_DEFAULT_AFS_SHIFT),
    drop(FILTER_DEFAULT_AFS_DROP),
    smooth(FILTER_DEFAULT_AFS_SMOOTH),
    force24(FILTER_DEFAULT_AFS_FORCE24),
    tune(FILTER_DEFAULT_AFS_TUNE) {
    check();
}

VppAfs::VppAfs(bool _enable,
    int _tb_order,
    AFS_SCAN_CLIP _clip,
    int _method_watershed,
    int _coeff_shift,
    int _thre_shift,
    int _thre_deint,
    int _thre_Ymotion,
    int _thre_Cmotion,
    int _analyze,
    bool _shift,
    bool _drop,
    bool _smooth,
    bool _force24,
    bool _tune) :
    enable(_enable),
    tb_order(_tb_order),
    clip(_clip),
    method_switch(_method_watershed),
    coeff_shift(_coeff_shift),
    thre_shift(_thre_shift),
    thre_deint(_thre_deint),
    thre_Ymotion(_thre_Ymotion),
    thre_Cmotion(_thre_Cmotion),
    analyze(_analyze),
    shift(_shift),
    drop(_drop),
    smooth(_smooth),
    force24(_force24),
    tune(_tune) {
    check();
}

void VppAfs::check() {
    if (!shift) {
        method_switch = 0;
        coeff_shift = 0;
    }
    drop &= shift;
    smooth &= drop;
}

