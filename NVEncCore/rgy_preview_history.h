#pragma once
#include <memory>
#include "rgy_prm.h"
#include "convert_csp.h"

#if ENCODER_NVENC
#include "NVEncFilter.h"
#include "NVEncFilterColorspace.h"
#include "NVEncFilterCurves.h"
#include "NVEncFilterDehalo.h"
#include "NVEncFilterDenoiseDct.h"
#include "NVEncFilterDenoiseKnn.h"
#include "NVEncFilterDenoiseNLMeans.h"
#include "NVEncFilterDenoisePmd.h"
#include "NVEncFilterEdgelevel.h"
#include "NVEncFilterFineDehalo.h"
#include "NVEncFilterHQDering.h"
#include "NVEncFilterNnedi.h"
#include "NVEncFilterNnediUpscale.h"
#include "NVEncFilterTransform.h"
#include "NVEncFilterTweak.h"
#include "NVEncFilterUnsharp.h"
#include "NVEncFilterWarpsharp.h"
using RGYPreviewHistoryFilter = NVEncFilter;
#define RGY_PREVIEW_HISTORY_FILTER(nvclass, clclass) std::make_unique<nvclass>()
#else
#include "rgy_filter_cl.h"
#include "rgy_filter_colorspace.h"
#include "rgy_filter_curves.h"
#include "rgy_filter_dehalo.h"
#include "rgy_filter_denoise_dct.h"
#include "rgy_filter_denoise_knn.h"
#include "rgy_filter_denoise_nlmeans.h"
#include "rgy_filter_denoise_pmd.h"
#include "rgy_filter_edgelevel.h"
#include "rgy_filter_finedehalo.h"
#include "rgy_filter_hqdering.h"
#include "rgy_filter_nnedi.h"
#include "rgy_filter_nnedi_upscale.h"
#include "rgy_filter_resize.h"
#include "rgy_filter_transform.h"
#include "rgy_filter_tweak.h"
#include "rgy_filter_unsharp.h"
#include "rgy_filter_warpsharp.h"
using RGYPreviewHistoryFilter = RGYFilter;
#define RGY_PREVIEW_HISTORY_FILTER(nvclass, clclass) std::make_unique<clclass>(nullptr)
#endif

// 対応表にない処理は保守的に履歴依存とし、判定は必ず実フィルタへ委譲する。
inline bool rgy_preview_filter_depends_on_history(const VppType type, const RGYParamVpp& vpp) {
    std::unique_ptr<RGYPreviewHistoryFilter> filter;
    switch (type) {
    case VppType::CL_CROP: filter = RGY_PREVIEW_HISTORY_FILTER(NVEncFilterCspCrop, RGYFilterCspCrop); break;
    case VppType::CL_PAD: filter = RGY_PREVIEW_HISTORY_FILTER(NVEncFilterPad, RGYFilterPad); break;
    case VppType::CL_COLORSPACE: filter = RGY_PREVIEW_HISTORY_FILTER(NVEncFilterColorspace, RGYFilterColorspace); break;
    case VppType::CL_TWEAK: filter = RGY_PREVIEW_HISTORY_FILTER(NVEncFilterTweak, RGYFilterTweak); break;
    case VppType::CL_CURVES: filter = RGY_PREVIEW_HISTORY_FILTER(NVEncFilterCurves, RGYFilterCurves); break;
    case VppType::CL_UNSHARP: filter = RGY_PREVIEW_HISTORY_FILTER(NVEncFilterUnsharp, RGYFilterUnsharp); break;
    case VppType::CL_EDGELEVEL: filter = RGY_PREVIEW_HISTORY_FILTER(NVEncFilterEdgelevel, RGYFilterEdgelevel); break;
    case VppType::CL_WARPSHARP: filter = RGY_PREVIEW_HISTORY_FILTER(NVEncFilterWarpsharp, RGYFilterWarpsharp); break;
    case VppType::CL_DENOISE_PMD: filter = RGY_PREVIEW_HISTORY_FILTER(NVEncFilterDenoisePmd, RGYFilterDenoisePmd); break;
    case VppType::CL_DENOISE_DCT: filter = RGY_PREVIEW_HISTORY_FILTER(NVEncFilterDenoiseDct, RGYFilterDenoiseDct); break;
    case VppType::CL_TRANSFORM: filter = RGY_PREVIEW_HISTORY_FILTER(NVEncFilterTransform, RGYFilterTransform); break;
    case VppType::CL_DENOISE_KNN: filter = RGY_PREVIEW_HISTORY_FILTER(NVEncFilterDenoiseKnn, RGYFilterDenoiseKnn); break;
    case VppType::CL_DENOISE_NLMEANS: filter = RGY_PREVIEW_HISTORY_FILTER(NVEncFilterDenoiseNLMeans, RGYFilterDenoiseNLMeans); break;
    case VppType::CL_NNEDI_UPSCALE: filter = RGY_PREVIEW_HISTORY_FILTER(NVEncFilterNnediUpscale, RGYFilterNnediUpscale); break;
    case VppType::CL_HQDERING: filter = RGY_PREVIEW_HISTORY_FILTER(NVEncFilterHQDering, RGYFilterDering); break;
    case VppType::CL_DEHALO: filter = RGY_PREVIEW_HISTORY_FILTER(NVEncFilterDehalo, RGYFilterDehalo); break;
    case VppType::CL_FINEDEHALO: filter = RGY_PREVIEW_HISTORY_FILTER(NVEncFilterFineDehalo, RGYFilterFineDehalo); break;
    case VppType::CL_NNEDI: filter = RGY_PREVIEW_HISTORY_FILTER(NVEncFilterNnedi, RGYFilterNnedi); break;
    case VppType::CL_RESIZE: filter = RGY_PREVIEW_HISTORY_FILTER(NVEncFilterResize, RGYFilterResize); break;
    default: return true;
    }
    return filter->dependsOnHistory(vpp);
}
#undef RGY_PREVIEW_HISTORY_FILTER

inline RGY_VPP_RESIZE_TYPE rgy_preview_resize_type(const VideoInfo& input, const RGYParamVpp& vpp) {
    if (input.dstWidth == 0 && input.dstHeight == 0) return RGY_VPP_RESIZE_TYPE_NONE;
    return getVppResizeType(vpp.resize_algo);
}

// 入力を開かないと経路を決められない処理は、暖機を省かず保守的に扱う。
template<typename Params>
inline bool rgy_preview_external_history(const Params& params) {
    return params.common.adaptResolution.first > 0 || params.common.adaptResolution.second > 0
        || params.common.AVSyncMode == RGY_AVSYNC_FORCE_CFR;
}
