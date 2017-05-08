// -----------------------------------------------------------------------------------------
// QSVEnc by rigaya
// -----------------------------------------------------------------------------------------
//
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

#include "rgy_err.h"

#if RGY_ERR_QSV
struct RGYErrMapMFX {
    RGY_ERR rgy;
    mfxStatus mfx;
};

#define MFX_MAP(x) { RGY_ ##x, MFX_ ##x }
static const RGYErrMapMFX ERR_MAP_MFX[] = {
    MFX_MAP(ERR_NONE),
    MFX_MAP(ERR_UNKNOWN),
    MFX_MAP(ERR_NULL_PTR),
    MFX_MAP(ERR_UNSUPPORTED),
    MFX_MAP(ERR_MEMORY_ALLOC),
    MFX_MAP(ERR_NOT_ENOUGH_BUFFER),
    MFX_MAP(ERR_INVALID_HANDLE),
    MFX_MAP(ERR_LOCK_MEMORY),
    MFX_MAP(ERR_NOT_INITIALIZED),
    MFX_MAP(ERR_NOT_FOUND),
    MFX_MAP(ERR_MORE_DATA),
    MFX_MAP(ERR_MORE_SURFACE),
    MFX_MAP(ERR_ABORTED),
    MFX_MAP(ERR_DEVICE_LOST),
    MFX_MAP(ERR_INCOMPATIBLE_VIDEO_PARAM),
    MFX_MAP(ERR_INVALID_VIDEO_PARAM),
    MFX_MAP(ERR_UNDEFINED_BEHAVIOR),
    MFX_MAP(ERR_DEVICE_FAILED),
    MFX_MAP(ERR_MORE_BITSTREAM),
    MFX_MAP(ERR_INCOMPATIBLE_AUDIO_PARAM),
    MFX_MAP(ERR_INVALID_AUDIO_PARAM),
    MFX_MAP(ERR_GPU_HANG),
    MFX_MAP(ERR_REALLOC_SURFACE),

    MFX_MAP(WRN_IN_EXECUTION),
    MFX_MAP(WRN_DEVICE_BUSY),
    MFX_MAP(WRN_VIDEO_PARAM_CHANGED),
    MFX_MAP(WRN_PARTIAL_ACCELERATION),
    MFX_MAP(WRN_INCOMPATIBLE_VIDEO_PARAM),
    MFX_MAP(WRN_VALUE_NOT_CHANGED),
    MFX_MAP(WRN_OUT_OF_RANGE),
    MFX_MAP(WRN_FILTER_SKIPPED),
    MFX_MAP(WRN_INCOMPATIBLE_AUDIO_PARAM),

    MFX_MAP(PRINT_OPTION_DONE),
    MFX_MAP(PRINT_OPTION_ERR),

    MFX_MAP(ERR_INVALID_COLOR_FORMAT),

    MFX_MAP(ERR_MORE_DATA_SUBMIT_TASK),
};
#undef MFX_MAP

mfxStatus err_to_mfx(RGY_ERR err) {
    const RGYErrMapMFX *ERR_MAP_FIN = (const RGYErrMapMFX *)ERR_MAP_MFX + _countof(ERR_MAP_MFX);
    auto ret = std::find_if((const RGYErrMapMFX *)ERR_MAP_MFX, ERR_MAP_FIN, [err](RGYErrMapMFX map) {
        return map.rgy == err;
    });
    return (ret == ERR_MAP_FIN) ? MFX_ERR_UNKNOWN : ret->mfx;
}

RGY_ERR err_to_rgy(mfxStatus err) {
    const RGYErrMapMFX *ERR_MAP_FIN = (const RGYErrMapMFX *)ERR_MAP_MFX + _countof(ERR_MAP_MFX);
    auto ret = std::find_if((const RGYErrMapMFX *)ERR_MAP_MFX, ERR_MAP_FIN, [err](RGYErrMapMFX map) {
        return map.mfx == err;
    });
    return (ret == ERR_MAP_FIN) ? RGY_ERR_UNKNOWN : ret->rgy;
}
#endif //#if RGY_ERR_QSV

#if RGY_ERR_AMF
struct RGYErrMapAMF {
    RGYErr rgy;
    AMF_ERR amf;
};

static const RGYErrMapAMF ERR_MAP_AMF[] = {
    { RGY_ERR_NONE, AMF_OK },
    { RGY_ERR_UNKNOWN, AMF_FAIL },
    { RGY_ERR_UNDEFINED_BEHAVIOR, AMF_UNEXPECTED },
    { RGY_ERR_ACCESS_DENIED, AMF_ACCESS_DENIED },
    { RGY_ERR_INVALID_PARAM, AMF_INVALID_ARG },
    { RGY_ERR_OUT_OF_RANGE, AMF_OUT_OF_RANGE },
    { RGY_ERR_NULL_PTR, AMF_INVALID_POINTER },
    { RGY_ERR_NULL_PTR, AMF_OUT_OF_MEMORY },
    { RGY_ERR_UNSUPPORTED, AMF_NO_INTERFACE },
    { RGY_ERR_UNSUPPORTED, AMF_NOT_IMPLEMENTED },
    { RGY_ERR_UNSUPPORTED, AMF_NOT_SUPPORTED },
    { RGY_ERR_NOT_FOUND, AMF_NOT_FOUND },
    { RGY_ERR_ALREADY_INITIALIZED, AMF_ALREADY_INITIALIZED },
    { RGY_ERR_NOT_INITIALIZED, AMF_NOT_INITIALIZED },
    { RGY_ERR_INVALID_FORMAT, AMF_INVALID_FORMAT },
    { RGY_ERR_WRONG_STATE, AMF_WRONG_STATE },
    { RGY_ERR_FILE_OPEN, AMF_FILE_NOT_OPEN },
    { RGY_ERR_NO_DEVICE, AMF_NO_DEVICE },
    { RGY_ERR_DEVICE_FAILED, AMF_DIRECTX_FAILED },
    { RGY_ERR_DEVICE_FAILED, AMF_OPENCL_FAILED },
    { RGY_ERR_DEVICE_FAILED, AMF_GLX_FAILED },
    { RGY_ERR_DEVICE_FAILED, AMF_ALSA_FAILED },
    { RGY_ERR_MORE_DATA, AMF_EOF },
    { RGY_ERR_UNKNOWN, AMF_REPEAT },
    { RGY_ERR_INPUT_FULL, AMF_INPUT_FULL },
    { RGY_WRN_VIDEO_PARAM_CHANGED, AMF_RESOLUTION_CHANGED },
    { RGY_WRN_VIDEO_PARAM_CHANGED, AMF_RESOLUTION_UPDATED },
    { RGY_ERR_INVALID_DATA_TYPE, AMF_INVALID_DATA_TYPE },
    { RGY_ERR_INVALID_RESOLUTION, AMF_INVALID_RESOLUTION },
    { RGY_ERR_INVALID_CODEC, AMF_CODEC_NOT_SUPPORTED },
    { RGY_ERR_INVALID_COLOR_FORMAT, AMF_SURFACE_FORMAT_NOT_SUPPORTED },
    { RGY_ERR_DEVICE_FAILED, AMF_SURFACE_MUST_BE_SHARED }
};

AMF_ERR err_to_amf(RGYErr err) {
    const RGYErrMapAMF *ERR_MAP_FIN = (const RGYErrMapAMF *)ERR_MAP_AMF + _countof(ERR_MAP_AMF);
    auto ret = std::find_if((const RGYErrMapAMF *)ERR_MAP_AMF, ERR_MAP_FIN, [err](const RGYErrMapAMF map) {
        return map.rgy == err;
    });
    return (ret == ERR_MAP_FIN) ? MFX_ERR_UNKNOWN : ret->amf;
}

RGYErr err_to_rgy(AMF_ERR err) {
    const RGYErrMapAMF *ERR_MAP_FIN = (const RGYErrMapAMF *)ERR_MAP_AMF + _countof(ERR_MAP_AMF);
    auto ret = std::find_if((const RGYErrMapAMF *)ERR_MAP_AMF, ERR_MAP_FIN, [err](const RGYErrMapAMF map) {
        return map.amf == err;
    });
    return (ret == ERR_MAP_FIN) ? RGY_ERR_UNKNOWN : ret->rgy;
}
#endif //#if RGY_ERR_AMF


#if RGY_ERR_NV
struct RGYErrMapNV {
    RGY_ERR rgy;
    NVENCSTATUS nv;
};

static const RGYErrMapNV ERR_MAP_NV[] = {
    { RGY_ERR_NONE, NV_ENC_SUCCESS },
    { RGY_ERR_UNKNOWN, NV_ENC_ERR_GENERIC },
    { RGY_ERR_ACCESS_DENIED, NV_ENC_ERR_INCOMPATIBLE_CLIENT_KEY },
    { RGY_ERR_INVALID_PARAM, NV_ENC_ERR_INVALID_EVENT },
    { RGY_ERR_INVALID_PARAM, NV_ENC_ERR_INVALID_PARAM },
    { RGY_ERR_INVALID_PARAM, NV_ENC_ERR_UNSUPPORTED_PARAM },
    { RGY_ERR_NOT_ENOUGH_BUFFER, NV_ENC_ERR_NOT_ENOUGH_BUFFER },
    { RGY_ERR_NULL_PTR, NV_ENC_ERR_INVALID_PTR },
    { RGY_ERR_NULL_PTR, NV_ENC_ERR_OUT_OF_MEMORY },
    { RGY_ERR_UNSUPPORTED, NV_ENC_ERR_UNIMPLEMENTED },
    { RGY_ERR_UNSUPPORTED, NV_ENC_ERR_UNSUPPORTED_DEVICE },
    { RGY_ERR_UNSUPPORTED, NV_ENC_ERR_INVALID_CALL },
    { RGY_WRN_DEVICE_BUSY, NV_ENC_ERR_LOCK_BUSY },
    { RGY_WRN_DEVICE_BUSY, NV_ENC_ERR_ENCODER_BUSY },
    { RGY_ERR_NO_DEVICE, NV_ENC_ERR_NO_ENCODE_DEVICE },
    { RGY_ERR_NOT_INITIALIZED, NV_ENC_ERR_ENCODER_NOT_INITIALIZED },
    { RGY_ERR_INVALID_VERSION, NV_ENC_ERR_INVALID_VERSION },
    { RGY_ERR_INVALID_DEVICE, NV_ENC_ERR_INVALID_ENCODERDEVICE },
    { RGY_ERR_INVALID_DEVICE, NV_ENC_ERR_INVALID_DEVICE },
    { RGY_ERR_NO_DEVICE, NV_ENC_ERR_DEVICE_NOT_EXIST },
    { RGY_ERR_MORE_DATA, NV_ENC_ERR_NEED_MORE_INPUT },
    { RGY_ERR_MAP_FAILED, NV_ENC_ERR_MAP_FAILED, }
};

NVENCSTATUS err_to_nv(RGY_ERR err) {
    const RGYErrMapNV *ERR_MAP_FIN = (const RGYErrMapNV *)ERR_MAP_NV + _countof(ERR_MAP_NV);
    auto ret = std::find_if((const RGYErrMapNV *)ERR_MAP_NV, ERR_MAP_FIN, [err](const RGYErrMapNV map) {
        return map.rgy == err;
    });
    return (ret == ERR_MAP_FIN) ? NV_ENC_ERR_GENERIC : ret->nv;
}

RGY_ERR err_to_rgy(NVENCSTATUS err) {
    const RGYErrMapNV *ERR_MAP_FIN = (const RGYErrMapNV *)ERR_MAP_NV + _countof(ERR_MAP_NV);
    auto ret = std::find_if((const RGYErrMapNV *)ERR_MAP_NV, ERR_MAP_FIN, [err](const RGYErrMapNV map) {
        return map.nv == err;
    });
    return (ret == ERR_MAP_FIN) ? RGY_ERR_UNKNOWN : ret->rgy;
}

#endif //#if RGY_ERR_NV

const TCHAR *get_err_mes(RGY_ERR sts) {
    switch (sts) {
    case RGY_ERR_NONE:                     return _T("no error.");
    case RGY_ERR_UNKNOWN:                  return _T("unknown error.");
    case RGY_ERR_NULL_PTR:                 return _T("null pointer.");
    case RGY_ERR_UNSUPPORTED:              return _T("undeveloped feature.");
    case RGY_ERR_MEMORY_ALLOC:             return _T("failed to allocate memory.");
    case RGY_ERR_NOT_ENOUGH_BUFFER:        return _T("insufficient buffer at input/output.");
    case RGY_ERR_INVALID_HANDLE:           return _T("invalid handle.");
    case RGY_ERR_LOCK_MEMORY:              return _T("failed to lock the memory block.");
    case RGY_ERR_NOT_INITIALIZED:          return _T("member function called before initialization.");
    case RGY_ERR_NOT_FOUND:                return _T("the specified object is not found.");
    case RGY_ERR_MORE_DATA:                return _T("expect more data at input.");
    case RGY_ERR_MORE_SURFACE:             return _T("expect more surface at output.");
    case RGY_ERR_ABORTED:                  return _T("operation aborted.");
    case RGY_ERR_DEVICE_LOST:              return _T("lose the HW acceleration device.");
    case RGY_ERR_INCOMPATIBLE_VIDEO_PARAM: return _T("incompatible video parameters.");
    case RGY_ERR_INVALID_VIDEO_PARAM:      return _T("invalid video parameters.");
    case RGY_ERR_UNDEFINED_BEHAVIOR:       return _T("undefined behavior.");
    case RGY_ERR_DEVICE_FAILED:            return _T("device operation failure.");
    case RGY_ERR_GPU_HANG:                 return _T("gpu hang.");
    case RGY_ERR_REALLOC_SURFACE:          return _T("failed to realloc surface.");
    case RGY_ERR_ACCESS_DENIED:            return _T("access denied");
    case RGY_ERR_INVALID_PARAM:            return _T("invalid param.");
    case RGY_ERR_OUT_OF_RANGE:             return _T("out pf range.");
    case RGY_ERR_ALREADY_INITIALIZED:      return _T("already initialized.");
    case RGY_ERR_INVALID_FORMAT:           return _T("invalid format.");
    case RGY_ERR_WRONG_STATE:              return _T("wrong state.");
    case RGY_ERR_FILE_OPEN:                return _T("file open error.");
    case RGY_ERR_INPUT_FULL:               return _T("input full.");
    case RGY_ERR_INVALID_CODEC:            return _T("invalid codec.");
    case RGY_ERR_INVALID_DATA_TYPE:        return _T("invalid data type.");
    case RGY_ERR_INVALID_RESOLUTION:       return _T("invaldi resolution.");
    case RGY_ERR_INVALID_DEVICE:           return _T("invalid devices.");
    case RGY_ERR_INVALID_CALL:             return _T("invalid call sequence.");
    case RGY_ERR_NO_DEVICE:                return _T("no deivce found.");
    case RGY_ERR_INVALID_VERSION:          return _T("invalid version.");
    case RGY_ERR_MAP_FAILED:               return _T("map failed.");
    case RGY_WRN_IN_EXECUTION:             return _T("the previous asynchrous operation is in execution.");
    case RGY_WRN_DEVICE_BUSY:              return _T("the HW acceleration device is busy.");
    case RGY_WRN_VIDEO_PARAM_CHANGED:      return _T("the video parameters are changed during decoding.");
    case RGY_WRN_PARTIAL_ACCELERATION:     return _T("SW is used.");
    case RGY_WRN_INCOMPATIBLE_VIDEO_PARAM: return _T("incompatible video parameters.");
    case RGY_WRN_VALUE_NOT_CHANGED:        return _T("the value is saturated based on its valid range.");
    case RGY_WRN_OUT_OF_RANGE:             return _T("the value is out of valid range.");
    default:                               return _T("unknown error.");
    }
}
