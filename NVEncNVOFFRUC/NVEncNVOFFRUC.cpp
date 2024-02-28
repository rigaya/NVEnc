// -----------------------------------------------------------------------------------------
// NVEnc by rigaya
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

#include "rgy_osdep.h"
#include <memory>
#include <array>
#include "rgy_err.h"

#define NVENC_NVOFFRUC_EXPORTS
#include "NVEncNVOFFRUC.h"

#if ENABLE_NVOFFRUC_HEADER
#include "NvOFFRUC.h"

#if defined(_WIN32) || defined(_WIN64)
static const TCHAR * NVOFFRUC_MODULENAME = _T("NvOFFRUC.dll");
#else
static const TCHAR * NVOFFRUC_MODULENAME = _T("libNvOFFRUC.so");
#endif


struct RGYErrMapNVOFFRUC {
    RGY_ERR rgy;
    NvOFFRUC_STATUS nv;
};
//grep '=' cudaErrors.txt | awk '{print "NPPERR_MAP(",$1,"),"}'
#define NPPERR_MAP(x) { RGY_ERR_NvOFFRUC_ ##x, NvOFFRUC_ERR_ ##x }
static const RGYErrMapNVOFFRUC ERR_MAP_NVOFFRUC[] = {
    { RGY_ERR_NONE, NvOFFRUC_SUCCESS },
    NPPERR_MAP(NvOFFRUC_NOT_SUPPORTED),
    NPPERR_MAP(INVALID_PTR),
    NPPERR_MAP(INVALID_PARAM),
    NPPERR_MAP(INVALID_HANDLE),
    NPPERR_MAP(OUT_OF_SYSTEM_MEMORY),
    NPPERR_MAP(OUT_OF_VIDEO_MEMORY),
    NPPERR_MAP(OPENCV_NOT_AVAILABLE),
    NPPERR_MAP(UNIMPLEMENTED),
    NPPERR_MAP(OF_FAILURE),
    NPPERR_MAP(DUPLICATE_RESOURCE),
    NPPERR_MAP(UNREGISTERED_RESOURCE),
    NPPERR_MAP(INCORRECT_API_SEQUENCE),
    NPPERR_MAP(WRITE_TODISK_FAILED),
    NPPERR_MAP(PIPELINE_EXECUTION_FAILURE),
    NPPERR_MAP(SYNC_WRITE_FAILED),
    NPPERR_MAP(GENERIC)
};

static NvOFFRUC_STATUS err_to_nvoffruc(RGY_ERR err) {
    if (err == RGY_ERR_NONE) return NvOFFRUC_SUCCESS;
    const RGYErrMapNVOFFRUC *ERR_MAP_FIN = (const RGYErrMapNVOFFRUC *)ERR_MAP_NVOFFRUC + _countof(ERR_MAP_NVOFFRUC);
    auto ret = std::find_if((const RGYErrMapNVOFFRUC *)ERR_MAP_NVOFFRUC, ERR_MAP_FIN, [err](const RGYErrMapNVOFFRUC map) {
        return map.rgy == err;
        });
    return (ret == ERR_MAP_FIN) ? NvOFFRUC_ERR_GENERIC : ret->nv;
}

static RGY_ERR err_to_rgy(NvOFFRUC_STATUS err) {
    if (err == NvOFFRUC_SUCCESS) return RGY_ERR_NONE;
    const RGYErrMapNVOFFRUC *ERR_MAP_FIN = (const RGYErrMapNVOFFRUC *)ERR_MAP_NVOFFRUC + _countof(ERR_MAP_NVOFFRUC);
    auto ret = std::find_if((const RGYErrMapNVOFFRUC *)ERR_MAP_NVOFFRUC, ERR_MAP_FIN, [err](const RGYErrMapNVOFFRUC map) {
        return map.nv == err;
        });
    return (ret == ERR_MAP_FIN) ? RGY_ERR_UNKNOWN : ret->rgy;
}

struct NVOFFRUCFunc {
    HMODULE hModule;
    PtrToFuncNvOFFRUCCreate create;
    PtrToFuncNvOFFRUCRegisterResource registerResource;
    PtrToFuncNvOFFRUCUnregisterResource unregisterResource;
    PtrToFuncNvOFFRUCProcess process;
    PtrToFuncNvOFFRUCDestroy destroy;
    NVOFFRUCFunc();
    ~NVOFFRUCFunc();
    RGY_ERR load();
    void close();
};

using unique_fruc_handle = std::unique_ptr<std::remove_pointer<NvOFFRUCHandle>::type, PtrToFuncNvOFFRUCDestroy>;

class NVEncNVOFFRUC {
public:
    NVEncNVOFFRUC();
    virtual ~NVEncNVOFFRUC();
    RGY_ERR createFRUCHandle(int width, int height);
    RGY_ERR registerResource(void *ptr0, void *ptr1, void *ptr2);
    RGY_ERR process(NVEncNVOFFRUCParams *prm);
    RGY_ERR closeFRUCHandle();
    void close();
protected:
    std::unique_ptr<NVOFFRUCFunc> m_func;
    unique_fruc_handle m_frucHandle;
    std::array<void*, 3> m_resource;
    int64_t m_firstTimestamp;
};

NVOFFRUCFunc::NVOFFRUCFunc() :
    hModule(nullptr),
    create(nullptr),
    registerResource(nullptr),
    unregisterResource(nullptr),
    process(nullptr),
    destroy(nullptr) {
}

NVOFFRUCFunc::~NVOFFRUCFunc() {
    close();
}

void NVOFFRUCFunc::close() {
    if (hModule) {
        RGY_FREE_LIBRARY(hModule);
        hModule = nullptr;
    }
}

RGY_ERR NVOFFRUCFunc::load() {
    hModule = RGY_LOAD_LIBRARY(NVOFFRUC_MODULENAME);
    if (!hModule) {
        return RGY_ERR_NULL_PTR;
    }

#define LOAD_PROC(proc, procType, procName) { \
    proc = (procType)RGY_GET_PROC_ADDRESS(hModule, procName); \
    if (!proc) { \
        close(); \
        return RGY_ERR_NULL_PTR; \
    } \
}

    LOAD_PROC(create,             PtrToFuncNvOFFRUCCreate,             CreateProcName);
    LOAD_PROC(registerResource,   PtrToFuncNvOFFRUCRegisterResource,   RegisterResourceProcName);
    LOAD_PROC(unregisterResource, PtrToFuncNvOFFRUCUnregisterResource, UnregisterResourceProcName);
    LOAD_PROC(process,            PtrToFuncNvOFFRUCProcess,            ProcessProcName);
    LOAD_PROC(destroy,            PtrToFuncNvOFFRUCDestroy,            DestroyProcName);
#undef LOAD_PROC
    return RGY_ERR_NONE;
}

NVEncNVOFFRUC::NVEncNVOFFRUC() :
    m_func(),
    m_frucHandle(unique_fruc_handle(nullptr, nullptr)),
    m_resource(),
    m_firstTimestamp(-1) {
}

NVEncNVOFFRUC::~NVEncNVOFFRUC() {
    close();
}

void NVEncNVOFFRUC::close() {
    closeFRUCHandle();
    m_func.reset();
}

RGY_ERR NVEncNVOFFRUC::createFRUCHandle(int width, int height) {
    closeFRUCHandle();

    NvOFFRUC_CREATE_PARAM create_param = { 0 };
    create_param.uiWidth = width;
    create_param.uiHeight = height;
    create_param.pDevice = nullptr;
    create_param.eResourceType = CudaResource;
    create_param.eSurfaceFormat = ARGBSurface;
    create_param.eCUDAResourceType = CudaResourceCuDevicePtr;
    NvOFFRUCHandle handle = nullptr;
    auto sts = err_to_rgy(m_func->create(&create_param, &handle));
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    m_frucHandle = unique_fruc_handle(handle, m_func->destroy);
    handle = nullptr;
    return sts;
}

RGY_ERR NVEncNVOFFRUC::registerResource(void *ptr0, void *ptr1, void *ptr2) {
    NvOFFRUC_REGISTER_RESOURCE_PARAM registerParam = { 0 };
    registerParam.pArrResource[0] = m_resource[0];
    registerParam.pArrResource[1] = m_resource[1];
    registerParam.pArrResource[2] = m_resource[2];
    registerParam.uiCount = 3;
    auto sts = err_to_rgy(m_func->registerResource(m_frucHandle.get(), &registerParam));
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    m_resource[0] = ptr0;
    m_resource[1] = ptr1;
    m_resource[2] = ptr2;
    return RGY_ERR_NONE;
}

RGY_ERR NVEncNVOFFRUC::closeFRUCHandle() {
    if (m_resource[0]) {
        NvOFFRUC_UNREGISTER_RESOURCE_PARAM unregisterParam = { 0 };
        unregisterParam.uiCount = 3;
        unregisterParam.pArrResource[0] = m_resource[0];
        unregisterParam.pArrResource[1] = m_resource[1];
        unregisterParam.pArrResource[2] = m_resource[2];
        m_func->unregisterResource(m_frucHandle.get(), &unregisterParam);
        m_frucHandle.reset();
        m_resource[0] = nullptr;
        m_resource[1] = nullptr;
        m_resource[2] = nullptr;
    }
    m_firstTimestamp = -1;
    return RGY_ERR_NONE;
}


RGY_ERR NVEncNVOFFRUC::process(NVEncNVOFFRUCParams *prm) {
    if (m_firstTimestamp < 0) {
        m_firstTimestamp = prm->timestampIn;
    }
    bool ignored = false;
    NvOFFRUC_PROCESS_IN_PARAMS in = { 0 };
    NvOFFRUC_PROCESS_OUT_PARAMS out = { 0 };
    in.stFrameDataInput.pFrame = prm->frameIn;
    in.stFrameDataInput.nTimeStamp = (double)(prm->timestampIn - m_firstTimestamp);
    out.stFrameDataOutput.pFrame = prm->frameOut;
    out.stFrameDataOutput.nTimeStamp = (double)(prm->timestampOut - m_firstTimestamp);
    out.stFrameDataOutput.bHasFrameRepetitionOccurred = &ignored;
    auto sts = err_to_rgy(m_func->process(m_frucHandle.get(), &in, &out));
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    return RGY_ERR_NONE;
}

BOOL APIENTRY DllMain([[maybe_unused]] HMODULE hModule, DWORD  ul_reason_for_call, [[maybe_unused]] LPVOID lpReserved) {
    switch (ul_reason_for_call) {
    case DLL_PROCESS_ATTACH:
    case DLL_THREAD_ATTACH:
    case DLL_THREAD_DETACH:
    case DLL_PROCESS_DETACH:
        break;
    }
    return TRUE;
};

#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */

NVENC_NVOFFRUC_API RGY_ERR __stdcall NVEncNVOptFlowCreate(NVEncNVOFFRUCHandle *ppNVOptFlow) {
    if (ppNVOptFlow == nullptr) {
        return RGY_ERR_NULL_PTR;
    }
    *ppNVOptFlow = new NVEncNVOFFRUC();
    return RGY_ERR_NONE;
}

NVENC_NVOFFRUC_API void __stdcall NVEncNVOptFlowDelete(NVEncNVOFFRUCHandle pNVOptFlow) {
    auto ptr = (NVEncNVOFFRUC *)pNVOptFlow;
    if (ptr) {
        delete ptr;
    }
}

NVENC_NVOFFRUC_API RGY_ERR __stdcall NVEncNVOptFlowCreateFURCHandle(NVEncNVOFFRUCHandle pNVOptFlow, int width, int height) {
    return ((NVEncNVOFFRUC *)pNVOptFlow)->createFRUCHandle(width, height);
}

NVENC_NVOFFRUC_API RGY_ERR __stdcall NVEncNVOptFlowCloseFURCHandle(NVEncNVOFFRUCHandle pNVOptFlow) {
    return ((NVEncNVOFFRUC *)pNVOptFlow)->closeFRUCHandle();
}

NVENC_NVOFFRUC_API RGY_ERR __stdcall NVEncNVOptFlowProc(NVEncNVOFFRUCHandle pNVOptFlow, NVEncNVOFFRUCParams *params) {
    return ((NVEncNVOFFRUC *)pNVOptFlow)->process(params);
}

#if defined(__cplusplus)
}
#endif /* __cplusplus */

#endif
