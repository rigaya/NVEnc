// -----------------------------------------------------------------------------------------
// QSVEnc/NVEnc by rigaya
// -----------------------------------------------------------------------------------------
// The MIT License
//
// Copyright (c) 2011-2016 rigaya
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
#include "rgy_osdep.h"

#if ENCODER_QSV
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
    MFX_MAP(ERR_NONE_PARTIAL_OUTPUT),

    //MFX_MAP(PRINT_OPTION_DONE),
    //MFX_MAP(PRINT_OPTION_ERR),

    //MFX_MAP(ERR_INVALID_COLOR_FORMAT),

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
#endif //#if ENCODER_QSV

#if ENCODER_NVENC || CUFILTERS
#if ENCODER_NVENC
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
#endif


#if ENABLE_NVVFX

struct RGYErrMapNVCV {
    RGY_ERR rgy;
    NvCV_Status nvcv;
};

#define NVCVERR_MAP(x) { RGY_ERR_NVCV_ ##x, NVCV_ERR_ ##x }
static const RGYErrMapNVCV ERR_MAP_NVCV[] = {
  NVCVERR_MAP(GENERAL),
  NVCVERR_MAP(UNIMPLEMENTED),
  NVCVERR_MAP(MEMORY),
  NVCVERR_MAP(EFFECT),
  NVCVERR_MAP(SELECTOR),
  NVCVERR_MAP(BUFFER),
  NVCVERR_MAP(PARAMETER),
  NVCVERR_MAP(MISMATCH),
  NVCVERR_MAP(PIXELFORMAT),
  NVCVERR_MAP(MODEL),
  NVCVERR_MAP(LIBRARY),
  NVCVERR_MAP(INITIALIZATION),
  NVCVERR_MAP(FILE),
  NVCVERR_MAP(FEATURENOTFOUND),
  NVCVERR_MAP(MISSINGINPUT),
  NVCVERR_MAP(RESOLUTION),
  NVCVERR_MAP(UNSUPPORTEDGPU),
  NVCVERR_MAP(WRONGGPU),
  NVCVERR_MAP(UNSUPPORTEDDRIVER),
  NVCVERR_MAP(MODELDEPENDENCIES),
  NVCVERR_MAP(PARSE),
  NVCVERR_MAP(MODELSUBSTITUTION),
  NVCVERR_MAP(READ),
  NVCVERR_MAP(WRITE),
  NVCVERR_MAP(PARAMREADONLY),
  NVCVERR_MAP(TRT_ENQUEUE),
  NVCVERR_MAP(TRT_BINDINGS),
  NVCVERR_MAP(TRT_CONTEXT),
  NVCVERR_MAP(TRT_INFER),
  NVCVERR_MAP(TRT_ENGINE),
  NVCVERR_MAP(NPP),
  NVCVERR_MAP(CONFIG),
  NVCVERR_MAP(TOOSMALL),
  NVCVERR_MAP(TOOBIG),
  NVCVERR_MAP(WRONGSIZE),
  NVCVERR_MAP(OBJECTNOTFOUND),
  NVCVERR_MAP(SINGULAR),
  NVCVERR_MAP(NOTHINGRENDERED),
  NVCVERR_MAP(CONVERGENCE),
  NVCVERR_MAP(OPENGL),
  NVCVERR_MAP(DIRECT3D),
  NVCVERR_MAP(CUDA_BASE),
  NVCVERR_MAP(CUDA_VALUE),
  NVCVERR_MAP(CUDA_MEMORY),
  NVCVERR_MAP(CUDA_PITCH),
  NVCVERR_MAP(CUDA_INIT),
  NVCVERR_MAP(CUDA_LAUNCH),
  NVCVERR_MAP(CUDA_KERNEL),
  NVCVERR_MAP(CUDA_DRIVER),
  NVCVERR_MAP(CUDA_UNSUPPORTED),
  NVCVERR_MAP(CUDA_ILLEGAL_ADDRESS),
  NVCVERR_MAP(CUDA)
};
#undef MFX_MAP

NvCV_Status err_to_nvcv(RGY_ERR err) {
    if (err == RGY_ERR_NONE) return NVCV_SUCCESS;
    const RGYErrMapNVCV *ERR_MAP_FIN = (const RGYErrMapNVCV *)ERR_MAP_NVCV + _countof(ERR_MAP_NVCV);
    auto ret = std::find_if((const RGYErrMapNVCV *)ERR_MAP_NVCV, ERR_MAP_FIN, [err](const RGYErrMapNVCV map) {
        return map.rgy == err;
        });
    return (ret == ERR_MAP_FIN) ? NVCV_ERR_GENERAL : ret->nvcv;
}

RGY_ERR err_to_rgy(NvCV_Status err) {
    if (err == NVCV_SUCCESS) return RGY_ERR_NONE;
    const RGYErrMapNVCV *ERR_MAP_FIN = (const RGYErrMapNVCV *)ERR_MAP_NVCV + _countof(ERR_MAP_NVCV);
    auto ret = std::find_if((const RGYErrMapNVCV *)ERR_MAP_NVCV, ERR_MAP_FIN, [err](const RGYErrMapNVCV map) {
        return map.nvcv == err;
        });
    return (ret == ERR_MAP_FIN) ? RGY_ERR_UNKNOWN : ret->rgy;
}

#endif //#if ENABLE_NVVFX

struct RGYErrMapCuda {
    RGY_ERR rgy;
    cudaError cuda;
};
//grep '=' cudaErrors.txt | awk '{print "CUDAERR_MAP(",$1,"),"}'
#define CUDAERR_MAP(x) { RGY_ERR_cudaError ##x, cudaError ##x }

static const RGYErrMapCuda ERR_MAP_CUDA[] = {
    { RGY_ERR_NONE, cudaSuccess },
    CUDAERR_MAP( InvalidValue ),
    CUDAERR_MAP( MemoryAllocation ),
    CUDAERR_MAP( InitializationError ),
    CUDAERR_MAP( CudartUnloading ),
    CUDAERR_MAP( ProfilerDisabled ),
    CUDAERR_MAP( ProfilerNotInitialized ),
    CUDAERR_MAP( ProfilerAlreadyStarted ),
    CUDAERR_MAP( ProfilerAlreadyStopped ),
    CUDAERR_MAP( InvalidConfiguration ),
    CUDAERR_MAP( InvalidPitchValue ),
    CUDAERR_MAP( InvalidSymbol ),
    CUDAERR_MAP( InvalidHostPointer ),
    CUDAERR_MAP( InvalidDevicePointer ),
    CUDAERR_MAP( InvalidTexture ),
    CUDAERR_MAP( InvalidTextureBinding ),
    CUDAERR_MAP( InvalidChannelDescriptor ),
    CUDAERR_MAP( InvalidMemcpyDirection ),
    CUDAERR_MAP( AddressOfConstant ),
    CUDAERR_MAP( TextureFetchFailed ),
    CUDAERR_MAP( TextureNotBound ),
    CUDAERR_MAP( SynchronizationError ),
    CUDAERR_MAP( InvalidFilterSetting ),
    CUDAERR_MAP( InvalidNormSetting ),
    CUDAERR_MAP( MixedDeviceExecution ),
    CUDAERR_MAP( NotYetImplemented ),
    CUDAERR_MAP( MemoryValueTooLarge ),
#if defined(cudaErrorStubLibrary)
    CUDAERR_MAP( StubLibrary ),
#endif
    CUDAERR_MAP( InsufficientDriver ),
#if defined(cudaErrorCallRequiresNewerDriver)
    CUDAERR_MAP( CallRequiresNewerDriver ),
#endif
    CUDAERR_MAP( InvalidSurface ),
    CUDAERR_MAP( DuplicateVariableName ),
    CUDAERR_MAP( DuplicateTextureName ),
    CUDAERR_MAP( DuplicateSurfaceName ),
    CUDAERR_MAP( DevicesUnavailable ),
    CUDAERR_MAP( IncompatibleDriverContext ),
    CUDAERR_MAP( MissingConfiguration ),
    CUDAERR_MAP( PriorLaunchFailure ),
    CUDAERR_MAP( LaunchMaxDepthExceeded ),
    CUDAERR_MAP( LaunchFileScopedTex ),
    CUDAERR_MAP( LaunchFileScopedSurf ),
    CUDAERR_MAP( SyncDepthExceeded ),
    CUDAERR_MAP( LaunchPendingCountExceeded ),
    CUDAERR_MAP( InvalidDeviceFunction ),
    CUDAERR_MAP( NoDevice ),
    CUDAERR_MAP( InvalidDevice ),
#if defined(cudaErrorDeviceNotLicensed)
    CUDAERR_MAP( DeviceNotLicensed ),
#endif
#if defined(cudaErrorSoftwareValidityNotEstablished)
    CUDAERR_MAP( SoftwareValidityNotEstablished ),
#endif
    CUDAERR_MAP( StartupFailure ),
    CUDAERR_MAP( InvalidKernelImage ),
#if defined(cudaErrorDeviceUninitialized)
    CUDAERR_MAP( DeviceUninitialized ),
#endif
    CUDAERR_MAP( MapBufferObjectFailed ),
    CUDAERR_MAP( UnmapBufferObjectFailed ),
    CUDAERR_MAP( ArrayIsMapped ),
    CUDAERR_MAP( AlreadyMapped ),
    CUDAERR_MAP( NoKernelImageForDevice ),
    CUDAERR_MAP( AlreadyAcquired ),
    CUDAERR_MAP( NotMapped ),
    CUDAERR_MAP( NotMappedAsArray ),
    CUDAERR_MAP( NotMappedAsPointer ),
    CUDAERR_MAP( ECCUncorrectable ),
    CUDAERR_MAP( UnsupportedLimit ),
    CUDAERR_MAP( DeviceAlreadyInUse ),
    CUDAERR_MAP( PeerAccessUnsupported ),
    CUDAERR_MAP( InvalidPtx ),
    CUDAERR_MAP( InvalidGraphicsContext ),
    CUDAERR_MAP( NvlinkUncorrectable ),
    CUDAERR_MAP( JitCompilerNotFound ),
#if defined(cudaErrorUnsupportedPtxVersion)
    CUDAERR_MAP( UnsupportedPtxVersion ),
#endif
#if defined(cudaErrorJitCompilationDisabled)
    CUDAERR_MAP( JitCompilationDisabled ),
#endif
#if defined(cudaErrorUnsupportedExecAffinity)
    CUDAERR_MAP( UnsupportedExecAffinity ),
#endif
    CUDAERR_MAP( InvalidSource ),
    CUDAERR_MAP( FileNotFound ),
    CUDAERR_MAP( SharedObjectSymbolNotFound ),
    CUDAERR_MAP( SharedObjectInitFailed ),
    CUDAERR_MAP( OperatingSystem ),
    CUDAERR_MAP( InvalidResourceHandle ),
    CUDAERR_MAP( IllegalState ),
    CUDAERR_MAP( SymbolNotFound ),
    CUDAERR_MAP( NotReady ),
    CUDAERR_MAP( IllegalAddress ),
    CUDAERR_MAP( LaunchOutOfResources ),
    CUDAERR_MAP( LaunchTimeout ),
    CUDAERR_MAP( LaunchIncompatibleTexturing ),
    CUDAERR_MAP( PeerAccessAlreadyEnabled ),
    CUDAERR_MAP( PeerAccessNotEnabled ),
    CUDAERR_MAP( SetOnActiveProcess ),
    CUDAERR_MAP( ContextIsDestroyed ),
    CUDAERR_MAP( Assert ),
    CUDAERR_MAP( TooManyPeers ),
    CUDAERR_MAP( HostMemoryAlreadyRegistered ),
    CUDAERR_MAP( HostMemoryNotRegistered ),
    CUDAERR_MAP( HardwareStackError ),
    CUDAERR_MAP( IllegalInstruction ),
    CUDAERR_MAP( MisalignedAddress ),
    CUDAERR_MAP( InvalidAddressSpace ),
    CUDAERR_MAP( InvalidPc ),
    CUDAERR_MAP( LaunchFailure ),
    CUDAERR_MAP( CooperativeLaunchTooLarge ),
    CUDAERR_MAP( NotPermitted ),
    CUDAERR_MAP( NotSupported ),
    CUDAERR_MAP( SystemNotReady ),
    CUDAERR_MAP( SystemDriverMismatch ),
    CUDAERR_MAP( CompatNotSupportedOnDevice ),
    //CUDAERR_MAP( MpsConnectionFailed ),
    //CUDAERR_MAP( MpsRpcFailure ),
    //CUDAERR_MAP( MpsServerNotReady ),
    //CUDAERR_MAP( MpsMaxClientsReached ),
    //CUDAERR_MAP( MpsMaxConnectionsReached ),
    CUDAERR_MAP( StreamCaptureUnsupported ),
    CUDAERR_MAP( StreamCaptureInvalidated ),
    CUDAERR_MAP( StreamCaptureMerge ),
    CUDAERR_MAP( StreamCaptureUnmatched ),
    CUDAERR_MAP( StreamCaptureUnjoined ),
    CUDAERR_MAP( StreamCaptureIsolation ),
    CUDAERR_MAP( StreamCaptureImplicit ),
    CUDAERR_MAP( CapturedEvent ),
    CUDAERR_MAP( StreamCaptureWrongThread ),
#if defined(cudaErrorTimeout)
    CUDAERR_MAP( Timeout ),
#endif
#if defined(cudaErrorGraphExecUpdateFailure)
    CUDAERR_MAP( GraphExecUpdateFailure ),
#endif
#if defined(cudaErrorExternalDevice)
    CUDAERR_MAP( ExternalDevice ),
#endif
    CUDAERR_MAP( Unknown ),
    CUDAERR_MAP( ApiFailureBase )
};

cudaError err_to_cuda(RGY_ERR err) {
    if (err == RGY_ERR_NONE) return cudaSuccess;
    const RGYErrMapCuda *ERR_MAP_FIN = (const RGYErrMapCuda *)ERR_MAP_CUDA + _countof(ERR_MAP_CUDA);
    auto ret = std::find_if((const RGYErrMapCuda *)ERR_MAP_CUDA, ERR_MAP_FIN, [err](const RGYErrMapCuda map) {
        return map.rgy == err;
        });
    return (ret == ERR_MAP_FIN) ? cudaErrorUnknown : ret->cuda;
}

RGY_ERR err_to_rgy(cudaError err) {
    if (err == cudaSuccess) return RGY_ERR_NONE;
    const RGYErrMapCuda *ERR_MAP_FIN = (const RGYErrMapCuda *)ERR_MAP_CUDA + _countof(ERR_MAP_CUDA);
    auto ret = std::find_if((const RGYErrMapCuda *)ERR_MAP_CUDA, ERR_MAP_FIN, [err](const RGYErrMapCuda map) {
        return map.cuda == err;
        });
    return (ret == ERR_MAP_FIN) ? RGY_ERR_UNKNOWN : ret->rgy;
}

struct RGYErrMapCudaDriver {
    RGY_ERR rgy;
    CUresult cuda;
};
//grep '=' cudaErrors.txt | awk '{print "CUDADRIVERERR_MAP(",$1,"),"}'
#define CUDADRIVERERR_MAP(x) { RGY_ERR_CUDA_ERROR_ ##x, CUDA_ERROR_ ##x }
static const RGYErrMapCudaDriver ERR_MAP_CUDA_DRIVER[] = {
    { RGY_ERR_NONE, CUDA_SUCCESS },
    CUDADRIVERERR_MAP( INVALID_VALUE ),
    CUDADRIVERERR_MAP( OUT_OF_MEMORY ),
    CUDADRIVERERR_MAP( NOT_INITIALIZED ),
    CUDADRIVERERR_MAP( DEINITIALIZED ),
    CUDADRIVERERR_MAP( PROFILER_DISABLED ),
    CUDADRIVERERR_MAP( PROFILER_NOT_INITIALIZED ),
    CUDADRIVERERR_MAP( PROFILER_ALREADY_STARTED ),
    CUDADRIVERERR_MAP( PROFILER_ALREADY_STOPPED ),
#if defined(CUDA_ERROR_STUB_LIBRARY)
    CUDADRIVERERR_MAP( STUB_LIBRARY ),
#endif
#if defined(CUDA_ERROR_DEVICE_UNAVAILABLE)
    CUDADRIVERERR_MAP( DEVICE_UNAVAILABLE ),
#endif
    CUDADRIVERERR_MAP( NO_DEVICE ),
    CUDADRIVERERR_MAP( INVALID_DEVICE ),
#if defined(CUDA_ERROR_DEVICE_NOT_LICENSED)
    CUDADRIVERERR_MAP( DEVICE_NOT_LICENSED ),
#endif
    CUDADRIVERERR_MAP( INVALID_IMAGE ),
    CUDADRIVERERR_MAP( INVALID_CONTEXT ),
    CUDADRIVERERR_MAP( CONTEXT_ALREADY_CURRENT ),
    CUDADRIVERERR_MAP( MAP_FAILED ),
    CUDADRIVERERR_MAP( UNMAP_FAILED ),
    CUDADRIVERERR_MAP( ARRAY_IS_MAPPED ),
    CUDADRIVERERR_MAP( ALREADY_MAPPED ),
    CUDADRIVERERR_MAP( NO_BINARY_FOR_GPU ),
    CUDADRIVERERR_MAP( ALREADY_ACQUIRED ),
    CUDADRIVERERR_MAP( NOT_MAPPED ),
    CUDADRIVERERR_MAP( NOT_MAPPED_AS_ARRAY ),
    CUDADRIVERERR_MAP( NOT_MAPPED_AS_POINTER ),
    CUDADRIVERERR_MAP( ECC_UNCORRECTABLE ),
    CUDADRIVERERR_MAP( UNSUPPORTED_LIMIT ),
    CUDADRIVERERR_MAP( CONTEXT_ALREADY_IN_USE ),
    CUDADRIVERERR_MAP( PEER_ACCESS_UNSUPPORTED ),
    CUDADRIVERERR_MAP( INVALID_PTX ),
    CUDADRIVERERR_MAP( INVALID_GRAPHICS_CONTEXT ),
    CUDADRIVERERR_MAP( NVLINK_UNCORRECTABLE ),
    CUDADRIVERERR_MAP( JIT_COMPILER_NOT_FOUND ),
#if defined(CUDA_ERROR_UNSUPPORTED_PTX_VERSION)
    CUDADRIVERERR_MAP( UNSUPPORTED_PTX_VERSION ),
#endif
#if defined(CUDA_ERROR_JIT_COMPILATION_DISABLED)
    CUDADRIVERERR_MAP( JIT_COMPILATION_DISABLED ),
#endif
#if defined(CUDA_ERROR_UNSUPPORTED_EXEC_AFFINITY)
    CUDADRIVERERR_MAP( UNSUPPORTED_EXEC_AFFINITY ),
#endif
    CUDADRIVERERR_MAP( INVALID_SOURCE ),
    CUDADRIVERERR_MAP( FILE_NOT_FOUND ),
    CUDADRIVERERR_MAP( SHARED_OBJECT_SYMBOL_NOT_FOUND ),
    CUDADRIVERERR_MAP( SHARED_OBJECT_INIT_FAILED ),
    CUDADRIVERERR_MAP( OPERATING_SYSTEM ),
    CUDADRIVERERR_MAP( INVALID_HANDLE ),
    CUDADRIVERERR_MAP( ILLEGAL_STATE ),
    CUDADRIVERERR_MAP( NOT_FOUND ),
    CUDADRIVERERR_MAP( NOT_READY ),
    CUDADRIVERERR_MAP( ILLEGAL_ADDRESS ),
    CUDADRIVERERR_MAP( LAUNCH_OUT_OF_RESOURCES ),
    CUDADRIVERERR_MAP( LAUNCH_TIMEOUT ),
    CUDADRIVERERR_MAP( LAUNCH_INCOMPATIBLE_TEXTURING ),
    CUDADRIVERERR_MAP( PEER_ACCESS_ALREADY_ENABLED ),
    CUDADRIVERERR_MAP( PEER_ACCESS_NOT_ENABLED ),
    CUDADRIVERERR_MAP( PRIMARY_CONTEXT_ACTIVE ),
    CUDADRIVERERR_MAP( CONTEXT_IS_DESTROYED ),
    CUDADRIVERERR_MAP( ASSERT ),
    CUDADRIVERERR_MAP( TOO_MANY_PEERS ),
    CUDADRIVERERR_MAP( HOST_MEMORY_ALREADY_REGISTERED ),
    CUDADRIVERERR_MAP( HOST_MEMORY_NOT_REGISTERED ),
    CUDADRIVERERR_MAP( HARDWARE_STACK_ERROR ),
    CUDADRIVERERR_MAP( ILLEGAL_INSTRUCTION ),
    CUDADRIVERERR_MAP( MISALIGNED_ADDRESS ),
    CUDADRIVERERR_MAP( INVALID_ADDRESS_SPACE ),
    CUDADRIVERERR_MAP( INVALID_PC ),
    CUDADRIVERERR_MAP( LAUNCH_FAILED ),
    CUDADRIVERERR_MAP( COOPERATIVE_LAUNCH_TOO_LARGE ),
    CUDADRIVERERR_MAP( NOT_PERMITTED ),
    CUDADRIVERERR_MAP( NOT_SUPPORTED ),
    CUDADRIVERERR_MAP( SYSTEM_NOT_READY ),
    CUDADRIVERERR_MAP( SYSTEM_DRIVER_MISMATCH ),
    CUDADRIVERERR_MAP( COMPAT_NOT_SUPPORTED_ON_DEVICE ),
    //CUDADRIVERERR_MAP( MPS_CONNECTION_FAILED ),
    //CUDADRIVERERR_MAP( MPS_RPC_FAILURE ),
    //CUDADRIVERERR_MAP( MPS_SERVER_NOT_READY ),
    //CUDADRIVERERR_MAP( MPS_MAX_CLIENTS_REACHED ),
    //CUDADRIVERERR_MAP( MPS_MAX_CONNECTIONS_REACHED ),
    CUDADRIVERERR_MAP( STREAM_CAPTURE_UNSUPPORTED ),
    CUDADRIVERERR_MAP( STREAM_CAPTURE_INVALIDATED ),
    CUDADRIVERERR_MAP( STREAM_CAPTURE_MERGE ),
    CUDADRIVERERR_MAP( STREAM_CAPTURE_UNMATCHED ),
    CUDADRIVERERR_MAP( STREAM_CAPTURE_UNJOINED ),
    CUDADRIVERERR_MAP( STREAM_CAPTURE_ISOLATION ),
    CUDADRIVERERR_MAP( STREAM_CAPTURE_IMPLICIT ),
    CUDADRIVERERR_MAP( CAPTURED_EVENT ),
    CUDADRIVERERR_MAP( STREAM_CAPTURE_WRONG_THREAD ),
#if defined(CUDA_ERROR_TIMEOUT)
    CUDADRIVERERR_MAP( TIMEOUT ),
#endif
#if defined(CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE)
    CUDADRIVERERR_MAP( GRAPH_EXEC_UPDATE_FAILURE ),
#endif
#if defined(CUDA_ERROR_EXTERNAL_DEVICE)
    CUDADRIVERERR_MAP( EXTERNAL_DEVICE ),
#endif
    CUDADRIVERERR_MAP( UNKNOWN ),
};

CUresult err_to_cuda_driver(RGY_ERR err) {
    if (err == RGY_ERR_NONE) return CUDA_SUCCESS;
    const RGYErrMapCudaDriver *ERR_MAP_FIN = (const RGYErrMapCudaDriver *)ERR_MAP_CUDA_DRIVER + _countof(ERR_MAP_CUDA_DRIVER);
    auto ret = std::find_if((const RGYErrMapCudaDriver *)ERR_MAP_CUDA_DRIVER, ERR_MAP_FIN, [err](const RGYErrMapCudaDriver map) {
        return map.rgy == err;
        });
    return (ret == ERR_MAP_FIN) ? CUDA_ERROR_UNKNOWN : ret->cuda;
}

RGY_ERR err_to_rgy(CUresult err) {
    if (err == CUDA_SUCCESS) return RGY_ERR_NONE;
    const RGYErrMapCudaDriver *ERR_MAP_FIN = (const RGYErrMapCudaDriver *)ERR_MAP_CUDA_DRIVER + _countof(ERR_MAP_CUDA_DRIVER);
    auto ret = std::find_if((const RGYErrMapCudaDriver *)ERR_MAP_CUDA_DRIVER, ERR_MAP_FIN, [err](const RGYErrMapCudaDriver map) {
        return map.cuda == err;
        });
    return (ret == ERR_MAP_FIN) ? RGY_ERR_UNKNOWN : ret->rgy;
}


struct RGYErrMapNpp {
    RGY_ERR rgy;
    NppStatus npp;
};
//grep '=' cudaErrors.txt | awk '{print "NPPERR_MAP(",$1,"),"}'
#define NPPERR_MAP(x) { RGY_ERR_NPP_ ##x, NPP_ ##x }
static const RGYErrMapNpp ERR_MAP_NPP[] = {
    { RGY_ERR_NONE, NPP_SUCCESS },
    NPPERR_MAP( NOT_SUPPORTED_MODE_ERROR ),
    NPPERR_MAP( INVALID_HOST_POINTER_ERROR ),
    NPPERR_MAP( INVALID_DEVICE_POINTER_ERROR ),
    NPPERR_MAP( LUT_PALETTE_BITSIZE_ERROR ),
    NPPERR_MAP( ZC_MODE_NOT_SUPPORTED_ERROR ),
    NPPERR_MAP( NOT_SUFFICIENT_COMPUTE_CAPABILITY ),
    NPPERR_MAP( TEXTURE_BIND_ERROR ),
    NPPERR_MAP( WRONG_INTERSECTION_ROI_ERROR ),
    NPPERR_MAP( HAAR_CLASSIFIER_PIXEL_MATCH_ERROR ),
    NPPERR_MAP( MEMFREE_ERROR ),
    NPPERR_MAP( MEMSET_ERROR ),
    NPPERR_MAP( MEMCPY_ERROR ),
    NPPERR_MAP( ALIGNMENT_ERROR ),
    NPPERR_MAP( CUDA_KERNEL_EXECUTION_ERROR ),
    NPPERR_MAP( ROUND_MODE_NOT_SUPPORTED_ERROR ),
    NPPERR_MAP( QUALITY_INDEX_ERROR ),
    NPPERR_MAP( RESIZE_NO_OPERATION_ERROR ),
    NPPERR_MAP( OVERFLOW_ERROR ),
    NPPERR_MAP( NOT_EVEN_STEP_ERROR ),
    NPPERR_MAP( HISTOGRAM_NUMBER_OF_LEVELS_ERROR ),
    NPPERR_MAP( LUT_NUMBER_OF_LEVELS_ERROR ),
    NPPERR_MAP( CORRUPTED_DATA_ERROR ),
    NPPERR_MAP( CHANNEL_ORDER_ERROR ),
    NPPERR_MAP( ZERO_MASK_VALUE_ERROR ),
    NPPERR_MAP( QUADRANGLE_ERROR ),
    NPPERR_MAP( RECTANGLE_ERROR ),
    NPPERR_MAP( COEFFICIENT_ERROR ),
    NPPERR_MAP( NUMBER_OF_CHANNELS_ERROR ),
    NPPERR_MAP( COI_ERROR ),
    NPPERR_MAP( DIVISOR_ERROR ),
    NPPERR_MAP( CHANNEL_ERROR ),
    NPPERR_MAP( STRIDE_ERROR ),
    NPPERR_MAP( ANCHOR_ERROR ),
    NPPERR_MAP( MASK_SIZE_ERROR ),
    NPPERR_MAP( RESIZE_FACTOR_ERROR ),
    NPPERR_MAP( INTERPOLATION_ERROR ),
    NPPERR_MAP( MIRROR_FLIP_ERROR ),
    NPPERR_MAP( MOMENT_00_ZERO_ERROR ),
    NPPERR_MAP( THRESHOLD_NEGATIVE_LEVEL_ERROR ),
    NPPERR_MAP( THRESHOLD_ERROR ),
    NPPERR_MAP( CONTEXT_MATCH_ERROR ),
    NPPERR_MAP( FFT_FLAG_ERROR ),
    NPPERR_MAP( FFT_ORDER_ERROR ),
    NPPERR_MAP( STEP_ERROR ),
    NPPERR_MAP( SCALE_RANGE_ERROR ),
    NPPERR_MAP( DATA_TYPE_ERROR ),
    NPPERR_MAP( OUT_OFF_RANGE_ERROR ),
    NPPERR_MAP( DIVIDE_BY_ZERO_ERROR ),
    NPPERR_MAP( MEMORY_ALLOCATION_ERR ),
    NPPERR_MAP( NULL_POINTER_ERROR ),
    NPPERR_MAP( RANGE_ERROR ),
    NPPERR_MAP( SIZE_ERROR ),
    NPPERR_MAP( BAD_ARGUMENT_ERROR ),
    NPPERR_MAP( NO_MEMORY_ERROR ),
    NPPERR_MAP( NOT_IMPLEMENTED_ERROR ),
    NPPERR_MAP( ERROR ),
    NPPERR_MAP( ERROR_RESERVED ),
    NPPERR_MAP( NO_ERROR ),
    NPPERR_MAP( SUCCESS ),
    //NPPERR_MAP( NO_OPERATION_WARNING ),
    //NPPERR_MAP( DIVIDE_BY_ZERO_WARNING ),
    //NPPERR_MAP( AFFINE_QUAD_INCORRECT_WARNING ),
    //NPPERR_MAP( WRONG_INTERSECTION_ROI_WARNING ),
    //NPPERR_MAP( WRONG_INTERSECTION_QUAD_WARNING ),
    //NPPERR_MAP( DOUBLE_SIZE_WARNING ),
    //NPPERR_MAP( MISALIGNED_DST_ROI_WARNING ),
};

NppStatus err_to_npp(RGY_ERR err) {
    if (err == RGY_ERR_NONE) return NPP_SUCCESS;
    const RGYErrMapNpp *ERR_MAP_FIN = (const RGYErrMapNpp *)ERR_MAP_NPP + _countof(ERR_MAP_NPP);
    auto ret = std::find_if((const RGYErrMapNpp *)ERR_MAP_NPP, ERR_MAP_FIN, [err](const RGYErrMapNpp map) {
        return map.rgy == err;
        });
    return (ret == ERR_MAP_FIN) ? NPP_ERROR : ret->npp;
}

RGY_ERR err_to_rgy(NppStatus err) {
    if (err == NPP_SUCCESS || err == NPP_NO_ERROR) return RGY_ERR_NONE;
    const RGYErrMapNpp *ERR_MAP_FIN = (const RGYErrMapNpp *)ERR_MAP_NPP + _countof(ERR_MAP_NPP);
    auto ret = std::find_if((const RGYErrMapNpp *)ERR_MAP_NPP, ERR_MAP_FIN, [err](const RGYErrMapNpp map) {
        return map.npp == err;
        });
    return (ret == ERR_MAP_FIN) ? RGY_ERR_UNKNOWN : ret->rgy;
}

#endif //#if ENCODER_NVENC

#if ENCODER_VCEENC
struct RGYErrMapAMF {
    RGY_ERR rgy;
    AMF_RESULT amf;
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
    { RGY_ERR_MORE_BITSTREAM, AMF_EOF },
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

AMF_RESULT err_to_amf(RGY_ERR err) {
    const RGYErrMapAMF *ERR_MAP_FIN = (const RGYErrMapAMF *)ERR_MAP_AMF + _countof(ERR_MAP_AMF);
    auto ret = std::find_if((const RGYErrMapAMF *)ERR_MAP_AMF, ERR_MAP_FIN, [err](const RGYErrMapAMF map) {
        return map.rgy == err;
    });
    return (ret == ERR_MAP_FIN) ? AMF_FAIL : ret->amf;
}

RGY_ERR err_to_rgy(AMF_RESULT err) {
    const RGYErrMapAMF *ERR_MAP_FIN = (const RGYErrMapAMF *)ERR_MAP_AMF + _countof(ERR_MAP_AMF);
    auto ret = std::find_if((const RGYErrMapAMF *)ERR_MAP_AMF, ERR_MAP_FIN, [err](const RGYErrMapAMF map) {
        return map.amf == err;
    });
    return (ret == ERR_MAP_FIN) ? RGY_ERR_UNKNOWN : ret->rgy;
}

#include "rgy_vulkan.h"

#if ENABLE_VULKAN

struct RGYErrMapVK {
    RGY_ERR rgy;
    VkResult vk;
};

static const RGYErrMapVK ERR_MAP_VK[] = {
    { RGY_ERR_NONE, VK_SUCCESS },
    { RGY_ERR_VK_NOT_READY, VK_NOT_READY },
    { RGY_ERR_VK_TIMEOUT, VK_TIMEOUT },
    { RGY_ERR_VK_EVENT_SET, VK_EVENT_SET },
    { RGY_ERR_VK_EVENT_RESET, VK_EVENT_RESET },
    { RGY_ERR_VK_INCOMPLETE, VK_INCOMPLETE },
    { RGY_ERR_VK_OUT_OF_HOST_MEMORY, VK_ERROR_OUT_OF_HOST_MEMORY },
    { RGY_ERR_VK_OUT_OF_DEVICE_MEMORY, VK_ERROR_OUT_OF_DEVICE_MEMORY },
    { RGY_ERR_VK_INITIALIZATION_FAILED, VK_ERROR_INITIALIZATION_FAILED },
    { RGY_ERR_VK_DEVICE_LOST, VK_ERROR_DEVICE_LOST },
    { RGY_ERR_VK_MEMORY_MAP_FAILED, VK_ERROR_MEMORY_MAP_FAILED },
    { RGY_ERR_VK_LAYER_NOT_PRESENT, VK_ERROR_LAYER_NOT_PRESENT },
    { RGY_ERR_VK_EXTENSION_NOT_PRESENT, VK_ERROR_EXTENSION_NOT_PRESENT },
    { RGY_ERR_VK_FEATURE_NOT_PRESENT, VK_ERROR_FEATURE_NOT_PRESENT },
    { RGY_ERR_VK_INCOMPATIBLE_DRIVER, VK_ERROR_INCOMPATIBLE_DRIVER },
    { RGY_ERR_VK_TOO_MANY_OBJECTS, VK_ERROR_TOO_MANY_OBJECTS },
    { RGY_ERR_VK_FORMAT_NOT_SUPPORTED, VK_ERROR_FORMAT_NOT_SUPPORTED },
    { RGY_ERR_VK_FRAGMENTED_POOL, VK_ERROR_FRAGMENTED_POOL },
    { RGY_ERR_VK_UNKNOWN, VK_ERROR_UNKNOWN },
    { RGY_ERR_VK_OUT_OF_POOL_MEMORY, VK_ERROR_OUT_OF_POOL_MEMORY },
    { RGY_ERR_VK_INVALID_EXTERNAL_HANDLE, VK_ERROR_INVALID_EXTERNAL_HANDLE },
    { RGY_ERR_VK_FRAGMENTATION, VK_ERROR_FRAGMENTATION },
    { RGY_ERR_VK_INVALID_OPAQUE_CAPTURE_ADDRESS, VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS },
    { RGY_ERR_VK_SURFACE_LOST_KHR, VK_ERROR_SURFACE_LOST_KHR },
    { RGY_ERR_VK_NATIVE_WINDOW_IN_USE_KHR, VK_ERROR_NATIVE_WINDOW_IN_USE_KHR },
    { RGY_ERR_VK__SUBOPTIMAL_KHR, VK_SUBOPTIMAL_KHR },
    { RGY_ERR_VK_OUT_OF_DATE_KHR, VK_ERROR_OUT_OF_DATE_KHR },
    { RGY_ERR_VK_INCOMPATIBLE_DISPLAY_KHR, VK_ERROR_INCOMPATIBLE_DISPLAY_KHR },
    { RGY_ERR_VK_VALIDATION_FAILED_EXT, VK_ERROR_VALIDATION_FAILED_EXT },
    { RGY_ERR_VK_INVALID_SHADER_NV, VK_ERROR_INVALID_SHADER_NV },
    { RGY_ERR_VK_INVALID_DRM_FORMAT_MODIFIER_PLANE_LAYOUT_EXT, VK_ERROR_INVALID_DRM_FORMAT_MODIFIER_PLANE_LAYOUT_EXT },
    { RGY_ERR_VK_NOT_PERMITTED_EXT, VK_ERROR_NOT_PERMITTED_EXT },
    { RGY_ERR_VK_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT, VK_ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT },
    //{ RGY_VK_THREAD_IDLE_KHR, VK_THREAD_IDLE_KHR },
    //{ RGY_VK_THREAD_DONE_KHR, VK_THREAD_DONE_KHR },
    //{ RGY_VK_OPERATION_DEFERRED_KHR, VK_OPERATION_DEFERRED_KHR },
    //{ RGY_VK_OPERATION_NOT_DEFERRED_KHR, VK_OPERATION_NOT_DEFERRED_KHR },
    //{ RGY_VK_PIPELINE_COMPILE_REQUIRED_EXT, VK_PIPELINE_COMPILE_REQUIRED_EXT },
    { RGY_ERR_VK_OUT_OF_POOL_MEMORY_KHR, VK_ERROR_OUT_OF_POOL_MEMORY_KHR },
    { RGY_ERR_VK_INVALID_EXTERNAL_HANDLE_KHR, VK_ERROR_INVALID_EXTERNAL_HANDLE_KHR },
    { RGY_ERR_VK_FRAGMENTATION_EXT, VK_ERROR_FRAGMENTATION_EXT },
    { RGY_ERR_VK_INVALID_DEVICE_ADDRESS_EXT, VK_ERROR_INVALID_DEVICE_ADDRESS_EXT },
    { RGY_ERR_VK_INVALID_OPAQUE_CAPTURE_ADDRESS_KHR, VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS_KHR },
    //{ RGY_ERR_VK_PIPELINE_COMPILE_REQUIRED_EXT, VK_ERROR_PIPELINE_COMPILE_REQUIRED_EXT },
};

VkResult err_to_vk(RGY_ERR err) {
    const RGYErrMapVK *ERR_MAP_VK_FIN = (const RGYErrMapVK *)ERR_MAP_VK + _countof(ERR_MAP_VK);
    auto ret = std::find_if((const RGYErrMapVK *)ERR_MAP_VK, ERR_MAP_VK_FIN, [err](const RGYErrMapVK map) {
        return map.rgy == err;
        });
    return (ret == ERR_MAP_VK_FIN) ? VK_ERROR_UNKNOWN : ret->vk;
}

RGY_ERR err_to_rgy(VkResult err) {
    const RGYErrMapVK *ERR_MAP_VK_FIN = (const RGYErrMapVK *)ERR_MAP_VK + _countof(ERR_MAP_VK);
    auto ret = std::find_if((const RGYErrMapVK *)ERR_MAP_VK, ERR_MAP_VK_FIN, [err](const RGYErrMapVK map) {
        return map.vk == err;
        });
    return (ret == ERR_MAP_VK_FIN) ? RGY_ERR_VK_UNKNOWN : ret->rgy;
}
#endif //#if ENABLE_VULKAN
#endif //#if ENCODER_VCEENC

#if ENCODER_MPP

struct RGYErrMapMPP {
    RGY_ERR rgy;
    MPP_RET mpp;
};

#define MPPERR_MAP(x) { RGY_ERR_MPP_ERR_ ##x, MPP_ERR_ ##x }
static const RGYErrMapMPP ERR_MAP_MPP[] = {
    { RGY_ERR_NONE, MPP_SUCCESS },
    { RGY_ERR_NONE, MPP_OK },
    { RGY_ERR_UNKNOWN, MPP_NOK },
    MPPERR_MAP(UNKNOW),
    MPPERR_MAP(NULL_PTR),
    MPPERR_MAP(MALLOC),
    MPPERR_MAP(OPEN_FILE),
    MPPERR_MAP(VALUE),
    MPPERR_MAP(READ_BIT),
    MPPERR_MAP(TIMEOUT),
    MPPERR_MAP(PERM),
    MPPERR_MAP(BASE),
    MPPERR_MAP(LIST_STREAM),
    MPPERR_MAP(INIT),
    MPPERR_MAP(VPU_CODEC_INIT),
    MPPERR_MAP(STREAM),
    MPPERR_MAP(FATAL_THREAD),
    MPPERR_MAP(NOMEM),
    MPPERR_MAP(PROTOL),
    { RGY_ERR_MPP_FAIL_SPLIT_FRAME, MPP_FAIL_SPLIT_FRAME },
    MPPERR_MAP(VPUHW),
    { RGY_ERR_MPP_EOS_STREAM_REACHED, MPP_EOS_STREAM_REACHED },
    MPPERR_MAP(BUFFER_FULL),
    MPPERR_MAP(DISPLAY_FULL)
};
#undef MPPERR_MAP

MPP_RET err_to_mpp(RGY_ERR err) {
    const RGYErrMapMPP *ERR_MAP_MPP_FIN = (const RGYErrMapMPP *)ERR_MAP_MPP + _countof(ERR_MAP_MPP);
    auto ret = std::find_if((const RGYErrMapMPP *)ERR_MAP_MPP, ERR_MAP_MPP_FIN, [err](const RGYErrMapMPP map) {
        return map.rgy == err;
        });
    return (ret == ERR_MAP_MPP_FIN) ? MPP_ERR_UNKNOW : ret->mpp;
}

RGY_ERR err_to_rgy(MPP_RET err) {
    const RGYErrMapMPP *ERR_MAP_MPP_FIN = (const RGYErrMapMPP *)ERR_MAP_MPP + _countof(ERR_MAP_MPP);
    auto ret = std::find_if((const RGYErrMapMPP *)ERR_MAP_MPP, ERR_MAP_MPP_FIN, [err](const RGYErrMapMPP map) {
        return map.mpp == err;
        });
    return (ret == ERR_MAP_MPP_FIN) ? RGY_ERR_UNKNOWN : ret->rgy;
}

struct RGYErrMapIM2D {
    RGY_ERR rgy;
    IM_STATUS im2d;
};
#define MPPERR_IM2D(x) { RGY_ERR_IM_STATUS_ ##x, IM_STATUS_ ##x }
static const RGYErrMapIM2D ERR_MAP_IM2D[] = {
    { RGY_ERR_NONE, IM_STATUS_NOERROR },
    { RGY_ERR_NONE, IM_STATUS_SUCCESS },
    MPPERR_IM2D(NOT_SUPPORTED),
    MPPERR_IM2D(OUT_OF_MEMORY),
    MPPERR_IM2D(INVALID_PARAM),
    MPPERR_IM2D(ILLEGAL_PARAM),
    MPPERR_IM2D(FAILED)
};
#undef MPPERR_IM2D

IM_STATUS err_to_im2d(RGY_ERR err) {
    const RGYErrMapIM2D *ERR_MAP_IM2D_FIN = (const RGYErrMapIM2D *)ERR_MAP_IM2D + _countof(ERR_MAP_IM2D);
    auto ret = std::find_if((const RGYErrMapIM2D *)ERR_MAP_IM2D, ERR_MAP_IM2D_FIN, [err](const RGYErrMapIM2D map) {
        return map.rgy == err;
        });
    return (ret == ERR_MAP_IM2D_FIN) ? IM_STATUS_FAILED : ret->im2d;
}

RGY_ERR err_to_rgy(IM_STATUS err) {
    const RGYErrMapIM2D *ERR_MAP_IM2D_FIN = (const RGYErrMapIM2D *)ERR_MAP_IM2D + _countof(ERR_MAP_IM2D);
    auto ret = std::find_if((const RGYErrMapIM2D *)ERR_MAP_IM2D, ERR_MAP_IM2D_FIN, [err](const RGYErrMapIM2D map) {
        return map.im2d == err;
        });
    return (ret == ERR_MAP_IM2D_FIN) ? RGY_ERR_UNKNOWN : ret->rgy;
}

#endif //#if ENCODER_MPP

const TCHAR *get_err_mes(RGY_ERR sts) {
    switch (sts) {
    case RGY_ERR_NONE:                            return _T("no error.");
    case RGY_ERR_UNKNOWN:                         return _T("unknown error.");
    case RGY_ERR_NULL_PTR:                        return _T("null pointer.");
    case RGY_ERR_UNSUPPORTED:                     return _T("undeveloped feature.");
    case RGY_ERR_MEMORY_ALLOC:                    return _T("failed to allocate memory.");
    case RGY_ERR_NOT_ENOUGH_BUFFER:               return _T("insufficient buffer at input/output.");
    case RGY_ERR_INVALID_HANDLE:                  return _T("invalid handle.");
    case RGY_ERR_LOCK_MEMORY:                     return _T("failed to lock the memory block.");
    case RGY_ERR_NOT_INITIALIZED:                 return _T("member function called before initialization.");
    case RGY_ERR_NOT_FOUND:                       return _T("the specified object is not found.");
    case RGY_ERR_MORE_DATA:                       return _T("expect more data at input.");
    case RGY_ERR_MORE_SURFACE:                    return _T("expect more surface at output.");
    case RGY_ERR_ABORTED:                         return _T("operation aborted.");
    case RGY_ERR_DEVICE_LOST:                     return _T("lose the HW acceleration device.");
    case RGY_ERR_INCOMPATIBLE_VIDEO_PARAM:        return _T("incompatible video parameters.");
    case RGY_ERR_INVALID_VIDEO_PARAM:             return _T("invalid video parameters.");
    case RGY_ERR_UNDEFINED_BEHAVIOR:              return _T("undefined behavior.");
    case RGY_ERR_DEVICE_FAILED:                   return _T("device operation failure.");
    case RGY_ERR_MORE_BITSTREAM:                  return _T("more bitstream required.");
    case RGY_ERR_INCOMPATIBLE_AUDIO_PARAM:        return _T("incompatible audio param.");
    case RGY_ERR_INVALID_AUDIO_PARAM:             return _T("invalid audio param.");
    case RGY_ERR_GPU_HANG:                        return _T("gpu hang.");
    case RGY_ERR_REALLOC_SURFACE:                 return _T("failed to realloc surface.");
    case RGY_ERR_ACCESS_DENIED:                   return _T("access denied");
    case RGY_ERR_INVALID_PARAM:                   return _T("invalid param.");
    case RGY_ERR_OUT_OF_RANGE:                    return _T("out of range.");
    case RGY_ERR_ALREADY_INITIALIZED:             return _T("already initialized.");
    case RGY_ERR_INVALID_FORMAT:                  return _T("invalid format.");
    case RGY_ERR_WRONG_STATE:                     return _T("wrong state.");
    case RGY_ERR_FILE_OPEN:                       return _T("file open error.");
    case RGY_ERR_INPUT_FULL:                      return _T("input full.");
    case RGY_ERR_INVALID_CODEC:                   return _T("invalid codec.");
    case RGY_ERR_INVALID_DATA_TYPE:               return _T("invalid data type.");
    case RGY_ERR_INVALID_RESOLUTION:              return _T("invalid resolution.");
    case RGY_ERR_INVALID_DEVICE:                  return _T("invalid devices.");
    case RGY_ERR_INVALID_CALL:                    return _T("invalid call sequence.");
    case RGY_ERR_NO_DEVICE:                       return _T("no deivce found.");
    case RGY_ERR_INVALID_VERSION:                 return _T("invalid version.");
    case RGY_ERR_MAP_FAILED:                      return _T("map failed.");
    case RGY_ERR_CUDA:                            return _T("error in cuda.");
    case RGY_ERR_RUN_PROCESS:                     return _T("running process failed.");
    case RGY_WRN_IN_EXECUTION:                    return _T("the previous asynchrous operation is in execution.");
    case RGY_WRN_DEVICE_BUSY:                     return _T("the HW acceleration device is busy.");
    case RGY_WRN_VIDEO_PARAM_CHANGED:             return _T("the video parameters are changed during decoding.");
    case RGY_WRN_PARTIAL_ACCELERATION:            return _T("partial acceleration.");
    case RGY_WRN_INCOMPATIBLE_VIDEO_PARAM:        return _T("incompatible video parameters.");
    case RGY_WRN_VALUE_NOT_CHANGED:               return _T("the value is saturated based on its valid range.");
    case RGY_WRN_OUT_OF_RANGE:                    return _T("the value is out of valid range.");
    case RGY_ERR_INVALID_PLATFORM:                return _T("invalid platform.");
    case RGY_ERR_INVALID_DEVICE_TYPE:             return _T("invalid device type.");
    case RGY_ERR_INVALID_CONTEXT:                 return _T("invalid context.");
    case RGY_ERR_INVALID_QUEUE_PROPERTIES:        return _T("invalid queue properties.");
    case RGY_ERR_INVALID_COMMAND_QUEUE:           return _T("invalid command queue.");
    case RGY_ERR_DEVICE_NOT_FOUND:                return _T("device not found.");
    case RGY_ERR_DEVICE_NOT_AVAILABLE:            return _T("device not available.");
    case RGY_ERR_COMPILER_NOT_AVAILABLE:          return _T("compiler not available.");
    case RGY_ERR_COMPILE_PROGRAM_FAILURE:         return _T("compile program failure.");
    case RGY_ERR_MEM_OBJECT_ALLOCATION_FAILURE:   return _T("pbject allocation failure.");
    case RGY_ERR_OUT_OF_RESOURCES:                return _T("out of resources.");
    case RGY_ERR_OUT_OF_HOST_MEMORY:              return _T("out of hots memory.");
    case RGY_ERR_PROFILING_INFO_NOT_AVAILABLE:    return _T("profiling info not available.");
    case RGY_ERR_MEM_COPY_OVERLAP:                return _T("memcpy overlap.");
    case RGY_ERR_IMAGE_FORMAT_MISMATCH:           return _T("image format mismatch.");
    case RGY_ERR_IMAGE_FORMAT_NOT_SUPPORTED:      return _T("image format not supported.");
    case RGY_ERR_BUILD_PROGRAM_FAILURE:           return _T("build program failure.");
    case RGY_ERR_MAP_FAILURE:                     return _T("map failure.");
    case RGY_ERR_INVALID_HOST_PTR:                return _T("invalid host ptr.");
    case RGY_ERR_INVALID_MEM_OBJECT:              return _T("invalid mem obejct.");
    case RGY_ERR_INVALID_IMAGE_FORMAT_DESCRIPTOR: return _T("invalid image format descripter.");
    case RGY_ERR_INVALID_IMAGE_SIZE:              return _T("invalid image size.");
    case RGY_ERR_INVALID_SAMPLER:                 return _T("invalid sampler.");
    case RGY_ERR_INVALID_BINARY:                  return _T("invalid binary.");
    case RGY_ERR_INVALID_BUILD_OPTIONS:           return _T("invalid build options.");
    case RGY_ERR_INVALID_PROGRAM:                 return _T("invalid program.");
    case RGY_ERR_INVALID_PROGRAM_EXECUTABLE:      return _T("invalid program executable.");
    case RGY_ERR_INVALID_KERNEL_NAME:             return _T("invalid kernel name.");
    case RGY_ERR_INVALID_KERNEL_DEFINITION:       return _T("invalid kernel definition.");
    case RGY_ERR_INVALID_KERNEL:                  return _T("invalid kernel.");
    case RGY_ERR_INVALID_ARG_INDEX:               return _T("invalid arg index.");
    case RGY_ERR_INVALID_ARG_VALUE:               return _T("invalid arg value.");
    case RGY_ERR_INVALID_ARG_SIZE:                return _T("invalid arg size.");
    case RGY_ERR_INVALID_KERNEL_ARGS:             return _T("invalid kernel args.");
    case RGY_ERR_INVALID_WORK_DIMENSION:          return _T("invalid work dimension.");
    case RGY_ERR_INVALID_WORK_GROUP_SIZE:         return _T("invalid work group size.");
    case RGY_ERR_INVALID_WORK_ITEM_SIZE:          return _T("invalid work item size.");
    case RGY_ERR_INVALID_GLOBAL_OFFSET:           return _T("invalid global offset.");
    case RGY_ERR_INVALID_EVENT_WAIT_LIST:         return _T("invalid event wait list.");
    case RGY_ERR_INVALID_EVENT:                   return _T("invalid event.");
    case RGY_ERR_INVALID_OPERATION:               return _T("invalid operation.");
    case RGY_ERR_INVALID_GL_OBJECT:               return _T("invalid gl object.");
    case RGY_ERR_INVALID_BUFFER_SIZE:             return _T("invalid buffer size.");
    case RGY_ERR_INVALID_MIP_LEVEL:               return _T("invalid mip level.");
    case RGY_ERR_INVALID_GLOBAL_WORK_SIZE:        return _T("invalid global work size.");
    case RGY_ERR_OPENCL_CRUSH:                    return _T("OpenCL crushed.");
    case RGY_ERR_VK_NOT_READY:                                      return _T("VK_NOT_READY");
    case RGY_ERR_VK_TIMEOUT:                                        return _T("VK_TIMEOUT");
    case RGY_ERR_VK_EVENT_SET:                                      return _T("VK_EVENT_SET");
    case RGY_ERR_VK_EVENT_RESET:                                    return _T("VK_EVENT_RESET");
    case RGY_ERR_VK_INCOMPLETE:                                     return _T("VK_INCOMPLETE");
    case RGY_ERR_VK_OUT_OF_HOST_MEMORY:                             return _T("VK_OUT_OF_HOST_MEMORY");
    case RGY_ERR_VK_OUT_OF_DEVICE_MEMORY:                           return _T("VK_OUT_OF_DEVICE_MEMORY");
    case RGY_ERR_VK_INITIALIZATION_FAILED:                          return _T("VK_INITIALIZATION_FAILED");
    case RGY_ERR_VK_DEVICE_LOST:                                    return _T("VK_DEVICE_LOST");
    case RGY_ERR_VK_MEMORY_MAP_FAILED:                              return _T("VK_MEMORY_MAP_FAILED");
    case RGY_ERR_VK_LAYER_NOT_PRESENT:                              return _T("VK_LAYER_NOT_PRESENT");
    case RGY_ERR_VK_EXTENSION_NOT_PRESENT:                          return _T("VK_EXTENSION_NOT_PRESENT");
    case RGY_ERR_VK_FEATURE_NOT_PRESENT:                            return _T("VK_FEATURE_NOT_PRESENT");
    case RGY_ERR_VK_INCOMPATIBLE_DRIVER:                            return _T("VK_INCOMPATIBLE_DRIVER");
    case RGY_ERR_VK_TOO_MANY_OBJECTS:                               return _T("VK_TOO_MANY_OBJECTS");
    case RGY_ERR_VK_FORMAT_NOT_SUPPORTED:                           return _T("VK_FORMAT_NOT_SUPPORTED");
    case RGY_ERR_VK_FRAGMENTED_POOL:                                return _T("VK_FRAGMENTED_POOL");
    case RGY_ERR_VK_UNKNOWN:                                        return _T("VK_UNKNOWN");
    case RGY_ERR_VK_OUT_OF_POOL_MEMORY:                             return _T("VK_OUT_OF_POOL_MEMORY");
    case RGY_ERR_VK_INVALID_EXTERNAL_HANDLE:                        return _T("VK_INVALID_EXTERNAL_HANDLE");
    case RGY_ERR_VK_FRAGMENTATION:                                  return _T("VK_FRAGMENTATION");
    case RGY_ERR_VK_INVALID_OPAQUE_CAPTURE_ADDRESS:                 return _T("VK_INVALID_OPAQUE_CAPTURE_ADDRESS");
    case RGY_ERR_VK_SURFACE_LOST_KHR:                               return _T("VK_SURFACE_LOST_KHR");
    case RGY_ERR_VK_NATIVE_WINDOW_IN_USE_KHR:                       return _T("VK_NATIVE_WINDOW_IN_USE_KHR");
    case RGY_ERR_VK__SUBOPTIMAL_KHR:                                return _T("VK_SUBOPTIMAL_KHR");
    case RGY_ERR_VK_OUT_OF_DATE_KHR:                                return _T("VK_OUT_OF_DATE_KHR");
    case RGY_ERR_VK_INCOMPATIBLE_DISPLAY_KHR:                       return _T("VK_INCOMPATIBLE_DISPLAY_KHR");
    case RGY_ERR_VK_VALIDATION_FAILED_EXT:                          return _T("VK_VALIDATION_FAILED_EXT");
    case RGY_ERR_VK_INVALID_SHADER_NV:                              return _T("VK_INVALID_SHADER_NV");
    case RGY_ERR_VK_INVALID_DRM_FORMAT_MODIFIER_PLANE_LAYOUT_EXT:   return _T("VK_INVALID_DRM_FORMAT_MODIFIER_PLANE_LAYOUT_EXT");
    case RGY_ERR_VK_NOT_PERMITTED_EXT:                              return _T("VK_NOT_PERMITTED_EXT");
    case RGY_ERR_VK_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT:            return _T("VK_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT");
    case RGY_VK_THREAD_IDLE_KHR:                                    return _T("VK_THREAD_IDLE_KHR");
    case RGY_VK_THREAD_DONE_KHR:                                    return _T("VK_THREAD_DONE_KHR");
    case RGY_VK_OPERATION_DEFERRED_KHR:                             return _T("VK_OPERATION_DEFERRED_KHR");
    case RGY_VK_OPERATION_NOT_DEFERRED_KHR:                         return _T("VK_OPERATION_NOT_DEFERRED_KHR");
    case RGY_VK_PIPELINE_COMPILE_REQUIRED_EXT:                      return _T("VK_PIPELINE_COMPILE_REQUIRED_EXT");
    case RGY_ERR_VK_OUT_OF_POOL_MEMORY_KHR:                         return _T("VK_OUT_OF_POOL_MEMORY_KHR");
    case RGY_ERR_VK_INVALID_EXTERNAL_HANDLE_KHR:                    return _T("VK_INVALID_EXTERNAL_HANDLE_KHR");
    case RGY_ERR_VK_FRAGMENTATION_EXT:                              return _T("VK_FRAGMENTATION_EXT");
    case RGY_ERR_VK_INVALID_DEVICE_ADDRESS_EXT:                     return _T("VK_INVALID_DEVICE_ADDRESS_EXT");
    case RGY_ERR_VK_INVALID_OPAQUE_CAPTURE_ADDRESS_KHR:             return _T("VK_INVALID_OPAQUE_CAPTURE_ADDRESS_KHR");
    case RGY_ERR_VK_PIPELINE_COMPILE_REQUIRED_EXT:                  return _T("VK_PIPELINE_COMPILE_REQUIRED_EXT");

    case RGY_ERR_NVCV_GENERAL:               return _T("An otherwise unspecified error has occurred.");
    case RGY_ERR_NVCV_UNIMPLEMENTED:         return _T("The requested feature is not yet implemented.");
    case RGY_ERR_NVCV_MEMORY:                return _T("There is not enough memory for the requested operation.");
    case RGY_ERR_NVCV_EFFECT:                return _T("An invalid effect handle has been supplied.");
    case RGY_ERR_NVCV_SELECTOR:              return _T("The given parameter selector is not valid in this effect filter.");
    case RGY_ERR_NVCV_BUFFER:                return _T("An image buffer has not been specified.");
    case RGY_ERR_NVCV_PARAMETER:             return _T("An invalid parameter value has been supplied for this effect+selector.");
    case RGY_ERR_NVCV_MISMATCH:              return _T("Some parameters are not appropriately matched.");
    case RGY_ERR_NVCV_PIXELFORMAT:           return _T("The specified pixel format is not accommodated.");
    case RGY_ERR_NVCV_MODEL:                 return _T("Error while loading the TRT model.");
    case RGY_ERR_NVCV_LIBRARY:               return _T("Error loading the dynamic library.");
    case RGY_ERR_NVCV_INITIALIZATION:        return _T("The effect has not been properly initialized.");
    case RGY_ERR_NVCV_FILE:                  return _T("The file could not be found.");
    case RGY_ERR_NVCV_FEATURENOTFOUND:       return _T("The requested feature was not found");
    case RGY_ERR_NVCV_MISSINGINPUT:          return _T("A required parameter was not set");
    case RGY_ERR_NVCV_RESOLUTION:            return _T("The specified image resolution is not supported.");
    case RGY_ERR_NVCV_UNSUPPORTEDGPU:        return _T("The GPU is not supported");
    case RGY_ERR_NVCV_WRONGGPU:              return _T("The current GPU is not the one selected.");
    case RGY_ERR_NVCV_UNSUPPORTEDDRIVER:     return _T("The currently installed graphics driver is not supported");
    case RGY_ERR_NVCV_MODELDEPENDENCIES:     return _T("There is no model with dependencies that match this system");
    case RGY_ERR_NVCV_PARSE:                 return _T("There has been a parsing or syntax error while reading a file");
    case RGY_ERR_NVCV_MODELSUBSTITUTION:     return _T("The specified model does not exist and has been substituted.");
    case RGY_ERR_NVCV_READ:                  return _T("An error occurred while reading a file.");
    case RGY_ERR_NVCV_WRITE:                 return _T("An error occurred while writing a file.");
    case RGY_ERR_NVCV_PARAMREADONLY:         return _T("The selected parameter is read-only.");
    case RGY_ERR_NVCV_TRT_ENQUEUE:           return _T("TensorRT enqueue failed.");
    case RGY_ERR_NVCV_TRT_BINDINGS:          return _T("Unexpected TensorRT bindings.");
    case RGY_ERR_NVCV_TRT_CONTEXT:           return _T("An error occurred while creating a TensorRT context.");
    case RGY_ERR_NVCV_TRT_INFER:             return _T("The was a problem creating the inference engine.");
    case RGY_ERR_NVCV_TRT_ENGINE:            return _T("There was a problem deserializing the inference runtime engine.");
    case RGY_ERR_NVCV_NPP:                   return _T("An error has occurred in the NPP library.");
    case RGY_ERR_NVCV_CONFIG:                return _T("No suitable model exists for the specified parameter configuration.");
    case RGY_ERR_NVCV_TOOSMALL:              return _T("A supplied parameter or buffer is not large enough.");
    case RGY_ERR_NVCV_TOOBIG:                return _T("A supplied parameter is too big.");
    case RGY_ERR_NVCV_WRONGSIZE:             return _T("A supplied parameter is not the expected size.");
    case RGY_ERR_NVCV_OBJECTNOTFOUND:        return _T("The specified object was not found.");
    case RGY_ERR_NVCV_SINGULAR:              return _T("A mathematical singularity has been encountered.");
    case RGY_ERR_NVCV_NOTHINGRENDERED:       return _T("Nothing was rendered in the specified region.");
    case RGY_ERR_NVCV_CONVERGENCE:           return _T("An iteration did not converge satisfactorily.");
    case RGY_ERR_NVCV_OPENGL:                return _T("An OpenGL error has occurred.");
    case RGY_ERR_NVCV_DIRECT3D:              return _T("A Direct3D error has occurred.");
    case RGY_ERR_NVCV_CUDA_BASE:             return _T("CUDA errors are offset from this value.");
    case RGY_ERR_NVCV_CUDA_VALUE:            return _T("A CUDA parameter is not within the acceptable range.");
    case RGY_ERR_NVCV_CUDA_MEMORY:           return _T("There is not enough CUDA memory for the requested operation.");
    case RGY_ERR_NVCV_CUDA_PITCH:            return _T("A CUDA pitch is not within the acceptable range.");
    case RGY_ERR_NVCV_CUDA_INIT:             return _T("The CUDA driver and runtime could not be initialized.");
    case RGY_ERR_NVCV_CUDA_LAUNCH:           return _T("The CUDA kernel launch has failed.");
    case RGY_ERR_NVCV_CUDA_KERNEL:           return _T("No suitable kernel image is available for the device.");
    case RGY_ERR_NVCV_CUDA_DRIVER:           return _T("The installed NVIDIA CUDA driver is older than the CUDA runtime library.");
    case RGY_ERR_NVCV_CUDA_UNSUPPORTED:      return _T("The CUDA operation is not supported on the current system or device.");
    case RGY_ERR_NVCV_CUDA_ILLEGAL_ADDRESS:  return _T("CUDA tried to load or store on an invalid memory address.");
    case RGY_ERR_NVCV_CUDA:                  return _T("An otherwise unspecified CUDA error has been reported.");

    case RGY_ERR_cudaErrorInvalidValue: return _T("cudaErrorInvalidValue: This indicates that one or more of the parameters passed to the API call is not within an acceptable range of values.");
    case RGY_ERR_cudaErrorMemoryAllocation: return _T("cudaErrorMemoryAllocation: The API call failed because it was unable to allocate enough memory to perform the requested operation.");
    case RGY_ERR_cudaErrorInitializationError: return _T("cudaErrorInitializationError: The API call failed because the CUDA driver and runtime could not be initialized.");
    case RGY_ERR_cudaErrorCudartUnloading: return _T("cudaErrorCudartUnloading: This indicates that a CUDA Runtime API call cannot be executed because it is being called during process shut down, at a point in time after CUDA driver has been unloaded.");
    case RGY_ERR_cudaErrorProfilerDisabled: return _T("cudaErrorProfilerDisabled: This indicates profiler is not initialized for this run. This can happen when the application is running with external profiling tools like visual profiler.");
    case RGY_ERR_cudaErrorProfilerNotInitialized: return _T("cudaErrorProfilerNotInitialized: This error return is deprecated as of CUDA 5.0. It is no longer an error to attempt to enable/disable the profiling via ::cudaProfilerStart or ::cudaProfilerStop without initialization.");
    case RGY_ERR_cudaErrorProfilerAlreadyStarted: return _T("cudaErrorProfilerAlreadyStarted: This error return is deprecated as of CUDA 5.0. It is no longer an error to call cudaProfilerStart() when profiling is already enabled.");case RGY_ERR_cudaErrorProfilerAlreadyStopped: return _T("cudaErrorProfilerAlreadyStopped: This error return is deprecated as of CUDA 5.0. It is no longer an error to call cudaProfilerStop() when profiling is already disabled.");case RGY_ERR_cudaErrorInvalidConfiguration: return _T("cudaErrorInvalidConfiguration: This indicates that a kernel launch is requesting resources that can never be satisfied by the current device. Requesting more shared memory per block than the device supports will trigger this error, as will requesting too many threads or blocks. See ::cudaDeviceProp for more device limitations.");
    case RGY_ERR_cudaErrorInvalidPitchValue: return _T("cudaErrorInvalidPitchValue: This indicates that one or more of the pitch-related parameters passed to the API call is not within the acceptable range for pitch.");
    case RGY_ERR_cudaErrorInvalidSymbol: return _T("cudaErrorInvalidSymbol: This indicates that the symbol name/identifier passed to the API call is not a valid name or identifier.");
    case RGY_ERR_cudaErrorInvalidHostPointer: return _T("cudaErrorInvalidHostPointer: This indicates that at least one host pointer passed to the API call is not a valid host pointer. This error return is deprecated as of CUDA 10.1.");
    case RGY_ERR_cudaErrorInvalidDevicePointer: return _T("cudaErrorInvalidDevicePointer: This indicates that at least one device pointer passed to the API call is not a valid device pointer. This error return is deprecated as of CUDA 10.1.");
    case RGY_ERR_cudaErrorInvalidTexture: return _T("cudaErrorInvalidTexture: This indicates that the texture passed to the API call is not a valid texture.");
    case RGY_ERR_cudaErrorInvalidTextureBinding: return _T("cudaErrorInvalidTextureBinding: This indicates that the texture binding is not valid. This occurs if you call ::cudaGetTextureAlignmentOffset() with an unbound texture.");
    case RGY_ERR_cudaErrorInvalidChannelDescriptor: return _T("cudaErrorInvalidChannelDescriptor: This indicates that the channel descriptor passed to the API call is not valid. This occurs if the format is not one of the formats specified by ::cudaChannelFormatKind, or if one of the dimensions is invalid.");
    case RGY_ERR_cudaErrorInvalidMemcpyDirection: return _T("cudaErrorInvalidMemcpyDirection: This indicates that the direction of the memcpy passed to the API call is not one of the types specified by ::cudaMemcpyKind.");
    case RGY_ERR_cudaErrorAddressOfConstant: return _T("cudaErrorAddressOfConstant: This indicated that the user has taken the address of a constant variable, which was forbidden up until the CUDA 3.1 release. This error return is deprecated as of CUDA 3.1. Variables in constant memory may now have their address taken by the runtime via ::cudaGetSymbolAddress().");
    case RGY_ERR_cudaErrorTextureFetchFailed: return _T("cudaErrorTextureFetchFailed: This indicated that a texture fetch was not able to be performed. This was previously used for device emulation of texture operations. This error return is deprecated as of CUDA 3.1. Device emulation mode was removed with the CUDA 3.1 release.");
    case RGY_ERR_cudaErrorTextureNotBound: return _T("cudaErrorTextureNotBound: This indicated that a texture was not bound for access. This was previously used for device emulation of texture operations. This error return is deprecated as of CUDA 3.1. Device emulation mode was removed with the CUDA 3.1 release.");
    case RGY_ERR_cudaErrorSynchronizationError: return _T("cudaErrorSynchronizationError: This indicated that a synchronization operation had failed. This was previously used for some device emulation functions. This error return is deprecated as of CUDA 3.1. Device emulation mode was removed with the CUDA 3.1 release.");
    case RGY_ERR_cudaErrorInvalidFilterSetting: return _T("cudaErrorInvalidFilterSetting: This indicates that a non-float texture was being accessed with linear filtering. This is not supported by CUDA.");
    case RGY_ERR_cudaErrorInvalidNormSetting: return _T("cudaErrorInvalidNormSetting: This indicates that an attempt was made to read a non-float texture as a normalized float. This is not supported by CUDA.");
    case RGY_ERR_cudaErrorMixedDeviceExecution: return _T("cudaErrorMixedDeviceExecution: Mixing of device and device emulation code was not allowed. This error return is deprecated as of CUDA 3.1. Device emulation mode was removed with the CUDA 3.1 release.");
    case RGY_ERR_cudaErrorNotYetImplemented: return _T("cudaErrorNotYetImplemented: This indicates that the API call is not yet implemented. Production releases of CUDA will never return this error. This error return is deprecated as of CUDA 4.1.");
    case RGY_ERR_cudaErrorMemoryValueTooLarge: return _T("cudaErrorMemoryValueTooLarge: This indicated that an emulated device pointer exceeded the 32-bit address range. This error return is deprecated as of CUDA 3.1. Device emulation mode was removed with the CUDA 3.1 release.");
    case RGY_ERR_cudaErrorStubLibrary: return _T("cudaErrorStubLibrary: This indicates that the CUDA driver that the application has loaded is a stub library. Applications that run with the stub rather than a real driver loaded will result in CUDA API returning this error.");
    case RGY_ERR_cudaErrorInsufficientDriver: return _T("cudaErrorInsufficientDriver: This indicates that the installed NVIDIA CUDA driver is older than the CUDA runtime library. This is not a supported configuration. Users should install an updated NVIDIA display driver to allow the application to run.");
    case RGY_ERR_cudaErrorCallRequiresNewerDriver: return _T("cudaErrorCallRequiresNewerDriver: This indicates that the API call requires a newer CUDA driver than the one currently installed. Users should install an updated NVIDIA CUDA driver to allow the API call to succeed.");
    case RGY_ERR_cudaErrorInvalidSurface: return _T("cudaErrorInvalidSurface: This indicates that the surface passed to the API call is not a valid surface.");
    case RGY_ERR_cudaErrorDuplicateVariableName: return _T("cudaErrorDuplicateVariableName: This indicates that multiple global or constant variables (across separate CUDA source files in the application) share the same string name.");
    case RGY_ERR_cudaErrorDuplicateTextureName: return _T("cudaErrorDuplicateTextureName: This indicates that multiple textures (across separate CUDA source files in the application) share the same string name.");
    case RGY_ERR_cudaErrorDuplicateSurfaceName: return _T("cudaErrorDuplicateSurfaceName: This indicates that multiple surfaces (across separate CUDA source files in the application) share the same string name.");
    case RGY_ERR_cudaErrorDevicesUnavailable: return _T("cudaErrorDevicesUnavailable: This indicates that all CUDA devices are busy or unavailable at the current time. Devices are often busy/unavailable due to use of ::cudaComputeModeProhibited, ::cudaComputeModeExclusiveProcess, or when long running CUDA kernels have filled up the GPU and are blocking new work from starting. They can also be unavailable due to memory constraints on a device that already has active CUDA work being performed.");
    case RGY_ERR_cudaErrorIncompatibleDriverContext: return _T("cudaErrorIncompatibleDriverContext: This indicates that the current context is not compatible with this the CUDA Runtime. This can only occur if you are using CUDA Runtime/Driver interoperability and have created an existing Driver context using the driver API. The Driver context may be incompatible either because the Driver context was created using an older version of the API, because the Runtime API call expects a primary driver context and the Driver context is not primary, or because the Driver context has been destroyed. Please see CUDART_DRIVER Interactions with the CUDA Driver API for more information.");
    case RGY_ERR_cudaErrorMissingConfiguration: return _T("cudaErrorMissingConfiguration: The device function being invoked (usually via ::cudaLaunchKernel()) was not previously configured via the ::cudaConfigureCall() function.");
    case RGY_ERR_cudaErrorPriorLaunchFailure: return _T("cudaErrorPriorLaunchFailure: This indicated that a previous kernel launch failed. This was previously used for device emulation of kernel launches. This error return is deprecated as of CUDA 3.1. Device emulation mode was removed with the CUDA 3.1 release.");
    case RGY_ERR_cudaErrorLaunchMaxDepthExceeded: return _T("cudaErrorLaunchMaxDepthExceeded: This error indicates that a device runtime grid launch did not occur because the depth of the child grid would exceed the maximum supported number of nested grid launches.");
    case RGY_ERR_cudaErrorLaunchFileScopedTex: return _T("cudaErrorLaunchFileScopedTex: This error indicates that a grid launch did not occur because the kernel uses file-scoped textures which are unsupported by the device runtime. Kernels launched via the device runtime only support textures created with the Texture Object API's.");
    case RGY_ERR_cudaErrorLaunchFileScopedSurf: return _T("cudaErrorLaunchFileScopedSurf: This error indicates that a grid launch did not occur because the kernel uses file-scoped surfaces which are unsupported by the device runtime. Kernels launched via the device runtime only support surfaces created with the Surface Object API's.");
    case RGY_ERR_cudaErrorSyncDepthExceeded: return _T("cudaErrorSyncDepthExceeded: This error indicates that a call to ::cudaDeviceSynchronize made from the device runtime failed because the call was made at grid depth greater than than either the default (2 levels of grids) or user specified device limit ::cudaLimitDevRuntimeSyncDepth. To be able to synchronize on launched grids at a greater depth successfully, the maximum nested depth at which ::cudaDeviceSynchronize will be called must be specified with the ::cudaLimitDevRuntimeSyncDepth limit to the ::cudaDeviceSetLimit api before the host-side launch of a kernel using the device runtime. Keep in mind that additional levels of sync depth require the runtime to reserve large amounts of device memory that cannot be used for user allocations.");
    case RGY_ERR_cudaErrorLaunchPendingCountExceeded: return _T("cudaErrorLaunchPendingCountExceeded: This error indicates that a device runtime grid launch failed because the launch would exceed the limit ::cudaLimitDevRuntimePendingLaunchCount. For this launch to proceed successfully, ::cudaDeviceSetLimit must be called to set the ::cudaLimitDevRuntimePendingLaunchCount to be higher than the upper bound of outstanding launches that can be issued to the device runtime. Keep in mind that raising the limit of pending device runtime launches will require the runtime to reserve device memory that cannot be used for user allocations.");
    case RGY_ERR_cudaErrorInvalidDeviceFunction: return _T("cudaErrorInvalidDeviceFunction: The requested device function does not exist or is not compiled for the proper device architecture.");
    case RGY_ERR_cudaErrorNoDevice: return _T("cudaErrorNoDevice: This indicates that no CUDA-capable devices were detected by the installed CUDA driver.");
    case RGY_ERR_cudaErrorInvalidDevice: return _T("cudaErrorInvalidDevice: This indicates that the device ordinal supplied by the user does not correspond to a valid CUDA device or that the action requested is invalid for the specified device.");
    case RGY_ERR_cudaErrorDeviceNotLicensed: return _T("cudaErrorDeviceNotLicensed: This indicates that the device doesn't have a valid Grid License.");
    case RGY_ERR_cudaErrorSoftwareValidityNotEstablished: return _T("cudaErrorSoftwareValidityNotEstablished: By default, the CUDA runtime may perform a minimal set of self-tests, as well as CUDA driver tests, to establish the validity of both. Introduced in CUDA 11.2, this error return indicates that at least one of these tests has failed and the validity of either the runtime or the driver could not be established.");
    case RGY_ERR_cudaErrorStartupFailure: return _T("cudaErrorStartupFailure: This indicates an internal startup failure in the CUDA runtime.");
    case RGY_ERR_cudaErrorInvalidKernelImage: return _T("cudaErrorInvalidKernelImage: This indicates that the device kernel image is invalid.");
    case RGY_ERR_cudaErrorDeviceUninitialized: return _T("cudaErrorDeviceUninitialized: This most frequently indicates that there is no context bound to the current thread. This can also be returned if the context passed to an API call is not a valid handle (such as a context that has had ::cuCtxDestroy() invoked on it). This can also be returned if a user mixes different API versions (i.e. 3010 context with 3020 API calls). See ::cuCtxGetApiVersion() for more details.");
    case RGY_ERR_cudaErrorMapBufferObjectFailed: return _T("cudaErrorMapBufferObjectFailed: This indicates that the buffer object could not be mapped.");
    case RGY_ERR_cudaErrorUnmapBufferObjectFailed: return _T("cudaErrorUnmapBufferObjectFailed: This indicates that the buffer object could not be unmapped.");
    case RGY_ERR_cudaErrorArrayIsMapped: return _T("cudaErrorArrayIsMapped: This indicates that the specified array is currently mapped and thus cannot be destroyed.");
    case RGY_ERR_cudaErrorAlreadyMapped: return _T("cudaErrorAlreadyMapped: This indicates that the resource is already mapped.");
    case RGY_ERR_cudaErrorNoKernelImageForDevice: return _T("cudaErrorNoKernelImageForDevice: This indicates that there is no kernel image available that is suitable for the device. This can occur when a user specifies code generation options for a particular CUDA source file that do not include the corresponding device configuration.");
    case RGY_ERR_cudaErrorAlreadyAcquired: return _T("cudaErrorAlreadyAcquired: This indicates that a resource has already been acquired.");
    case RGY_ERR_cudaErrorNotMapped: return _T("cudaErrorNotMapped: This indicates that a resource is not mapped.");
    case RGY_ERR_cudaErrorNotMappedAsArray: return _T("cudaErrorNotMappedAsArray: This indicates that a mapped resource is not available for access as an array.");
    case RGY_ERR_cudaErrorNotMappedAsPointer: return _T("cudaErrorNotMappedAsPointer: This indicates that a mapped resource is not available for access as a pointer.");
    case RGY_ERR_cudaErrorECCUncorrectable: return _T("cudaErrorECCUncorrectable: This indicates that an uncorrectable ECC error was detected during execution.");
    case RGY_ERR_cudaErrorUnsupportedLimit: return _T("cudaErrorUnsupportedLimit: This indicates that the ::cudaLimit passed to the API call is not supported by the active device.");
    case RGY_ERR_cudaErrorDeviceAlreadyInUse: return _T("cudaErrorDeviceAlreadyInUse: This indicates that a call tried to access an exclusive-thread device that is already in use by a different thread.");
    case RGY_ERR_cudaErrorPeerAccessUnsupported: return _T("cudaErrorPeerAccessUnsupported: This error indicates that P2P access is not supported across the given devices.");
    case RGY_ERR_cudaErrorInvalidPtx: return _T("cudaErrorInvalidPtx: A PTX compilation failed. The runtime may fall back to compiling PTX if an application does not contain a suitable binary for the current device.");
    case RGY_ERR_cudaErrorInvalidGraphicsContext: return _T("cudaErrorInvalidGraphicsContext: This indicates an error with the OpenGL or DirectX context.");
    case RGY_ERR_cudaErrorNvlinkUncorrectable: return _T("cudaErrorNvlinkUncorrectable: This indicates that an uncorrectable NVLink error was detected during the execution.");
    case RGY_ERR_cudaErrorJitCompilerNotFound: return _T("cudaErrorJitCompilerNotFound: This indicates that the PTX JIT compiler library was not found. The JIT Compiler library is used for PTX compilation. The runtime may fall back to compiling PTX if an application does not contain a suitable binary for the current device.");
    case RGY_ERR_cudaErrorUnsupportedPtxVersion: return _T("cudaErrorUnsupportedPtxVersion: This indicates that the provided PTX was compiled with an unsupported toolchain. The most common reason for this, is the PTX was generated by a compiler newer than what is supported by the CUDA driver and PTX JIT compiler.");
    case RGY_ERR_cudaErrorJitCompilationDisabled: return _T("cudaErrorJitCompilationDisabled: This indicates that the JIT compilation was disabled. The JIT compilation compiles PTX. The runtime may fall back to compiling PTX if an application does not contain a suitable binary for the current device.");
    case RGY_ERR_cudaErrorUnsupportedExecAffinity: return _T("cudaErrorUnsupportedExecAffinity: This indicates that the provided execution affinity is not supported by the device.");
    case RGY_ERR_cudaErrorInvalidSource: return _T("cudaErrorInvalidSource: This indicates that the device kernel source is invalid.");
    case RGY_ERR_cudaErrorFileNotFound: return _T("cudaErrorFileNotFound: This indicates that the file specified was not found.");
    case RGY_ERR_cudaErrorSharedObjectSymbolNotFound: return _T("cudaErrorSharedObjectSymbolNotFound: This indicates that a link to a shared object failed to resolve.");
    case RGY_ERR_cudaErrorSharedObjectInitFailed: return _T("cudaErrorSharedObjectInitFailed: This indicates that initialization of a shared object failed.");
    case RGY_ERR_cudaErrorOperatingSystem: return _T("cudaErrorOperatingSystem: This error indicates that an OS call failed.");
    case RGY_ERR_cudaErrorInvalidResourceHandle: return _T("cudaErrorInvalidResourceHandle: This indicates that a resource handle passed to the API call was not valid. Resource handles are opaque types like ::cudaStream_t and ::cudaEvent_t.");
    case RGY_ERR_cudaErrorIllegalState: return _T("cudaErrorIllegalState: This indicates that a resource required by the API call is not in a valid state to perform the requested operation.");
    case RGY_ERR_cudaErrorSymbolNotFound: return _T("cudaErrorSymbolNotFound: This indicates that a named symbol was not found. Examples of symbols are global/constant variable names, driver function names, texture names, and surface names.");
    case RGY_ERR_cudaErrorNotReady: return _T("cudaErrorNotReady: This indicates that asynchronous operations issued previously have not completed yet. This result is not actually an error, but must be indicated differently than ::cudaSuccess (which indicates completion). Calls that may return this value include ::cudaEventQuery() and ::cudaStreamQuery().");
    case RGY_ERR_cudaErrorIllegalAddress: return _T("cudaErrorIllegalAddress: The device encountered a load or store instruction on an invalid memory address. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.");
    case RGY_ERR_cudaErrorLaunchOutOfResources: return _T("cudaErrorLaunchOutOfResources: This indicates that a launch did not occur because it did not have appropriate resources. Although this error is similar to ::cudaErrorInvalidConfiguration, this error usually indicates that the user has attempted to pass too many arguments to the device kernel, or the kernel launch specifies too many threads for the kernel's register count.");
    case RGY_ERR_cudaErrorLaunchTimeout: return _T("cudaErrorLaunchTimeout: This indicates that the device kernel took too long to execute. This can only occur if timeouts are enabled - see the device property ::cudaDeviceProp::kernelExecTimeoutEnabled for more information. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.");
    case RGY_ERR_cudaErrorLaunchIncompatibleTexturing: return _T("cudaErrorLaunchIncompatibleTexturing: This error indicates a kernel launch that uses an incompatible texturing mode.");
    case RGY_ERR_cudaErrorPeerAccessAlreadyEnabled: return _T("cudaErrorPeerAccessAlreadyEnabled: This error indicates that a call to ::cudaDeviceEnablePeerAccess() is trying to re-enable peer addressing on from a context which has already had peer addressing enabled.");
    case RGY_ERR_cudaErrorPeerAccessNotEnabled: return _T("cudaErrorPeerAccessNotEnabled: This error indicates that ::cudaDeviceDisablePeerAccess() is trying to disable peer addressing which has not been enabled yet via ::cudaDeviceEnablePeerAccess().");
    case RGY_ERR_cudaErrorSetOnActiveProcess: return _T("cudaErrorSetOnActiveProcess: This indicates that the user has called ::cudaSetValidDevices(), ::cudaSetDeviceFlags(), ::cudaD3D9SetDirect3DDevice(), ::cudaD3D10SetDirect3DDevice, ::cudaD3D11SetDirect3DDevice(), or ::cudaVDPAUSetVDPAUDevice() after initializing the CUDA runtime by calling non-device management operations (allocating memory and launching kernels are examples of non-device management operations). This error can also be returned if using runtime/driver interoperability and there is an existing ::CUcontext active on the host thread.");
    case RGY_ERR_cudaErrorContextIsDestroyed: return _T("cudaErrorContextIsDestroyed: This error indicates that the context current to the calling thread has been destroyed using ::cuCtxDestroy, or is a primary context which has not yet been initialized.");
    case RGY_ERR_cudaErrorAssert: return _T("cudaErrorAssert: An assert triggered in device code during kernel execution. The device cannot be used again. All existing allocations are invalid. To continue using CUDA, the process must be terminated and relaunched.");
    case RGY_ERR_cudaErrorTooManyPeers: return _T("cudaErrorTooManyPeers: This error indicates that the hardware resources required to enable peer access have been exhausted for one or more of the devices passed to ::cudaEnablePeerAccess().");
    case RGY_ERR_cudaErrorHostMemoryAlreadyRegistered: return _T("cudaErrorHostMemoryAlreadyRegistered: This error indicates that the memory range passed to ::cudaHostRegister() has already been registered.");
    case RGY_ERR_cudaErrorHostMemoryNotRegistered: return _T("cudaErrorHostMemoryNotRegistered: This error indicates that the pointer passed to ::cudaHostUnregister() does not correspond to any currently registered memory region.");
    case RGY_ERR_cudaErrorHardwareStackError: return _T("cudaErrorHardwareStackError: Device encountered an error in the call stack during kernel execution, possibly due to stack corruption or exceeding the stack size limit. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.");
    case RGY_ERR_cudaErrorIllegalInstruction: return _T("cudaErrorIllegalInstruction: The device encountered an illegal instruction during kernel execution This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.");
    case RGY_ERR_cudaErrorMisalignedAddress: return _T("cudaErrorMisalignedAddress: The device encountered a load or store instruction on a memory address which is not aligned. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.");
    case RGY_ERR_cudaErrorInvalidAddressSpace: return _T("cudaErrorInvalidAddressSpace: While executing a kernel, the device encountered an instruction which can only operate on memory locations in certain address spaces (global, shared, or local), but was supplied a memory address not belonging to an allowed address space. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.");
    case RGY_ERR_cudaErrorInvalidPc: return _T("cudaErrorInvalidPc: The device encountered an invalid program counter. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.");
    case RGY_ERR_cudaErrorLaunchFailure: return _T("cudaErrorLaunchFailure: An exception occurred on the device while executing a kernel. Common causes include dereferencing an invalid device pointer and accessing out of bounds shared memory. Less common cases can be system specific - more information about these cases can be found in the system specific user guide. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.");
    case RGY_ERR_cudaErrorCooperativeLaunchTooLarge: return _T("cudaErrorCooperativeLaunchTooLarge: This error indicates that the number of blocks launched per grid for a kernel that was launched via either ::cudaLaunchCooperativeKernel or ::cudaLaunchCooperativeKernelMultiDevice exceeds the maximum number of blocks as allowed by ::cudaOccupancyMaxActiveBlocksPerMultiprocessor or ::cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags times the number of multiprocessors as specified by the device attribute ::cudaDevAttrMultiProcessorCount.");
    case RGY_ERR_cudaErrorNotPermitted: return _T("cudaErrorNotPermitted: This error indicates the attempted operation is not permitted.");
    case RGY_ERR_cudaErrorNotSupported: return _T("cudaErrorNotSupported: This error indicates the attempted operation is not supported on the current system or device.");
    case RGY_ERR_cudaErrorSystemNotReady: return _T("cudaErrorSystemNotReady: This error indicates that the system is not yet ready to start any CUDA work.  To continue using CUDA, verify the system configuration is in a valid state and all required driver daemons are actively running. More information about this error can be found in the system specific user guide.");
    case RGY_ERR_cudaErrorSystemDriverMismatch: return _T("cudaErrorSystemDriverMismatch: This error indicates that there is a mismatch between the versions of the display driver and the CUDA driver. Refer to the compatibility documentation for supported versions.");
    case RGY_ERR_cudaErrorCompatNotSupportedOnDevice: return _T("cudaErrorCompatNotSupportedOnDevice: This error indicates that the system was upgraded to run with forward compatibility but the visible hardware detected by CUDA does not support this configuration. Refer to the compatibility documentation for the supported hardware matrix or ensure that only supported hardware is visible during initialization via the CUDA_VISIBLE_DEVICES environment variable.");
    case RGY_ERR_cudaErrorMpsConnectionFailed: return _T("cudaErrorMpsConnectionFailed: This error indicates that the MPS client failed to connect to the MPS control daemon or the MPS server.");
    case RGY_ERR_cudaErrorMpsRpcFailure: return _T("cudaErrorMpsRpcFailure: This error indicates that the remote procedural call between the MPS server and the MPS client failed.");
    case RGY_ERR_cudaErrorMpsServerNotReady: return _T("cudaErrorMpsServerNotReady: This error indicates that the MPS server is not ready to accept new MPS client requests. This error can be returned when the MPS server is in the process of recovering from a fatal failure.");
    case RGY_ERR_cudaErrorMpsMaxClientsReached: return _T("cudaErrorMpsMaxClientsReached: This error indicates that the hardware resources required to create MPS client have been exhausted.");
    case RGY_ERR_cudaErrorMpsMaxConnectionsReached: return _T("cudaErrorMpsMaxConnectionsReached: This error indicates the the hardware resources required to device connections have been exhausted.");
    case RGY_ERR_cudaErrorStreamCaptureUnsupported: return _T("cudaErrorStreamCaptureUnsupported: The operation is not permitted when the stream is capturing.");
    case RGY_ERR_cudaErrorStreamCaptureInvalidated: return _T("cudaErrorStreamCaptureInvalidated: The current capture sequence on the stream has been invalidated due to a previous error.");
    case RGY_ERR_cudaErrorStreamCaptureMerge: return _T("cudaErrorStreamCaptureMerge: The operation would have resulted in a merge of two independent capture sequences.");
    case RGY_ERR_cudaErrorStreamCaptureUnmatched: return _T("cudaErrorStreamCaptureUnmatched: The capture was not initiated in this stream.");
    case RGY_ERR_cudaErrorStreamCaptureUnjoined: return _T("cudaErrorStreamCaptureUnjoined: The capture sequence contains a fork that was not joined to the primary stream.");
    case RGY_ERR_cudaErrorStreamCaptureIsolation: return _T("cudaErrorStreamCaptureIsolation: A dependency would have been created which crosses the capture sequence boundary. Only implicit in-stream ordering dependencies are allowed to cross the boundary.");
    case RGY_ERR_cudaErrorStreamCaptureImplicit: return _T("cudaErrorStreamCaptureImplicit: The operation would have resulted in a disallowed implicit dependency on a current capture sequence from cudaStreamLegacy.");
    case RGY_ERR_cudaErrorCapturedEvent: return _T("cudaErrorCapturedEvent: The operation is not permitted on an event which was last recorded in a capturing stream.");
    case RGY_ERR_cudaErrorStreamCaptureWrongThread: return _T("cudaErrorStreamCaptureWrongThread: A stream capture sequence not initiated with the ::cudaStreamCaptureModeRelaxed argument to ::cudaStreamBeginCapture was passed to ::cudaStreamEndCapture in a different thread.");
    case RGY_ERR_cudaErrorTimeout: return _T("cudaErrorTimeout: This indicates that the wait operation has timed out.");
    case RGY_ERR_cudaErrorGraphExecUpdateFailure: return _T("cudaErrorGraphExecUpdateFailure: This error indicates that the graph update was not performed because it included changes which violated constraints specific to instantiated graph update.");
    case RGY_ERR_cudaErrorExternalDevice: return _T("cudaErrorExternalDevice: This indicates that an async error has occurred in a device outside of CUDA. If CUDA was waiting for an external device's signal before consuming shared data, the external device signaled an error indicating that the data is not valid for consumption. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.");
    case RGY_ERR_cudaErrorUnknown: return _T("cudaErrorUnknown: This indicates that an unknown internal error has occurred.");
    case RGY_ERR_cudaErrorApiFailureBase: return _T("cudaErrorApiFailureBase: Any unhandled CUDA driver error is added to this value and returned via the runtime. Production releases of CUDA should not return such errors. This error return is deprecated as of CUDA 4.1.");

    case RGY_ERR_CUDA_ERROR_INVALID_VALUE: return _T("CUDA_ERROR_INVALID_VALUE: This indicates that one or more of the parameters passed to the API call is not within an acceptable range of values.");
    case RGY_ERR_CUDA_ERROR_OUT_OF_MEMORY: return _T("CUDA_ERROR_OUT_OF_MEMORY: The API call failed because it was unable to allocate enough memory to perform the requested operation.");
    case RGY_ERR_CUDA_ERROR_NOT_INITIALIZED: return _T("CUDA_ERROR_NOT_INITIALIZED: This indicates that the CUDA driver has not been initialized with ::cuInit() or that initialization has failed.");
    case RGY_ERR_CUDA_ERROR_DEINITIALIZED: return _T("CUDA_ERROR_DEINITIALIZED: This indicates that the CUDA driver is in the process of shutting down.");
    case RGY_ERR_CUDA_ERROR_PROFILER_DISABLED: return _T("CUDA_ERROR_PROFILER_DISABLED: This indicates profiler is not initialized for this run. This can happen when the application is running with external profiling tools like visual profiler.");
    case RGY_ERR_CUDA_ERROR_PROFILER_NOT_INITIALIZED: return _T("CUDA_ERROR_PROFILER_NOT_INITIALIZED: This error return is deprecated as of CUDA 5.0. It is no longer an error to attempt to enable/disable the profiling via ::cuProfilerStart or ::cuProfilerStop without initialization.");
    case RGY_ERR_CUDA_ERROR_PROFILER_ALREADY_STARTED: return _T("CUDA_ERROR_PROFILER_ALREADY_STARTED: This error return is deprecated as of CUDA 5.0. It is no longer an error to call cuProfilerStart() when profiling is already enabled.");
    case RGY_ERR_CUDA_ERROR_PROFILER_ALREADY_STOPPED: return _T("CUDA_ERROR_PROFILER_ALREADY_STOPPED: This error return is deprecated as of CUDA 5.0. It is no longer an error to call cuProfilerStop() when profiling is already disabled.");
    case RGY_ERR_CUDA_ERROR_STUB_LIBRARY: return _T("CUDA_ERROR_STUB_LIBRARY: This indicates that the CUDA driver that the application has loaded is a stub library. Applications that run with the stub rather than a real driver loaded will result in CUDA API returning this error.");
    case RGY_ERR_CUDA_ERROR_DEVICE_UNAVAILABLE: return _T("CUDA_ERROR_DEVICE_UNAVAILABLE: This indicates that requested CUDA device is unavailable at the current time. Devices are often unavailable due to use of ::CU_COMPUTEMODE_EXCLUSIVE_PROCESS or ::CU_COMPUTEMODE_PROHIBITED.");
    case RGY_ERR_CUDA_ERROR_NO_DEVICE: return _T("CUDA_ERROR_NO_DEVICE: This indicates that no CUDA-capable devices were detected by the installed CUDA driver.");
    case RGY_ERR_CUDA_ERROR_INVALID_DEVICE: return _T("CUDA_ERROR_INVALID_DEVICE: This indicates that the device ordinal supplied by the user does not correspond to a valid CUDA device or that the action requested is invalid for the specified device.");
    case RGY_ERR_CUDA_ERROR_DEVICE_NOT_LICENSED: return _T("CUDA_ERROR_DEVICE_NOT_LICENSED: This error indicates that the Grid license is not applied.");
    case RGY_ERR_CUDA_ERROR_INVALID_IMAGE: return _T("CUDA_ERROR_INVALID_IMAGE: This indicates that the device kernel image is invalid. This can also indicate an invalid CUDA module.");
    case RGY_ERR_CUDA_ERROR_INVALID_CONTEXT: return _T("CUDA_ERROR_INVALID_CONTEXT: This most frequently indicates that there is no context bound to the current thread. This can also be returned if the context passed to an API call is not a valid handle (such as a context that has had ::cuCtxDestroy() invoked on it). This can also be returned if a user mixes different API versions (i.e. 3010 context with 3020 API calls). See ::cuCtxGetApiVersion() for more details.");
    case RGY_ERR_CUDA_ERROR_CONTEXT_ALREADY_CURRENT: return _T("CUDA_ERROR_CONTEXT_ALREADY_CURRENT: This indicated that the context being supplied as a parameter to the API call was already the active context. This error return is deprecated as of CUDA 3.2. It is no longer an error to attempt to push the active context via ::cuCtxPushCurrent().");
    case RGY_ERR_CUDA_ERROR_MAP_FAILED: return _T("CUDA_ERROR_MAP_FAILED: This indicates that a map or register operation has failed.");
    case RGY_ERR_CUDA_ERROR_UNMAP_FAILED: return _T("CUDA_ERROR_UNMAP_FAILED: This indicates that an unmap or unregister operation has failed.");
    case RGY_ERR_CUDA_ERROR_ARRAY_IS_MAPPED: return _T("CUDA_ERROR_ARRAY_IS_MAPPED: This indicates that the specified array is currently mapped and thus cannot be destroyed.");
    case RGY_ERR_CUDA_ERROR_ALREADY_MAPPED: return _T("CUDA_ERROR_ALREADY_MAPPED: This indicates that the resource is already mapped.");
    case RGY_ERR_CUDA_ERROR_NO_BINARY_FOR_GPU: return _T("CUDA_ERROR_NO_BINARY_FOR_GPU: This indicates that there is no kernel image available that is suitable for the device. This can occur when a user specifies code generation options for a particular CUDA source file that do not include the corresponding device configuration.");
    case RGY_ERR_CUDA_ERROR_ALREADY_ACQUIRED: return _T("CUDA_ERROR_ALREADY_ACQUIRED: This indicates that a resource has already been acquired.");
    case RGY_ERR_CUDA_ERROR_NOT_MAPPED: return _T("CUDA_ERROR_NOT_MAPPED: This indicates that a resource is not mapped.");
    case RGY_ERR_CUDA_ERROR_NOT_MAPPED_AS_ARRAY: return _T("CUDA_ERROR_NOT_MAPPED_AS_ARRAY: This indicates that a mapped resource is not available for access as an array.");
    case RGY_ERR_CUDA_ERROR_NOT_MAPPED_AS_POINTER: return _T("CUDA_ERROR_NOT_MAPPED_AS_POINTER: This indicates that a mapped resource is not available for access as a pointer.");
    case RGY_ERR_CUDA_ERROR_ECC_UNCORRECTABLE: return _T("CUDA_ERROR_ECC_UNCORRECTABLE: This indicates that an uncorrectable ECC error was detected during execution.");
    case RGY_ERR_CUDA_ERROR_UNSUPPORTED_LIMIT: return _T("CUDA_ERROR_UNSUPPORTED_LIMIT: This indicates that the ::CUlimit passed to the API call is not supported by the active device.");
    case RGY_ERR_CUDA_ERROR_CONTEXT_ALREADY_IN_USE: return _T("CUDA_ERROR_CONTEXT_ALREADY_IN_USE: This indicates that the ::CUcontext passed to the API call can only be bound to a single CPU thread at a time but is already bound to a CPU thread.");
    case RGY_ERR_CUDA_ERROR_PEER_ACCESS_UNSUPPORTED: return _T("CUDA_ERROR_PEER_ACCESS_UNSUPPORTED: This indicates that peer access is not supported across the given devices.");
    case RGY_ERR_CUDA_ERROR_INVALID_PTX: return _T("CUDA_ERROR_INVALID_PTX: This indicates that a PTX JIT compilation failed.");
    case RGY_ERR_CUDA_ERROR_INVALID_GRAPHICS_CONTEXT: return _T("CUDA_ERROR_INVALID_GRAPHICS_CONTEXT: This indicates an error with OpenGL or DirectX context.");
    case RGY_ERR_CUDA_ERROR_NVLINK_UNCORRECTABLE: return _T("CUDA_ERROR_NVLINK_UNCORRECTABLE: This indicates that an uncorrectable NVLink error was detected during the execution.");
    case RGY_ERR_CUDA_ERROR_JIT_COMPILER_NOT_FOUND: return _T("CUDA_ERROR_JIT_COMPILER_NOT_FOUND: This indicates that the PTX JIT compiler library was not found.");
    case RGY_ERR_CUDA_ERROR_UNSUPPORTED_PTX_VERSION: return _T("CUDA_ERROR_UNSUPPORTED_PTX_VERSION: This indicates that the provided PTX was compiled with an unsupported toolchain.");
    case RGY_ERR_CUDA_ERROR_JIT_COMPILATION_DISABLED: return _T("CUDA_ERROR_JIT_COMPILATION_DISABLED: This indicates that the PTX JIT compilation was disabled.");
    case RGY_ERR_CUDA_ERROR_UNSUPPORTED_EXEC_AFFINITY: return _T("CUDA_ERROR_UNSUPPORTED_EXEC_AFFINITY: This indicates that the ::CUexecAffinityType passed to the API call is not supported by the active device.");
    case RGY_ERR_CUDA_ERROR_INVALID_SOURCE: return _T("CUDA_ERROR_INVALID_SOURCE: This indicates that the device kernel source is invalid. This includes compilation/linker errors encountered in device code or user error.");
    case RGY_ERR_CUDA_ERROR_FILE_NOT_FOUND: return _T("CUDA_ERROR_FILE_NOT_FOUND: This indicates that the file specified was not found.");
    case RGY_ERR_CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND: return _T("CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND: This indicates that a link to a shared object failed to resolve.");
    case RGY_ERR_CUDA_ERROR_SHARED_OBJECT_INIT_FAILED: return _T("CUDA_ERROR_SHARED_OBJECT_INIT_FAILED: This indicates that initialization of a shared object failed.");
    case RGY_ERR_CUDA_ERROR_OPERATING_SYSTEM: return _T("CUDA_ERROR_OPERATING_SYSTEM: This indicates that an OS call failed.");
    case RGY_ERR_CUDA_ERROR_INVALID_HANDLE: return _T("CUDA_ERROR_INVALID_HANDLE: This indicates that a resource handle passed to the API call was not valid. Resource handles are opaque types like ::CUstream and ::CUevent.");
    case RGY_ERR_CUDA_ERROR_ILLEGAL_STATE: return _T("CUDA_ERROR_ILLEGAL_STATE: This indicates that a resource required by the API call is not in a valid state to perform the requested operation.");
    case RGY_ERR_CUDA_ERROR_NOT_FOUND: return _T("CUDA_ERROR_NOT_FOUND: This indicates that a named symbol was not found. Examples of symbols are global/constant variable names, driver function names, texture names, and surface names.");
    case RGY_ERR_CUDA_ERROR_NOT_READY: return _T("CUDA_ERROR_NOT_READY: This indicates that asynchronous operations issued previously have not completed yet. This result is not actually an error, but must be indicated differently than ::CUDA_SUCCESS (which indicates completion). Calls that may return this value include ::cuEventQuery() and ::cuStreamQuery().");
    case RGY_ERR_CUDA_ERROR_ILLEGAL_ADDRESS: return _T("CUDA_ERROR_ILLEGAL_ADDRESS: While executing a kernel, the device encountered a load or store instruction on an invalid memory address. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.");
    case RGY_ERR_CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES: return _T("CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES: This indicates that a launch did not occur because it did not have appropriate resources. This error usually indicates that the user has attempted to pass too many arguments to the device kernel, or the kernel launch specifies too many threads for the kernel's register count. Passing arguments of the wrong size (i.e. a 64-bit pointer when a 32-bit int is expected) is equivalent to passing too many arguments and can also result in this error.");
    case RGY_ERR_CUDA_ERROR_LAUNCH_TIMEOUT: return _T("CUDA_ERROR_LAUNCH_TIMEOUT: This indicates that the device kernel took too long to execute. This can only occur if timeouts are enabled - see the device attribute ::CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT for more information. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.");
    case RGY_ERR_CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING: return _T("CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING: This error indicates a kernel launch that uses an incompatible texturing mode.");
    case RGY_ERR_CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED: return _T("CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED: This error indicates that a call to ::cuCtxEnablePeerAccess() is trying to re-enable peer access to a context which has already had peer access to it enabled.");
    case RGY_ERR_CUDA_ERROR_PEER_ACCESS_NOT_ENABLED: return _T("CUDA_ERROR_PEER_ACCESS_NOT_ENABLED: This error indicates that ::cuCtxDisablePeerAccess() is trying to disable peer access which has not been enabled yet via ::cuCtxEnablePeerAccess().");
    case RGY_ERR_CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE: return _T("CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE: This error indicates that the primary context for the specified device has already been initialized.");
    case RGY_ERR_CUDA_ERROR_CONTEXT_IS_DESTROYED: return _T("CUDA_ERROR_CONTEXT_IS_DESTROYED: This error indicates that the context current to the calling thread has been destroyed using ::cuCtxDestroy, or is a primary context which has not yet been initialized.");
    case RGY_ERR_CUDA_ERROR_ASSERT: return _T("CUDA_ERROR_ASSERT: A device-side assert triggered during kernel execution. The context cannot be used anymore, and must be destroyed. All existing device memory allocations from this context are invalid and must be reconstructed if the program is to continue using CUDA.");
    case RGY_ERR_CUDA_ERROR_TOO_MANY_PEERS: return _T("CUDA_ERROR_TOO_MANY_PEERS: This error indicates that the hardware resources required to enable peer access have been exhausted for one or more of the devices passed to ::cuCtxEnablePeerAccess().");
    case RGY_ERR_CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED: return _T("CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED: This error indicates that the memory range passed to ::cuMemHostRegister() has already been registered.");
    case RGY_ERR_CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED: return _T("CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED: This error indicates that the pointer passed to ::cuMemHostUnregister() does not correspond to any currently registered memory region.");
    case RGY_ERR_CUDA_ERROR_HARDWARE_STACK_ERROR: return _T("CUDA_ERROR_HARDWARE_STACK_ERROR: While executing a kernel, the device encountered a stack error. This can be due to stack corruption or exceeding the stack size limit. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.");
    case RGY_ERR_CUDA_ERROR_ILLEGAL_INSTRUCTION: return _T("CUDA_ERROR_ILLEGAL_INSTRUCTION: While executing a kernel, the device encountered an illegal instruction. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.");
    case RGY_ERR_CUDA_ERROR_MISALIGNED_ADDRESS: return _T("CUDA_ERROR_MISALIGNED_ADDRESS: While executing a kernel, the device encountered a load or store instruction on a memory address which is not aligned. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.");
    case RGY_ERR_CUDA_ERROR_INVALID_ADDRESS_SPACE: return _T("CUDA_ERROR_INVALID_ADDRESS_SPACE: While executing a kernel, the device encountered an instruction which can only operate on memory locations in certain address spaces (global, shared, or local), but was supplied a memory address not belonging to an allowed address space. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.");
    case RGY_ERR_CUDA_ERROR_INVALID_PC: return _T("CUDA_ERROR_INVALID_PC: While executing a kernel, the device program counter wrapped its address space. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.");
    case RGY_ERR_CUDA_ERROR_LAUNCH_FAILED: return _T("CUDA_ERROR_LAUNCH_FAILED: An exception occurred on the device while executing a kernel. Common causes include dereferencing an invalid device pointer and accessing out of bounds shared memory. Less common cases can be system specific - more information about these cases can be found in the system specific user guide. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.");
    case RGY_ERR_CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE: return _T("CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE: This error indicates that the number of blocks launched per grid for a kernel that was launched via either ::cuLaunchCooperativeKernel or ::cuLaunchCooperativeKernelMultiDevice exceeds the maximum number of blocks as allowed by ::cuOccupancyMaxActiveBlocksPerMultiprocessor or ::cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags times the number of multiprocessors as specified by the device attribute ::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT.");
    case RGY_ERR_CUDA_ERROR_NOT_PERMITTED: return _T("CUDA_ERROR_NOT_PERMITTED: This error indicates that the attempted operation is not permitted.");
    case RGY_ERR_CUDA_ERROR_NOT_SUPPORTED: return _T("CUDA_ERROR_NOT_SUPPORTED: This error indicates that the attempted operation is not supported on the current system or device.");
    case RGY_ERR_CUDA_ERROR_SYSTEM_NOT_READY: return _T("CUDA_ERROR_SYSTEM_NOT_READY: This error indicates that the system is not yet ready to start any CUDA work.  To continue using CUDA, verify the system configuration is in a valid state and all required driver daemons are actively running. More information about this error can be found in the system specific user guide.");
    case RGY_ERR_CUDA_ERROR_SYSTEM_DRIVER_MISMATCH: return _T("CUDA_ERROR_SYSTEM_DRIVER_MISMATCH: This error indicates that there is a mismatch between the versions of the display driver and the CUDA driver. Refer to the compatibility documentation for supported versions.");
    case RGY_ERR_CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE: return _T("CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE: This error indicates that the system was upgraded to run with forward compatibility but the visible hardware detected by CUDA does not support this configuration. Refer to the compatibility documentation for the supported hardware matrix or ensure that only supported hardware is visible during initialization via the CUDA_VISIBLE_DEVICES environment variable.");
    case RGY_ERR_CUDA_ERROR_MPS_CONNECTION_FAILED: return _T("CUDA_ERROR_MPS_CONNECTION_FAILED: This error indicates that the MPS client failed to connect to the MPS control daemon or the MPS server.");
    case RGY_ERR_CUDA_ERROR_MPS_RPC_FAILURE: return _T("CUDA_ERROR_MPS_RPC_FAILURE: This error indicates that the remote procedural call between the MPS server and the MPS client failed.");
    case RGY_ERR_CUDA_ERROR_MPS_SERVER_NOT_READY: return _T("CUDA_ERROR_MPS_SERVER_NOT_READY: This error indicates that the MPS server is not ready to accept new MPS client requests. This error can be returned when the MPS server is in the process of recovering from a fatal failure.");
    case RGY_ERR_CUDA_ERROR_MPS_MAX_CLIENTS_REACHED: return _T("CUDA_ERROR_MPS_MAX_CLIENTS_REACHED: This error indicates that the hardware resources required to create MPS client have been exhausted.");
    case RGY_ERR_CUDA_ERROR_MPS_MAX_CONNECTIONS_REACHED: return _T("CUDA_ERROR_MPS_MAX_CONNECTIONS_REACHED: This error indicates the the hardware resources required to support device connections have been exhausted.");
    case RGY_ERR_CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED: return _T("CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED: This error indicates that the operation is not permitted when the stream is capturing.");
    case RGY_ERR_CUDA_ERROR_STREAM_CAPTURE_INVALIDATED: return _T("CUDA_ERROR_STREAM_CAPTURE_INVALIDATED: This error indicates that the current capture sequence on the stream has been invalidated due to a previous error.");
    case RGY_ERR_CUDA_ERROR_STREAM_CAPTURE_MERGE: return _T("CUDA_ERROR_STREAM_CAPTURE_MERGE: This error indicates that the operation would have resulted in a merge of two independent capture sequences.");
    case RGY_ERR_CUDA_ERROR_STREAM_CAPTURE_UNMATCHED: return _T("CUDA_ERROR_STREAM_CAPTURE_UNMATCHED: This error indicates that the capture was not initiated in this stream.");
    case RGY_ERR_CUDA_ERROR_STREAM_CAPTURE_UNJOINED: return _T("CUDA_ERROR_STREAM_CAPTURE_UNJOINED: This error indicates that the capture sequence contains a fork that was not joined to the primary stream.");
    case RGY_ERR_CUDA_ERROR_STREAM_CAPTURE_ISOLATION: return _T("CUDA_ERROR_STREAM_CAPTURE_ISOLATION: This error indicates that a dependency would have been created which crosses the capture sequence boundary. Only implicit in-stream ordering dependencies are allowed to cross the boundary.");
    case RGY_ERR_CUDA_ERROR_STREAM_CAPTURE_IMPLICIT: return _T("CUDA_ERROR_STREAM_CAPTURE_IMPLICIT: This error indicates a disallowed implicit dependency on a current capture sequence from cudaStreamLegacy.");
    case RGY_ERR_CUDA_ERROR_CAPTURED_EVENT: return _T("CUDA_ERROR_CAPTURED_EVENT: This error indicates that the operation is not permitted on an event which was last recorded in a capturing stream.");
    case RGY_ERR_CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD: return _T("CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD: A stream capture sequence not initiated with the ::CU_STREAM_CAPTURE_MODE_RELAXED argument to ::cuStreamBeginCapture was passed to ::cuStreamEndCapture in a different thread.");
    case RGY_ERR_CUDA_ERROR_TIMEOUT: return _T("CUDA_ERROR_TIMEOUT: This error indicates that the timeout specified for the wait operation has lapsed.");
    case RGY_ERR_CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE: return _T("CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE: This error indicates that the graph update was not performed because it included changes which violated constraints specific to instantiated graph update.");
    case RGY_ERR_CUDA_ERROR_EXTERNAL_DEVICE: return _T("CUDA_ERROR_EXTERNAL_DEVICE: This indicates that an async error has occurred in a device outside of CUDA. If CUDA was waiting for an external device's signal before consuming shared data, the external device signaled an error indicating that the data is not valid for consumption. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.");
    case RGY_ERR_CUDA_ERROR_UNKNOWN: return _T("CUDA_ERROR_UNKNOWN: This indicates that an unknown internal error has occurred.");

#define CASE_ERR_NPP(x) case RGY_ERR_ ## x: return _T(#x);
    CASE_ERR_NPP(NPP_NOT_SUPPORTED_MODE_ERROR);
    CASE_ERR_NPP(NPP_INVALID_HOST_POINTER_ERROR);
    CASE_ERR_NPP(NPP_INVALID_DEVICE_POINTER_ERROR);
    CASE_ERR_NPP(NPP_LUT_PALETTE_BITSIZE_ERROR);
    CASE_ERR_NPP(NPP_ZC_MODE_NOT_SUPPORTED_ERROR);
    CASE_ERR_NPP(NPP_NOT_SUFFICIENT_COMPUTE_CAPABILITY);
    CASE_ERR_NPP(NPP_TEXTURE_BIND_ERROR);
    CASE_ERR_NPP(NPP_WRONG_INTERSECTION_ROI_ERROR);
    CASE_ERR_NPP(NPP_HAAR_CLASSIFIER_PIXEL_MATCH_ERROR);
    CASE_ERR_NPP(NPP_MEMFREE_ERROR);
    CASE_ERR_NPP(NPP_MEMSET_ERROR);
    CASE_ERR_NPP(NPP_MEMCPY_ERROR);
    CASE_ERR_NPP(NPP_ALIGNMENT_ERROR);
    CASE_ERR_NPP(NPP_CUDA_KERNEL_EXECUTION_ERROR);
    CASE_ERR_NPP(NPP_ROUND_MODE_NOT_SUPPORTED_ERROR);
    CASE_ERR_NPP(NPP_QUALITY_INDEX_ERROR);
    CASE_ERR_NPP(NPP_RESIZE_NO_OPERATION_ERROR);
    CASE_ERR_NPP(NPP_OVERFLOW_ERROR);
    CASE_ERR_NPP(NPP_NOT_EVEN_STEP_ERROR);
    CASE_ERR_NPP(NPP_HISTOGRAM_NUMBER_OF_LEVELS_ERROR);
    CASE_ERR_NPP(NPP_LUT_NUMBER_OF_LEVELS_ERROR);
    CASE_ERR_NPP(NPP_CORRUPTED_DATA_ERROR);
    CASE_ERR_NPP(NPP_CHANNEL_ORDER_ERROR);
    CASE_ERR_NPP(NPP_ZERO_MASK_VALUE_ERROR);
    CASE_ERR_NPP(NPP_QUADRANGLE_ERROR);
    CASE_ERR_NPP(NPP_RECTANGLE_ERROR);
    CASE_ERR_NPP(NPP_COEFFICIENT_ERROR);
    CASE_ERR_NPP(NPP_NUMBER_OF_CHANNELS_ERROR);
    CASE_ERR_NPP(NPP_COI_ERROR);
    CASE_ERR_NPP(NPP_DIVISOR_ERROR);
    CASE_ERR_NPP(NPP_CHANNEL_ERROR);
    CASE_ERR_NPP(NPP_STRIDE_ERROR);
    CASE_ERR_NPP(NPP_ANCHOR_ERROR);
    CASE_ERR_NPP(NPP_MASK_SIZE_ERROR);
    CASE_ERR_NPP(NPP_RESIZE_FACTOR_ERROR);
    CASE_ERR_NPP(NPP_INTERPOLATION_ERROR);
    CASE_ERR_NPP(NPP_MIRROR_FLIP_ERROR);
    CASE_ERR_NPP(NPP_MOMENT_00_ZERO_ERROR);
    CASE_ERR_NPP(NPP_THRESHOLD_NEGATIVE_LEVEL_ERROR);
    CASE_ERR_NPP(NPP_THRESHOLD_ERROR);
    CASE_ERR_NPP(NPP_CONTEXT_MATCH_ERROR);
    CASE_ERR_NPP(NPP_FFT_FLAG_ERROR);
    CASE_ERR_NPP(NPP_FFT_ORDER_ERROR);
    CASE_ERR_NPP(NPP_STEP_ERROR);
    CASE_ERR_NPP(NPP_SCALE_RANGE_ERROR);
    CASE_ERR_NPP(NPP_DATA_TYPE_ERROR);
    CASE_ERR_NPP(NPP_OUT_OFF_RANGE_ERROR);
    CASE_ERR_NPP(NPP_DIVIDE_BY_ZERO_ERROR);
    CASE_ERR_NPP(NPP_MEMORY_ALLOCATION_ERR);
    CASE_ERR_NPP(NPP_NULL_POINTER_ERROR);
    CASE_ERR_NPP(NPP_RANGE_ERROR);
    CASE_ERR_NPP(NPP_SIZE_ERROR);
    CASE_ERR_NPP(NPP_BAD_ARGUMENT_ERROR);
    CASE_ERR_NPP(NPP_NO_MEMORY_ERROR);
    CASE_ERR_NPP(NPP_NOT_IMPLEMENTED_ERROR);
    CASE_ERR_NPP(NPP_ERROR);
    CASE_ERR_NPP(NPP_ERROR_RESERVED);
    //CASE_ERR_NPP(NPP_NO_ERROR);
    //CASE_ERR_NPP(NPP_SUCCESS);
    //CASE_ERR_NPP(NPP_NO_OPERATION_WARNING);
    //CASE_ERR_NPP(NPP_DIVIDE_BY_ZERO_WARNING);
    //CASE_ERR_NPP(NPP_AFFINE_QUAD_INCORRECT_WARNING);
    //CASE_ERR_NPP(NPP_WRONG_INTERSECTION_ROI_WARNING);
    //CASE_ERR_NPP(NPP_WRONG_INTERSECTION_QUAD_WARNING);
    //CASE_ERR_NPP(NPP_DOUBLE_SIZE_WARNING);
    //CASE_ERR_NPP(NPP_MISALIGNED_DST_ROI_WARNING);
#undef CASE_ERR_NPP

#define CASE_ERR_NVOFFRUC(x) case RGY_ERR_NvOFFRUC_ ## x: return _T("NvOFFRUC_") _T(#x);
    CASE_ERR_NVOFFRUC(NvOFFRUC_NOT_SUPPORTED);
    CASE_ERR_NVOFFRUC(INVALID_PTR);
    CASE_ERR_NVOFFRUC(INVALID_PARAM);
    CASE_ERR_NVOFFRUC(INVALID_HANDLE);
    CASE_ERR_NVOFFRUC(OUT_OF_SYSTEM_MEMORY);
    CASE_ERR_NVOFFRUC(OUT_OF_VIDEO_MEMORY);
    CASE_ERR_NVOFFRUC(OPENCV_NOT_AVAILABLE);
    CASE_ERR_NVOFFRUC(UNIMPLEMENTED);
    CASE_ERR_NVOFFRUC(OF_FAILURE);
    CASE_ERR_NVOFFRUC(DUPLICATE_RESOURCE);
    CASE_ERR_NVOFFRUC(UNREGISTERED_RESOURCE);
    CASE_ERR_NVOFFRUC(INCORRECT_API_SEQUENCE);
    CASE_ERR_NVOFFRUC(WRITE_TODISK_FAILED);
    CASE_ERR_NVOFFRUC(PIPELINE_EXECUTION_FAILURE);
    CASE_ERR_NVOFFRUC(SYNC_WRITE_FAILED);
    CASE_ERR_NVOFFRUC(GENERIC);
#undef CASE_ERR_NVOFFRUC

#define CASE_ERR_MPP(x) case RGY_ERR_ ## x: return _T(#x);
    CASE_ERR_MPP(MPP_ERR_UNKNOW);
    CASE_ERR_MPP(MPP_ERR_NULL_PTR);
    CASE_ERR_MPP(MPP_ERR_MALLOC);
    CASE_ERR_MPP(MPP_ERR_OPEN_FILE);
    CASE_ERR_MPP(MPP_ERR_VALUE);
    CASE_ERR_MPP(MPP_ERR_READ_BIT);
    CASE_ERR_MPP(MPP_ERR_TIMEOUT);
    CASE_ERR_MPP(MPP_ERR_PERM);
    CASE_ERR_MPP(MPP_ERR_BASE);
    CASE_ERR_MPP(MPP_ERR_LIST_STREAM);
    CASE_ERR_MPP(MPP_ERR_INIT);
    CASE_ERR_MPP(MPP_ERR_VPU_CODEC_INIT);
    CASE_ERR_MPP(MPP_ERR_STREAM);
    CASE_ERR_MPP(MPP_ERR_FATAL_THREAD);
    CASE_ERR_MPP(MPP_ERR_NOMEM);
    CASE_ERR_MPP(MPP_ERR_PROTOL);
    CASE_ERR_MPP(MPP_FAIL_SPLIT_FRAME);
    CASE_ERR_MPP(MPP_ERR_VPUHW);
    CASE_ERR_MPP(MPP_EOS_STREAM_REACHED);
    CASE_ERR_MPP(MPP_ERR_BUFFER_FULL);
    CASE_ERR_MPP(MPP_ERR_DISPLAY_FULL);
#undef CASE_ERR_MPP

    case RGY_ERR_NVSDK_NGX_FeatureNotSupported: return _T("NVSDK NGX ERR: Feature is not supported on current hardware");
    case RGY_ERR_NVSDK_NGX_PlatformError: return _T("NVSDK NGX ERR: Platform error - check d3d12 debug layer log for more information");
    case RGY_ERR_NVSDK_NGX_FeatureAlreadyExists: return _T("NVSDK NGX ERR: Feature with given parameters already exists");
    case RGY_ERR_NVSDK_NGX_FeatureNotFound: return _T("NVSDK NGX ERR: Feature with provided handle does not exist");
    case RGY_ERR_NVSDK_NGX_InvalidParameter: return _T("NVSDK NGX ERR: Invalid parameter was provided");
    case RGY_ERR_NVSDK_NGX_ScratchBufferTooSmall: return _T("NVSDK NGX ERR: Provided buffer is too small, please use size provided by NVSDK_NGX_GetScratchBufferSize");
    case RGY_ERR_NVSDK_NGX_NotInitialized: return _T("NVSDK NGX ERR: SDK was not initialized properly");
    case RGY_ERR_NVSDK_NGX_UnsupportedInputFormat: return _T("NVSDK NGX ERR: Unsupported format used for input/output buffers");
    case RGY_ERR_NVSDK_NGX_RWFlagMissing: return _T("NVSDK NGX ERR: Feature input/output needs RW access (UAV) (d3d11/d3d12 specific)");
    case RGY_ERR_NVSDK_NGX_MissingInput: return _T("NVSDK NGX ERR: Feature was created with specific input but none is provided at evaluation");
    case RGY_ERR_NVSDK_NGX_UnableToInitializeFeature: return _T("NVSDK NGX ERR: Feature is not available on the system");
    case RGY_ERR_NVSDK_NGX_OutOfDate: return _T("NVSDK NGX ERR: NGX system libraries are old and need an update");
    case RGY_ERR_NVSDK_NGX_OutOfGPUMemory: return _T("NVSDK NGX ERR: Feature requires more GPU memory than it is available on system");
    case RGY_ERR_NVSDK_NGX_UnsupportedFormat: return _T("NVSDK NGX ERR: Format used in input buffer(s) is not supported by feature");
    case RGY_ERR_NVSDK_NGX_UnableToWriteToAppDataPath: return _T("NVSDK NGX ERR: Path provided in InApplicationDataPath cannot be written to");
    case RGY_ERR_NVSDK_NGX_UnsupportedParameter: return _T("NVSDK NGX ERR: Unsupported parameter was provided (e.g. specific scaling factor is unsupported)");
    case RGY_ERR_NVSDK_NGX_Denied: return _T("NVSDK NGX ERR: The feature or application was denied (contact NVIDIA for further details)");
    case RGY_ERR_NVSDK_NGX_NotImplemented: return _T("NVSDK NGX ERR: The feature or functionality is not implemented");

    default:                                      return _T("unknown error.");
    }
}

