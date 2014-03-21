/*
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#include <dynlink/cuda_runtime_dynlink.h>

tcudaMalloc3D                   *cudaMalloc3D;
tcudaMalloc3DArray              *cudaMalloc3DArray;
tcudaMemset3D                   *cudaMemset3D;
tcudaMemcpy3D                   *cudaMemcpy3D;
tcudaMemcpy3DAsync              *cudaMemcpy3DAsync;
tcudaMalloc                     *cudaMalloc;
tcudaMallocHost                 *cudaMallocHost;
tcudaMallocPitch                *cudaMallocPitch;
tcudaMallocArray                *cudaMallocArray;
tcudaFree                       *cudaFree;
tcudaFreeHost                   *cudaFreeHost;
tcudaFreeArray                  *cudaFreeArray;
tcudaMemcpy                     *cudaMemcpy;
tcudaMemcpyToArray              *cudaMemcpyToArray;
tcudaMemcpyFromArray            *cudaMemcpyFromArray;
tcudaMemcpyArrayToArray         *cudaMemcpyArrayToArray;
tcudaMemcpy2D                   *cudaMemcpy2D;
tcudaMemcpy2DToArray            *cudaMemcpy2DToArray;
tcudaMemcpy2DFromArray          *cudaMemcpy2DFromArray;
tcudaMemcpy2DArrayToArray       *cudaMemcpy2DArrayToArray;
tcudaMemcpyToSymbol             *cudaMemcpyToSymbol;
tcudaMemcpyFromSymbol           *cudaMemcpyFromSymbol;
tcudaMemcpyAsync                *cudaMemcpyAsync;
tcudaMemcpyToArrayAsync         *cudaMemcpyToArrayAsync;
tcudaMemcpyFromArrayAsync       *cudaMemcpyFromArrayAsync;
tcudaMemcpy2DAsync              *cudaMemcpy2DAsync;
tcudaMemcpy2DToArrayAsync       *cudaMemcpy2DToArrayAsync;
tcudaMemcpy2DFromArrayAsync     *cudaMemcpy2DFromArrayAsync;
tcudaMemcpyToSymbolAsync        *cudaMemcpyToSymbolAsync;
tcudaMemcpyFromSymbolAsync      *cudaMemcpyFromSymbolAsync;
tcudaMemset                     *cudaMemset;
tcudaMemset2D                   *cudaMemset2D;
tcudaGetSymbolAddress           *cudaGetSymbolAddress;
tcudaGetSymbolSize              *cudaGetSymbolSize;
tcudaGetDeviceCount             *cudaGetDeviceCount;
tcudaGetDeviceProperties        *cudaGetDeviceProperties;
tcudaChooseDevice               *cudaChooseDevice;
tcudaSetDevice                  *cudaSetDevice;
tcudaGetDevice                  *cudaGetDevice;
tcudaBindTexture                *cudaBindTexture;
tcudaBindTextureToArray         *cudaBindTextureToArray;
tcudaUnbindTexture              *cudaUnbindTexture;
tcudaGetTextureAlignmentOffset  *cudaGetTextureAlignmentOffset;
tcudaGetTextureReference        *cudaGetTextureReference;
tcudaGetChannelDesc             *cudaGetChannelDesc;
tcudaCreateChannelDesc          *cudaCreateChannelDesc;
tcudaGetLastError               *cudaGetLastError;
tcudaGetErrorString             *cudaGetErrorString;
tcudaConfigureCall              *cudaConfigureCall;
tcudaSetupArgument              *cudaSetupArgument;
tcudaLaunch                     *cudaLaunch;
tcudaStreamCreate               *cudaStreamCreate;
tcudaStreamDestroy              *cudaStreamDestroy;
tcudaStreamSynchronize          *cudaStreamSynchronize;
tcudaStreamQuery                *cudaStreamQuery;
tcudaEventCreate                *cudaEventCreate;
tcudaEventRecord                *cudaEventRecord;
tcudaEventQuery                 *cudaEventQuery;
tcudaEventSynchronize           *cudaEventSynchronize;
tcudaEventDestroy               *cudaEventDestroy;
tcudaEventElapsedTime           *cudaEventElapsedTime;
tcudaSetDoubleForDevice         *cudaSetDoubleForDevice;
tcudaSetDoubleForHost           *cudaSetDoubleForHost;
tcudaDeviceReset                *cudaDeviceReset;
tcudaDeviceSynchronize          *cudaDeviceSynchronize;



#if defined(_WIN32) || defined(_WIN64)
#define _WIN    1
#define _OS _WIN
#elif defined(__unix__)
#define _UNIX   2
#define _OS _UNIX
#endif


#if (_OS == _WIN)

#include <Windows.h>

__host__ cudaError_t CUDARTAPI cudaRuntimeDynload(void)
{

#define QUOTE(x)        #x
#define GET_PROC(name)  name = (t##name *)GetProcAddress(CudaRtLib, QUOTE(name)); if (name == NULL) return cudaErrorUnknown

    HMODULE CudaRtLib = LoadLibrary(L"cudart.dll");

    if (CudaRtLib == NULL)
    {
        return cudaErrorUnknown;
    }
    else
    {
        GET_PROC(cudaMalloc3D);
        GET_PROC(cudaMalloc3DArray);
        GET_PROC(cudaMemset3D);
        GET_PROC(cudaMemcpy3D);
        GET_PROC(cudaMemcpy3DAsync);

        GET_PROC(cudaMalloc);
        GET_PROC(cudaMallocHost);
        GET_PROC(cudaMallocPitch);
        GET_PROC(cudaMallocArray);
        GET_PROC(cudaFree);
        GET_PROC(cudaFreeHost);
        GET_PROC(cudaFreeArray);

        GET_PROC(cudaMemcpy);
        GET_PROC(cudaMemcpyToArray);
        GET_PROC(cudaMemcpyFromArray);
        GET_PROC(cudaMemcpyArrayToArray);
        GET_PROC(cudaMemcpy2D);
        GET_PROC(cudaMemcpy2DToArray);
        GET_PROC(cudaMemcpy2DFromArray);
        GET_PROC(cudaMemcpy2DArrayToArray);
        GET_PROC(cudaMemcpyToSymbol);
        GET_PROC(cudaMemcpyFromSymbol);

        GET_PROC(cudaMemcpyAsync);
        GET_PROC(cudaMemcpyToArrayAsync);
        GET_PROC(cudaMemcpyFromArrayAsync);
        GET_PROC(cudaMemcpy2DAsync);
        GET_PROC(cudaMemcpy2DToArrayAsync);
        GET_PROC(cudaMemcpy2DFromArrayAsync);
        GET_PROC(cudaMemcpyToSymbolAsync);
        GET_PROC(cudaMemcpyFromSymbolAsync);

        GET_PROC(cudaMemset);
        GET_PROC(cudaMemset2D);

        GET_PROC(cudaGetSymbolAddress);
        GET_PROC(cudaGetSymbolSize);

        GET_PROC(cudaGetDeviceCount);
        GET_PROC(cudaGetDeviceProperties);
        GET_PROC(cudaChooseDevice);
        GET_PROC(cudaSetDevice);
        GET_PROC(cudaGetDevice);

        GET_PROC(cudaBindTexture);
        GET_PROC(cudaBindTextureToArray);
        GET_PROC(cudaUnbindTexture);
        GET_PROC(cudaGetTextureAlignmentOffset);
        GET_PROC(cudaGetTextureReference);

        GET_PROC(cudaGetChannelDesc);
        GET_PROC(cudaCreateChannelDesc);

        GET_PROC(cudaGetLastError);
        GET_PROC(cudaGetErrorString);

        GET_PROC(cudaConfigureCall);
        GET_PROC(cudaSetupArgument);
        GET_PROC(cudaLaunch);

        GET_PROC(cudaStreamCreate);
        GET_PROC(cudaStreamDestroy);
        GET_PROC(cudaStreamSynchronize);
        GET_PROC(cudaStreamQuery);

        GET_PROC(cudaEventCreate);
        GET_PROC(cudaEventRecord);
        GET_PROC(cudaEventQuery);
        GET_PROC(cudaEventSynchronize);
        GET_PROC(cudaEventDestroy);
        GET_PROC(cudaEventElapsedTime);

        GET_PROC(cudaSetDoubleForDevice);
        GET_PROC(cudaSetDoubleForHost);

        GET_PROC(cudaDeviceReset);
        GET_PROC(cudaDeviceSynchronize);
    }

    return cudaSuccess;

#undef QUOTE
#undef GET_PROC

}

#elif (_OS == _UNIX)

#include <dlfcn.h>

__host__ cudaError_t CUDARTAPI cudaRuntimeDynload(void)
{
#define QUOTE(x)        #x
#define GET_PROC(name)  name = (t##name *)dlsym(CudaRtLib, QUOTE(name)); if (name == NULL) return cudaErrorUnknown

    void *CudaRtLib = dlopen("libcudart.so", RTLD_LAZY);

    if (CudaRtLib == NULL)
    {
        return cudaErrorUnknown;
    }
    else
    {
        GET_PROC(cudaMalloc3D);
        GET_PROC(cudaMalloc3DArray);
        GET_PROC(cudaMemset3D);
        GET_PROC(cudaMemcpy3D);
        GET_PROC(cudaMemcpy3DAsync);

        GET_PROC(cudaMalloc);
        GET_PROC(cudaMallocHost);
        GET_PROC(cudaMallocPitch);
        GET_PROC(cudaMallocArray);
        GET_PROC(cudaFree);
        GET_PROC(cudaFreeHost);
        GET_PROC(cudaFreeArray);

        GET_PROC(cudaMemcpy);
        GET_PROC(cudaMemcpyToArray);
        GET_PROC(cudaMemcpyFromArray);
        GET_PROC(cudaMemcpyArrayToArray);
        GET_PROC(cudaMemcpy2D);
        GET_PROC(cudaMemcpy2DToArray);
        GET_PROC(cudaMemcpy2DFromArray);
        GET_PROC(cudaMemcpy2DArrayToArray);
        GET_PROC(cudaMemcpyToSymbol);
        GET_PROC(cudaMemcpyFromSymbol);

        GET_PROC(cudaMemcpyAsync);
        GET_PROC(cudaMemcpyToArrayAsync);
        GET_PROC(cudaMemcpyFromArrayAsync);
        GET_PROC(cudaMemcpy2DAsync);
        GET_PROC(cudaMemcpy2DToArrayAsync);
        GET_PROC(cudaMemcpy2DFromArrayAsync);
        GET_PROC(cudaMemcpyToSymbolAsync);
        GET_PROC(cudaMemcpyFromSymbolAsync);

        GET_PROC(cudaMemset);
        GET_PROC(cudaMemset2D);

        GET_PROC(cudaGetSymbolAddress);
        GET_PROC(cudaGetSymbolSize);

        GET_PROC(cudaGetDeviceCount);
        GET_PROC(cudaGetDeviceProperties);
        GET_PROC(cudaChooseDevice);
        GET_PROC(cudaSetDevice);
        GET_PROC(cudaGetDevice);

        GET_PROC(cudaBindTexture);
        GET_PROC(cudaBindTextureToArray);
        GET_PROC(cudaUnbindTexture);
        GET_PROC(cudaGetTextureAlignmentOffset);
        GET_PROC(cudaGetTextureReference);

        GET_PROC(cudaGetChannelDesc);
        GET_PROC(cudaCreateChannelDesc);

        GET_PROC(cudaGetLastError);
        GET_PROC(cudaGetErrorString);

        GET_PROC(cudaConfigureCall);
        GET_PROC(cudaSetupArgument);
        GET_PROC(cudaLaunch);

        GET_PROC(cudaStreamCreate);
        GET_PROC(cudaStreamDestroy);
        GET_PROC(cudaStreamSynchronize);
        GET_PROC(cudaStreamQuery);

        GET_PROC(cudaEventCreate);
        GET_PROC(cudaEventRecord);
        GET_PROC(cudaEventQuery);
        GET_PROC(cudaEventSynchronize);
        GET_PROC(cudaEventDestroy);
        GET_PROC(cudaEventElapsedTime);

        GET_PROC(cudaSetDoubleForDevice);
        GET_PROC(cudaSetDoubleForHost);

        GET_PROC(cudaDeviceReset);
        GET_PROC(cudaDeviceSynchronize);
    }

    return cudaSuccess;

#undef QUOTE
#undef GET_PROC
}



#endif






