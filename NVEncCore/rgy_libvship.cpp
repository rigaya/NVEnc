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

#include "rgy_libvship.h"

#if ENABLE_LIBVSHIP

#if defined(_WIN32) || defined(_WIN64)
const TCHAR *RGY_LIBVSHIP_DLL_NAME = _T("libvship.dll");
#else
const TCHAR *RGY_LIBVSHIP_DLL_NAME = _T("libvship.so");
#endif // #if defined(_WIN32) || defined(_WIN64)

RGYLibVshipLoader::RGYLibVshipLoader() :
    m_hModule(nullptr),
    m_loaded(false),
    m_Vship_GetVersion(nullptr),
    m_Vship_GetDeviceCount(nullptr),
    m_Vship_SetDevice(nullptr),
    m_Vship_GetDeviceInfo(nullptr),
    m_Vship_GPUFullCheck(nullptr),
    m_Vship_GetErrorMessage(nullptr),
    m_Vship_GetDetailedLastError(nullptr),
    m_Vship_PinnedMalloc(nullptr),
    m_Vship_PinnedFree(nullptr),
    m_Vship_SSIMU2Init2(nullptr),
    m_Vship_SSIMU2Free(nullptr),
    m_Vship_ComputeSSIMU2(nullptr),
    m_Vship_SSIMU2GetDetailedLastError(nullptr),
    m_Vship_ButteraugliInit2(nullptr),
    m_Vship_ButteraugliFree(nullptr),
    m_Vship_ComputeButteraugli(nullptr),
    m_Vship_ButteraugliGetDetailedLastError(nullptr),
    m_Vship_CVVDPInit3(nullptr),
    m_Vship_CVVDPFree(nullptr),
    m_Vship_ResetCVVDP(nullptr),
    m_Vship_ResetScoreCVVDP(nullptr),
    m_Vship_ComputeCVVDP(nullptr),
    m_Vship_CVVDPGetDetailedLastError(nullptr)
{
}

RGYLibVshipLoader::~RGYLibVshipLoader() {
    close();
}

bool RGYLibVshipLoader::load() {
    if (m_loaded) {
        return true;
    }

    if ((m_hModule = RGY_LOAD_LIBRARY(RGY_LIBVSHIP_DLL_NAME)) == nullptr) {
        return false;
    }

    auto loadFunc = [this](const char *funcName, void **func) {
        if ((*func = RGY_GET_PROC_ADDRESS(m_hModule, funcName)) == nullptr) {
            return false;
        }
        return true;
    };

    if (!loadFunc("Vship_GetVersion", (void **)&m_Vship_GetVersion)) { close(); return false; }
    if (!loadFunc("Vship_GetDeviceCount", (void **)&m_Vship_GetDeviceCount)) { close(); return false; }
    if (!loadFunc("Vship_SetDevice", (void **)&m_Vship_SetDevice)) { close(); return false; }
    if (!loadFunc("Vship_GetDeviceInfo", (void **)&m_Vship_GetDeviceInfo)) { close(); return false; }
    if (!loadFunc("Vship_GPUFullCheck", (void **)&m_Vship_GPUFullCheck)) { close(); return false; }
    if (!loadFunc("Vship_GetErrorMessage", (void **)&m_Vship_GetErrorMessage)) { close(); return false; }
    if (!loadFunc("Vship_GetDetailedLastError", (void **)&m_Vship_GetDetailedLastError)) { close(); return false; }
    if (!loadFunc("Vship_PinnedMalloc", (void **)&m_Vship_PinnedMalloc)) { close(); return false; }
    if (!loadFunc("Vship_PinnedFree", (void **)&m_Vship_PinnedFree)) { close(); return false; }
    if (!loadFunc("Vship_SSIMU2Init2", (void **)&m_Vship_SSIMU2Init2)) { close(); return false; }
    if (!loadFunc("Vship_SSIMU2Free", (void **)&m_Vship_SSIMU2Free)) { close(); return false; }
    if (!loadFunc("Vship_ComputeSSIMU2", (void **)&m_Vship_ComputeSSIMU2)) { close(); return false; }
    if (!loadFunc("Vship_SSIMU2GetDetailedLastError", (void **)&m_Vship_SSIMU2GetDetailedLastError)) { close(); return false; }
    if (!loadFunc("Vship_ButteraugliInit2", (void **)&m_Vship_ButteraugliInit2)) { close(); return false; }
    if (!loadFunc("Vship_ButteraugliFree", (void **)&m_Vship_ButteraugliFree)) { close(); return false; }
    if (!loadFunc("Vship_ComputeButteraugli", (void **)&m_Vship_ComputeButteraugli)) { close(); return false; }
    if (!loadFunc("Vship_ButteraugliGetDetailedLastError", (void **)&m_Vship_ButteraugliGetDetailedLastError)) { close(); return false; }
    if (!loadFunc("Vship_CVVDPInit3", (void **)&m_Vship_CVVDPInit3)) { close(); return false; }
    if (!loadFunc("Vship_CVVDPFree", (void **)&m_Vship_CVVDPFree)) { close(); return false; }
    if (!loadFunc("Vship_ResetCVVDP", (void **)&m_Vship_ResetCVVDP)) { close(); return false; }
    if (!loadFunc("Vship_ResetScoreCVVDP", (void **)&m_Vship_ResetScoreCVVDP)) { close(); return false; }
    if (!loadFunc("Vship_ComputeCVVDP", (void **)&m_Vship_ComputeCVVDP)) { close(); return false; }
    if (!loadFunc("Vship_CVVDPGetDetailedLastError", (void **)&m_Vship_CVVDPGetDetailedLastError)) { close(); return false; }

    m_loaded = true;
    return true;
}

void RGYLibVshipLoader::close() {
    if (m_hModule) {
        RGY_FREE_LIBRARY(m_hModule);
        m_hModule = nullptr;
    }
    m_loaded = false;

    m_Vship_GetVersion = nullptr;
    m_Vship_GetDeviceCount = nullptr;
    m_Vship_SetDevice = nullptr;
    m_Vship_GetDeviceInfo = nullptr;
    m_Vship_GPUFullCheck = nullptr;
    m_Vship_GetErrorMessage = nullptr;
    m_Vship_GetDetailedLastError = nullptr;
    m_Vship_PinnedMalloc = nullptr;
    m_Vship_PinnedFree = nullptr;
    m_Vship_SSIMU2Init2 = nullptr;
    m_Vship_SSIMU2Free = nullptr;
    m_Vship_ComputeSSIMU2 = nullptr;
    m_Vship_SSIMU2GetDetailedLastError = nullptr;
    m_Vship_ButteraugliInit2 = nullptr;
    m_Vship_ButteraugliFree = nullptr;
    m_Vship_ComputeButteraugli = nullptr;
    m_Vship_ButteraugliGetDetailedLastError = nullptr;
    m_Vship_CVVDPInit3 = nullptr;
    m_Vship_CVVDPFree = nullptr;
    m_Vship_ResetCVVDP = nullptr;
    m_Vship_ResetScoreCVVDP = nullptr;
    m_Vship_ComputeCVVDP = nullptr;
    m_Vship_CVVDPGetDetailedLastError = nullptr;
}

#endif //#if ENABLE_LIBVSHIP
