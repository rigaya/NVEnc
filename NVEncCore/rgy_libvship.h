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

#ifndef __RGY_LIBVSHIP_H__
#define __RGY_LIBVSHIP_H__

#include "rgy_version.h"

#if ENABLE_LIBVSHIP

#include "rgy_osdep.h"
#include "rgy_tchar.h"
#include "rgy_util.h"

#pragma warning (push)
#pragma warning (disable: 4244)
#pragma warning (disable: 4819)
#include "VshipColor.h"
#include "VshipAPI.h"
#pragma warning (pop)

extern const TCHAR *RGY_LIBVSHIP_DLL_NAME;

class RGYLibVshipLoader {
private:
    HMODULE m_hModule;
    bool m_loaded;

    decltype(&Vship_GetVersion) m_Vship_GetVersion;
    decltype(&Vship_GetDeviceCount) m_Vship_GetDeviceCount;
    decltype(&Vship_SetDevice) m_Vship_SetDevice;
    decltype(&Vship_GetDeviceInfo) m_Vship_GetDeviceInfo;
    decltype(&Vship_GPUFullCheck) m_Vship_GPUFullCheck;
    decltype(&Vship_GetErrorMessage) m_Vship_GetErrorMessage;
    decltype(&Vship_GetDetailedLastError) m_Vship_GetDetailedLastError;
    decltype(&Vship_PinnedMalloc) m_Vship_PinnedMalloc;
    decltype(&Vship_PinnedFree) m_Vship_PinnedFree;
    decltype(&Vship_SSIMU2Init2) m_Vship_SSIMU2Init2;
    decltype(&Vship_SSIMU2Free) m_Vship_SSIMU2Free;
    decltype(&Vship_ComputeSSIMU2) m_Vship_ComputeSSIMU2;
    decltype(&Vship_SSIMU2GetDetailedLastError) m_Vship_SSIMU2GetDetailedLastError;
    decltype(&Vship_ButteraugliInit2) m_Vship_ButteraugliInit2;
    decltype(&Vship_ButteraugliFree) m_Vship_ButteraugliFree;
    decltype(&Vship_ComputeButteraugli) m_Vship_ComputeButteraugli;
    decltype(&Vship_ButteraugliGetDetailedLastError) m_Vship_ButteraugliGetDetailedLastError;
    decltype(&Vship_CVVDPInit3) m_Vship_CVVDPInit3;
    decltype(&Vship_CVVDPFree) m_Vship_CVVDPFree;
    decltype(&Vship_ResetCVVDP) m_Vship_ResetCVVDP;
    decltype(&Vship_ResetScoreCVVDP) m_Vship_ResetScoreCVVDP;
    decltype(&Vship_ComputeCVVDP) m_Vship_ComputeCVVDP;
    decltype(&Vship_CVVDPGetDetailedLastError) m_Vship_CVVDPGetDetailedLastError;

public:
    RGYLibVshipLoader();
    ~RGYLibVshipLoader();

    bool load();
    void close();
    bool loaded() const { return m_loaded; }

    auto p_Vship_GetVersion() const { return m_Vship_GetVersion; }
    auto p_Vship_GetDeviceCount() const { return m_Vship_GetDeviceCount; }
    auto p_Vship_SetDevice() const { return m_Vship_SetDevice; }
    auto p_Vship_GetDeviceInfo() const { return m_Vship_GetDeviceInfo; }
    auto p_Vship_GPUFullCheck() const { return m_Vship_GPUFullCheck; }
    auto p_Vship_GetErrorMessage() const { return m_Vship_GetErrorMessage; }
    auto p_Vship_GetDetailedLastError() const { return m_Vship_GetDetailedLastError; }
    auto p_Vship_PinnedMalloc() const { return m_Vship_PinnedMalloc; }
    auto p_Vship_PinnedFree() const { return m_Vship_PinnedFree; }
    auto p_Vship_SSIMU2Init2() const { return m_Vship_SSIMU2Init2; }
    auto p_Vship_SSIMU2Free() const { return m_Vship_SSIMU2Free; }
    auto p_Vship_ComputeSSIMU2() const { return m_Vship_ComputeSSIMU2; }
    auto p_Vship_SSIMU2GetDetailedLastError() const { return m_Vship_SSIMU2GetDetailedLastError; }
    auto p_Vship_ButteraugliInit2() const { return m_Vship_ButteraugliInit2; }
    auto p_Vship_ButteraugliFree() const { return m_Vship_ButteraugliFree; }
    auto p_Vship_ComputeButteraugli() const { return m_Vship_ComputeButteraugli; }
    auto p_Vship_ButteraugliGetDetailedLastError() const { return m_Vship_ButteraugliGetDetailedLastError; }
    auto p_Vship_CVVDPInit3() const { return m_Vship_CVVDPInit3; }
    auto p_Vship_CVVDPFree() const { return m_Vship_CVVDPFree; }
    auto p_Vship_ResetCVVDP() const { return m_Vship_ResetCVVDP; }
    auto p_Vship_ResetScoreCVVDP() const { return m_Vship_ResetScoreCVVDP; }
    auto p_Vship_ComputeCVVDP() const { return m_Vship_ComputeCVVDP; }
    auto p_Vship_CVVDPGetDetailedLastError() const { return m_Vship_CVVDPGetDetailedLastError; }
};

#endif //#if ENABLE_LIBVSHIP

#endif // __RGY_LIBVSHIP_H__
