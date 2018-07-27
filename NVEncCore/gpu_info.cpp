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

#include "rgy_tchar.h"
#include <string>
#include <vector>
#include <random>
#include <future>
#include <algorithm>
#include "cl_func.h"
#include "rgy_osdep.h"
#include "rgy_util.h"

#if ENABLE_OPENCL

static std::basic_string<TCHAR> to_tchar(const char *string) {
#if UNICODE
    int required_length = MultiByteToWideChar(CP_ACP, 0, string, -1, NULL, 0);
    std::basic_string<TCHAR> str(1+required_length, _T('\0'));
    MultiByteToWideChar(CP_ACP, 0, string, -1, &str[0], (int)str.size());
#else
    std::basic_string<char> str = string;
#endif
    return str;
};

cl_int cl_get_driver_version(const cl_data_t *cl_data, const cl_func_t *cl, TCHAR *buffer, unsigned int buffer_size) {
    cl_int ret = CL_SUCCESS;
    char cl_info_buffer[1024] = { 0 };
    #if defined(_WIN32) || defined(_WIN64)
    __try {
    #endif //#if defined(_WIN32) || defined(_WIN64)
        ret = cl->getDeviceInfo(cl_data->deviceID, CL_DRIVER_VERSION, _countof(cl_info_buffer), cl_info_buffer, NULL);
    #if defined(_WIN32) || defined(_WIN64)
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        _ftprintf(stderr, _T("Crush (getDeviceInfo)\n"));
        ret = CL_INVALID_VALUE;
    }
    #endif //#if defined(_WIN32) || defined(_WIN64)
    if (ret == CL_SUCCESS) {
        _tcscpy_s(buffer, buffer_size, to_tchar(cl_info_buffer).c_str());
    } else {
        _tcscpy_s(buffer, buffer_size, _T("Unknown"));
    }
    return ret;
}

static cl_int cl_create_info_string(cl_data_t *cl_data, const cl_func_t *cl, TCHAR *buffer, unsigned int buffer_size) {
    cl_int ret = CL_SUCCESS;
    char cl_info_buffer[1024] = { 0 };
    #if defined(_WIN32) || defined(_WIN64)
    __try {
    #endif //#if defined(_WIN32) || defined(_WIN64)
        ret = cl->getDeviceInfo(cl_data->deviceID, CL_DEVICE_NAME, _countof(cl_info_buffer), cl_info_buffer, NULL);
    #if defined(_WIN32) || defined(_WIN64)
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        _ftprintf(stderr, _T("Crush (getDeviceInfo)\n"));
        ret = CL_INVALID_VALUE;
    }
    #endif //#if defined(_WIN32) || defined(_WIN64)
    if (ret != CL_SUCCESS) {
        _tcscpy_s(buffer, buffer_size, _T("Unknown (error on OpenCL clGetDeviceInfo)"));
        return ret;
    } else {
        _tcscpy_s(buffer, buffer_size, to_tchar(cl_info_buffer).c_str());
        const int max_device_frequency = cl_get_device_max_clock_frequency_mhz(cl_data, cl);
        if (CL_SUCCESS == cl->getDeviceInfo(cl_data->deviceID, CL_DEVICE_MAX_COMPUTE_UNITS, _countof(cl_info_buffer), cl_info_buffer, NULL)) {
            _stprintf_s(buffer + _tcslen(buffer), buffer_size - _tcslen(buffer), _T(" (%d EU)"), *(cl_uint *)cl_info_buffer);
        }
        if (max_device_frequency) {
            _stprintf_s(buffer + _tcslen(buffer), buffer_size - _tcslen(buffer), _T(" @ %d MHz"), max_device_frequency);
        }
        if (CL_SUCCESS == cl->getDeviceInfo(cl_data->deviceID, CL_DRIVER_VERSION, _countof(cl_info_buffer), cl_info_buffer, NULL)) {
            _stprintf_s(buffer + _tcslen(buffer), buffer_size - _tcslen(buffer), _T(" (%s)"), to_tchar(cl_info_buffer).c_str());
        }
    }
    return ret;
}

#endif //ENABLE_OPENCL

#pragma warning (push)
#pragma warning (disable: 4100)
int getGPUInfo(const char *VendorName, TCHAR *buffer, unsigned int buffer_size, bool driver_version_only) {
#if !ENABLE_OPENCL
    _stprintf_s(buffer, buffer_size, _T("Unknown (not compiled with OpenCL support)"));
    return 0;
#else
    int ret = CL_SUCCESS;
    cl_func_t cl = { 0 };
    cl_data_t data = { 0 };

    if (CL_SUCCESS != (ret = cl_get_func(&cl))) {
        _tcscpy_s(buffer, buffer_size, _T("Unknown (Failed to load OpenCL.dll)"));
    } else if (CL_SUCCESS != (ret = cl_get_platform_and_device(VendorName, CL_DEVICE_TYPE_GPU, &data, &cl))) {
        _stprintf_s(buffer, buffer_size, _T("Unknown (Failed to find %s GPU)"), to_tchar(VendorName).c_str());
    } else {
        if (driver_version_only)
            cl_get_driver_version(&data, &cl, buffer, buffer_size);
        else
            cl_create_info_string(&data, &cl, buffer, buffer_size);
    }
    cl_release(&data, &cl);
    return ret;
#endif // !ENABLE_OPENCL
}
#pragma warning (pop)
