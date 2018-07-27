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

static cl_int cl_create_kernel(cl_data_t *cl_data, const cl_func_t *cl) {
    cl_int ret = CL_SUCCESS;
    cl_data->contextCL = cl->createContext(0, 1, &cl_data->deviceID, NULL, NULL, &ret);
    if (CL_SUCCESS != ret)
        return ret;

    cl_data->commands = cl->createCommandQueue(cl_data->contextCL, cl_data->deviceID, NULL, &ret);
    if (CL_SUCCESS != ret)
        return ret;

    //OpenCLのカーネル用のコードはリソース埋め込みにしているので、それを呼び出し
    HRSRC hResource = NULL;
    HGLOBAL hResourceData = NULL;
    const char *clSourceFile = NULL;
    if (   NULL == (hResource = FindResource(NULL, _T("CLDATA"), _T("KERNEL_DATA")))
        || NULL == (hResourceData = LoadResource(NULL, hResource))
        || NULL == (clSourceFile = (const char *)LockResource(hResourceData))) {
        return 1;
    }
    size_t programLength = strlen(clSourceFile);
    cl_data->program = cl->createProgramWithSource(cl_data->contextCL, 1, (const char**)&clSourceFile, &programLength, &ret);
    if (CL_SUCCESS != ret)
        return ret;

    if (CL_SUCCESS != (ret = cl->buildProgram(cl_data->program, 1, &cl_data->deviceID, NULL, NULL, NULL))) {
        char buffer[2048];
        size_t length = 0;
        cl->getProgramBuildInfo(cl_data->program, cl_data->deviceID, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &length);
        fprintf(stderr, "%s\n", buffer);
        return ret;
    }
    cl_data->kernel = cl->createKernel(cl_data->program, "dummy_calc", &ret);
    if (CL_SUCCESS != ret)
        return ret;

    return ret;
}

static cl_int cl_calc(const cl_data_t *cl_data, const cl_func_t *cl) {
    using namespace std;
    const int LOOKAROUND = 10;
    const int BUFFER_X = 1024 * 8;
    const int BUFFER_Y = 1024;
    const size_t BUFFER_BYTE_SIZE = BUFFER_X * BUFFER_Y * sizeof(float);
    cl_int ret = CL_SUCCESS;
    cl_mem bufA = cl->createBuffer(cl_data->contextCL, CL_MEM_READ_ONLY,  BUFFER_BYTE_SIZE, NULL, &ret);
    cl_mem bufB = cl->createBuffer(cl_data->contextCL, CL_MEM_READ_ONLY,  BUFFER_BYTE_SIZE, NULL, &ret);
    cl_mem bufC = cl->createBuffer(cl_data->contextCL, CL_MEM_WRITE_ONLY, BUFFER_BYTE_SIZE, NULL, &ret);

    vector<float> arrayA(BUFFER_BYTE_SIZE);
    vector<float> arrayB(BUFFER_BYTE_SIZE);
    vector<float> arrayC(BUFFER_BYTE_SIZE, 0.0);

    random_device rd;
    mt19937 mt(rd());
    uniform_real_distribution<float> random(0.0f, 10.0f);
    generate(arrayA.begin(), arrayA.end(), [&random, &mt]() { return random(mt); });
    generate(arrayB.begin(), arrayB.end(), [&random, &mt]() { return random(mt); });

    cl->enqueueWriteBuffer(cl_data->commands, bufA, CL_FALSE, 0, BUFFER_BYTE_SIZE, &arrayA[0], 0, NULL, NULL);
    cl->enqueueWriteBuffer(cl_data->commands, bufB, CL_FALSE, 0, BUFFER_BYTE_SIZE, &arrayB[0], 0, NULL, NULL);
    cl->setKernelArg(cl_data->kernel, 0, sizeof(cl_mem), &bufA);
    cl->setKernelArg(cl_data->kernel, 1, sizeof(cl_mem), &bufB);
    cl->setKernelArg(cl_data->kernel, 2, sizeof(cl_mem), &bufC);
    cl->setKernelArg(cl_data->kernel, 3, sizeof(cl_int), &BUFFER_X);
    cl->setKernelArg(cl_data->kernel, 4, sizeof(cl_int), &BUFFER_Y);
    cl->setKernelArg(cl_data->kernel, 5, sizeof(cl_int), &LOOKAROUND);

    size_t data_size = BUFFER_X * BUFFER_Y;
    cl->enqueueNDRangeKernel(cl_data->commands, cl_data->kernel, 1, 0, &data_size, NULL, 0, NULL, NULL);
    cl->enqueueReadBuffer(cl_data->commands, bufC, CL_TRUE, 0, BUFFER_BYTE_SIZE, &arrayC[0], 0, NULL, NULL);
    cl->finish(cl_data->commands);

    cl->releaseMemObject(bufA);
    cl->releaseMemObject(bufB);
    cl->releaseMemObject(bufC);

    return ret;
}

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
#if 0
        bool abort_get_frequency_loop = false;
        std::future<int> f_max_frequency = std::async([&]() {
            int max_frequency = 0;
            while (false == abort_get_frequency_loop) {
                Sleep(1);
                max_frequency = (std::max)(max_frequency, cl_get_device_max_clock_frequency_mhz(cl_data, cl));
            }
            return max_frequency;
        });
        Sleep(20);
        cl_int ret = CL_SUCCESS;
        if (   CL_SUCCESS != (ret = cl_create_kernel(cl_data, cl))
            || CL_SUCCESS != (ret = cl_calc(cl_data, cl))) {
            ;
        }
        abort_get_frequency_loop = true;
        const int max_device_frequency = f_max_frequency.get(); //f_max_frequencyにセットされる値を待つ (async終了同期)
#else
        const int max_device_frequency = cl_get_device_max_clock_frequency_mhz(cl_data, cl);
#endif
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
