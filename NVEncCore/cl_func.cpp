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

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>
#include <tchar.h>
#include <vector>
#include "cl_func.h"

#if ENABLE_OPENCL

//OpenCLのドライバは場合によってはクラッシュする可能性がある
//クラッシュしたことがあれば、このフラグを立て、以降OpenCLを使用しないようにする
static bool opencl_crush = false;

static void to_tchar(TCHAR *buf, uint32_t buf_size, const char *string) {
#if UNICODE
    MultiByteToWideChar(CP_ACP, 0, string, -1, buf, buf_size);
#else
    strcpy_s(buf, buf_size, string);
#endif
};

static inline const char *strichr(const char *str, int c) {
    c = tolower(c);
    for (; *str; str++)
        if (c == tolower(*str))
            return str;
    return NULL;
}
static inline const char *stristr(const char *str, const char *substr) {
    size_t len = 0;
    if (substr && (len = strlen(substr)) != NULL)
        for (; (str = strichr(str, substr[0])) != NULL; str++)
            if (_strnicmp(str, substr, len) == NULL)
                return str;
    return NULL;
}

bool cl_check_vendor_name(const char *str, const char *VendorName) {
    if (NULL != stristr(str, VendorName))
        return true;
    if (NULL != stristr(VendorName, "AMD"))
        return NULL != stristr(str, "Advanced Micro Devices");
    return false;
}
cl_int cl_get_func(cl_func_t *cl) {
    ZeroMemory(cl, sizeof(cl_func_t));
    if (NULL == (cl->hdll = LoadLibrary(_T("OpenCL.dll")))) {
        return 1;
    }

    std::vector<std::pair<void**, const char*>> cl_func_list = {
        { (void **)&cl->getPlatformIDs, "clGetPlatformIDs" },
        { (void **)&cl->getPlatformInfo, "clGetPlatformInfo" },
        { (void **)&cl->getDeviceIDs, "clGetDeviceIDs" },
        { (void **)&cl->getDeviceInfo, "clGetDeviceInfo" },
        { (void **)&cl->createProgramWithSource, "clCreateProgramWithSource" },
        { (void **)&cl->buildProgram, "clBuildProgram" },
        { (void **)&cl->getProgramBuildInfo, "clGetProgramBuildInfo" },
        { (void **)&cl->releaseProgram, "clReleaseProgram" },
        { (void **)&cl->createContext, "clCreateContext" },
        { (void **)&cl->releaseContext, "clReleaseContext" },
        { (void **)&cl->createCommandQueue, "clCreateCommandQueue" },
        { (void **)&cl->releaseCommandQueue, "clReleaseCommandQueue" },
        { (void **)&cl->createBuffer, "clCreateBuffer" },
        { (void **)&cl->releaseMemObject, "clReleaseMemObject" },
        { (void **)&cl->createKernel, "clCreateKernel" },
        { (void **)&cl->releaseKernel, "clReleaseKernel" },
        { (void **)&cl->setKernelArg, "clSetKernelArg" },
        { (void **)&cl->enqueueTask, "clEnqueueTask" },
        { (void **)&cl->enqueueNDRangeKernel, "clEnqueueNDRangeKernel" },
        { (void **)&cl->finish, "clFinish" },
        { (void **)&cl->enqueueReadBuffer, "clEnqueueReadBuffer" },
        { (void **)&cl->enqueueWriteBuffer, "clEnqueueWriteBuffer" },
    };

    for (auto func : cl_func_list) {
        if (NULL == (*(func.first) = GetProcAddress(cl->hdll, func.second))) {
            return 1;
        }
    }
    return CL_SUCCESS;
}
void cl_release_func(cl_func_t *cl) {
    if (cl && cl->hdll) {
        FreeLibrary(cl->hdll);
    }
    ZeroMemory(cl, sizeof(cl_func_t));
}

cl_int cl_get_platform_and_device(const char *VendorName, cl_int device_type, cl_data_t *cl_data, const cl_func_t *cl) {
    if (opencl_crush) {
        return CL_INVALID_VALUE;
    }

    cl_uint platform_count = 0;
    cl_int ret = CL_SUCCESS;

    //OpenCLのドライバは場合によってはクラッシュする可能性がある
    //その対策として、構造化例外を使用して回避を試みる
    #if defined(_WIN32) || defined(_WIN64)
    __try {
    #endif //#if defined(_WIN32) || defined(_WIN64)
        if (CL_SUCCESS != (ret = cl->getPlatformIDs(0, NULL, &platform_count))) {
            _ftprintf(stderr, _T("Error (clGetPlatformIDs): %d\n"), ret);
            return ret;
        }
    #if defined(_WIN32) || defined(_WIN64)
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        _ftprintf(stderr, _T("Crush (clGetPlatformIDs)\n"));
        opencl_crush = true; //クラッシュフラグを立てる
        return CL_INVALID_VALUE;
    }
    #endif //#if defined(_WIN32) || defined(_WIN64)
    if (platform_count > 0) {
        cl_platform_id *platform_list = (cl_platform_id *)malloc(sizeof(platform_list[0]) * platform_count);
        if (platform_list == nullptr) {
            return CL_OUT_OF_HOST_MEMORY;
        }
        #if defined(_WIN32) || defined(_WIN64)
        __try {
        #endif //#if defined(_WIN32) || defined(_WIN64)
            if (CL_SUCCESS != (ret = cl->getPlatformIDs(platform_count, platform_list, &platform_count))) {
                _ftprintf(stderr, _T("Error (getPlatformIDs): %d\n"), ret);
                return ret;
            }
        #if defined(_WIN32) || defined(_WIN64)
        } __except (EXCEPTION_EXECUTE_HANDLER) {
            _ftprintf(stderr, _T("Crush (getPlatformIDs)\n"));
            opencl_crush = true; //クラッシュフラグを立てる
            ret = CL_INVALID_VALUE;
            platform_count = 0;
        }
        #endif //#if defined(_WIN32) || defined(_WIN64)

        for (uint32_t i = 0; i < platform_count; i++) {
            bool targetVendor = true;
            if (VendorName != nullptr) {
                char buf[1024] = { 0 };
                #if defined(_WIN32) || defined(_WIN64)
                __try {
                #endif //#if defined(_WIN32) || defined(_WIN64)
                    targetVendor = (CL_SUCCESS == cl->getPlatformInfo(platform_list[i], CL_PLATFORM_VENDOR, _countof(buf), buf, NULL)
                        && cl_check_vendor_name(buf, VendorName));
                #if defined(_WIN32) || defined(_WIN64)
                } __except (EXCEPTION_EXECUTE_HANDLER) {
                    _ftprintf(stderr, _T("Crush (getPlatformInfo)\n"));
                    opencl_crush = true; //クラッシュフラグを立てる
                    ret = CL_INVALID_VALUE;
                    break;
                }
                #endif //#if defined(_WIN32) || defined(_WIN64)
            }
            if (targetVendor) {
                cl_uint device_count = 0;
                #if defined(_WIN32) || defined(_WIN64)
                __try {
                #endif //#if defined(_WIN32) || defined(_WIN64)
                    if (CL_SUCCESS != (ret = cl->getDeviceIDs(platform_list[i], device_type, 0, NULL, &device_count))) {
                        continue;
                    }
                #if defined(_WIN32) || defined(_WIN64)
                } __except (EXCEPTION_EXECUTE_HANDLER) {
                    _ftprintf(stderr, _T("Crush (getDeviceIDs)\n"));
                    ret = CL_INVALID_VALUE;
                    break;
                }
                #endif //#if defined(_WIN32) || defined(_WIN64)
                if (device_count == 0) {
                    continue;
                }
                bool got_result = false;
                cl_device_id *device_list = (cl_device_id *)malloc(sizeof(device_list[0]) * device_count);
                if (device_list != nullptr) {
                    #if defined(_WIN32) || defined(_WIN64)
                    __try {
                    #endif //#if defined(_WIN32) || defined(_WIN64)
                        ret = cl->getDeviceIDs(platform_list[i], device_type, device_count, &device_list[0], &device_count);
                    #if defined(_WIN32) || defined(_WIN64)
                    } __except (EXCEPTION_EXECUTE_HANDLER) {
                        _ftprintf(stderr, _T("Crush (getDeviceIDs)\n"));
                        opencl_crush = true; //クラッシュフラグを立てる
                        ret = CL_INVALID_VALUE;
                    }
                    #endif //#if defined(_WIN32) || defined(_WIN64)
                    if (ret == CL_SUCCESS) {
                        cl_data->platformID = platform_list[i];
                        cl_data->deviceID = device_list[0];
                        got_result = true;
                    }
                }
                if (device_list) free(device_list);
                if (!got_result) continue;
                break;
            }
        }
        if (platform_list) free(platform_list);
    }
    return ret;
}

int cl_get_device_max_clock_frequency_mhz(const cl_data_t *cl_data, const cl_func_t *cl) {
    int frequency = 0;
    if (opencl_crush) {
        return frequency;
    }
    #if defined(_WIN32) || defined(_WIN64)
    __try {
    #endif //#if defined(_WIN32) || defined(_WIN64)
        char cl_info_buffer[1024] = { 0 };
        if (CL_SUCCESS == cl->getDeviceInfo(cl_data->deviceID, CL_DEVICE_MAX_CLOCK_FREQUENCY, _countof(cl_info_buffer), cl_info_buffer, NULL)) {
            frequency = *(cl_uint *)cl_info_buffer;
        }
    #if defined(_WIN32) || defined(_WIN64)
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        _ftprintf(stderr, _T("Crush (getDeviceIDs)\n"));
        opencl_crush = true;
    }
    #endif //#if defined(_WIN32) || defined(_WIN64)
    return frequency;
}

int cl_get_device_max_compute_units(const cl_data_t *cl_data, const cl_func_t *cl) {
    int cu = 0;
    if (opencl_crush) {
        return cu;
    }
    #if defined(_WIN32) || defined(_WIN64)
    __try {
    #endif //#if defined(_WIN32) || defined(_WIN64)
        char cl_info_buffer[1024] = { 0 };
        if (CL_SUCCESS == cl->getDeviceInfo(cl_data->deviceID, CL_DEVICE_MAX_COMPUTE_UNITS, _countof(cl_info_buffer), cl_info_buffer, NULL)) {
            cu = *(cl_uint *)cl_info_buffer;
        }
    #if defined(_WIN32) || defined(_WIN64)
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        _ftprintf(stderr, _T("Crush (getDeviceIDs)\n"));
        opencl_crush = true;
    }
    #endif //#if defined(_WIN32) || defined(_WIN64)
    return cu;
}

cl_int cl_get_device_name(const cl_data_t *cl_data, const cl_func_t *cl, TCHAR *buffer, unsigned int buffer_size) {
    if (opencl_crush) {
        return CL_INVALID_VALUE;
    }
    cl_int ret = CL_SUCCESS;
    char cl_info_buffer[1024] = { 0 };
    #if defined(_WIN32) || defined(_WIN64)
    __try {
    #endif //#if defined(_WIN32) || defined(_WIN64)
        ret = cl->getDeviceInfo(cl_data->deviceID, CL_DEVICE_NAME, _countof(cl_info_buffer), cl_info_buffer, NULL);
    #if defined(_WIN32) || defined(_WIN64)
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        _ftprintf(stderr, _T("Crush (getDeviceInfo)\n"));
        opencl_crush = true;
        ret = CL_INVALID_VALUE;
    }
    #endif //#if defined(_WIN32) || defined(_WIN64)
    if (ret == CL_SUCCESS) {
        to_tchar(buffer, buffer_size, cl_info_buffer);
    } else {
        _tcscpy_s(buffer, buffer_size, _T("Unknown"));
    }
    return ret;
}

cl_int cl_get_driver_version(const cl_data_t *cl_data, const cl_func_t *cl, TCHAR *buffer, unsigned int buffer_size) {
    if (opencl_crush) {
        return CL_INVALID_VALUE;
    }
    cl_int ret = CL_SUCCESS;
    char cl_info_buffer[1024] = { 0 };
    #if defined(_WIN32) || defined(_WIN64)
    __try {
    #endif //#if defined(_WIN32) || defined(_WIN64)
        ret = cl->getDeviceInfo(cl_data->deviceID, CL_DRIVER_VERSION, _countof(cl_info_buffer), cl_info_buffer, NULL);
    #if defined(_WIN32) || defined(_WIN64)
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        _ftprintf(stderr, _T("Crush (getDeviceInfo)\n"));
        opencl_crush = true;
        ret = CL_INVALID_VALUE;
    }
    #endif //#if defined(_WIN32) || defined(_WIN64)
    if (ret == CL_SUCCESS) {
        to_tchar(buffer, buffer_size, cl_info_buffer);
    } else {
        _tcscpy_s(buffer, buffer_size, _T("Unknown"));
    }
    return ret;
}

void cl_release(cl_data_t *cl_data, cl_func_t *cl) {
    if (cl) {
        if (cl_data) {
            if (cl_data->kernel) cl->releaseKernel(cl_data->kernel);
            if (cl_data->program) cl->releaseProgram(cl_data->program);
            if (cl_data->commands) cl->releaseCommandQueue(cl_data->commands);
            if (cl_data->contextCL) cl->releaseContext(cl_data->contextCL);
        }
        cl_release_func(cl);
    }
}

#endif
