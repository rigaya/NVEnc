//  -----------------------------------------------------------------------------------------
//    QSVEnc by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  -----------------------------------------------------------------------------------------

#include <tchar.h>
#include <vector>
#include "cl_func.h"

#if ENABLE_OPENCL

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
		return NULL != stristr(VendorName, "Advanced Micro Devices");
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
	if (cl->hdll) {
		FreeLibrary(cl->hdll);
	}
	ZeroMemory(cl, sizeof(cl_func_t));
}

cl_int cl_get_platform_and_device(const char *VendorName, cl_int device_type, cl_data_t *cl_data, const cl_func_t *cl) {
	using namespace std;
	cl_uint size = 0;
	cl_int ret = CL_SUCCESS;

	if (CL_SUCCESS != (ret = cl->getPlatformIDs(0, NULL, &size))) {
		_ftprintf(stderr, _T("Error (clGetPlatformIDs): %d\n"), ret);
		return ret;
	}

	vector<cl_platform_id> platform_list(size);

	if (CL_SUCCESS != (ret = cl->getPlatformIDs(size, &platform_list[0], &size))) {
		_ftprintf(stderr, _T("Error (clGetPlatformIDs): %d\n"), ret);
		return ret;
	}

	auto checkPlatformForVendor = [cl, VendorName](cl_platform_id platform_id) {
		char buf[1024] = { 0 };
		return (CL_SUCCESS == cl->getPlatformInfo(platform_id, CL_PLATFORM_VENDOR, _countof(buf), buf, NULL)
			&& cl_check_vendor_name(buf, VendorName));
	};

	for (auto platform : platform_list) {
		if (checkPlatformForVendor(platform)) {
			if (CL_SUCCESS != (ret = cl->getDeviceIDs(platform, device_type, 0, NULL, &size))) {
				_ftprintf(stderr, _T("Error (clGetDeviceIDs): %d\n"), ret);
				return ret;
			}
			vector<cl_device_id> device_list(size);
			if (CL_SUCCESS != (ret = cl->getDeviceIDs(platform, device_type, size, &device_list[0], &size))) {
				_ftprintf(stderr, _T("Error (clGetDeviceIDs): %d\n"), ret);
				return ret;
			}
			cl_data->platformID = platform;
			cl_data->deviceID = device_list[0];
			break;
		}
	}

	return ret;
}

int cl_get_device_max_clock_frequency_mhz(const cl_data_t *cl_data, const cl_func_t *cl) {
	int frequency = 0;
	char cl_info_buffer[1024] = { 0 };
	if (CL_SUCCESS == cl->getDeviceInfo(cl_data->deviceID, CL_DEVICE_MAX_CLOCK_FREQUENCY, _countof(cl_info_buffer), cl_info_buffer, NULL)) {
		frequency = *(cl_uint *)cl_info_buffer;
	}
	return frequency;
}

void cl_release(cl_data_t *cl_data, cl_func_t *cl) {
	if (cl_data->kernel) cl->releaseKernel(cl_data->kernel);
	if (cl_data->program) cl->releaseProgram(cl_data->program);
	if (cl_data->commands) cl->releaseCommandQueue(cl_data->commands);
	if (cl_data->contextCL) cl->releaseContext(cl_data->contextCL);
	cl_release_func(cl);
}

#endif
