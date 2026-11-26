#include "rgy_version.h"

#if ENABLE_PROBE_BUILD
#include <algorithm>
#include <cstdio>
#include <cstdint>
#include "rgy_util.h"
#include "NVEncDevice.h"
#include "NVEncProbe.h"

namespace {

using rgy_probe::Json;

Json cuda_pci_bdf(CUdevice device) {
    char busId[64] = {};
    if (cuDeviceGetPCIBusId(busId, sizeof(busId), device) != CUDA_SUCCESS) {
        return nullptr;
    }
    return rgy_probe::identity_string(rgy_gpu_inventory::normalize_pci_bdf(busId));
}

Json cuda_uuid(CUdevice device) {
    CUuuid uuid = {};
    if (cuDeviceGetUuid(&uuid, device) != CUDA_SUCCESS) {
        return nullptr;
    }
    static const char digits[] = "0123456789abcdef";
    std::string id = "GPU-";
    for (size_t i = 0; i < sizeof(uuid.bytes); ++i) {
        if (i == 4 || i == 6 || i == 8 || i == 10) {
            id += '-';
        }
        const auto byte = static_cast<unsigned char>(uuid.bytes[i]);
        id += digits[byte >> 4];
        id += digits[byte & 0x0f];
    }
    return id;
}

Json cuda_luid([[maybe_unused]] CUdevice device) {
#if defined(_WIN32)
    const auto module = GetModuleHandle(_T("nvcuda.dll"));
    const auto getLuid = module ? reinterpret_cast<decltype(&cuDeviceGetLuid)>(GetProcAddress(module, "cuDeviceGetLuid")) : nullptr;
    char luid[8] = {};
    unsigned int nodeMask = 0;
    if (getLuid && getLuid(luid, &nodeMask, device) == CUDA_SUCCESS) {
        // CUDAのバイト列はWindowsのLUIDと同じリトルエンディアン。
        LUID adapter = {};
        static_assert(sizeof(adapter) == sizeof(luid));
        std::memcpy(&adapter, luid, sizeof(adapter));
        return rgy_gpu_inventory::luid_string(adapter);
    }
#endif
    return nullptr;
}

} // namespace

rgy_probe::Json nvenc_probe_cuda_failure(const char *reason, const char *api, CUresult error) {
    switch (error) {
    case CUDA_ERROR_NO_DEVICE:
    case CUDA_ERROR_INVALID_DEVICE:
        reason = "device_not_visible";
        break;
    case CUDA_ERROR_NOT_PERMITTED:
        reason = "permission_denied";
        break;
    case CUDA_ERROR_SYSTEM_DRIVER_MISMATCH:
    case CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE:
        reason = "driver_api_mismatch";
        break;
    case CUDA_ERROR_STUB_LIBRARY:
        reason = "library_load_failed";
        break;
    default:
        break;
    }
    const char *message = nullptr;
    cuGetErrorString(error, &message);
    return rgy_probe::failure(reason, api, static_cast<int>(error), message ? Json(message) : Json(nullptr));
}

rgy_probe::Json nvenc_probe_device_identity(int index, CUdevice device) {
    return {
        {"encoder_device_index", index},
        {"pci_bdf", cuda_pci_bdf(device)},
        {"luid", cuda_luid(device)},
        {"drm_render_node", nullptr}
    };
}

int nvenc_probe_inventory() {
    auto devices = Json::array();
    rgy_probe::event("load_library");
    if (!check_if_nvcuda_dll_available()) {
        rgy_probe::finish_list(std::move(devices), rgy_probe::get_failure());
        return 1;
    }

    rgy_probe::event("enumerate");
    const auto initResult = cuInit(0);
    if (initResult != CUDA_SUCCESS) {
        rgy_probe::finish_list(std::move(devices), nvenc_probe_cuda_failure("device_not_visible", "cuInit", initResult));
        return 1;
    }
    int count = 0;
    const auto countResult = cuDeviceGetCount(&count);
    if (countResult != CUDA_SUCCESS) {
        rgy_probe::finish_list(std::move(devices), nvenc_probe_cuda_failure("device_not_visible", "cuDeviceGetCount", countResult));
        return 1;
    }

    Json failure = nullptr;
    for (int index = 0; index < count; ++index) {
        CUdevice device = 0;
        const auto deviceResult = cuDeviceGet(&device, index);
        if (deviceResult != CUDA_SUCCESS) {
            if (failure.is_null()) {
                failure = nvenc_probe_cuda_failure("device_not_visible", "cuDeviceGet", deviceResult);
            }
            continue;
        }

        char name[256] = {};
        const auto nameResult = cuDeviceGetName(name, sizeof(name), device);
        size_t bytes = 0;
        const auto memResult = cuDeviceTotalMem(&bytes, device);
        int integrated = 0;
        const auto integratedResult = cuDeviceGetAttribute(&integrated, CU_DEVICE_ATTRIBUTE_INTEGRATED, device);
        devices.push_back({
            {"cuda_index", index},
            {"name", nameResult == CUDA_SUCCESS ? Json(rgy_probe::utf8(char_to_tstring(name).c_str())) : Json(nullptr)},
            {"pci_bdf", cuda_pci_bdf(device)},
            {"uuid", cuda_uuid(device)},
            {"total_mem_bytes", memResult == CUDA_SUCCESS ? Json(bytes) : Json(nullptr)},
            {"integrated", integratedResult == CUDA_SUCCESS ? Json(integrated != 0) : Json(nullptr)}
        });
    }
    const bool failed = devices.empty() && !failure.is_null();
    rgy_probe::finish_list(std::move(devices), std::move(failure));
    return failed ? 1 : 0;
}

int nvenc_probe_devices() {
    auto devices = Json::array();
    rgy_probe::event("load_library");
    NVEncCtrl ctrl;
    const auto vulkan = ENABLE_VULKAN ? RGYParamInitVulkan::TargetVendor : RGYParamInitVulkan::Disable;
    if (ctrl.Initialize(-1, vulkan, RGY_LOG_INFO) != RGY_ERR_NONE) {
        rgy_probe::finish_list(std::move(devices), rgy_probe::get_failure());
        return 1;
    }

    std::vector<std::unique_ptr<NVGPUInfo>> gpuList;
    const auto status = ctrl.GetDeviceList(gpuList, 0, false, [&](int index, RGY_ERR error) {
        auto failure = rgy_probe::get_failure();
        if (failure.is_null()) {
            failure = rgy_probe::failure("encoder_init_failed", nullptr, static_cast<int>(error), rgy_probe::utf8(get_err_mes(error)));
        }

        // NVEncの照会に失敗しても、CUDAから識別子を取得できればGPUの行を残す。
        CUdevice cudaDevice = 0;
        if (cuDeviceGet(&cudaDevice, index) != CUDA_SUCCESS) {
            return;
        }
        auto device = nvenc_probe_device_identity(index, cudaDevice);
        char name[256] = {};
        device["name"] = cuDeviceGetName(name, sizeof(name), cudaDevice) == CUDA_SUCCESS
            ? Json(rgy_probe::utf8(char_to_tstring(name).c_str())) : Json(nullptr);
        device["encode_test_visible"] = false;
        rgy_probe::set_encode_failure(device, std::move(failure));
        devices.push_back(std::move(device));
    });
    if (status != RGY_ERR_NONE) {
        auto failure = devices.empty() ? rgy_probe::get_failure() : devices.front().at("encode_failure");
        if (failure.is_null()) {
            failure = rgy_probe::failure("device_not_visible");
        }
        rgy_probe::finish_list(std::move(devices), std::move(failure));
        return 1;
    }
    for (const auto &gpu : gpuList) {
        auto device = nvenc_probe_device_identity(gpu->id(), gpu->cudevicetx());
        device["name"] = rgy_probe::utf8(gpu->name().c_str());
        // --check-hwと同じNVEnc GUIDの一覧から判定する。
        const auto& features = gpu->nvenc_codec_features();
        const auto supports = [&](const GUID& codec) {
            return std::any_of(features.begin(), features.end(), [&](const auto& feature) {
                return feature.codec == codec;
            });
        };
        rgy_probe::set_encode_caps(device, supports(NV_ENC_CODEC_H264_GUID),
            supports(NV_ENC_CODEC_HEVC_GUID), supports(NV_ENC_CODEC_AV1_GUID));
        devices.push_back(std::move(device));
    }
    std::sort(devices.begin(), devices.end(), [](const Json& a, const Json& b) {
        return a.at("encoder_device_index").get<int>() < b.at("encoder_device_index").get<int>();
    });
    rgy_probe::finish_list(std::move(devices));
    return 0;
}
#endif
