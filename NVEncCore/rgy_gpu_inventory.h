#pragma once

#include <charconv>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <string>
#include <string_view>

#if defined(_WIN32)
#include <windows.h>
#include <winternl.h>
#include <d3dkmthk.h>
#pragma comment(lib, "gdi32.lib")
#elif defined(__linux__)
#include <filesystem>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#endif

namespace rgy_gpu_inventory {

inline std::optional<std::string> format_pci_bdf(unsigned domain, unsigned bus, unsigned device, unsigned function) {
    if (domain > 0xffff || bus > 0xff || device > 0x1f || function > 7) {
        return std::nullopt;
    }
    char text[13] = {};
    std::snprintf(text, sizeof(text), "%04x:%02x:%02x.%x", domain, bus, device, function);
    return std::string(text);
}

inline std::optional<std::string> normalize_pci_bdf(std::string_view text) {
    unsigned fields[4] = {};
    constexpr std::string_view separators = "::.";
    for (size_t i = 0; i < 4; ++i) {
        const auto end = i < separators.size() ? text.find(separators[i]) : text.size();
        if (end == 0 || end == std::string_view::npos) {
            return std::nullopt;
        }
        const auto part = text.substr(0, end);
        const auto parsed = std::from_chars(part.data(), part.data() + part.size(), fields[i], 16);
        if (parsed.ec != std::errc() || parsed.ptr != part.data() + part.size()) {
            return std::nullopt;
        }
        text.remove_prefix(end + (i < separators.size() ? 1 : 0));
    }
    return format_pci_bdf(fields[0], fields[1], fields[2], fields[3]);
}

#if defined(_WIN32)
inline std::string luid_string(const LUID& luid) {
    const auto value = (static_cast<uint64_t>(static_cast<uint32_t>(luid.HighPart)) << 32) | luid.LowPart;
    char text[17] = {};
    std::snprintf(text, sizeof(text), "%016llx", static_cast<unsigned long long>(value));
    return text;
}

inline bool query_adapter_bdf(const LUID& luid, UINT& bus, UINT& device, UINT& function) {
    D3DKMT_OPENADAPTERFROMLUID open = {};
    open.AdapterLuid = luid;
    if (D3DKMTOpenAdapterFromLuid(&open) != 0) {
        return false;
    }
    D3DKMT_ADAPTERADDRESS address = {};
    D3DKMT_QUERYADAPTERINFO query = {};
    query.hAdapter = open.hAdapter;
    query.Type = KMTQAITYPE_ADAPTERADDRESS;
    query.pPrivateDriverData = &address;
    query.PrivateDriverDataSize = sizeof(address);
    const auto status = D3DKMTQueryAdapterInfo(&query);
    D3DKMT_CLOSEADAPTER close = {};
    close.hAdapter = open.hAdapter;
    D3DKMTCloseAdapter(&close);
    if (status != 0) {
        return false;
    }
    bus = address.BusNumber;
    device = address.DeviceNumber;
    function = address.FunctionNumber;
    return true;
}

inline std::optional<std::string> pci_bdf_from_luid(const LUID& luid) {
    UINT bus = 0, device = 0, function = 0;
    if (!query_adapter_bdf(luid, bus, device, function)) {
        return std::nullopt;
    }
    // Windowsのアダプターアドレスにドメイン番号の欄はないため、ドメイン0の表記に揃える。
    return format_pci_bdf(0, bus, device, function);
}
#elif defined(__linux__)
inline bool drm_node_name(std::string_view name, std::string_view prefix) {
    return name.size() > prefix.size() && name.substr(0, prefix.size()) == prefix
        && name.find_first_not_of("0123456789", prefix.size()) == std::string_view::npos;
}

inline std::optional<std::string> device_name_from_link(const std::filesystem::path& link) {
    std::error_code error;
    const auto target = std::filesystem::canonical(link, error);
    return error ? std::nullopt : std::optional<std::string>(target.filename().string());
}

inline std::optional<std::string> pci_bdf_from_device_link(const std::filesystem::path& link) {
    const auto name = device_name_from_link(link);
    return name ? normalize_pci_bdf(*name) : std::nullopt;
}

inline std::optional<std::string> drm_device_name_from_fd(int fd, const std::filesystem::path& sysfs = "/sys") {
    struct stat info = {};
    if (fstat(fd, &info) != 0 || !S_ISCHR(info.st_mode)) {
        return std::nullopt;
    }
    // 実際に開いたキャラクターデバイスの番号から対応するPCIデバイスを照合する。
    const auto node = std::to_string(major(info.st_rdev)) + ":" + std::to_string(minor(info.st_rdev));
    return device_name_from_link(sysfs / "dev" / "char" / node / "device");
}

inline std::optional<std::string> pci_bdf_from_drm_fd(int fd, const std::filesystem::path& sysfs = "/sys") {
    const auto name = drm_device_name_from_fd(fd, sysfs);
    return name ? normalize_pci_bdf(*name) : std::nullopt;
}

inline std::optional<std::string> drm_render_node_from_fd(int fd) {
    std::error_code error;
    const auto path = std::filesystem::read_symlink("/proc/self/fd/" + std::to_string(fd), error);
    if (error || !path.is_absolute() || !drm_node_name(path.filename().string(), "renderD")) {
        return std::nullopt;
    }
    return path.string();
}
#endif

}
