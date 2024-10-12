// -----------------------------------------------------------------------------------------
//     VCEEnc by rigaya
// -----------------------------------------------------------------------------------------
// The MIT License
//
// Copyright (c) 2021 rigaya
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
// IABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//
// ------------------------------------------------------------------------------------------

#include "vce_device_vulkan.h"
#include "vce_util.h"

#if ENABLE_VULKAN

DeviceVulkan::DeviceVulkan() :
    m_name(_T("devVulkan")),
    m_VulkanDev(),
    m_displayDeviceName(),
    m_vk(),
    m_uQueueGraphicsFamilyIndex(std::numeric_limits<decltype(m_uQueueGraphicsFamilyIndex)>::max()),
    m_uQueueComputeFamilyIndex(std::numeric_limits<decltype(m_uQueueComputeFamilyIndex)>::max()),
    m_hQueueGraphics(NULL),
    m_hQueueCompute(NULL),
    m_log() {
	m_VulkanDev = {};
	m_VulkanDev.cbSizeof = sizeof(amf::AMFVulkanDevice);
}
DeviceVulkan::~DeviceVulkan() {
    Terminate();
}

amf::AMFVulkanDevice* DeviceVulkan::GetDevice() {
    return &m_VulkanDev;
}

int DeviceVulkan::adapterCount() {
    RGY_ERR res = RGY_ERR_NONE;
    if (m_vk.init() != 0) {
        return 0;
    }

    if ((res = CreateInstance()) != RGY_ERR_NONE) {
        return 0;
    }

    uint32_t physicalDeviceCount = 0;
    if ((res = err_to_rgy(GetVulkan()->vkEnumeratePhysicalDevices(m_VulkanDev.hInstance, &physicalDeviceCount, nullptr))) != RGY_ERR_NONE) {
        return 0;
    }
    return (int)physicalDeviceCount;
}

RGY_ERR DeviceVulkan::Init(int adapterID, amf::AMFContext *pContext, std::shared_ptr<RGYLog> log) {
    RGY_ERR res = RGY_ERR_NONE;
    m_log = log;

    amf::AMFContext1Ptr pContext1(pContext);
    amf_size nCount = 0;
    pContext1->GetVulkanDeviceExtensions(&nCount, NULL);
    std::vector<const char*> deviceExtensions;
    if (nCount > 0) { 
        deviceExtensions.resize(nCount);
        pContext1->GetVulkanDeviceExtensions(&nCount, deviceExtensions.data());
    }

    if (m_vk.init() != 0) {
        AddMessage(RGY_LOG_ERROR, _T("LoadFunctionsTable() failed - check if the proper Vulkan SDK is installed\n"));
        return RGY_ERR_NULL_PTR;
    }

    if ((res = CreateInstance()) != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("CreateInstance() failed\n"));
        return res;
    }

#if defined(_DEBUG) && defined(ENABLE_VALIDATION)
    const bool bDebug = true;
#else
    const bool bDebug = false;
#endif
    if (m_vk.load(m_VulkanDev.hInstance, bDebug) != 0) {
        AddMessage(RGY_LOG_ERROR, _T("LoadInstanceFunctionsTableExt() failed - check if the proper Vulkan SDK is installed\n"));
        return RGY_ERR_NULL_PTR;
    }

    // load instance based functions
    if (CreateDeviceAndFindQueues(adapterID, deviceExtensions) != 0) {
        AddMessage(RGY_LOG_ERROR, _T("CreateDeviceAndFindQueues() failed\n"));
        return RGY_ERR_NULL_PTR;
    }

    if (m_vk.load(m_VulkanDev.hDevice) != 0) {
        AddMessage(RGY_LOG_ERROR, _T("LoadDeviceFunctionsTableExt() failed - check if the proper Vulkan SDK is installed\n"));
        return RGY_ERR_NULL_PTR;
    }

    return RGY_ERR_NONE;
}

RGY_ERR DeviceVulkan::Terminate() {
    m_hQueueGraphics = NULL;
    m_hQueueCompute = NULL;

    if (m_VulkanDev.hDevice != VK_NULL_HANDLE) {
        GetVulkan()->vkDestroyDevice(m_VulkanDev.hDevice, nullptr);
    }
    if (m_VulkanDev.hInstance != VK_NULL_HANDLE) {
        GetVulkan()->vkDestroyInstance(m_VulkanDev.hInstance, nullptr);
    }
	m_VulkanDev = {};
	m_VulkanDev.cbSizeof = sizeof(amf::AMFVulkanDevice);
    return RGY_ERR_NONE;
}

//-------------------------------------------------------------------------------------------------
std::vector<const char*> DeviceVulkan::GetDebugInstanceExtensionNames() {
    uint32_t extensionCount = 0;
    GetVulkan()->vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);

    std::vector<VkExtensionProperties> instanceExtensions{ extensionCount };
    GetVulkan()->vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, instanceExtensions.data());

    std::vector<const char*> result;
    for (const VkExtensionProperties& e : instanceExtensions) {
        if (strcmp(e.extensionName, "VK_EXT_debug_report") == 0) {
            result.push_back("VK_EXT_debug_report");
        }
    }
    return result;
}
//-------------------------------------------------------------------------------------------------
std::vector<const char*> DeviceVulkan::GetDebugInstanceLayerNames() {
    uint32_t layerCount = 0;
    GetVulkan()->vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
    std::vector<VkLayerProperties> instanceLayers{ layerCount };
    GetVulkan()->vkEnumerateInstanceLayerProperties(&layerCount, instanceLayers.data());
    std::vector<const char*> result;
    for (const VkLayerProperties& p : instanceLayers) {
        if (strcmp(p.layerName, "VK_LAYER_LUNARG_standard_validation") == 0) {
            result.push_back("VK_LAYER_LUNARG_standard_validation");
        }
    }
    return result;
}
//-------------------------------------------------------------------------------------------------
std::vector<const char*> DeviceVulkan::GetDebugDeviceLayerNames(VkPhysicalDevice device) {
    uint32_t layerCount = 0;
    GetVulkan()->vkEnumerateDeviceLayerProperties(device, &layerCount, nullptr);
    std::vector<VkLayerProperties> deviceLayers{ layerCount };
    GetVulkan()->vkEnumerateDeviceLayerProperties(device, &layerCount, deviceLayers.data());
    std::vector<const char*> result;
    for (const VkLayerProperties& p : deviceLayers) {
        if (strcmp(p.layerName, "VK_LAYER_LUNARG_standard_validation") == 0) {
            result.push_back("VK_LAYER_LUNARG_standard_validation");
        }
    }
    return result;
}
RGY_ERR DeviceVulkan::CreateInstance() {
    RGY_ERR res = RGY_ERR_NONE;

    if (m_VulkanDev.hInstance != VK_NULL_HANDLE) {
        return RGY_ERR_NONE;
    }
    // VkInstanceCreateInfo
    ///////////////////////
    VkInstanceCreateInfo instanceCreateInfo = {};
    instanceCreateInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;

    std::vector<const char*> instanceExtensions = {
#if defined(_WIN32)
        VK_KHR_SURFACE_EXTENSION_NAME,
        VK_KHR_WIN32_SURFACE_EXTENSION_NAME,
#elif defined(__linux)
        "VK_KHR_surface",
        VK_KHR_XLIB_SURFACE_EXTENSION_NAME,
#endif
    };

#if defined(_DEBUG) && defined(ENABLE_VALIDATION)
	std::vector<const char*> debugInstanceExtensionNames = GetDebugInstanceExtensionNames();

    instanceExtensions.insert(instanceExtensions.end(),
        debugInstanceExtensionNames.begin(),
        debugInstanceExtensionNames.end());
#endif

    instanceCreateInfo.ppEnabledExtensionNames = instanceExtensions.data();
    instanceCreateInfo.enabledExtensionCount = static_cast<uint32_t> (instanceExtensions.size());

    std::vector<const char*> instanceLayers;

#if defined(_DEBUG) && defined(ENABLE_VALIDATION)
	std::vector<const char*> debugInstanceLayerNames = GetDebugInstanceLayerNames();

    instanceLayers.insert(instanceLayers.end(),
        debugInstanceLayerNames.begin(),
        debugInstanceLayerNames.end());
#endif

    instanceCreateInfo.ppEnabledLayerNames = instanceLayers.data();
    instanceCreateInfo.enabledLayerCount = static_cast<uint32_t> (instanceLayers.size());

    // VkApplicationInfo
    ///////////////////////
    VkApplicationInfo applicationInfo = {};
    applicationInfo.sType               = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    applicationInfo.apiVersion          = VK_API_VERSION_1_0;
    applicationInfo.applicationVersion  = VK_MAKE_VERSION(1, 0, 0);
    applicationInfo.engineVersion       = VK_MAKE_VERSION(1, 0, 0);
    applicationInfo.pApplicationName    = "AMF Vulkan application";
    applicationInfo.pEngineName         = "AMD Vulkan Sample Engine";

    instanceCreateInfo.pApplicationInfo = &applicationInfo;
    
    if ((res = err_to_rgy(GetVulkan()->vkCreateInstance(&instanceCreateInfo, nullptr, &m_VulkanDev.hInstance))) != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("CreateInstance() failed to vkCreateInstance, Error=%s\n"), get_err_mes(res));
        return res;
    }

    return RGY_ERR_NONE;
}

RGY_ERR DeviceVulkan::CreateDeviceAndFindQueues(int adapterID, std::vector<const char*> &deviceExtensions) {
    RGY_ERR res = RGY_ERR_NONE;
    uint32_t physicalDeviceCount = 0;
    if ((res = err_to_rgy(GetVulkan()->vkEnumeratePhysicalDevices(m_VulkanDev.hInstance, &physicalDeviceCount, nullptr))) != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("CreateDeviceAndQueues() failed to vkEnumeratePhysicalDevices, Error=%s\n"), get_err_mes(res));
        return res;
    }

    std::vector<VkPhysicalDevice> physicalDevices{ physicalDeviceCount };
    if ((res = err_to_rgy(GetVulkan()->vkEnumeratePhysicalDevices(m_VulkanDev.hInstance, &physicalDeviceCount, physicalDevices.data()))) != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("CreateDeviceAndQueues() failed to vkEnumeratePhysicalDevices, Error=%s\n"), get_err_mes(res));
        return res;
    }

    if (adapterID < 0) {
        adapterID = 0;
    }
    if (adapterID >= (int)physicalDeviceCount) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid Adapter ID: %d"), adapterID);
        return RGY_ERR_UNKNOWN;
    }

    m_VulkanDev.hPhysicalDevice = physicalDevices[adapterID];
    std::vector<VkDeviceQueueCreateInfo> deviceQueueCreateInfoItems;
    std::vector<float>                   queuePriorities;

    uint32_t queueFamilyPropertyCount = 0;

    GetVulkan()->vkGetPhysicalDeviceQueueFamilyProperties(m_VulkanDev.hPhysicalDevice, &queueFamilyPropertyCount, nullptr);

    std::vector<VkQueueFamilyProperties> queueFamilyProperties{ queueFamilyPropertyCount };

    GetVulkan()->vkGetPhysicalDeviceQueueFamilyProperties(m_VulkanDev.hPhysicalDevice, &queueFamilyPropertyCount, queueFamilyProperties.data());

    m_uQueueGraphicsFamilyIndex  = std::numeric_limits<uint32_t>::max();
    m_uQueueComputeFamilyIndex = std::numeric_limits<uint32_t>::max();


    uint32_t uQueueGraphicsIndex = std::numeric_limits<uint32_t>::max();
    uint32_t uQueueComputeIndex = std::numeric_limits<uint32_t>::max();
    for (uint32_t i = 0; i < queueFamilyPropertyCount; i++) {
        VkQueueFamilyProperties &queueFamilyProperty = queueFamilyProperties[i];
        if (queuePriorities.size() < queueFamilyProperty.queueCount) {
            queuePriorities.resize(queueFamilyProperty.queueCount, 1.0f);
        }
    }
    for (uint32_t i = 0; i < queueFamilyPropertyCount; i++) {
        VkQueueFamilyProperties &queueFamilyProperty = queueFamilyProperties[i];
        VkDeviceQueueCreateInfo queueCreateInfo = {};


        queueCreateInfo.pQueuePriorities = &queuePriorities[0];
        queueCreateInfo.queueFamilyIndex = i;
        queueCreateInfo.queueCount = queueFamilyProperty.queueCount;
        queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;

        if ((queueFamilyProperty.queueFlags & VK_QUEUE_COMPUTE_BIT) && (queueFamilyProperty.queueFlags & VK_QUEUE_GRAPHICS_BIT) == 0 && m_uQueueComputeFamilyIndex == std::numeric_limits<decltype(m_uQueueComputeFamilyIndex)>::max()) {
            m_uQueueComputeFamilyIndex = i;
            uQueueComputeIndex = 0;
        }
        if ((queueFamilyProperty.queueFlags & VK_QUEUE_GRAPHICS_BIT) && m_uQueueGraphicsFamilyIndex == std::numeric_limits<decltype(m_uQueueGraphicsFamilyIndex)>::max()) {
            m_uQueueGraphicsFamilyIndex = i;
            uQueueGraphicsIndex = 0;
        }
        if (queueFamilyProperty.queueCount > 1) {
            queueCreateInfo.queueCount = 1;
        }

        deviceQueueCreateInfoItems.push_back(queueCreateInfo);
    }
    VkDeviceCreateInfo deviceCreateInfo = {};

    deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceCreateInfo.queueCreateInfoCount = static_cast<uint32_t>(deviceQueueCreateInfoItems.size());
    deviceCreateInfo.pQueueCreateInfos = &deviceQueueCreateInfoItems[0];

    VkPhysicalDeviceFeatures features = {};
	GetVulkan()->vkGetPhysicalDeviceFeatures(m_VulkanDev.hPhysicalDevice, &features);
    deviceCreateInfo.pEnabledFeatures = &features;

    std::vector<const char*> deviceLayers;

#if defined(_DEBUG) && defined(ENABLE_VALIDATION)
	std::vector<const char*> debugDeviceLayerNames = GetDebugDeviceLayerNames(physicalDevices[0]);

    deviceLayers.insert(deviceLayers.end(),
        debugDeviceLayerNames.begin(),
        debugDeviceLayerNames.end());
#endif

    deviceCreateInfo.ppEnabledLayerNames = deviceLayers.data();
    deviceCreateInfo.enabledLayerCount = static_cast<uint32_t> (deviceLayers.size());

    deviceExtensions.insert(deviceExtensions.begin(), "VK_KHR_swapchain");

    deviceCreateInfo.ppEnabledExtensionNames = deviceExtensions.data();
    deviceCreateInfo.enabledExtensionCount = static_cast<uint32_t> (deviceExtensions.size());

    if ((res = err_to_rgy(GetVulkan()->vkCreateDevice(m_VulkanDev.hPhysicalDevice, &deviceCreateInfo, nullptr, &m_VulkanDev.hDevice)))!= RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("CreateDeviceAndQueues() vkCreateDevice() failed, Error=%s\n"), get_err_mes(res));
        return res;
    }
    if (m_VulkanDev.hDevice == nullptr) {
        AddMessage(RGY_LOG_ERROR, _T("CreateDeviceAndQueues() vkCreateDevice() returned nullptr\n"));
        return RGY_ERR_NULL_PTR;
    }
    
	GetVulkan()->vkGetDeviceQueue(m_VulkanDev.hDevice, m_uQueueGraphicsFamilyIndex, uQueueGraphicsIndex, &m_hQueueGraphics);
	GetVulkan()->vkGetDeviceQueue(m_VulkanDev.hDevice, m_uQueueComputeFamilyIndex, uQueueComputeIndex, &m_hQueueCompute);

    return RGY_ERR_NONE;
}

RGYVulkanFuncs * DeviceVulkan::GetVulkan() {
	return &m_vk;
}

void DeviceVulkan::AddMessage(RGYLogLevel log_level, const tstring &str) {
    if (m_log == nullptr || log_level < m_log->getLogLevel(RGY_LOGT_DEV)) {
        return;
    }
    auto lines = split(str, _T("\n"));
    for (const auto &line : lines) {
        if (line[0] != _T('\0')) {
            m_log->write(log_level, RGY_LOGT_DEV, (m_name + _T(": ") + line + _T("\n")).c_str());
        }
    }
}
void DeviceVulkan::AddMessage(RGYLogLevel log_level, const TCHAR *format, ...) {
    if (m_log == nullptr || log_level < m_log->getLogLevel(RGY_LOGT_DEV)) {
        return;
    }

    va_list args;
    va_start(args, format);
    int len = _vsctprintf(format, args) + 1; // _vscprintf doesn't count terminating '\0'
    tstring buffer;
    buffer.resize(len, _T('\0'));
    _vstprintf_s(&buffer[0], len, format, args);
    va_end(args);
    AddMessage(log_level, buffer);
}
#endif
