// -----------------------------------------------------------------------------------------
// VCEEnc by rigaya
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
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//
// --------------------------------------------------------------------------------------------

#include "NVEncFilterVulkan.h"
#include "rgy_device_vulkan.h"

#if ENABLE_VULKAN

CUDAVulkanFrame::CUDAVulkanFrame() :
    m_vk(nullptr),
    m_buffer(nullptr),
    m_bufferMemory(nullptr),
    m_bufferSize(0),
    m_cudaMem(nullptr),
    m_cudaPtr(nullptr),
    m_width(0),
    m_height(0) {
}

CUDAVulkanFrame::~CUDAVulkanFrame() {
    release();
}

VkExternalMemoryHandleTypeFlagBits CUDAVulkanFrame::getDefaultMemHandleType() const {
#if defined(_WIN32) || defined(_WIN64)
  return VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#else
  return VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif //defined(_WIN32) || defined(_WIN64)
}

uint32_t CUDAVulkanFrame::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memProperties;
    m_vk->GetVulkan()->vkGetPhysicalDeviceMemoryProperties(m_vk->GetPhysicalDevice(), &memProperties);
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if (typeFilter & (1 << i) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    return ~0;
}

void *CUDAVulkanFrame::getMemHandle(VkExternalMemoryHandleTypeFlagBits handleType) {
#if defined(_WIN32) || defined(_WIN64)
    HANDLE handle = 0;

    VkMemoryGetWin32HandleInfoKHR vkMemoryGetWin32HandleInfoKHR = {};
    vkMemoryGetWin32HandleInfoKHR.sType =
        VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
    vkMemoryGetWin32HandleInfoKHR.pNext = NULL;
    vkMemoryGetWin32HandleInfoKHR.memory = m_bufferMemory;
    vkMemoryGetWin32HandleInfoKHR.handleType = handleType;

    if (m_vk->GetVulkan()->vkGetMemoryWin32HandleKHR(m_device, &vkMemoryGetWin32HandleInfoKHR, &handle) != VK_SUCCESS) {
        return nullptr;
    }
    return (void *)handle;
#else
    int fd = -1;

    VkMemoryGetFdInfoKHR vkMemoryGetFdInfoKHR = {};
    vkMemoryGetFdInfoKHR.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
    vkMemoryGetFdInfoKHR.pNext = NULL;
    vkMemoryGetFdInfoKHR.memory = m_bufferMemory;
    vkMemoryGetFdInfoKHR.handleType = handleType;
    if (m_vk->GetVulkan()->vkGetMemoryFdKHR(m_vk->GetDevice(), &vkMemoryGetFdInfoKHR, &fd) != VK_SUCCESS) {
        return nullptr;
    }
    return (void *)(uintptr_t)fd;
#endif //defined(_WIN32) || defined(_WIN64)
}

RGY_ERR CUDAVulkanFrame::importCudaExternalMemory(VkExternalMemoryHandleTypeFlagBits handleType) {
    cudaExternalMemoryHandleDesc externalMemoryHandleDesc = {};

    if (handleType & VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT) {
        externalMemoryHandleDesc.type = cudaExternalMemoryHandleTypeOpaqueWin32;
    } else if (handleType & VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT) {
        externalMemoryHandleDesc.type = cudaExternalMemoryHandleTypeOpaqueWin32Kmt;
    } else if (handleType & VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT) {
        externalMemoryHandleDesc.type = cudaExternalMemoryHandleTypeOpaqueFd;
    } else {
        return RGY_ERR_INVALID_ARG_VALUE;
    }

    externalMemoryHandleDesc.size = m_bufferSize;

#if defined(_WIN32) || defined(_WIN64)
    externalMemoryHandleDesc.handle.win32.handle = (HANDLE)getMemHandle(handleType);
#else
    externalMemoryHandleDesc.handle.fd = (int)(uintptr_t)getMemHandle(handleType);
#endif

	if (auto err = cudaImportExternalMemory(&m_cudaMem, &externalMemoryHandleDesc); err != cudaSuccess) {
        return err_to_rgy(err);
    }

    cudaExternalMemoryBufferDesc externalMemBufferDesc = {};
    externalMemBufferDesc.offset = 0;
    externalMemBufferDesc.size = m_bufferSize;
    externalMemBufferDesc.flags = 0;

	if (auto err = cudaExternalMemoryGetMappedBuffer(&m_cudaPtr, m_cudaMem, &externalMemBufferDesc); err != cudaSuccess) {
        return err_to_rgy(err);
    }
	return RGY_ERR_NONE;
}

RGY_ERR CUDAVulkanFrame::create(DeviceVulkan *vk, const int width, const int height, const RGY_DATA_TYPE dataType) {
    m_vk = vk;
    m_width = width;
    m_height = height;
    m_pitch = ALIGN(width, 128) * bytesPerPix(dataType);
	m_bufferSize = m_pitch * height;

    const VkBufferUsageFlags usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    const VkMemoryPropertyFlags properties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    const auto extMemHandleType = getDefaultMemHandleType();

    VkBufferCreateInfo bufferInfo = {};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkExternalMemoryBufferCreateInfo externalMemoryBufferInfo = {};
    externalMemoryBufferInfo.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
    externalMemoryBufferInfo.handleTypes = extMemHandleType;
    bufferInfo.pNext = &externalMemoryBufferInfo;
  
    if (auto err = m_vk->GetVulkan()->vkCreateBuffer(m_vk->GetDevice(), &bufferInfo, nullptr, &m_buffer); err != VK_SUCCESS) {
        return err_to_rgy(err);
    }
  
    VkMemoryRequirements memRequirements;
    m_vk->GetVulkan()->vkGetBufferMemoryRequirements(m_vk->GetDevice(), m_buffer, &memRequirements);

    VkExportMemoryAllocateInfoKHR vulkanExportMemoryAllocateInfoKHR = {};
    vulkanExportMemoryAllocateInfoKHR.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO_KHR;

#if defined(_WIN32) || defined(_WIN64)
    WindowsSecurityAttributes winSecurityAttributes;
  
    VkExportMemoryWin32HandleInfoKHR vulkanExportMemoryWin32HandleInfoKHR = {};
    vulkanExportMemoryWin32HandleInfoKHR.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_WIN32_HANDLE_INFO_KHR;
    vulkanExportMemoryWin32HandleInfoKHR.pNext = nullptr;
    vulkanExportMemoryWin32HandleInfoKHR.pAttributes = &winSecurityAttributes;
    vulkanExportMemoryWin32HandleInfoKHR.dwAccess = DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE;
    vulkanExportMemoryWin32HandleInfoKHR.name = (LPCWSTR)nullptr;

    vulkanExportMemoryAllocateInfoKHR.pNext = extMemHandleType & VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT_KHR ? &vulkanExportMemoryWin32HandleInfoKHR : nullptr;
    vulkanExportMemoryAllocateInfoKHR.handleTypes = extMemHandleType;
#else
    vulkanExportMemoryAllocateInfoKHR.pNext = nullptr;
    vulkanExportMemoryAllocateInfoKHR.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif  //defined(_WIN32) || defined(_WIN64)

    VkMemoryAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.pNext = &vulkanExportMemoryAllocateInfoKHR;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

    if (auto err = m_vk->GetVulkan()->vkAllocateMemory(m_vk->GetDevice(), &allocInfo, nullptr, &m_bufferMemory); err != VK_SUCCESS) {
        return err_to_rgy(err);
    }
    m_vk->GetVulkan()->vkBindBufferMemory(m_vk->GetDevice(), m_buffer, m_bufferMemory, 0);
}

RGY_ERR CUDAVulkanFrame::registerTexture() {
    if (auto err = importCudaExternalMemory(extMemHandleType); err != RGY_ERR_NONE) {
        return err;
    }
    return RGY_ERR_NONE;
}

RGY_ERR CUDAVulkanFrame::unregisterTexture() {
    if (m_cudaMem) {
        cudaDestroyExternalMemory(m_cudaMem);
        m_cudaMem = nullptr;
    }
    m_cudaPtr = nullptr;
    return RGY_ERR_NONE;
}

void CUDAVulkanFrame::release() {
    unregisterTexture();
    if (m_buffer) {
        m_vk->GetVulkan()->vkDestroyBuffer(m_vk->GetDevice(), m_buffer, nullptr);
        m_buffer = nullptr;
    }
    if (m_bufferMemory) {
        m_vk->GetVulkan()->vkFreeMemory(m_vk->GetDevice(), m_bufferMemory, nullptr);
        m_bufferMemory = nullptr;
    }
}

CUDAVulkanSemaphore::CUDAVulkanSemaphore() :
    m_vk(nullptr),
    m_semaphore(nullptr),
    m_cudaSem(nullptr),
    m_waitValue(1),
    m_signalValue(2) {
}

CUDAVulkanSemaphore::~CUDAVulkanSemaphore() {
    release();
}

VkExternalSemaphoreHandleTypeFlagBits CUDAVulkanSemaphore::getDefaultSemaphoreHandleType() const {
#if defined(_WIN32) || defined(_WIN64)
  return VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#else
  return VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif //defined(_WIN32) || defined(_WIN64)
}

RGY_ERR CUDAVulkanSemaphore::create(DeviceVulkan *vk) {
    m_vk = vk;

    VkSemaphoreCreateInfo semaphoreInfo = {};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VkExportSemaphoreCreateInfoKHR exportSemaphoreCreateInfo = {};
    exportSemaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO_KHR;

    VkSemaphoreTypeCreateInfo timelineCreateInfo;
    timelineCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
    timelineCreateInfo.pNext = NULL;
    timelineCreateInfo.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    timelineCreateInfo.initialValue = 0;
    exportSemaphoreCreateInfo.pNext = &timelineCreateInfo;

    if (auto err = m_vk->GetVulkan()->vkCreateSemaphore(m_vk->GetDevice(), &semaphoreInfo, nullptr, &m_semaphore); err != VK_SUCCESS) {
        return err_to_rgy(err);
    }
    if (auto err = importCudaExternalSemaphore(getDefaultSemaphoreHandleType()); err != RGY_ERR_NONE) {
        return err;
    }
    m_waitValue = 1;
    m_signalValue = 2;
    return RGY_ERR_NONE;
}

RGY_ERR CUDAVulkanSemaphore::importCudaExternalSemaphore(VkExternalSemaphoreHandleTypeFlagBits handleType) {
    cudaExternalSemaphoreHandleDesc externalSemaphoreHandleDesc = {};

    if (handleType & VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT) {
        externalSemaphoreHandleDesc.type = cudaExternalSemaphoreHandleTypeOpaqueWin32;
    } else if (handleType & VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT) {
        externalSemaphoreHandleDesc.type = cudaExternalSemaphoreHandleTypeOpaqueWin32Kmt;
    } else if (handleType & VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT) {
        externalSemaphoreHandleDesc.type = cudaExternalSemaphoreHandleTypeOpaqueFd;
    } else {
        return RGY_ERR_INVALID_ARG_VALUE;
    }


#if defined(_WIN32) || defined(_WIN64)
    externalSemaphoreHandleDesc.handle.win32.handle = (HANDLE)getSemaphoreHandle(handleType);
#else
    externalSemaphoreHandleDesc.handle.fd = (int)(uintptr_t)getSemaphoreHandle(handleType);
#endif //defined(_WIN32) || defined(_WIN64)

    externalSemaphoreHandleDesc.flags = 0;

    return err_to_rgy(cudaImportExternalSemaphore(&m_cudaSem, &externalSemaphoreHandleDesc));
}

void CUDAVulkanSemaphore::release() {
    if (m_cudaSem) {
        cudaDestroyExternalSemaphore(m_cudaSem);
        m_cudaSem = nullptr;
    }
    if (m_semaphore) {
        m_vk->GetVulkan()->vkDestroySemaphore(m_vk->GetDevice(), m_semaphore, nullptr);
        m_semaphore = nullptr;
    }
}

void *CUDAVulkanSemaphore::getSemaphoreHandle(VkExternalSemaphoreHandleTypeFlagBits handleType) {
#if defined(_WIN32) || defined(_WIN64)
    HANDLE handle = 0;

    VkSemaphoreGetWin32HandleInfoKHR vkSemaphoreGetWin32HandleInfoKHR = {};
    vkSemaphoreGetWin32HandleInfoKHR.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR;
    vkSemaphoreGetWin32HandleInfoKHR.pNext = NULL;
    vkSemaphoreGetWin32HandleInfoKHR.semaphore = m_semaphore;
    vkSemaphoreGetWin32HandleInfoKHR.handleType = handleType;

    if (auto err = m_vk->GetVulkan()->vkGetSemaphoreWin32HandleKHR(m_vk->GetDevice(), &vkSemaphoreGetWin32HandleInfoKHR, &handle); err != VK_SUCCESS) {
        return nullptr;
    }
    return (void *)handle;
#else
    int fd = -1;

    VkSemaphoreGetFdInfoKHR vkSemaphoreGetFdInfoKHR = {};
    vkSemaphoreGetFdInfoKHR.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR;
    vkSemaphoreGetFdInfoKHR.pNext = NULL;
    vkSemaphoreGetFdInfoKHR.semaphore = m_semaphore;
    vkSemaphoreGetFdInfoKHR.handleType = handleType;
    if (m_vk->GetVulkan()->vkGetSemaphoreFdKHR(m_vk->GetDevice(), &vkSemaphoreGetFdInfoKHR, &fd) != VK_SUCCESS) {
        return nullptr;
    }
    return (void *)(uintptr_t)fd;
#endif //defined(_WIN32) || defined(_WIN64)
}

RGY_ERR CUDAVulkanSemaphore::wait(cudaStream_t stream) {
    cudaExternalSemaphoreWaitParams waitParams = {};
    waitParams.flags = 0;
    waitParams.params.fence.value = m_waitValue;
    m_waitValue += 2;

    if (auto err = cudaWaitExternalSemaphoresAsync(&m_cudaSem, &waitParams, 1, stream); err != cudaSuccess) {
        return err_to_rgy(err);
    }
    return RGY_ERR_NONE;
}

RGY_ERR CUDAVulkanSemaphore::signal(cudaStream_t stream) {
    cudaExternalSemaphoreSignalParams signalParams = {};
    signalParams.flags = 0;
    signalParams.params.fence.value = m_signalValue;
    m_signalValue += 2;
    if (auto err = cudaSignalExternalSemaphoresAsync(&m_cudaSem, &signalParams, 1, stream); err != cudaSuccess) {
        return err_to_rgy(err);
    }
    return RGY_ERR_NONE;
}

#endif //ENABLE_VULKAN
