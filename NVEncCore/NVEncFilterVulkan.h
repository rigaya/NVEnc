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

#pragma once
#ifndef __NVENC_FILTER_VULKAN_H__
#define __NVENC_FILTER_VULKAN_H__

class DeviceVulkan;

#include "rgy_version.h"
#if ENABLE_VULKAN
#include "rgy_vulkan.h"
#include "rgy_err.h"

class DeviceVulkan;

class CUDAVulkanFrame {
    DeviceVulkan *m_vk;

    VkFormat m_format;
    VkBufferUsageFlags m_usage;
    VkImage m_image;
    VkDeviceMemory m_bufferMemory;

    cudaExternalMemory_t m_cudaMem;
    cudaArray *m_cudaArray;
    cudaMipmappedArray *m_cudaMipmappedArray;

    int m_width;
    int m_height;
    int m_pitch;
    uint64_t m_bufferSize;
public:
    CUDAVulkanFrame();
    ~CUDAVulkanFrame();
    RGY_ERR create(DeviceVulkan *vk, const int width, const int height, const VkFormat dataType);
    RGY_ERR registerTexture();
    RGY_ERR unregisterTexture();
    cudaArray *getMappedArray() { return m_cudaArray; }

    VkImage image() { return m_image; }
    VkFormat format() const { return m_format; }
    VkBufferUsageFlags usage() const { return m_usage; }

    void release();

    int width() const { return m_width; }
    int height() const { return m_height; }
    int pitch() const { return m_pitch; }
    int getTextureBytePerPix() const;
protected:
    VkExternalMemoryHandleTypeFlagBits getDefaultMemHandleType() const;
    RGY_ERR importCudaExternalMemory(VkExternalMemoryHandleTypeFlagBits handleType);
    void *getMemHandle(VkExternalMemoryHandleTypeFlagBits handleType);
    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);
};

class CUDAVulkanSemaphore {
    DeviceVulkan *m_vk;
    VkSemaphore m_semaphore;
    cudaExternalSemaphore_t m_cudaSem;
    uint64_t m_waitValue;
    uint64_t m_signalValue;
public:
    CUDAVulkanSemaphore();
    ~CUDAVulkanSemaphore();
    RGY_ERR create(DeviceVulkan *vk);
    void release();
    RGY_ERR wait(cudaStream_t stream);
    RGY_ERR signal(cudaStream_t stream);
protected:
    VkExternalSemaphoreHandleTypeFlagBits getDefaultSemaphoreHandleType() const;
    RGY_ERR importCudaExternalSemaphore(VkExternalSemaphoreHandleTypeFlagBits handleType);
    void *getSemaphoreHandle(VkExternalSemaphoreHandleTypeFlagBits handleType);
};


#endif //#if ENABLE_VULKAN

#endif //#ifndef __NVENC_FILTER_VULKAN_H__
