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
#ifndef __VCE_DEVICE_VULKAN_H__
#define __VCE_DEVICE_VULKAN_H__

#include <unordered_map>

#include "rgy_version.h"
#include "rgy_err.h"
#include "rgy_util.h"
#include "rgy_log.h"
#include "rgy_opencl.h"
#include "rgy_device.h"
#include "vce_param.h"

#pragma warning(push)
#pragma warning(disable:4100)
RGY_DISABLE_WARNING_PUSH
RGY_DISABLE_WARNING_STR("-Wclass-memaccess")
#include "Factory.h"
#include "Context.h"
#include "Trace.h"
RGY_DISABLE_WARNING_POP
#pragma warning(pop)

#if ENABLE_VULKAN
#include "rgy_vulkan.h"
#include "VulkanAMF.h"

class DeviceVulkan {
public:
    DeviceVulkan();
    virtual ~DeviceVulkan();

    RGY_ERR Init(int adapterID, amf::AMFContext *pContext, std::shared_ptr<RGYLog> log);
    RGY_ERR Terminate();

    RGYVulkanFuncs* GetVulkan();
    amf::AMFVulkanDevice*      GetDevice();
    std::wstring GetDisplayDeviceName() { return m_displayDeviceName; }

    int GetQueueGraphicFamilyIndex() { return m_uQueueGraphicsFamilyIndex; }
    VkQueue    GetQueueGraphicQueue() { return m_hQueueGraphics; }

    int GetQueueComputeFamilyIndex() { return m_uQueueComputeFamilyIndex; }
    VkQueue    GetQueueComputeQueue() { return m_hQueueCompute; }

    int adapterCount();

protected:
    void AddMessage(RGYLogLevel log_level, const tstring &str);
    void AddMessage(RGYLogLevel log_level, const TCHAR *format, ...);
private:
    RGY_ERR CreateInstance();
    RGY_ERR CreateDeviceAndFindQueues(int adapterID, std::vector<const char*> &deviceExtensions);

    std::vector<const char*> GetDebugInstanceExtensionNames();
    std::vector<const char*> GetDebugInstanceLayerNames();
    std::vector<const char*> GetDebugDeviceLayerNames(VkPhysicalDevice device);

    tstring                         m_name;
    amf::AMFVulkanDevice            m_VulkanDev;
    std::wstring                    m_displayDeviceName;
    RGYVulkanFuncs                  m_vk;

    int                      m_uQueueGraphicsFamilyIndex;
    int                      m_uQueueComputeFamilyIndex;

    VkQueue                         m_hQueueGraphics;
    VkQueue                         m_hQueueCompute;

    std::shared_ptr<RGYLog> m_log;
};
#endif //#if ENABLE_VULKAN

#endif //#if __VCE_DEVICE_VULKAN_H__
