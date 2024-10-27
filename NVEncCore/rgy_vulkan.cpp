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

#include "rgy_vulkan.h"
#if ENABLE_VULKAN
#include "rgy_tchar.h"

#if defined(_WIN32) || defined(_WIN64)
static const TCHAR *VULKAN_DLL = _T("vulkan-1.dll");
static const TCHAR *VULKAN_DLL2 = nullptr;
#else
static const TCHAR *VULKAN_DLL = _T("libvulkan.so.1");
static const TCHAR *VULKAN_DLL2 = _T("libvulkan.so");
#endif

RGYVulkanFuncs::RGYVulkanFuncs() :
    vkCreateInstance(nullptr),
    vkDestroyInstance(nullptr),
    vkEnumeratePhysicalDevices(nullptr),
    vkGetPhysicalDeviceFeatures(nullptr),
    vkGetPhysicalDeviceFormatProperties(nullptr),
    vkGetPhysicalDeviceImageFormatProperties(nullptr),
    vkGetPhysicalDeviceProperties(nullptr),
    vkGetPhysicalDeviceProperties2(nullptr),
    vkGetPhysicalDeviceQueueFamilyProperties(nullptr),
    vkGetPhysicalDeviceMemoryProperties(nullptr),
    vkGetInstanceProcAddr(nullptr),
    vkGetDeviceProcAddr(nullptr),
    vkCreateDevice(nullptr),
    vkDestroyDevice(nullptr),
    vkEnumerateInstanceExtensionProperties(nullptr),
    vkEnumerateDeviceExtensionProperties(nullptr),
    vkEnumerateInstanceLayerProperties(nullptr),
    vkEnumerateDeviceLayerProperties(nullptr),
    vkGetDeviceQueue(nullptr),
    vkQueueSubmit(nullptr),
    vkQueueWaitIdle(nullptr),
    vkDeviceWaitIdle(nullptr),
    vkAllocateMemory(nullptr),
    vkFreeMemory(nullptr),
    vkMapMemory(nullptr),
    vkUnmapMemory(nullptr),
    vkFlushMappedMemoryRanges(nullptr),
    vkInvalidateMappedMemoryRanges(nullptr),
    vkGetDeviceMemoryCommitment(nullptr),
    vkBindBufferMemory(nullptr),
    vkBindImageMemory(nullptr),
    vkGetBufferMemoryRequirements(nullptr),
    vkGetImageMemoryRequirements(nullptr),
    vkGetImageSparseMemoryRequirements(nullptr),
    vkGetPhysicalDeviceSparseImageFormatProperties(nullptr),
    vkQueueBindSparse(nullptr),
    vkCreateFence(nullptr),
    vkDestroyFence(nullptr),
    vkResetFences(nullptr),
    vkGetFenceStatus(nullptr),
    vkWaitForFences(nullptr),
    vkCreateSemaphore(nullptr),
    vkDestroySemaphore(nullptr),
    vkWaitSemaphores(nullptr),
    vkSignalSemaphore(nullptr),
    vkCreateEvent(nullptr),
    vkDestroyEvent(nullptr),
    vkGetEventStatus(nullptr),
    vkSetEvent(nullptr),
    vkResetEvent(nullptr),
    vkCreateQueryPool(nullptr),
    vkDestroyQueryPool(nullptr),
    vkGetQueryPoolResults(nullptr),
    vkCreateBuffer(nullptr),
    vkDestroyBuffer(nullptr),
    vkCreateBufferView(nullptr),
    vkDestroyBufferView(nullptr),
    vkCreateImage(nullptr),
    vkDestroyImage(nullptr),
    vkGetImageSubresourceLayout(nullptr),
    vkCreateImageView(nullptr),
    vkDestroyImageView(nullptr),
    vkCreateShaderModule(nullptr),
    vkDestroyShaderModule(nullptr),
    vkCreatePipelineCache(nullptr),
    vkDestroyPipelineCache(nullptr),
    vkGetPipelineCacheData(nullptr),
    vkMergePipelineCaches(nullptr),
    vkCreateGraphicsPipelines(nullptr),
    vkCreateComputePipelines(nullptr),
    vkDestroyPipeline(nullptr),
    vkCreatePipelineLayout(nullptr),
    vkDestroyPipelineLayout(nullptr),
    vkCreateSampler(nullptr),
    vkDestroySampler(nullptr),
    vkCreateDescriptorSetLayout(nullptr),
    vkDestroyDescriptorSetLayout(nullptr),
    vkCreateDescriptorPool(nullptr),
    vkDestroyDescriptorPool(nullptr),
    vkResetDescriptorPool(nullptr),
    vkAllocateDescriptorSets(nullptr),
    vkFreeDescriptorSets(nullptr),
    vkUpdateDescriptorSets(nullptr),
    vkCreateFramebuffer(nullptr),
    vkDestroyFramebuffer(nullptr),
    vkCreateRenderPass(nullptr),
    vkDestroyRenderPass(nullptr),
    vkGetRenderAreaGranularity(nullptr),
    vkCreateCommandPool(nullptr),
    vkDestroyCommandPool(nullptr),
    vkResetCommandPool(nullptr),
    vkAllocateCommandBuffers(nullptr),
    vkFreeCommandBuffers(nullptr),
    vkBeginCommandBuffer(nullptr),
    vkEndCommandBuffer(nullptr),
    vkResetCommandBuffer(nullptr),
    vkCmdBindPipeline(nullptr),
    vkCmdSetViewport(nullptr),
    vkCmdSetScissor(nullptr),
    vkCmdSetLineWidth(nullptr),
    vkCmdSetDepthBias(nullptr),
    vkCmdSetBlendConstants(nullptr),
    vkCmdSetDepthBounds(nullptr),
    vkCmdSetStencilCompareMask(nullptr),
    vkCmdSetStencilWriteMask(nullptr),
    vkCmdSetStencilReference(nullptr),
    vkCmdBindDescriptorSets(nullptr),
    vkCmdBindIndexBuffer(nullptr),
    vkCmdBindVertexBuffers(nullptr),
    vkCmdDraw(nullptr),
    vkCmdDrawIndexed(nullptr),
    vkCmdDrawIndirect(nullptr),
    vkCmdDrawIndexedIndirect(nullptr),
    vkCmdDispatch(nullptr),
    vkCmdDispatchIndirect(nullptr),
    vkCmdCopyBuffer(nullptr),
    vkCmdCopyImage(nullptr),
    vkCmdBlitImage(nullptr),
    vkCmdCopyBufferToImage(nullptr),
    vkCmdCopyImageToBuffer(nullptr),
    vkCmdUpdateBuffer(nullptr),
    vkCmdFillBuffer(nullptr),
    vkCmdClearColorImage(nullptr),
    vkCmdClearDepthStencilImage(nullptr),
    vkCmdClearAttachments(nullptr),
    vkCmdResolveImage(nullptr),
    vkCmdSetEvent(nullptr),
    vkCmdResetEvent(nullptr),
    vkCmdWaitEvents(nullptr),
    vkCmdPipelineBarrier(nullptr),
    vkCmdBeginQuery(nullptr),
    vkCmdEndQuery(nullptr),
    vkCmdResetQueryPool(nullptr),
    vkCmdWriteTimestamp(nullptr),
    vkCmdCopyQueryPoolResults(nullptr),
    vkCmdPushConstants(nullptr),
    vkCmdBeginRenderPass(nullptr),
    vkCmdNextSubpass(nullptr),
    vkCmdEndRenderPass(nullptr),
    vkCmdExecuteCommands(nullptr),
    vkDestroySurfaceKHR(nullptr),
    vkGetPhysicalDeviceSurfaceSupportKHR(nullptr),
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(nullptr),
    vkGetPhysicalDeviceSurfaceFormatsKHR(nullptr),
    vkGetPhysicalDeviceSurfacePresentModesKHR(nullptr),
    vkCreateSwapchainKHR(nullptr),
    vkDestroySwapchainKHR(nullptr),
    vkGetSwapchainImagesKHR(nullptr),
    vkAcquireNextImageKHR(nullptr),
    vkQueuePresentKHR(nullptr),
#if defined(_WIN32) || defined(_WIN64)
    vkCreateWin32SurfaceKHR(nullptr),
    vkGetMemoryWin32HandleKHR(nullptr),
    vkGetSemaphoreWin32HandleKHR(nullptr),
#else
    vkCreateXlibSurfaceKHR(nullptr),
    vkGetMemoryFdKHR(nullptr),
    vkGetSemaphoreFdKHR(nullptr),
#endif
    vkCreateDebugReportCallbackEXT(nullptr),
    vkDebugReportMessageEXT(nullptr),
    vkDestroyDebugReportCallbackEXT(nullptr),
    m_hVulkanDll(nullptr) {
}

RGYVulkanFuncs::~RGYVulkanFuncs() {
    if (m_hVulkanDll != nullptr) {
        RGY_FREE_LIBRARY(m_hVulkanDll);
    }
    m_hVulkanDll = nullptr;
}

int RGYVulkanFuncs::init() {
    if (m_hVulkanDll != nullptr) {
        return 0;
    }
    m_hVulkanDll = RGY_LOAD_LIBRARY(VULKAN_DLL);
    if (!m_hVulkanDll) {
        fprintf(stderr, "Failed to load %s.\n", VULKAN_DLL);
        if (VULKAN_DLL2) {
            m_hVulkanDll = RGY_LOAD_LIBRARY(VULKAN_DLL2);
        }
        if (!m_hVulkanDll) {
            fprintf(stderr, "Failed to load %s.\n", VULKAN_DLL2);
            return 1;
        }
    }

#define LOAD(w) w = reinterpret_cast<PFN_##w>(RGY_GET_PROC_ADDRESS(m_hVulkanDll, #w)); if(w==nullptr) \
    { RGY_FREE_LIBRARY(m_hVulkanDll); m_hVulkanDll = nullptr; fprintf(stderr, "Failed to load %s.\n", #w); return 1; };
    LOAD(vkCreateInstance);
    LOAD(vkCreateInstance);
    LOAD(vkDestroyInstance);
    LOAD(vkEnumeratePhysicalDevices);
    LOAD(vkGetPhysicalDeviceFeatures);
    LOAD(vkGetPhysicalDeviceFormatProperties);
    LOAD(vkGetPhysicalDeviceImageFormatProperties);
    LOAD(vkGetPhysicalDeviceProperties);
    LOAD(vkGetPhysicalDeviceProperties2);
    LOAD(vkGetPhysicalDeviceQueueFamilyProperties);
    LOAD(vkGetPhysicalDeviceMemoryProperties);
    LOAD(vkGetInstanceProcAddr);
    LOAD(vkGetDeviceProcAddr);
    LOAD(vkCreateDevice);
    LOAD(vkDestroyDevice);
    LOAD(vkEnumerateInstanceExtensionProperties);
    LOAD(vkEnumerateDeviceExtensionProperties);
    LOAD(vkEnumerateInstanceLayerProperties);
    LOAD(vkEnumerateDeviceLayerProperties);
    LOAD(vkGetDeviceQueue);
    LOAD(vkQueueSubmit);
    LOAD(vkQueueWaitIdle);
    LOAD(vkDeviceWaitIdle);
    LOAD(vkAllocateMemory);
    LOAD(vkFreeMemory);
    LOAD(vkMapMemory);
    LOAD(vkUnmapMemory);
    LOAD(vkFlushMappedMemoryRanges);
    LOAD(vkInvalidateMappedMemoryRanges);
    LOAD(vkGetDeviceMemoryCommitment);
    LOAD(vkBindBufferMemory);
    LOAD(vkBindImageMemory);
    LOAD(vkGetBufferMemoryRequirements);
    LOAD(vkGetImageMemoryRequirements);
    LOAD(vkGetImageSparseMemoryRequirements);
    LOAD(vkGetPhysicalDeviceSparseImageFormatProperties);
    LOAD(vkQueueBindSparse);
    LOAD(vkCreateFence);
    LOAD(vkDestroyFence);
    LOAD(vkResetFences);
    LOAD(vkGetFenceStatus);
    LOAD(vkWaitForFences);
    LOAD(vkCreateSemaphore);
    LOAD(vkDestroySemaphore);
    LOAD(vkWaitSemaphores);
    LOAD(vkSignalSemaphore);
    LOAD(vkCreateEvent);
    LOAD(vkDestroyEvent);
    LOAD(vkGetEventStatus);
    LOAD(vkSetEvent);
    LOAD(vkResetEvent);
    LOAD(vkCreateQueryPool);
    LOAD(vkDestroyQueryPool);
    LOAD(vkGetQueryPoolResults);
    LOAD(vkCreateBuffer);
    LOAD(vkDestroyBuffer);
    LOAD(vkCreateBufferView);
    LOAD(vkDestroyBufferView);
    LOAD(vkCreateImage);
    LOAD(vkDestroyImage);
    LOAD(vkGetImageSubresourceLayout);
    LOAD(vkCreateImageView);
    LOAD(vkDestroyImageView);
    LOAD(vkCreateShaderModule);
    LOAD(vkDestroyShaderModule);
    LOAD(vkCreatePipelineCache);
    LOAD(vkDestroyPipelineCache);
    LOAD(vkGetPipelineCacheData);
    LOAD(vkMergePipelineCaches);
    LOAD(vkCreateGraphicsPipelines);
    LOAD(vkCreateComputePipelines);
    LOAD(vkDestroyPipeline);
    LOAD(vkCreatePipelineLayout);
    LOAD(vkDestroyPipelineLayout);
    LOAD(vkCreateSampler);
    LOAD(vkDestroySampler);
    LOAD(vkCreateDescriptorSetLayout);
    LOAD(vkDestroyDescriptorSetLayout);
    LOAD(vkCreateDescriptorPool);
    LOAD(vkDestroyDescriptorPool);
    LOAD(vkResetDescriptorPool);
    LOAD(vkAllocateDescriptorSets);
    LOAD(vkFreeDescriptorSets);
    LOAD(vkUpdateDescriptorSets);
    LOAD(vkCreateFramebuffer);
    LOAD(vkDestroyFramebuffer);
    LOAD(vkCreateRenderPass);
    LOAD(vkDestroyRenderPass);
    LOAD(vkGetRenderAreaGranularity);
    LOAD(vkCreateCommandPool);
    LOAD(vkDestroyCommandPool);
    LOAD(vkResetCommandPool);
    LOAD(vkAllocateCommandBuffers);
    LOAD(vkFreeCommandBuffers);
    LOAD(vkBeginCommandBuffer);
    LOAD(vkEndCommandBuffer);
    LOAD(vkResetCommandBuffer);
    LOAD(vkCmdBindPipeline);
    LOAD(vkCmdSetViewport);
    LOAD(vkCmdSetScissor);
    LOAD(vkCmdSetLineWidth);
    LOAD(vkCmdSetDepthBias);
    LOAD(vkCmdSetBlendConstants);
    LOAD(vkCmdSetDepthBounds);
    LOAD(vkCmdSetStencilCompareMask);
    LOAD(vkCmdSetStencilWriteMask);
    LOAD(vkCmdSetStencilReference);
    LOAD(vkCmdBindDescriptorSets);
    LOAD(vkCmdBindIndexBuffer);
    LOAD(vkCmdBindVertexBuffers);
    LOAD(vkCmdDraw);
    LOAD(vkCmdDrawIndexed);
    LOAD(vkCmdDrawIndirect);
    LOAD(vkCmdDrawIndexedIndirect);
    LOAD(vkCmdDispatch);
    LOAD(vkCmdDispatchIndirect);
    LOAD(vkCmdCopyBuffer);
    LOAD(vkCmdCopyImage);
    LOAD(vkCmdBlitImage);
    LOAD(vkCmdCopyBufferToImage);
    LOAD(vkCmdCopyImageToBuffer);
    LOAD(vkCmdUpdateBuffer);
    LOAD(vkCmdFillBuffer);
    LOAD(vkCmdClearColorImage);
    LOAD(vkCmdClearDepthStencilImage);
    LOAD(vkCmdClearAttachments);
    LOAD(vkCmdResolveImage);
    LOAD(vkCmdSetEvent);
    LOAD(vkCmdResetEvent);
    LOAD(vkCmdWaitEvents);
    LOAD(vkCmdPipelineBarrier);
    LOAD(vkCmdBeginQuery);
    LOAD(vkCmdEndQuery);
    LOAD(vkCmdResetQueryPool);
    LOAD(vkCmdWriteTimestamp);
    LOAD(vkCmdCopyQueryPoolResults);
    LOAD(vkCmdPushConstants);
    LOAD(vkCmdBeginRenderPass);
    LOAD(vkCmdNextSubpass);
    LOAD(vkCmdEndRenderPass);
    LOAD(vkCmdExecuteCommands);

    LOAD(vkGetPhysicalDeviceSurfaceSupportKHR);
    LOAD(vkGetPhysicalDeviceSurfaceCapabilitiesKHR);
    LOAD(vkGetPhysicalDeviceSurfaceFormatsKHR);
    LOAD(vkGetPhysicalDeviceSurfacePresentModesKHR);
    LOAD(vkDestroySurfaceKHR);

#ifdef WIN32
    LOAD(vkCreateWin32SurfaceKHR);
#else
    LOAD(vkCreateXlibSurfaceKHR);
#endif
    return 0;

#undef LOAD
}

int RGYVulkanFuncs::load(VkInstance instance, bool bDebug) {
#define LOAD(w) w = reinterpret_cast<PFN_##w>(vkGetInstanceProcAddr(instance, #w)); if(w==nullptr) { return 1; };
    if (bDebug) {
        LOAD(vkCreateDebugReportCallbackEXT);
        LOAD(vkDebugReportMessageEXT);
        LOAD(vkDestroyDebugReportCallbackEXT);
    }
    return 0;
#undef LOAD
}

//-------------------------------------------------------------------------------------------------
int RGYVulkanFuncs::load(VkDevice device) {
#define LOAD(w) w = reinterpret_cast<PFN_##w>(vkGetDeviceProcAddr(device, #w)); if(w==nullptr) { return 1; };
    LOAD(vkCreateSwapchainKHR);
    LOAD(vkDestroySwapchainKHR);
    LOAD(vkGetSwapchainImagesKHR);
    LOAD(vkAcquireNextImageKHR);
    LOAD(vkQueuePresentKHR);
#ifdef WIN32
    LOAD(vkGetMemoryWin32HandleKHR);
    LOAD(vkGetSemaphoreWin32HandleKHR);
#else
    LOAD(vkGetMemoryFdKHR);
    LOAD(vkGetSemaphoreFdKHR);
#endif
    return 0;
#undef LOAD
}

#endif //#if ENABLE_VULKAN
