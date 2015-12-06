/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#ifndef NVCPUOPSYS_H
#define NVCPUOPSYS_H


#if defined(_WIN32) || defined(_WIN16)
#   define NV_WINDOWS
#endif

#if (defined(__unix__) || defined(__unix) ) && !defined(nvmacosx) && !defined(vxworks) && !defined(__DJGPP__) && !defined(NV_UNIX) && !defined(__QNX__) && !defined(__QNXNTO__)/* XXX until removed from Makefiles */
#   define NV_UNIX
#endif /* defined(__unix__) */

#if defined(__linux__) && !defined(NV_LINUX) && !defined(NV_VMWARE)
#   define NV_LINUX
#endif  /* defined(__linux__) */

#endif
