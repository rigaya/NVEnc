// -----------------------------------------------------------------------------------------
// QSVEnc/NVEnc by rigaya
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
// -------------------------------------------------------------------------------------------

#pragma once
#ifndef __RGY_THREAD_AFFINITY_H__
#define __RGY_THREAD_AFFINITY_H__

#include <cstdint>
#include <array>
#include <limits>
#include "rgy_tchar.h"

#if defined(_WIN32) || defined(_WIN64)
typedef void* RGYThreadHandle;
#else
#include <pthread.h>
typedef pthread_t RGYThreadHandle;
#endif

enum class RGYThreadPriority : int {
    BackgroundBeign = 0x00010000,
    BackgroundEnd   = 0x00020000,
    Idle            = -15,
    Lowest          = -2,
    BelowNormal     = -1,
    Normal          = 0,
    AboveNormal     = 1,
    Highest         = 2,
    TimeCritical    = 15,

    Unknwon         = 0xffff,
};

static const std::array<std::pair<RGYThreadPriority, const TCHAR*>, 7> RGY_THREAD_PRIORITY_STR = {
    std::pair<RGYThreadPriority, const TCHAR*>{ RGYThreadPriority::BackgroundBeign, _T("background")},
    std::pair<RGYThreadPriority, const TCHAR*>{ RGYThreadPriority::Idle,            _T("idle")},
    std::pair<RGYThreadPriority, const TCHAR*>{ RGYThreadPriority::Lowest,          _T("lowest")},
    std::pair<RGYThreadPriority, const TCHAR*>{ RGYThreadPriority::BelowNormal,     _T("belownormal")},
    std::pair<RGYThreadPriority, const TCHAR*>{ RGYThreadPriority::Normal,          _T("normal")},
    std::pair<RGYThreadPriority, const TCHAR*>{ RGYThreadPriority::AboveNormal,     _T("abovenormal")},
    std::pair<RGYThreadPriority, const TCHAR*>{ RGYThreadPriority::Highest,         _T("highest")}
};

const TCHAR* rgy_thread_priority_mode_to_str(RGYThreadPriority mode);
RGYThreadPriority rgy_str_to_thread_priority_mode(const TCHAR* str);

enum class RGYThreadPowerThrottlingMode {
    Unset    = -2,
    Auto     = -1,
    Disabled = 0,
    Enabled  = 1,

    END
};

static const std::array<std::pair<RGYThreadPowerThrottlingMode, const TCHAR*>, (int)RGYThreadPowerThrottlingMode::END - (int)RGYThreadPowerThrottlingMode::Unset> RGY_THREAD_POWER_THROTTOLING_MODE_STR = {
    std::pair<RGYThreadPowerThrottlingMode, const TCHAR*>{ RGYThreadPowerThrottlingMode::Unset,    _T("unset")},
    std::pair<RGYThreadPowerThrottlingMode, const TCHAR*>{ RGYThreadPowerThrottlingMode::Auto,     _T("auto")},
    std::pair<RGYThreadPowerThrottlingMode, const TCHAR*>{ RGYThreadPowerThrottlingMode::Disabled, _T("off")},
    std::pair<RGYThreadPowerThrottlingMode, const TCHAR*>{ RGYThreadPowerThrottlingMode::Enabled,  _T("on")}
};

const TCHAR* rgy_thread_power_throttoling_mode_to_str(RGYThreadPowerThrottlingMode mode);
RGYThreadPowerThrottlingMode rgy_str_to_thread_power_throttoling_mode(const TCHAR* str);

enum class RGYThreadAffinityMode {
    ALL,
    PCORE,
    ECORE,
    LOGICAL,
    PHYSICAL,
    CACHEL2,
    CACHEL3,
    CUSTOM,
    END
};

static const std::array<std::pair<const TCHAR *, RGYThreadAffinityMode>, (int)RGYThreadAffinityMode::END - (int)RGYThreadAffinityMode::ALL> RGY_THREAD_AFFINITY_MODE_STR = {
    std::pair<const TCHAR *, RGYThreadAffinityMode>{ _T("all"),      RGYThreadAffinityMode::ALL      },
    std::pair<const TCHAR *, RGYThreadAffinityMode>{ _T("pcore"),    RGYThreadAffinityMode::PCORE    },
    std::pair<const TCHAR *, RGYThreadAffinityMode>{ _T("ecore"),    RGYThreadAffinityMode::ECORE    },
    std::pair<const TCHAR *, RGYThreadAffinityMode>{ _T("logical"),  RGYThreadAffinityMode::LOGICAL  },
    std::pair<const TCHAR *, RGYThreadAffinityMode>{ _T("physical"), RGYThreadAffinityMode::PHYSICAL },
    std::pair<const TCHAR *, RGYThreadAffinityMode>{ _T("cachel2"),  RGYThreadAffinityMode::CACHEL2  },
    std::pair<const TCHAR *, RGYThreadAffinityMode>{ _T("cachel3"),  RGYThreadAffinityMode::CACHEL3  },
    std::pair<const TCHAR *, RGYThreadAffinityMode>{ _T("custom"),   RGYThreadAffinityMode::CUSTOM   }
};

const TCHAR *rgy_thread_affnity_mode_to_str(RGYThreadAffinityMode mode);
RGYThreadAffinityMode rgy_str_to_thread_affnity_mode(const TCHAR *str);

struct RGYThreadAffinity {
    RGYThreadAffinityMode mode;
    uint64_t custom;

    RGYThreadAffinity();
    RGYThreadAffinity(RGYThreadAffinityMode m);
    RGYThreadAffinity(RGYThreadAffinityMode m, uint64_t customAffinity);
    uint64_t getMask() const;
    uint64_t getMask(int idx) const;
    tstring to_string() const;
    bool operator==(const RGYThreadAffinity &x) const;
    bool operator!=(const RGYThreadAffinity &x) const;
};

uint64_t selectMaskFromLowerBit(uint64_t mask, const int idx);

enum class RGYThreadType {
    ALL,
    PROCESS,
    MAIN,
    DEC,
    ENC,
    CSP,
    INPUT,
    OUTUT,
    AUDIO,
    PERF_MONITOR,
    VIDEO_QUALITY,

    END
};

static const std::array<std::pair<RGYThreadType, const TCHAR *>, (int)RGYThreadType::END - (int)RGYThreadType::ALL> RGY_THREAD_TYPE_STR = {
    std::pair<RGYThreadType, const TCHAR *>{ RGYThreadType::ALL,           _T("all")},
    std::pair<RGYThreadType, const TCHAR *>{ RGYThreadType::PROCESS,       _T("process")},
    std::pair<RGYThreadType, const TCHAR *>{ RGYThreadType::MAIN,          _T("main")},
    std::pair<RGYThreadType, const TCHAR *>{ RGYThreadType::DEC,           _T("decoder")},
    std::pair<RGYThreadType, const TCHAR *>{ RGYThreadType::ENC,           _T("encoder")},
    std::pair<RGYThreadType, const TCHAR *>{ RGYThreadType::CSP,           _T("csp")},
    std::pair<RGYThreadType, const TCHAR *>{ RGYThreadType::INPUT,         _T("input")},
    std::pair<RGYThreadType, const TCHAR *>{ RGYThreadType::OUTUT,         _T("output")},
    std::pair<RGYThreadType, const TCHAR *>{ RGYThreadType::AUDIO,         _T("audio")},
    std::pair<RGYThreadType, const TCHAR *>{ RGYThreadType::PERF_MONITOR,  _T("perfmonitor")},
    std::pair<RGYThreadType, const TCHAR *>{ RGYThreadType::VIDEO_QUALITY, _T("videoquality")}
};

const TCHAR *rgy_thread_type_to_str(RGYThreadType type);

enum class RGYParamThreadType {
    all,
    affinity,
    priority,
    throttling,
};

struct RGYParamThread {
    RGYThreadAffinity affinity;
    RGYThreadPriority priority;
    RGYThreadPowerThrottlingMode throttling;

    RGYParamThread();
    uint32_t getPriorityCalss();
    tstring to_string(RGYParamThreadType type) const;
    tstring desc() const;
    void set(RGYThreadAffinity affinity, RGYThreadPriority priority, RGYThreadPowerThrottlingMode throttling);
    bool apply(RGYThreadHandle threadHandle) const;
    bool operator==(const RGYParamThread& x) const;
    bool operator!=(const RGYParamThread& x) const;
};

struct RGYParamThreads {
    RGYParamThread process;
    RGYParamThread main;
    RGYParamThread dec;
    RGYParamThread enc;
    RGYParamThread csp;
    RGYParamThread input;
    RGYParamThread output;
    RGYParamThread audio;
    RGYParamThread perfmonitor;
    RGYParamThread videoquality;

    RGYParamThreads();
    RGYParamThread& get(RGYThreadType type);
    const RGYParamThread& get(RGYThreadType type) const;
    void set(const RGYThreadAffinity affinity, RGYThreadType type);
    void set(const RGYThreadPriority priority, RGYThreadType type);
    void set(const RGYThreadPowerThrottlingMode mode, RGYThreadType type);
    void apply_unset();
    tstring to_string(RGYParamThreadType type) const;
    bool operator==(const RGYParamThreads&x) const;
    bool operator!=(const RGYParamThreads&x) const;
};

bool SetThreadPriorityForModule(const uint32_t TargetProcessId, const TCHAR *TargetModule, const RGYThreadPriority ThreadPriority);
bool SetThreadAffinityForModule(const uint32_t TargetProcessId, const TCHAR *TargetModule, const uint64_t ThreadAffinityMask);

bool SetThreadPowerThrottolingMode(RGYThreadHandle threadHandle, const RGYThreadPowerThrottlingMode mode);
bool SetThreadPowerThrottolingModeForModule(const uint32_t TargetProcessId, const TCHAR* TargetModule, const RGYThreadPowerThrottlingMode mode);

#endif //__RGY_THREAD_AFFINITY_H__
