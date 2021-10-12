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

struct RGYParamThreadAffinity {
    RGYThreadAffinity process;
    RGYThreadAffinity main;
    RGYThreadAffinity dec;
    RGYThreadAffinity enc;
    RGYThreadAffinity csp;
    RGYThreadAffinity input;
    RGYThreadAffinity output;
    RGYThreadAffinity audio;
    RGYThreadAffinity perfmonitor;
    RGYThreadAffinity videoquality;

    RGYParamThreadAffinity();
    RGYThreadAffinity get(RGYThreadType type) const;
    void set(RGYThreadAffinity affinity, RGYThreadType type);
    tstring to_string() const;
    bool operator==(const RGYParamThreadAffinity &x) const;
    bool operator!=(const RGYParamThreadAffinity &x) const;
};

#endif //__RGY_THREAD_AFFINITY_H__
