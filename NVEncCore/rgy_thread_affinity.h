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
#include "rgy_tchar.h"
#include "rgy_def.h"

enum class RGYThreadAffinityMode {
    ALL,
    PCORE,
    ECORE,
#if defined(_WIN32) || defined(_WIN64)
    LOGICAL,
    PHYSICAL,
    CACHEL2,
    CACHEL3,
#endif //#if defined(_WIN32) || defined(_WIN64)
    CUSTOM,
    END
};

const CX_DESC list_thread_affinity_mode[] = {
    { _T("all"),      (int)RGYThreadAffinityMode::ALL      },
    { _T("pcore"),    (int)RGYThreadAffinityMode::PCORE    },
    { _T("ecore"),    (int)RGYThreadAffinityMode::ECORE    },
#if defined(_WIN32) || defined(_WIN64)
    { _T("logical"),  (int)RGYThreadAffinityMode::LOGICAL  },
    { _T("physical"), (int)RGYThreadAffinityMode::PHYSICAL },
    { _T("cachel2"),  (int)RGYThreadAffinityMode::CACHEL2  },
    { _T("cachel3"),  (int)RGYThreadAffinityMode::CACHEL3  },
#endif //#if defined(_WIN32) || defined(_WIN64)
    { _T("custom"),   (int)RGYThreadAffinityMode::CUSTOM   },
    { NULL, 0 }
};

struct RGYThreadAffinity {
    RGYThreadAffinityMode mode;
    uint64_t custom;

    RGYThreadAffinity();
    RGYThreadAffinity(RGYThreadAffinityMode m);
    RGYThreadAffinity(RGYThreadAffinityMode m, uint64_t customAffinity);
    uint64_t getMask() const;
    tstring to_string() const;
    bool operator==(const RGYThreadAffinity &x) const;
    bool operator!=(const RGYThreadAffinity &x) const;
};

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
