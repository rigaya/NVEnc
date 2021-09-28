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
// --------------------------------------------------------------------------------------------

#include <sstream>
#include "rgy_thread_affinity.h"
#include "rgy_osdep.h"
#include "cpu_info.h"

RGYThreadAffinity::RGYThreadAffinity() : mode(), custom(std::numeric_limits<decltype(custom)>::max()) {};

RGYThreadAffinity::RGYThreadAffinity(RGYThreadAffinityMode affinityMode) : mode(affinityMode), custom(std::numeric_limits<decltype(custom)>::max()) {};

RGYThreadAffinity::RGYThreadAffinity(RGYThreadAffinityMode m, uint64_t customAffinity) : mode(m), custom(customAffinity) {};

tstring RGYThreadAffinity::to_string() const {
    if (mode == RGYThreadAffinityMode::CUSTOM) {
        TCHAR buf[64];
        _stprintf_s(buf, _T("0x%llx"), custom);
        return buf;
    }
    auto modeStr = get_cx_desc(list_thread_affinity_mode, (int)mode);
#if defined(_WIN32) || defined(_WIN64)
    if (mode == RGYThreadAffinityMode::LOGICAL
        || mode == RGYThreadAffinityMode::PHYSICAL
        || mode == RGYThreadAffinityMode::CACHEL2
        || mode == RGYThreadAffinityMode::CACHEL3) {
        const auto cpu_info = get_cpu_info();
        int targetCount = 0;
        if (mode == RGYThreadAffinityMode::LOGICAL) {
            targetCount = cpu_info.logical_cores;
        } else if (mode == RGYThreadAffinityMode::PHYSICAL) {
            targetCount = cpu_info.physical_cores;
        } else if (mode == RGYThreadAffinityMode::CACHEL2) {
            targetCount = cpu_info.cache_count[1];
        } else if (mode == RGYThreadAffinityMode::CACHEL3) {
            targetCount = cpu_info.cache_count[2];
        }
        std::basic_stringstream<TCHAR> tmp;
        for (int id = 0; id < targetCount; id++) {
            const auto target = 1llu << id;
            if (target & custom) {
                tmp << _T(":") << id;
            }
        }
        if (!tmp.str().empty()) {
            return modeStr + tstring(_T("#")) + tmp.str().substr(1);
        } else {
            return modeStr;
        }
    }
#endif
    return modeStr;
}

bool RGYThreadAffinity::operator==(const RGYThreadAffinity &x) const {
    return mode == x.mode
        && custom == x.custom;
}
bool RGYThreadAffinity::operator!=(const RGYThreadAffinity &x) const {
    return !(*this == x);
}

uint64_t RGYThreadAffinity::getMask() const {
    uint64_t mask = 0;
    const auto cpu_info = get_cpu_info();
    switch (mode) {
    case RGYThreadAffinityMode::PCORE:  mask = (cpu_info.maskCoreP) ? cpu_info.maskCoreP : cpu_info.maskSystem; break;
    case RGYThreadAffinityMode::ECORE:  mask = (cpu_info.maskCoreE) ? cpu_info.maskCoreE : cpu_info.maskSystem; break;
#if defined(_WIN32) || defined(_WIN64)
    case RGYThreadAffinityMode::LOGICAL:
        for (int i = 0; i < cpu_info.logical_cores; i++) {
            const auto target = 1llu << i;
            if (target & custom) {
                mask |= get_mask(&cpu_info, RGYUnitType::Core, (int)RGYCoreType::Logical, i);
            }
        }
        break;
    case RGYThreadAffinityMode::PHYSICAL:
        for (int i = 0; i < cpu_info.physical_cores; i++) {
            const auto target = 1llu << i;
            if (target & custom) {
                mask |= get_mask(&cpu_info, RGYUnitType::Core, (int)RGYCoreType::Physical, i);
            }
        }
        break;
    case RGYThreadAffinityMode::CACHEL2:
        for (int i = 0; i < cpu_info.cache_count[1]; i++) {
            const auto target = 1llu << i;
            if (target & custom) {
                mask |= get_mask(&cpu_info, RGYUnitType::Cache, (int)RGYCacheLevel::L2, i);
            }
        }
        break;
    case RGYThreadAffinityMode::CACHEL3:
        for (int i = 0; i < cpu_info.cache_count[2]; i++) {
            const auto target = 1llu << i;
            if (target & custom) {
                mask |= get_mask(&cpu_info, RGYUnitType::Cache, (int)RGYCacheLevel::L3, i);
            }
        }
        break;
#endif
    case RGYThreadAffinityMode::CUSTOM: mask = (custom) ? custom & cpu_info.maskSystem : cpu_info.maskSystem; break;
    case RGYThreadAffinityMode::ALL:
    default: mask = cpu_info.maskSystem; break;
    }
    return (mask) ? mask : std::numeric_limits<decltype(mask)>::max();
}

RGYParamThreadAffinity::RGYParamThreadAffinity() :
    process(),
    main(),
    dec(),
    enc(),
    csp(),
    input(),
    output(),
    audio(),
    perfmonitor(),
    videoquality() {
}

const TCHAR *rgy_thread_type_to_str(RGYThreadType type) {
    for (const auto& p : RGY_THREAD_TYPE_STR) {
        if (p.first == type) return p.second;
    }
    return nullptr;
}

RGYThreadAffinity RGYParamThreadAffinity::get(RGYThreadType type) const {
    switch (type) {
    case RGYThreadType::MAIN: return main;
    case RGYThreadType::DEC:  return dec;
    case RGYThreadType::ENC:  return enc;
    case RGYThreadType::CSP:  return csp;
    case RGYThreadType::INPUT: return input;
    case RGYThreadType::OUTUT: return output;
    case RGYThreadType::AUDIO: return audio;
    case RGYThreadType::PERF_MONITOR: return perfmonitor;
    case RGYThreadType::VIDEO_QUALITY: return videoquality;
    case RGYThreadType::PROCESS: return process;
    case RGYThreadType::ALL:
    default: return process;
    }
}

void RGYParamThreadAffinity::set(RGYThreadAffinity affinity, RGYThreadType type) {
    switch (type) {
    case RGYThreadType::PROCESS: process = affinity; break;
    case RGYThreadType::MAIN: main = affinity; break;
    case RGYThreadType::DEC:  dec = affinity; break;
    case RGYThreadType::ENC:  enc = affinity; break;
    case RGYThreadType::CSP:  csp = affinity; break;
    case RGYThreadType::INPUT: input = affinity; break;
    case RGYThreadType::OUTUT: output = affinity; break;
    case RGYThreadType::AUDIO: audio = affinity; break;
    case RGYThreadType::PERF_MONITOR: perfmonitor = affinity; break;
    case RGYThreadType::VIDEO_QUALITY: videoquality = affinity; break;
    case RGYThreadType::ALL:
        process = affinity;
        main = affinity;
        dec = affinity;
        enc = affinity;
        csp = affinity;
        input = affinity;
        output = affinity;
        audio = affinity;
        perfmonitor = affinity;
        videoquality = affinity; break;
    default: break;
    }
}

tstring RGYParamThreadAffinity::to_string() const {
    std::basic_stringstream<TCHAR> tmp;
#define RGY_THREAD_AFF_ADD_TYPE(TYPE, VAR) { tmp << _T(",") << rgy_thread_type_to_str(TYPE) << _T("=") << VAR.to_string(); }
    RGY_THREAD_AFF_ADD_TYPE(RGYThreadType::PROCESS, process);
    RGY_THREAD_AFF_ADD_TYPE(RGYThreadType::MAIN, main);
    RGY_THREAD_AFF_ADD_TYPE(RGYThreadType::DEC, dec);
    RGY_THREAD_AFF_ADD_TYPE(RGYThreadType::ENC, enc);
    RGY_THREAD_AFF_ADD_TYPE(RGYThreadType::INPUT, input);
    RGY_THREAD_AFF_ADD_TYPE(RGYThreadType::OUTUT, output);
    RGY_THREAD_AFF_ADD_TYPE(RGYThreadType::AUDIO, audio);
    RGY_THREAD_AFF_ADD_TYPE(RGYThreadType::PERF_MONITOR, perfmonitor);
    RGY_THREAD_AFF_ADD_TYPE(RGYThreadType::VIDEO_QUALITY, videoquality);
#undef LOG_LEVEL_ADD_TYPE
    return tmp.str();
}

bool RGYParamThreadAffinity::operator==(const RGYParamThreadAffinity &x) const {
    return process == x.process
        && main == x.main
        && dec == x.dec
        && enc == x.enc
        && csp == x.csp
        && input == x.input
        && output == x.output
        && audio == x.audio
        && perfmonitor == x.perfmonitor
        && videoquality == x.videoquality;
}
bool RGYParamThreadAffinity::operator!=(const RGYParamThreadAffinity &x) const {
    return !(*this == x);
}
