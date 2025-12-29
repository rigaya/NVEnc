// -----------------------------------------------------------------------------------------
// QSVEnc/NVEnc by rigaya
// -----------------------------------------------------------------------------------------
// The MIT License
//
// Copyright (c) 2011-2016 rigaya
//
// ------------------------------------------------------------------------------------------

#pragma once
#ifndef __RGY_VAPOURSYNTH_WRAPPER_H__
#define __RGY_VAPOURSYNTH_WRAPPER_H__

#include <cstdint>
#include <memory>
#include <string>

#include "rgy_tchar.h"
#include "rgy_err.h"

class RGYLog;

struct RGYVapourSynthVideoInfo {
    int width;
    int height;
    int numFrames;
    int64_t fpsNum;
    int64_t fpsDen;
    int bitsPerSample;
    int subSamplingW;
    int subSamplingH;
    bool isYUV;
    bool isInteger;
    int api;        // core api version (3/4)
    int numThreads; // core thread count
    std::string versionString;
};

using RGYVapourSynthFrameDoneCallback = void (*)(void *userData, const void *frame, int n, const char *errorMsg);

class RGYVapourSynthWrapper {
public:
    virtual ~RGYVapourSynthWrapper() = default;

    virtual int apiMajor() const = 0; // 3 or 4
    virtual const RGYVapourSynthVideoInfo& videoInfo() const = 0;

    virtual RGY_ERR openScriptFromBuffer(const std::string& script, const std::string& scriptFilenameUtf8) = 0;

    virtual void getFrameAsync(int n, RGYVapourSynthFrameDoneCallback cb, void *userData) = 0;
    virtual const uint8_t *getReadPtr(const void *frame, int plane) const = 0;
    virtual ptrdiff_t getStride(const void *frame, int plane) const = 0;
    virtual void freeFrame(const void *frame) const = 0;

    virtual void close() = 0;
};

// Factory: tries v4 first, then v3.
std::unique_ptr<RGYVapourSynthWrapper> CreateVapourSynthWrapper(const tstring& vsdir, RGYLog *log);

#endif //__RGY_VAPOURSYNTH_WRAPPER_H__


