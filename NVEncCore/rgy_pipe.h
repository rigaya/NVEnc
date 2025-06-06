// -----------------------------------------------------------------------------------------
// QSVEnc/NVEnc by rigaya
// -----------------------------------------------------------------------------------------
// The MIT License
//
// Copyright (c) 2011-2016 rigaya
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

#pragma once
#ifndef __RGY_PIPE_H__
#define __RGY_PIPE_H__

#include <cstdint>
#include <cstdio>
#include <vector>
#include "rgy_osdep.h"
#include "rgy_util.h"
#include "rgy_tchar.h"

std::vector<tstring> SplitCommandLine(const tstring& cmdLine);

enum RGYPipeMode : uint32_t {
    PIPE_MODE_DISABLE   = 0x00,
    PIPE_MODE_ENABLE    = 0x01,
    PIPE_MODE_ENABLE_FP = 0x02,
    PIPE_MODE_MUXED     = 0x04, //Stderrのモードに使用し、StderrをStdOutに混合する
};

static RGYPipeMode operator|(RGYPipeMode a, RGYPipeMode b) {
    return (RGYPipeMode)((uint32_t)a | (uint32_t)b);
}

static RGYPipeMode operator|=(RGYPipeMode& a, RGYPipeMode b) {
    a = a | b;
    return a;
}

static RGYPipeMode operator&(RGYPipeMode a, RGYPipeMode b) {
    return (RGYPipeMode)((uint32_t)a & (uint32_t)b);
}

static RGYPipeMode operator&=(RGYPipeMode& a, RGYPipeMode b) {
    a = (RGYPipeMode)((uint32_t)a & (uint32_t)b);
    return a;
}

static const int RGY_PIPE_STDOUT_BUFSIZE_DEFAULT = 512 * 1024;
static const int RGY_PIPE_STDERR_BUFSIZE_DEFAULT = 4 * 1024;

#if defined(_WIN32) || defined(_WIN64)
typedef HANDLE PIPE_HANDLE;
typedef HANDLE PROCESS_HANDLE;
#else
typedef int PIPE_HANDLE;
typedef pid_t PROCESS_HANDLE;
#endif

struct RGYPipeSet {
    PIPE_HANDLE h_read;
    PIPE_HANDLE h_write;
    RGYPipeMode mode;
    uint32_t bufferSize;
    FILE *fp;

    RGYPipeSet() : h_read(0), h_write(0), mode(PIPE_MODE_DISABLE), bufferSize(0), fp(nullptr) {};
};

struct RGYProcessPipe {
    RGYPipeSet stdIn;
    RGYPipeSet stdOut;
    RGYPipeSet stdErr;

    RGYProcessPipe() : stdIn(), stdOut(), stdErr() { };
};

class RGYPipeProcess {
public:
    RGYPipeProcess() : m_phandle(0), m_pipe(), m_stdOutBuffer(), m_stdErrBuffer() { };
    virtual ~RGYPipeProcess() { };

    void init(RGYPipeMode stdin_, RGYPipeMode stdout_, RGYPipeMode stderr_) {
        m_pipe.stdIn.mode = stdin_;
        m_pipe.stdOut.mode = stdout_;
        m_pipe.stdErr.mode = stderr_;
    };
    void setStdOutBufferSize(uint32_t size) {
        m_pipe.stdOut.bufferSize = size;
    }
    void setStdErrBufferSize(uint32_t size) {
        m_pipe.stdErr.bufferSize = size;
    }
    virtual int run(const tstring& cmd_line, const TCHAR *exedir, uint32_t priority, bool hidden, bool minimized) = 0;
    virtual int run(const std::vector<tstring>& args, const TCHAR *exedir, uint32_t priority, bool hidden, bool minimized) = 0;
    virtual void close() = 0;
    virtual bool processAlive() = 0;
    virtual std::string getOutput() = 0;
    virtual int stdInClose() = 0;
    virtual int stdInWrite(const void *data, const size_t dataSize) = 0;
    virtual int stdInWrite(const std::vector<uint8_t>& buffer) = 0;
    virtual int stdOutRead(std::vector<uint8_t>& buffer) = 0;
    virtual int stdErrRead(std::vector<uint8_t>& buffer) = 0;
    virtual size_t stdInFpWrite(const void *data, const size_t dataSize) = 0;
    virtual int stdInFpFlush() = 0;
    virtual int stdInFpClose() = 0;
    virtual size_t stdOutFpRead(void *data, const size_t dataSize) = 0;
    virtual int stdOutFpClose() = 0;
    virtual size_t stdErrFpRead(void *data, const size_t dataSize) = 0;
    virtual int stdErrFpClose() = 0;
    virtual int wait(uint32_t timeout) = 0;
    virtual int waitAndGetExitCode() = 0;
    virtual int pid() const = 0;
protected:
    virtual int startPipes() = 0;
    PROCESS_HANDLE m_phandle;
    RGYProcessPipe m_pipe;
    std::vector<uint8_t> m_stdOutBuffer;
    std::vector<uint8_t> m_stdErrBuffer;
};

#if defined(_WIN32) || defined(_WIN64)
class RGYPipeProcessWin : public RGYPipeProcess {
public:
    RGYPipeProcessWin();
    virtual ~RGYPipeProcessWin();

    virtual int run(const tstring& cmd_line, const TCHAR *exedir, uint32_t priority, bool hidden, bool minimized) override;
    virtual int run(const std::vector<tstring>& args, const TCHAR *exedir, uint32_t priority, bool hidden, bool minimized) override;
    virtual void close() override;
    virtual bool processAlive() override;
    virtual std::string getOutput() override;
    virtual int stdInClose() override;
    virtual int stdInWrite(const void *data, const size_t dataSize) override;
    virtual int stdInWrite(const std::vector<uint8_t>& buffer) override;
    virtual int stdOutRead(std::vector<uint8_t>& buffer) override;
    virtual int stdErrRead(std::vector<uint8_t>& buffer) override;
    virtual size_t stdInFpWrite(const void *data, const size_t dataSize) override;
    virtual int stdInFpFlush() override;
    virtual int stdInFpClose() override;
    virtual size_t stdOutFpRead(void *data, const size_t dataSize) override;
    virtual int stdOutFpClose() override;
    virtual size_t stdErrFpRead(void *data, const size_t dataSize) override;
    virtual int stdErrFpClose() override;
    virtual int wait(uint32_t timeout) override;
    virtual int waitAndGetExitCode() override;
    virtual int pid() const override;
    const PROCESS_INFORMATION& getProcessInfo();
protected:
    virtual int startPipes() override;
    PROCESS_INFORMATION m_pi;
};
#else

#ifndef INFINITE
#define INFINITE (UINT_MAX)
#endif
class RGYPipeProcessLinux : public RGYPipeProcess {
public:
    RGYPipeProcessLinux();
    virtual ~RGYPipeProcessLinux();

    virtual int run(const tstring& cmd_line, const TCHAR *exedir, uint32_t priority, bool hidden, bool minimized) override;
    virtual int run(const std::vector<tstring>& args, const TCHAR *exedir, uint32_t priority, bool hidden, bool minimized) override;
    virtual void close() override;
    virtual bool processAlive() override;
    virtual std::string getOutput() override;
    virtual int stdInClose() override;
    virtual int stdInWrite(const void *data, const size_t dataSize) override;
    virtual int stdInWrite(const std::vector<uint8_t>& buffer) override;
    virtual int stdOutRead(std::vector<uint8_t>& buffer) override;
    virtual int stdErrRead(std::vector<uint8_t>& buffer) override;
    virtual size_t stdInFpWrite(const void *data, const size_t dataSize) override;
    virtual int stdInFpFlush() override;
    virtual int stdInFpClose() override;
    virtual size_t stdOutFpRead(void *data, const size_t dataSize) override;
    virtual int stdOutFpClose() override;
    virtual size_t stdErrFpRead(void *data, const size_t dataSize) override;
    virtual int stdErrFpClose() override;
    virtual int wait(uint32_t timeout) override;
    virtual int waitAndGetExitCode() override;
    virtual int pid() const override;
protected:
    virtual int startPipes() override;
};
#endif //#if defined(_WIN32) || defined(_WIN64)

std::unique_ptr<RGYPipeProcess> createRGYPipeProcess();

#endif //__RGY_PIPE_H__
