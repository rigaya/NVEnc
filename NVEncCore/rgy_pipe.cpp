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

#include "rgy_pipe.h"
#include "rgy_util.h"
#if defined(_WIN32) || defined(_WIN64)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>
#include <fcntl.h>
#include <io.h>
#include <cstring>

RGYPipeProcessWin::RGYPipeProcessWin() :
    RGYPipeProcess(),
    m_pi() {
}

RGYPipeProcessWin::~RGYPipeProcessWin() {
    close();
}

int RGYPipeProcessWin::startPipes() {
    if (m_pipe.stdOut.mode & PIPE_MODE_ENABLE) {
        SECURITY_ATTRIBUTES sa = { sizeof(SECURITY_ATTRIBUTES), NULL, TRUE };
        if (!CreatePipe(&m_pipe.stdOut.h_read, &m_pipe.stdOut.h_write, &sa, m_pipe.stdOut.bufferSize) ||
            !SetHandleInformation(m_pipe.stdOut.h_read, HANDLE_FLAG_INHERIT, 0))
            return 1;
        if ((m_pipe.stdOut.mode & PIPE_MODE_ENABLE_FP)
            && (m_pipe.stdOut.fp = _fdopen(_open_osfhandle((intptr_t)m_pipe.stdOut.h_read, _O_BINARY), "rb")) == NULL) {
            return 1;
        }
        auto bufsize = m_pipe.stdOut.bufferSize ? m_pipe.stdOut.bufferSize : RGY_PIPE_STDOUT_BUFSIZE_DEFAULT;
        m_stdOutBuffer.resize(bufsize);
    }
    if (m_pipe.stdErr.mode & PIPE_MODE_ENABLE) {
        SECURITY_ATTRIBUTES sa = { sizeof(SECURITY_ATTRIBUTES), NULL, TRUE };
        if (!CreatePipe(&m_pipe.stdErr.h_read, &m_pipe.stdErr.h_write, &sa, m_pipe.stdErr.bufferSize) ||
            !SetHandleInformation(m_pipe.stdErr.h_read, HANDLE_FLAG_INHERIT, 0))
            return 1;
        if ((m_pipe.stdErr.mode & PIPE_MODE_ENABLE_FP)
            && (m_pipe.stdErr.fp = _fdopen(_open_osfhandle((intptr_t)m_pipe.stdErr.h_read, _O_BINARY), "rb")) == NULL) {
            return 1;
        }
        auto bufsize = m_pipe.stdErr.bufferSize ? m_pipe.stdErr.bufferSize : RGY_PIPE_STDERR_BUFSIZE_DEFAULT;
        m_stdErrBuffer.resize(bufsize);
    }
    if (m_pipe.stdIn.mode & PIPE_MODE_ENABLE) {
        SECURITY_ATTRIBUTES sa = { sizeof(SECURITY_ATTRIBUTES), NULL, TRUE };
        if (!CreatePipe(&m_pipe.stdIn.h_read, &m_pipe.stdIn.h_write, &sa, m_pipe.stdIn.bufferSize) ||
            !SetHandleInformation(m_pipe.stdIn.h_write, HANDLE_FLAG_INHERIT, 0))
            return 1;
        if ((m_pipe.stdIn.mode & PIPE_MODE_ENABLE_FP)
            && (m_pipe.stdIn.fp = _fdopen(_open_osfhandle((intptr_t)m_pipe.stdIn.h_write, _O_BINARY), "wb")) == NULL) {
            return 1;
        }
    }
    return 0;
}

int RGYPipeProcessWin::run(const tstring& cmd_line, const TCHAR *exedir, uint32_t priority, bool hidden, bool minimized) {
    BOOL Inherit = FALSE;
    DWORD flag = priority;
    STARTUPINFO si;
    memset(&si, 0, sizeof(STARTUPINFO));
    memset(&m_pi, 0, sizeof(PROCESS_INFORMATION));
    si.cb = sizeof(STARTUPINFO);

    startPipes();

    if (m_pipe.stdOut.mode)
        si.hStdOutput = m_pipe.stdOut.h_write;
    if (m_pipe.stdErr.mode)
        si.hStdError = ((m_pipe.stdErr.mode & (PIPE_MODE_ENABLE | PIPE_MODE_MUXED)) == (PIPE_MODE_ENABLE | PIPE_MODE_MUXED)) ? m_pipe.stdOut.h_write : m_pipe.stdErr.h_write;
    if (m_pipe.stdIn.mode)
        si.hStdInput = m_pipe.stdIn.h_read;
    si.dwFlags |= STARTF_USESTDHANDLES;
    Inherit = TRUE;
    //flag |= DETACHED_PROCESS; //このフラグによるコンソール抑制よりCREATE_NO_WINDOWの抑制を使用する
    if (minimized) {
        si.dwFlags |= STARTF_USESHOWWINDOW;
        si.wShowWindow |= SW_SHOWMINNOACTIVE;
    }
    if (hidden)
        flag |= CREATE_NO_WINDOW;

    int ret = (CreateProcess(NULL, (TCHAR *)cmd_line.c_str(), NULL, NULL, Inherit, flag, NULL, exedir, &si, &m_pi)) ? 0 : 1;
    m_phandle = m_pi.hProcess;
    if (m_pipe.stdOut.mode) {
        CloseHandle(m_pipe.stdOut.h_write);
        m_pipe.stdOut.h_write = nullptr;
        if (ret) {
            CloseHandle(m_pipe.stdOut.h_read);
            m_pipe.stdOut.h_read = nullptr;
            m_pipe.stdOut.mode = PIPE_MODE_DISABLE;
        }
    }
    if (m_pipe.stdErr.mode) {
        CloseHandle(m_pipe.stdErr.h_write);
        m_pipe.stdErr.h_write = nullptr;
        if (ret) {
            CloseHandle(m_pipe.stdErr.h_read);
            m_pipe.stdErr.h_read = nullptr;
            m_pipe.stdErr.mode = PIPE_MODE_DISABLE;
        }
    }
    if (m_pipe.stdIn.mode) {
        CloseHandle(m_pipe.stdIn.h_read);
        m_pipe.stdIn.h_read = nullptr;
        if (ret) {
            CloseHandle(m_pipe.stdIn.h_write);
            m_pipe.stdIn.h_write = nullptr;
            m_pipe.stdIn.mode = PIPE_MODE_DISABLE;
        }
    }
    return ret;
}

int RGYPipeProcessWin::run(const std::vector<tstring>& args, const TCHAR *exedir, uint32_t priority, bool hidden, bool minimized) {
    tstring cmd_line;
    for (auto arg : args) {
        if (!arg.empty()) {
            cmd_line += _T("\"") + tstring(arg) + _T("\" ");
        }
    }
    return run(cmd_line, exedir, priority, hidden, minimized);
}

int RGYPipeProcessWin::stdInClose() {
    if (m_pipe.stdIn.h_write) {
        CloseHandle(m_pipe.stdIn.h_write);
        m_pipe.stdIn.h_write = nullptr;
    }
    return 0;
}

size_t RGYPipeProcessWin::stdInFpWrite(const void *data, const size_t dataSize) {
    return _fwrite_nolock(data, 1, dataSize, m_pipe.stdIn.fp);
}

int RGYPipeProcessWin::stdInFpFlush() {
    return fflush(m_pipe.stdIn.fp);
}

int RGYPipeProcessWin::stdInFpClose() {
    int ret = 0;
    if (m_pipe.stdIn.fp) {
        ret = fclose(m_pipe.stdIn.fp);
        m_pipe.stdIn.fp = nullptr;
        m_pipe.stdIn.h_write = nullptr;
    }
    return ret;
}

size_t RGYPipeProcessWin::stdOutFpRead(void *data, const size_t dataSize) {
    return _fread_nolock(data, 1, dataSize, m_pipe.stdOut.fp);
}

int RGYPipeProcessWin::stdOutFpClose() {
    int ret = 0;
    if (m_pipe.stdOut.fp) {
        ret = fclose(m_pipe.stdOut.fp);
        m_pipe.stdOut.fp = nullptr;
        m_pipe.stdOut.h_read = nullptr;
    }
    return ret;
}

size_t RGYPipeProcessWin::stdErrFpRead(void *data, const size_t dataSize) {
    return _fread_nolock(data, 1, dataSize, m_pipe.stdErr.fp);
}

int RGYPipeProcessWin::stdErrFpClose() {
    int ret = 0;
    if (m_pipe.stdErr.fp) {
        ret = fclose(m_pipe.stdErr.fp);
        m_pipe.stdErr.fp = nullptr;
        m_pipe.stdErr.h_read = nullptr;
    }
    return ret;
}

int RGYPipeProcessWin::stdInWrite(const void *data, const size_t dataSize) {
    DWORD pipe_write = 0;
    if (!WriteFile(m_pipe.stdIn.h_write, data, (DWORD)dataSize, &pipe_write, NULL)) {
        return -1;
    }
    if (dataSize != pipe_write) {
        return -1;
    }
    return (int)pipe_write;
}

int RGYPipeProcessWin::stdInWrite(const std::vector<uint8_t>& buffer) {
    return stdInWrite(buffer.data(), buffer.size());
}

int RGYPipeProcessWin::stdOutRead(std::vector<uint8_t>& buffer) {
    auto read_from_pipe = [&]() {
        DWORD pipe_read = 0;
        //if (!PeekNamedPipe(m_pipe.stdOut.h_read, NULL, 0, NULL, &pipe_read, NULL))
        //    return -1;
        //if (pipe_read) {
            if (!ReadFile(m_pipe.stdOut.h_read, m_stdOutBuffer.data(), (DWORD)m_stdOutBuffer.size(), &pipe_read, NULL)) {
                return -1;
            }
            // パイプはWriteFileにゼロを渡すとReadFileがゼロで帰ってくることがある
            if (pipe_read > 0) {
                buffer.insert(buffer.end(), m_stdOutBuffer.data(), m_stdOutBuffer.data() + pipe_read);
            }
        //}
        return (int)pipe_read;
    };
    int ret = 0;
    for (;;) {
        ret = read_from_pipe();
        if (ret != 0) {
            break;
        }
    }
    return ret < 0 ? -1 : (int)buffer.size();
}
int RGYPipeProcessWin::stdErrRead(std::vector<uint8_t>& buffer) {
    auto read_from_pipe = [&]() {
        DWORD pipe_read = 0;
        //if (!PeekNamedPipe(m_pipe.stdErr.h_read, NULL, 0, NULL, &pipe_read, NULL))
        //    return -1;
        //if (pipe_read) {
            if (!ReadFile(m_pipe.stdErr.h_read, m_stdErrBuffer.data(), (DWORD)m_stdErrBuffer.size(), &pipe_read, NULL)) {
                return -1;
            }
            // パイプはWriteFileにゼロを渡すとReadFileがゼロで帰ってくることがある
            if (pipe_read > 0) {
                buffer.insert(buffer.end(), m_stdErrBuffer.data(), m_stdErrBuffer.data() + pipe_read);
            }
        //}
        return (int)pipe_read;
    };

    int ret = 0;
    for (;;) {
        ret = read_from_pipe();
        if (ret != 0) {
            break;
        }
    }
    return ret < 0 ? -1 : (int)buffer.size();
}

std::string RGYPipeProcessWin::getOutput() {
    std::string outstr;
    auto read_from_pipe = [&]() {
        DWORD pipe_read = 0;
        if (!PeekNamedPipe(m_pipe.stdOut.h_read, NULL, 0, NULL, &pipe_read, NULL))
            return -1;
        if (pipe_read) {
            char read_buf[1024] = { 0 };
            ReadFile(m_pipe.stdOut.h_read, read_buf, sizeof(read_buf) - 1, &pipe_read, NULL);
            outstr += read_buf;
        }
        return (int)pipe_read;
    };

    while (WAIT_TIMEOUT == WaitForSingleObject(m_phandle, 10)) {
        read_from_pipe();
    }
    for (;;) {
        if (read_from_pipe() <= 0) {
            break;
        }
    }
    return outstr;
}

int RGYPipeProcessWin::wait(uint32_t timeout) {
    return WaitForSingleObject(m_phandle, timeout);
}

int RGYPipeProcessWin::waitAndGetExitCode() {
    if (WaitForSingleObject(m_phandle, INFINITE) == WAIT_OBJECT_0) {
        DWORD exitCode = 0;
        if (GetExitCodeProcess(m_phandle, &exitCode)) {
            return (int)exitCode;
        }
    }
    return -1;
}

const PROCESS_INFORMATION& RGYPipeProcessWin::getProcessInfo() {
    return m_pi;
}

void RGYPipeProcessWin::close() {
    if (m_pipe.stdIn.mode & PIPE_MODE_ENABLE_FP) {
        stdInFpClose();
    }
    if (m_pipe.stdOut.mode & PIPE_MODE_ENABLE_FP) {
        stdOutFpClose();
    }
    if (m_pipe.stdErr.mode & PIPE_MODE_ENABLE_FP) {
        stdErrFpClose();
    }
    if (m_pipe.stdIn.h_read) {
        CloseHandle(m_pipe.stdIn.h_read);
        m_pipe.stdIn.h_read = nullptr;
    }
    if (m_pipe.stdIn.h_write) {
        CloseHandle(m_pipe.stdIn.h_write);
        m_pipe.stdIn.h_write = nullptr;
    }
    if (m_pipe.stdOut.h_read) {
        CloseHandle(m_pipe.stdOut.h_read);
        m_pipe.stdOut.h_read = nullptr;
    }
    if (m_pipe.stdOut.h_write) {
        CloseHandle(m_pipe.stdOut.h_write);
        m_pipe.stdOut.h_write = nullptr;
    }
    if (m_pipe.stdErr.h_read) {
        CloseHandle(m_pipe.stdErr.h_read);
        m_pipe.stdErr.h_read = nullptr;
    }
    if (m_pipe.stdErr.h_write) {
        CloseHandle(m_pipe.stdErr.h_write);
        m_pipe.stdErr.h_write = nullptr;
    }
    if (m_pi.hProcess) {
        CloseHandle(m_pi.hProcess);
        m_pi.hProcess = nullptr;
    }
    if (m_pi.hThread) {
        CloseHandle(m_pi.hThread);
        m_pi.hThread = nullptr;
    }
    memset(&m_pi, 0, sizeof(m_pi));
}

int RGYPipeProcessWin::pid() const {
    return m_pi.dwProcessId;
}

bool RGYPipeProcessWin::processAlive() {
    return WAIT_TIMEOUT == WaitForSingleObject(m_phandle, 0);
}


#endif //defined(_WIN32) || defined(_WIN64)


std::unique_ptr<RGYPipeProcess> createRGYPipeProcess() {
#if defined(_WIN32) || defined(_WIN64)
    auto process = std::make_unique<RGYPipeProcessWin>();
#else
    auto process = std::make_unique<RGYPipeProcessLinux>();
#endif
    return std::move(process);
}

// コマンドライン文字列を引数リストに分割するヘルパー関数
std::vector<tstring> SplitCommandLine(const tstring& cmdLine) {
    std::vector<tstring> args;

#if defined(_WIN32) || defined(_WIN64)
    // Windows版の実装
    int argc = 0;
    LPWSTR* argv = CommandLineToArgvW(cmdLine.c_str(), &argc);
    if (argv) {
        for (int i = 0; i < argc; i++) {
            args.push_back(argv[i]);
        }
        LocalFree(argv);
    }
#else
    // Linux/Unix版の実装
    // 単純なシェル風の解析を行う
    const auto isSpace = [](TCHAR c) { return c == _T(' ') || c == _T('\t'); };

    bool inQuote = false;
    TCHAR quoteChar = 0;
    tstring currentArg;

    for (size_t i = 0; i < cmdLine.length(); i++) {
        TCHAR c = cmdLine[i];

        // クオート処理
        if (c == _T('\'') || c == _T('"')) {
            if (!inQuote) {
                inQuote = true;
                quoteChar = c;
            } else if (c == quoteChar) {
                inQuote = false;
                quoteChar = 0;
            } else {
                currentArg += c;
            }
            continue;
        }

        // バックスラッシュによるエスケープ
        if (c == _T('\\') && i + 1 < cmdLine.length()) {
            TCHAR nextChar = cmdLine[i + 1];
            if (nextChar == _T('\'') || nextChar == _T('"') || nextChar == _T('\\')) {
                currentArg += nextChar;
                i++; // 次の文字をスキップ
                continue;
            }
        }

        // 空白文字の処理
        if (isSpace(c) && !inQuote) {
            if (!currentArg.empty()) {
                args.push_back(currentArg);
                currentArg.clear();
            }
        } else {
            currentArg += c;
        }
    }

    // 最後の引数を追加
    if (!currentArg.empty()) {
        args.push_back(currentArg);
    }
#endif

    // 空のリストにならないように、少なくとも空の引数を1つ入れる
    if (args.empty()) {
        args.push_back(_T(""));
    }

    return args;
}
