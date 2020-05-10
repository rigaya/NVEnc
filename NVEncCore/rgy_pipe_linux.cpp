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

#if !(defined(_WIN32) || defined(_WIN64))
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <sys/wait.h>
#include "rgy_pipe.h"
#include "rgy_tchar.h"

RGYPipeProcessLinux::RGYPipeProcessLinux() {
}

RGYPipeProcessLinux::~RGYPipeProcessLinux() {

}

void RGYPipeProcessLinux::init() {
    close();
}


int RGYPipeProcessLinux::startPipes(ProcessPipe *pipes) {
    if (pipes->stdOut.mode) {
        if (-1 == (pipe((int *)&pipes->stdOut.h_read)))
            return 1;
        pipes->f_stdout = fdopen(pipes->stdOut.h_read, "r");
    }
    if (pipes->stdErr.mode) {
        if (-1 == (pipe((int *)&pipes->stdErr.h_read)))
            return 1;
        pipes->f_stderr = fdopen(pipes->stdErr.h_read, "r");
    }
    if (pipes->stdIn.mode) {
        if (-1 == (pipe((int *)&pipes->stdIn.h_read)))
            return 1;
        pipes->f_stdin = fdopen(pipes->stdIn.h_write, "w");
    }
    return 0;
}

int RGYPipeProcessLinux::run(const std::vector<const TCHAR *>& args, const TCHAR *exedir, ProcessPipe *pipes, uint32_t priority, bool hidden, bool minimized) {
    startPipes(pipes);

    m_phandle = fork();
    if (m_phandle < 0) {
        return 1;
    }

    if (m_phandle == 0) {
        //子プロセス
        if (pipes->stdIn.mode) {
            ::close(pipes->stdIn.h_write);
            dup2(pipes->stdIn.h_read, STDIN_FILENO);
        }
        if (pipes->stdOut.mode) {
            ::close(pipes->stdOut.h_read);
            dup2(pipes->stdOut.h_write, STDOUT_FILENO);
        }
        if (pipes->stdErr.mode) {
            ::close(pipes->stdErr.h_read);
            dup2(pipes->stdErr.h_write, STDERR_FILENO);
        }
        int ret = execvp(args[0], (char *const *)args.data());
        exit(-1);
    }
    //親プロセス
    if (pipes->stdIn.mode) {
        ::close(pipes->stdIn.h_read);
        pipes->stdIn.h_read = 0;
    }
    if (pipes->stdOut.mode) {
        ::close(pipes->stdOut.h_write);
        pipes->stdOut.h_write = 0;
    }
    if (pipes->stdErr.mode) {
        ::close(pipes->stdErr.h_write);
        pipes->stdErr.h_write = 0;
    }
    return 0;
}

void RGYPipeProcessLinux::close() {
}

tstring RGYPipeProcessLinux::getOutput(ProcessPipe *pipes) {
    std::string outstr;
    auto read_from_pipe = [&]() {
        char buf[4096];
        int ret = fread(buf, sizeof(buf[0]), _countof(buf), pipes->f_stdout);
        outstr += ret;
        return (int)ret;
    };

    while (processAlive()) {
        read_from_pipe();
    }
    for (;;) {
        if (read_from_pipe() <= 0) {
            break;
        }
    }
    return outstr;
}

bool RGYPipeProcessLinux::processAlive() {
    int status = 0;
    return 0 == waitpid(m_phandle, &status, WNOHANG);
}
#endif //#if !(defined(_WIN32) || defined(_WIN64))
