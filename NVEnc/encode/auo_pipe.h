// -----------------------------------------------------------------------------------------
// NVEnc by rigaya
// -----------------------------------------------------------------------------------------
// The MIT License
//
// Copyright (c) 2014-2016 rigaya
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
// ------------------------------------------------------------------------------------------
#ifndef _AUO_PIPE_H_
#define _AUO_PIPE_H_

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>
#include <stdio.h>

enum {
    RP_USE_NO_PIPE = -1,
    RP_SUCCESS = 0,
    RP_ERROR_OPEN_PIPE,
    RP_ERROR_GET_STDIN_FILE_HANDLE,
    RP_ERROR_CREATE_PROCESS,
};

enum AUO_PIPE_MODE {
    AUO_PIPE_DISABLE = 0,
    AUO_PIPE_ENABLE,
    AUO_PIPE_MUXED, //Stderrのモードに使用し、StderrをStdOutに混合する
};

const int PIPE_READ_BUF = 2048;

typedef struct {
    HANDLE h_read;
    HANDLE h_write;
    AUO_PIPE_MODE mode;
    DWORD bufferSize;
} PIPE;

typedef struct {
    PIPE stdIn;
    PIPE stdOut;
    PIPE stdErr;
    FILE *f_stdin;
    DWORD buf_len;
    char read_buf[PIPE_READ_BUF];
} PIPE_SET;

void InitPipes(PIPE_SET *pipes);
int RunProcess(char *args, const char *exe_dir, PROCESS_INFORMATION *pi, PIPE_SET *pipes, DWORD priority, BOOL hidden, BOOL minimized);
void CloseStdIn(PIPE_SET *pipes);
BOOL get_exe_message(const char *exe_path, const char *args, char *buf, size_t nSize, AUO_PIPE_MODE from_stderr);
BOOL get_exe_message_to_file(const char *exe_path, const char *args, const char *filepath, AUO_PIPE_MODE from_stderr, DWORD loop_ms);

#endif //_AUO_PIPE_H_
