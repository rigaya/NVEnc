//  -----------------------------------------------------------------------------------------
//    拡張 x264 出力(GUI) Ex  v1.xx by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  -----------------------------------------------------------------------------------------

#ifndef _AUO_PIPE_H_
#define _AUO_PIPE_H_

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
