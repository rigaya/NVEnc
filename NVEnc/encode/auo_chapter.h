//  -----------------------------------------------------------------------------------------
//    拡張 x264/x265 出力(GUI) Ex  v1.xx/2.xx/3.xx by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  -----------------------------------------------------------------------------------------

#ifndef _AUO_CHAPTER_H_
#define _AUO_CHAPTER_H_

#include <Windows.h>

typedef struct {
	WCHAR *name;
	int h, m, s, ms;
} chapter_t;

typedef struct {
	chapter_t *data;
	int count;
} chapter_list_t;

enum {
	CHAP_TYPE_ANOTHER = -1,
	CHAP_TYPE_UNKNOWN = 0,
	CHAP_TYPE_NERO    = 1,
	CHAP_TYPE_APPLE   = 2,
};

enum {
	AUO_CHAP_ERR_NONE = 0,
	AUO_CHAP_ERR_FILE_OPEN,
	AUO_CHAP_ERR_FILE_WRITE,
	AUO_CHAP_ERR_FILE_READ,
	AUO_CHAP_ERR_FILE_SWAP,
	AUO_CHAP_ERR_NULL_PTR,
	AUO_CHAP_ERR_INIT_IMUL2,
	AUO_CHAP_ERR_CONVERTION,
	AUO_CHAP_ERR_CP_DETECT,
	AUO_CHAP_ERR_INVALID_FMT,
	AUO_CHAP_ERR_INIT_XML_PARSER,
	AUO_CHAP_ERR_INIT_READ_STREAM,
	AUO_CHAP_ERR_FAIL_SET_STREAM,
	AUO_CHAP_ERR_PARSE_XML
};

int get_chapter_list(chapter_list_t *chap_list, const char *filename, DWORD orig_code_page);
void free_chapter_list(chapter_list_t *chap_list);
double get_chap_second(chapter_t *chap);

//チャプターファイルの変換を行う
//基本的にはorig_nero_filename(nero形式) から new_apple_filename(apple形式) へ
//orig_fileがapple形式の場合、nero形式を出力してファイル名をスワップする
int convert_chapter(const char *new_apple_filename, const char *orig_nero_filename, DWORD orig_code_page, double duration, int out_chap_type = CHAP_TYPE_ANOTHER, bool nero_out_utf8 = false);

int create_chapter_file_delayed_by_add_vframe(const char *new_filename, const char *orig_filename, int delay_ms);

#endif //_AUO_CHAPTER_H_