//  -----------------------------------------------------------------------------------------
//    拡張 x264/x265 出力(GUI) Ex  v1.xx/2.xx/3.xx by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  -----------------------------------------------------------------------------------------

#include <Windows.h>
#include <stdio.h>
#include <vector>
#include <fstream>

#include <objbase.h>
#pragma comment(lib, "ole32.lib")
#include <mlang.h>
#include <xmllite.h>
#pragma comment (lib, "xmllite.lib")
#include <shlwapi.h>
#pragma comment (lib, "shlwapi.lib")

#include "auo.h"
#include "auo_util.h"
#include "auo_chapter.h"

double get_chap_second(chapter_t *chap) {
	return (double)(chap->h * 3600 + chap->m * 60 + chap->s) + chap->ms * 0.001;
}

//bufのWCHAR文字列をutf-8に変換しfpに出力
static int write_utf8(FILE *fp, IMultiLanguage2 *pImul, WCHAR *buf, UINT *buf_len) {
	if (fp == NULL || pImul == NULL || buf == NULL)
		return AUO_CHAP_ERR_NULL_PTR;
	DWORD encMode = 0;
	std::vector<char> dst_buf;
	dst_buf.resize(wcslen(buf) * 4, '\0');
	UINT i_dst_buf = (UINT)dst_buf.size();

	UINT buf_len_in_byte = (buf_len) ? *buf_len * sizeof(WCHAR) : -1;
	DWORD last_len = buf_len_in_byte;
	if (S_OK != pImul->ConvertString(&encMode, CODE_PAGE_UTF16_LE, CODE_PAGE_UTF8, (BYTE *)buf, &buf_len_in_byte, (BYTE *)&dst_buf[0], &i_dst_buf))
		return AUO_CHAP_ERR_CONVERTION;
	if (buf_len) {
		//変換されなかったものを先頭に移動(そんなことないと思うけど念のため)
		memmove(buf, buf + buf_len_in_byte, last_len - buf_len_in_byte + sizeof(WCHAR));
		buf_len_in_byte = last_len - buf_len_in_byte;
		*buf_len = buf_len_in_byte / sizeof(WCHAR);
	}
	return (fwrite(&dst_buf[0], 1, i_dst_buf, fp) == i_dst_buf) ? AUO_CHAP_ERR_NONE : AUO_CHAP_ERR_FILE_WRITE;
}

static int write_chapter_apple_header(FILE *fp, IMultiLanguage2 *pImul) {
	if (fp == NULL || pImul == NULL)
		return AUO_CHAP_ERR_NULL_PTR;
	fwrite(UTF8_BOM, 1, sizeof(UTF8_BOM), fp);
	return write_utf8(fp, pImul, 
	L"<?xml version=\"1.0\" encoding=\"UTF-8\" ?>\r\n" 
	L"<TextStream version=\"1.1\">\r\n" 
	L"<TextStreamHeader>\r\n"
	L"<TextSampleDescription>\r\n"
	L"</TextSampleDescription>\r\n" 
	L"</TextStreamHeader>\r\n",
	 NULL);
}

//終端 長さも指定する(しないとわけのわからんdurationになる)
static int write_chapter_apple_foot(FILE *fp, IMultiLanguage2 *pImul, double duration) {
	if (fp == NULL)
		return AUO_CHAP_ERR_NULL_PTR;
	WCHAR chap_foot[256] = { 0 };
	DWORD duration_ms = (DWORD)(duration * 1000.0 + 0.5);
	swprintf_s(chap_foot, sizeof(chap_foot) / sizeof(WCHAR), 
		L"<TextSample sampleTime=\"%02d:%02d:%02d.%03d\" text=\"\" />\r\n</TextStream>",
		duration_ms / (60*60*1000),
		(duration_ms % (60*60*1000)) / (60*1000),
		(duration_ms % (60*1000)) / 1000,
		duration_ms % 1000
		);
	return write_utf8(fp, pImul, chap_foot, NULL);
}

int write_apple_chap(const char *filename, IMultiLanguage2 *pImul, chapter_list_t *chap_list, double duration) {
	int sts = AUO_CHAP_ERR_NONE;
	FILE *fp = NULL;
	if (fopen_s(&fp, filename, "wb") || fp == NULL) {
		sts = AUO_CHAP_ERR_FILE_OPEN;
	} else {
		std::vector<WCHAR> wchar_buffer;
		write_chapter_apple_header(fp, pImul);
		for (int i = 0; i < chap_list->count; i++) {
			const DWORD chapter_name_length = wcslen(chap_list->data[i].name);
			if (wchar_buffer.size() < chapter_name_length + 128)
				wchar_buffer.resize(chapter_name_length + 128);

			swprintf_s(&wchar_buffer[0], wchar_buffer.size(), L"<TextSample sampleTime=\"%02d:%02d:%02d.%03d\">%s</TextSample>\r\n",
				chap_list->data[i].h, chap_list->data[i].m, chap_list->data[i].s, chap_list->data[i].ms, chap_list->data[i].name);
			write_utf8(fp, pImul, &wchar_buffer[0], NULL);
		}

		if (duration <= 0.0 && 0 < chap_list->count) {
			chapter_t *last_chap = &chap_list->data[chap_list->count-1];
			duration = last_chap->h * 3600.0 + last_chap->m * 60.0 + last_chap->s * 1.0 + last_chap->ms * 0.001;
			duration += 0.001;
		}

		write_chapter_apple_foot(fp, pImul, duration);
		fclose(fp);
	}
	return sts;
}

int write_nero_chap(const char *filename, IMultiLanguage2 *pImul, chapter_list_t *chap_list, bool utf8) {
	int sts = AUO_CHAP_ERR_NONE;
	FILE *fp = NULL;
	if (fopen_s(&fp, filename, "wb") || fp == NULL) {
		sts = AUO_CHAP_ERR_FILE_OPEN;
	} else {
		if (utf8)
			fwrite(UTF8_BOM, 1, sizeof(UTF8_BOM), fp);

		const DWORD output_codepage = (utf8) ? CODE_PAGE_UTF8 : CODE_PAGE_SJIS;
		std::vector<char> char_buffer;
		for (int i = 0; i < chap_list->count; i++) {
			static const char * const KEY_BASE = "CHAPTER";
			static const char * const KEY_NAME = "NAME";
			const DWORD chapter_name_length = wcslen(chap_list->data[i].name) + 1;
			if (char_buffer.size() < chapter_name_length * 4)
				char_buffer.resize(chapter_name_length * 4);
			memset(&char_buffer[0], 0, char_buffer.size() * sizeof(char_buffer[0]));

			DWORD encMode = 0;
			UINT buf_len_in_byte = char_buffer.size();
			if (S_OK != pImul->ConvertString(&encMode, CODE_PAGE_UTF16_LE, output_codepage, (BYTE *)chap_list->data[i].name, NULL, (BYTE *)&char_buffer[0], &buf_len_in_byte))
				return AUO_CHAP_ERR_CONVERTION;

			fprintf(fp, "%s%02d=%02d:%02d:%02d:%03d\r\n", KEY_BASE, i+1, chap_list->data[i].h, chap_list->data[i].m, chap_list->data[i].s, chap_list->data[i].ms);
			fprintf(fp, "%s%02d%s=%s\r\n", KEY_BASE, i+1, KEY_NAME, &char_buffer[0]);
		}
		fclose(fp);
	}
	return sts;
}

void free_chapter_list(chapter_list_t *chap_list) {
	if (chap_list) {
		if (chap_list->data) {
			for (int i = 0; i < chap_list->count; i++)
				if (chap_list->data[i].name)
					free(chap_list->data[i].name);
			free(chap_list->data);
		}
		chap_list->data = NULL;
	}
}

static DWORD check_code_page(const char *src_buf, DWORD src_buf_len, IMultiLanguage2 *pImul, DWORD orig_code_page) {
	DetectEncodingInfo dEnc = { 0 };
	int denc_count = 1;
	if (   CODE_PAGE_UNSET == orig_code_page //指定があればスキップ
		&& CODE_PAGE_UNSET == (dEnc.nCodePage = get_code_page(src_buf, src_buf_len)) //まず主に日本語をチェック
		&& S_OK != pImul->DetectInputCodepage(MLDETECTCP_NONE, 0, (char *)src_buf, (int *)&src_buf_len, &dEnc, &denc_count) //IMultiLanguage2で判定してみる
		&& TRUE != fix_ImulL_WesternEurope(&dEnc.nCodePage))
		return CODE_PAGE_UNSET;

	return dEnc.nCodePage;
}

static int get_unicode_data(std::vector<WCHAR> *wchar_data, char *src_data, UINT src_length, DWORD orig_code_page, IMultiLanguage2 *pImul) {
	int sts = AUO_CHAP_ERR_NONE;
	wchar_data->resize((src_length + 1) * 2, 0);
	if (0 == wchar_data->size())
		return AUO_CHAP_ERR_NULL_PTR;

	if (orig_code_page == CODE_PAGE_UTF16_LE) {
		memcpy(&((*wchar_data)[0]), src_data, src_length);
	} else {
		DWORD encMode = 0;
		UINT dst_output_len = wchar_data->size();
		if (S_OK != pImul->ConvertStringToUnicode(&encMode, orig_code_page, src_data, &src_length, &((*wchar_data)[0]), &dst_output_len)) {
			sts = AUO_CHAP_ERR_CONVERTION;
		}
	}
	return sts;
}

static int get_unicode_data_from_file(std::vector<WCHAR> *wchar_data, const char *filename, DWORD orig_code_page, IMultiLanguage2 *pImul) {
	//ファイルを一気に読み込み
	std::vector<char> apple_data;
	std::ifstream inputFile(filename, std::ios::in | std::ios::binary);
	if (!inputFile.good()) {
		return AUO_CHAP_ERR_FILE_OPEN;
	}
	apple_data.resize((size_t)inputFile.seekg(0, std::ios::end).tellg());
	inputFile.seekg(0, std::ios::beg).read(&apple_data[0], static_cast<std::streamsize>(apple_data.size()));

	if (0 == apple_data.size()) {
		return AUO_CHAP_ERR_FILE_OPEN;
	}
	
	//文字コード判定
	if (CODE_PAGE_UNSET == (orig_code_page = check_code_page(&apple_data[0], apple_data.size(), pImul, orig_code_page)))
		return AUO_CHAP_ERR_CP_DETECT;

	//文字コード変換
	return get_unicode_data(wchar_data, &apple_data[0], apple_data.size(), orig_code_page, pImul);
}

static int read_apple_chap(const char *apple_filename, IMultiLanguage2 *pImul, chapter_list_t *chap_list) {
	int sts = AUO_CHAP_ERR_NONE;
	if (chap_list == NULL || apple_filename == NULL || pImul == NULL)
		return AUO_CHAP_ERR_NULL_PTR;

	static const WCHAR * const ELEMENT_NAME = L"TextSample";
	static const WCHAR * const ATTRIBUTE_NAME = L"sampleTime";

	IXmlReader *pReader = NULL;
	IStream *pStream = NULL;
	chapter_t chap = { 0 };
	std::vector<chapter_t> list_of_chapter;

	CoInitialize(NULL);

	if (S_OK != CreateXmlReader(IID_PPV_ARGS(&pReader), NULL))
		sts = AUO_CHAP_ERR_INIT_XML_PARSER;
	else if (S_OK != SHCreateStreamOnFile(apple_filename, STGM_READ, &pStream))
		sts = AUO_CHAP_ERR_INIT_READ_STREAM;
	else if (S_OK != pReader->SetInput(pStream))
		sts = AUO_CHAP_ERR_FAIL_SET_STREAM;
	else {
		const WCHAR *pwLocalName = NULL, *pwValue = NULL;
		XmlNodeType nodeType;
		BOOL flag_next_line_is_time = TRUE; //次は時間を取得するべき

		while (S_OK == pReader->Read(&nodeType)) {
			switch (nodeType) {
				case XmlNodeType_Element:
					if (S_OK != pReader->GetLocalName(&pwLocalName, NULL))
						return AUO_CHAP_ERR_PARSE_XML;
					if (wcscmp(ELEMENT_NAME, pwLocalName))
						break;
					if (S_OK != pReader->MoveToFirstAttribute())
						break;
					do {
						const WCHAR *pwAttributeName = NULL;
						const WCHAR *pwAttributeValue = NULL;
						if (S_OK != pReader->GetLocalName(&pwAttributeName, NULL))
							break;
						if (_wcsicmp(ATTRIBUTE_NAME, pwAttributeName))
							break;
						if (S_OK != pReader->GetValue(&pwAttributeValue, NULL))
							break;
						//必要ならバッファ拡張(想定される最大限必要なバッファに設定)
						if (   4 != swscanf_s(pwAttributeValue, L"%d:%d:%d:%d\r\n", &chap.h, &chap.m, &chap.s, &chap.ms)
							&& 4 != swscanf_s(pwAttributeValue, L"%d:%d:%d.%d\r\n", &chap.h, &chap.m, &chap.s, &chap.ms)
							&& 4 != swscanf_s(pwAttributeValue, L"%d:%d.%d.%d\r\n", &chap.h, &chap.m, &chap.s, &chap.ms)
							&& 4 != swscanf_s(pwAttributeValue, L"%d.%d.%d.%d\r\n", &chap.h, &chap.m, &chap.s, &chap.ms))
							return AUO_CHAP_ERR_PARSE_XML;
						flag_next_line_is_time = FALSE;
					} while (S_OK == pReader->MoveToNextAttribute());
					break;
				case XmlNodeType_Text:
					if (S_OK != pReader->GetValue(&pwValue, NULL))
						break;
					if (pwLocalName == NULL || wcscmp(pwLocalName, ELEMENT_NAME))
						break;
					if (flag_next_line_is_time)
						break;
					//変換
					{
						const int length_of_name = wcslen(pwValue);
						if (NULL == (chap.name = (WCHAR *)calloc(length_of_name + 1, sizeof(chap.name[0]))))
							return AUO_CHAP_ERR_NULL_PTR;
						memcpy(chap.name, pwValue, sizeof(chap.name[0]) * length_of_name);
					}
					flag_next_line_is_time = TRUE;
					list_of_chapter.push_back(chap);
					ZeroMemory(&chap, sizeof(chap));
					break;
				default:
					break;
			}
		}
	}
	
	//リソース解放
	if (pReader)
		pReader->Release();
	if (pStream)
		pStream->Release();
	CoUninitialize();
	

	//配列に格納
	if (NULL == (chap_list->data = (chapter_t *)malloc(list_of_chapter.size() * sizeof(chap_list->data[0])))) {
		sts = AUO_CHAP_ERR_NULL_PTR;
	} else {
		chap_list->count = list_of_chapter.size();
		memcpy(chap_list->data, &list_of_chapter[0], sizeof(chap_list->data[0]) * list_of_chapter.size());
	}

	return sts;
}
static int read_nero_chap(const char *nero_filename, IMultiLanguage2 *pImul, DWORD orig_code_page, chapter_list_t *chap_list) {
	int sts = AUO_CHAP_ERR_NONE;
	if (chap_list == NULL || nero_filename == NULL || pImul == NULL)
		return AUO_CHAP_ERR_NULL_PTR;
	
	//文字コード変換してファイル内容を取得
	std::vector<WCHAR> wchar_data;
	if (AUO_CHAP_ERR_NONE != (sts = get_unicode_data_from_file(&wchar_data, nero_filename, orig_code_page, pImul)))
		return sts;

	//行単位に分解
	WCHAR *qw, *wchar_buf = &wchar_data[0];
	std::vector<WCHAR*> pw_line; //各行へのポインタ
	const WCHAR * const delim = (wcschr(wchar_buf, L'\n')) ? L"\n" : L"\r"; //適切な改行コードを見つける
	for (WCHAR *pw = wchar_buf + (check_bom(wchar_buf) != CODE_PAGE_UNSET); (pw = wcstok_s(pw, delim, &qw)) != NULL; ) {
		pw_line.push_back(pw);
		pw = NULL;
	}
	
	//読み取り
	std::vector<chapter_t> list_of_chapter;
	static const WCHAR * const CHAP_KEY = L"CHAPTER";
	const WCHAR *pw_key[] = { CHAP_KEY, NULL, NULL }; // 時間行, 名前行, ダミー
	WCHAR *pw_data[2];
	const int total_lines = pw_line.size();
	for (int i = 0; i < total_lines && !sts; i++) {
		deleteCRLFSpace_at_End(pw_line[i]);
		if (wcsncmp(pw_line[i], pw_key[i&1], wcslen(pw_key[i&1])) != NULL)
			return AUO_CHAP_ERR_INVALID_FMT;
		pw_key[(i&1) + 1] = pw_line[i];//CHAPTER KEY名を保存
		pw_data[i&1] = wcschr(pw_line[i], L'='); 
		*pw_data[i&1] = L'\0'; //CHAPTER KEY名を一つの文字列として扱えるように
		pw_data[i&1]++; //データは'='の次から
		if (i&1) {
			//読み取り
			chapter_t chap = { 0 };
			if (   4 != swscanf_s(pw_data[0], L"%d:%d:%d:%d", &chap.h, &chap.m, &chap.s, &chap.ms)
				&& 4 != swscanf_s(pw_data[0], L"%d:%d:%d.%d", &chap.h, &chap.m, &chap.s, &chap.ms)
				&& 4 != swscanf_s(pw_data[0], L"%d:%d.%d.%d", &chap.h, &chap.m, &chap.s, &chap.ms)
				&& 4 != swscanf_s(pw_data[0], L"%d.%d.%d.%d", &chap.h, &chap.m, &chap.s, &chap.ms)) {
				sts = AUO_CHAP_ERR_INVALID_FMT;
			} else {
				DWORD chap_name_length = wcslen(pw_data[1]);
				if (NULL == (chap.name = (WCHAR *)calloc(chap_name_length + 1, sizeof(WCHAR)))) {
					sts = AUO_CHAP_ERR_NULL_PTR;
				} else {
					memcpy(chap.name, pw_data[1], sizeof(WCHAR) * chap_name_length);
					list_of_chapter.push_back(chap);
				}
			}
		}
	}

	//配列に格納
	if (NULL == (chap_list->data = (chapter_t *)malloc(list_of_chapter.size() * sizeof(chap_list->data[0])))) {
		sts = AUO_CHAP_ERR_NULL_PTR;
	} else {
		chap_list->count = list_of_chapter.size();
		memcpy(chap_list->data, &list_of_chapter[0], sizeof(chap_list->data[0]) * list_of_chapter.size());
	}
	return sts;
}

//チャプターの種類を大雑把に判別
static int check_chap_type(const WCHAR *data) {
	if (data == NULL || wcslen(data) == 0)
		return CHAP_TYPE_UNKNOWN;
	const WCHAR *rw, *qw, *pw = wcsstr(data, L"CHAPTER");
	if (pw != NULL) {
		qw = wcschr(pw, L'=');
		rw = wcschr(qw, L'\n');
		if (rw == NULL)
			rw = wcschr(qw, L'\r');
		if (qw && rw && wcsncmp(rw+1, pw, qw - pw) == NULL)
			return CHAP_TYPE_NERO;
	}
	if (wcsstr(data, L"<TextStream") && wcsstr(data, L"<TextSample"))
		return CHAP_TYPE_APPLE;

	return CHAP_TYPE_UNKNOWN;
}

//チャプターの種類を大雑把に判別
static int check_chap_type_from_file(const char *filename, IMultiLanguage2 *pImul, DWORD orig_code_page) {
	int sts = AUO_CHAP_ERR_NONE;
	if (filename == NULL || pImul == NULL)
		return CHAP_TYPE_UNKNOWN;
	//文字コード変換してファイル内容を取得
	std::vector<WCHAR> wchar_data;
	if (AUO_CHAP_ERR_NONE != (sts = get_unicode_data_from_file(&wchar_data, filename, orig_code_page, pImul)))
		return CHAP_TYPE_UNKNOWN;

	return check_chap_type(&wchar_data[0]);
}

static int get_chapter_list(chapter_list_t *chap_list, const char *filename, IMultiLanguage2 *pImul, DWORD orig_code_page) {
	int chapter_type = CHAP_TYPE_UNKNOWN;
	if (CHAP_TYPE_UNKNOWN == (chapter_type = check_chap_type_from_file(filename, pImul, orig_code_page)))
		return AUO_CHAP_ERR_INVALID_FMT;

	return (chapter_type == CHAP_TYPE_NERO) ? read_nero_chap(filename, pImul, orig_code_page, chap_list)
		                                    : read_apple_chap(filename, pImul, chap_list);
}

int get_chapter_list(chapter_list_t *chap_list, const char *filename, DWORD orig_code_page) {
	IMultiLanguage2 *pImul = NULL;
	//COM用初期化
	CoInitialize(NULL);

	int sts = AUO_CHAP_ERR_NONE;
	if (S_OK != CoCreateInstance(CLSID_CMultiLanguage, NULL, CLSCTX_INPROC_SERVER, IID_IMultiLanguage2, (void**)&pImul) || pImul == NULL) {
		sts = AUO_CHAP_ERR_INIT_IMUL2;
	} else {
		sts = get_chapter_list(chap_list, filename, pImul, orig_code_page); 
	}
	if (pImul)
		pImul->Release();
	CoUninitialize();
	return sts;
}

static double get_fake_duration(chapter_list_t *chap_list) {
	if (chap_list == NULL || chap_list->count == 0)
		return 0.0;

	chapter_t *last_chap = &chap_list->data[chap_list->count - 1];
	return (double)(last_chap->h * 3600 + last_chap->m * 60 + last_chap->s) + (last_chap->ms + 1.1) * 0.001;
}

int convert_chapter(const char *new_filename, const char *orig_filename, DWORD orig_code_page, double duration, int out_chap_type, bool nero_out_utf8) {
	if (new_filename == NULL || orig_filename == NULL || new_filename == orig_filename)
		return AUO_CHAP_ERR_NULL_PTR;

	int sts = AUO_CHAP_ERR_NONE;
	IMultiLanguage2 *pImul = NULL;
	int chap_type = CHAP_TYPE_UNKNOWN;

	//COM用初期化
	CoInitialize(NULL);

	if (S_OK != CoCreateInstance(CLSID_CMultiLanguage, NULL, CLSCTX_INPROC_SERVER, IID_IMultiLanguage2, (void**)&pImul) || pImul == NULL) {
		sts = AUO_CHAP_ERR_INIT_IMUL2;
	} else {
		chapter_list_t chap_list = { 0 };
		if (CHAP_TYPE_UNKNOWN == (chap_type = check_chap_type_from_file(orig_filename, pImul, orig_code_page)))
			sts = AUO_CHAP_ERR_INVALID_FMT;

		if (AUO_CHAP_ERR_NONE == sts)
			sts = (chap_type == CHAP_TYPE_NERO) ? read_nero_chap(orig_filename, pImul, orig_code_page, &chap_list)
										        : read_apple_chap(orig_filename, pImul, &chap_list);

		if (AUO_CHAP_ERR_NONE == sts && duration <= 0.0)
			duration = get_fake_duration(&chap_list);

		if (CHAP_TYPE_ANOTHER == out_chap_type)
			out_chap_type = (CHAP_TYPE_APPLE == chap_type) ? CHAP_TYPE_NERO : CHAP_TYPE_APPLE;

		if (AUO_CHAP_ERR_NONE == sts)
			sts = (CHAP_TYPE_NERO == out_chap_type) ? write_nero_chap(new_filename, pImul, &chap_list, nero_out_utf8)
										            : write_apple_chap(new_filename, pImul, &chap_list, duration);

		free_chapter_list(&chap_list);
	}

	//開放処理
	if (pImul)
		pImul->Release();

	//Apple -> Nero変換をしたのなら、ファイル名を入れ替える
	if (chap_type == CHAP_TYPE_APPLE)
		if (!swap_file(orig_filename, new_filename))
			sts = AUO_CHAP_ERR_FILE_SWAP;

	CoUninitialize();

	return sts;
}

int create_chapter_file_delayed_by_add_vframe(const char *new_filename, const char *orig_filename, int delay_ms) {
	chapter_list_t chap_list = { 0 };
	int chapter_type = CHAP_TYPE_UNKNOWN;
	IMultiLanguage2 *pImul = NULL;
	//COM用初期化
	CoInitialize(NULL);

	int sts = AUO_CHAP_ERR_NONE;
	if (S_OK != CoCreateInstance(CLSID_CMultiLanguage, NULL, CLSCTX_INPROC_SERVER, IID_IMultiLanguage2, (void**)&pImul) || pImul == NULL) {
		sts = AUO_CHAP_ERR_INIT_IMUL2;
	} else if (CHAP_TYPE_UNKNOWN == (chapter_type = check_chap_type_from_file(orig_filename, pImul, CODE_PAGE_UNSET))) {
		sts = AUO_CHAP_ERR_INVALID_FMT;
	} else if (AUO_CHAP_ERR_NONE == (sts = (chapter_type == CHAP_TYPE_NERO) ? read_nero_chap(orig_filename, pImul, CODE_PAGE_UNSET, &chap_list)
										                                    : read_apple_chap(orig_filename, pImul, &chap_list))) {

		for (int i = 0; i < chap_list.count; i++) {
			chapter_t *data = &chap_list.data[i];
			INT64 chap_time_ms = (((INT64)(data->h * 60 + data->m) * 60) + data->s) * 1000 + data->ms;
			if (0 < chap_time_ms)
				chap_time_ms += delay_ms;
			data->h = (int)(chap_time_ms / (3600 * 1000));
			chap_time_ms -= data->h * (3600 * 1000);
			data->m = (int)(chap_time_ms / (60 * 1000));
			chap_time_ms -= data->m * (60 * 1000);
			data->s = (int)(chap_time_ms / 1000);
			chap_time_ms -= data->s * 1000;
			data->ms = (int)chap_time_ms;
		}

		sts = (chapter_type == CHAP_TYPE_NERO) ? write_nero_chap(new_filename, pImul, &chap_list, false)
										       : write_apple_chap(new_filename, pImul, &chap_list, 0.0);

		free_chapter_list(&chap_list);
	}
	//開放処理
	if (pImul)
		pImul->Release();

	CoUninitialize();

	return sts;
}
