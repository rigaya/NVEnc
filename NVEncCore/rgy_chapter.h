// -----------------------------------------------------------------------------------------
// QSVEnc/NVEnc by rigaya
// -----------------------------------------------------------------------------------------
// The MIT License
//
// Copyright (c) 2019 rigaya
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

#ifndef __RGY_CHAPTER_H__
#define __RGY_CHAPTER_H__

#include <cstdint>
#include <string>
#include <memory>
#include <vector>
#include "rgy_tchar.h"

namespace tinyxml2 {
    class XMLElement;
}

enum ChapType {
    CHAP_TYPE_ANOTHER = -1,
    CHAP_TYPE_UNKNOWN = 0,
    CHAP_TYPE_NERO,
    CHAP_TYPE_APPLE,
    CHAP_TYPE_MATROSKA,
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

class ChapData {
public:
    std::string name;
    int h = 0;
    int m = 0;
    int s = 0;
    int ms = 0;

    //チャプターの時間をミリ秒に
    int get_ms() {
        return (h * 3600 + m * 60 + s) * 1000 + ms;
    }
    //チャプターの時間を変更
    void set_ms(int chap_time_ms) {
        h = (int)(chap_time_ms / (3600 * 1000));
        chap_time_ms -= h * (3600 * 1000);
        m = (int)(chap_time_ms / (60 * 1000));
        chap_time_ms -= m * (60 * 1000);
        s = (int)(chap_time_ms / 1000);
        chap_time_ms -= s * 1000;
        ms = (int)chap_time_ms;
    }
};

class ChapterRW {
private:
    std::string m_filedata;   //チャプターファイルの文字列をwcharに変換したもの
    const TCHAR *m_filepath;  //読み込むチャプターファイル
    ChapType m_chapter_type;  //読み込んだチャプターの種類
    uint32_t m_code_page;     //読み込んだチャプターの文字コード
    double m_duration;        //動画の長さ情報 (秒)
    std::vector<std::unique_ptr<ChapData>> chapters; //読み込んだチャプターのリスト
public:
    ChapterRW();
    ~ChapterRW();

    //読み込んだチャプターの種類(m_chapter_type)を返す
    int file_chapter_type();

    //読み込んだチャプターの文字コード(m_code_page)を返す
    uint32_t file_code_page();

    //チャプターファイルの読み込み
    //filepath ...読み込むファイル
    //m_code_page...ファイルの文字コードを指定
    //duration ...動画の長さ情報(秒)を渡す
    int read_file(const TCHAR *filepath, uint32_t code_page, double duration);

    //チャプターファイルの書き出し
    //out_filepath...出力先
    //out_chapter_type...出力するチャプターの種類
    //nero_in_utf8...出力するチャプターがneroの場合に、utf-8で出力する
    int write_file(const TCHAR *out_filepath, ChapType out_chapter_type, bool nero_in_utf8);

    //チャプターファイルを上書きする
    //失敗した場合にはなにもしない
    //out_chapter_type...出力するチャプターの種類
    //nero_in_utf8...出力するチャプターがneroの場合に、utf-8で出力する
    int overwrite_file(ChapType out_chapter_type, bool nero_in_utf8);

    //チャプターリストの先頭に、0秒の位置にダミーチャプターを追加する
    void add_dummy_chap_zero_pos();

    //時間0秒の位置にあるチャプター以外に遅延を追加する
    void delay_chapter(int delay_ms);

    const std::vector<std::unique_ptr<ChapData>>& chapterlist() {
        return chapters;
    }

private:

    void init(); //初期化
    void close(); //終了、リソース開放
    int read_file(); //ファイルの読み込み

    //内部のm_code_pageに従って文字データをwcharに変換
    int get_unicode_data(std::string& data, std::vector<char>& src);

    //文字データの文字コードをチェック、判定した文字コードを返す
    uint32_t check_code_page(std::vector<char>& src, uint32_t orig_code_page);

    //ファイルからデータを取得し、
    // 1. check_code_pageを使って、内部のm_code_pageに判定結果をセット
    // 2. get_unicode_dataで文字列をUTF-8変換し、m_filedataにセット
    int get_unicode_data_from_file(std::string& data);

    //読み込んだチャプターの種類を判定
    ChapType check_chap_type_from_file();

    //文字列データからチャプターの種類を判定
    ChapType check_chap_type(const std::string& data);

    //ファイルを読み込み、チャプターリスト(chapters)を作成
    //m_chapter_typeに従って処理を振り分け
    int read_chapter();

    //neroチャプターから、チャプターリスト(chapters)を作成
    int read_chapter_nero();

    //appleチャプターから、チャプターリスト(chapters)を作成
    int read_chapter_apple();

    int read_chapter_matroska_chapter_atom(tinyxml2::XMLElement *elem, int &count);

    //matroskaチャプターから、チャプターリスト(chapters)を作成
    int read_chapter_matroska();

    //appleチャプターのヘッダー部分を作成
    int write_chapter_apple_header(std::ostream& ostream);

    //appleチャプターのフッター部分を作成
    int write_chapter_apple_foot(std::ostream& ostream);

    //チャプターリスト(chapters)からappleチャプターを作成
    int write_chapter_apple(const TCHAR *filepath);

    //チャプターリスト(chapters)からneroチャプターを作成
    int write_chapter_nero(const TCHAR *filepath, bool utf8);
};

#endif //__RGY_CHAPTER_H__
