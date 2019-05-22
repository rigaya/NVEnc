//  -----------------------------------------------------------------------------------------
//    拡張 x264 出力(GUI) Ex  v1.xx/2.xx by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  -----------------------------------------------------------------------------------------

#ifndef _AUO_CHAPTER_H_
#define _AUO_CHAPTER_H_

#include <cstdint>
#include <string>
#include <memory>
#include <vector>

#ifndef TCHAR
#ifdef UNICODE
typedef wchar_t TCHAR;
#else
typedef char TCHAR;
#endif
#endif

namespace tinyxml2 {
    class XMLElement;
}

//日本語環境の一般的なコードページ一覧
enum : uint32_t {
    CODE_PAGE_SJIS        = 932, //Shift-JIS
    CODE_PAGE_JIS         = 50220,
    CODE_PAGE_EUC_JP      = 51932,
    CODE_PAGE_UTF8        = 65001,
    CODE_PAGE_UTF16_LE    = 1200, //WindowsのUnicode WCHAR のコードページ
    CODE_PAGE_UTF16_BE    = 1201,
    CODE_PAGE_US_ASCII    = 20127,
    CODE_PAGE_WEST_EUROPE = 1252,  //厄介な西ヨーロッパ言語
    CODE_PAGE_UNSET       = 0xffffffff,
};

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
    std::wstring name;
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
    std::wstring m_wchar_filedata;                //チャプターファイルの文字列をwcharに変換したもの
    const TCHAR *m_filepath = nullptr;             //読み込むチャプターファイル
    ChapType m_chapter_type = CHAP_TYPE_UNKNOWN;  //読み込んだチャプターの種類
    uint32_t m_code_page = 0;                     //読み込んだチャプターの文字コード
    double m_duration = 0.0;                      //動画の長さ情報 (秒)
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
    std::vector<std::unique_ptr<ChapData>> chapters; //読み込んだチャプターのリスト

    void init(); //初期化
    void close(); //終了、リソース開放
    int read_file(); //ファイルの読み込み

    //内部のm_code_pageに従って文字データをwcharに変換
    int get_unicode_data(std::wstring& wchar_data, std::vector<char>& src);

    //文字データの文字コードをチェック、判定した文字コードを返す
    uint32_t check_code_page(std::vector<char>& src, uint32_t orig_code_page);

    //ファイルからデータを取得し、
    // 1. check_code_pageを使って、内部のm_code_pageに判定結果をセット
    // 2. get_unicode_dataで文字列をwcharに変換し、m_wchar_filedataにセット
    int get_unicode_data_from_file(std::wstring& wchar_data);

    //読み込んだチャプターの種類を判定
    ChapType check_chap_type_from_file();

    //文字列データからチャプターの種類を判定
    ChapType check_chap_type(const std::wstring& data);

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


#endif //_AUO_CHAPTER_H_