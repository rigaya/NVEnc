//  -----------------------------------------------------------------------------------------
//    拡張 x264/x265 出力(GUI) Ex  v1.xx/2.xx/3.xx by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  -----------------------------------------------------------------------------------------

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>
#include <malloc.h>
#include <stdlib.h>
#include <string.h>
#include "auo.h"
#include "auo_util.h"
#include "auo_video.h"
#include "auo_frm.h"
#include "convert.h"

//音声の16bit->8bit変換の選択
func_audio_16to8 get_audio_16to8_func(BOOL split) {
    static const func_audio_16to8 FUNC_CONVERT_AUDIO[][2] = {
        { convert_audio_16to8,      split_audio_16to8x2      },
        { convert_audio_16to8_sse2, split_audio_16to8x2_sse2 },
        { convert_audio_16to8_avx2, split_audio_16to8x2_avx2 },
    };
    int simd = 0;
    if (0 == (simd = (!!check_avx2() * 2)))
        simd = check_sse2();
    return FUNC_CONVERT_AUDIO[simd][!!split];
}

#if 0

enum eInterlace {
    A = -1, //区別の必要なし
    P = 0,  //プログレッシブ用
    I = 1   //インターレース用
};

typedef struct {
    int        input_from_aviutl; //Aviutlからの入力に使用する
    int        output_csp;        //出力色空間
    int        bit_depth;         //bit深度
    eInterlace for_interlaced;    //インタレース用関数であるかどうか
    DWORD      mod;               //幅(横解像)に制限(割り切れるかどうか)
    DWORD      SIMD;              //対応するSIMD
    func_convert_frame func;      //関数へのポインタ
} COVERT_FUNC_INFO;

//なんの数字かわかりやすいようにこう定義する
static const int BIT_8 =  8;
static const int BIT10 = 10;
static const int BIT12 = 12;
static const int BIT16 = 16;

#define ENABLE_NV12 1

//変換関数のテーブル
//上からチェックするので、より厳しい条件で速い関数を上に書くこと
static const COVERT_FUNC_INFO FUNC_TABLE[] = {
    //YUY2をそのまま渡す
    { CF_YUY2, OUT_CSP_YUY2,   BIT_8, A,  1,  SSE2,                  copy_yuy2_sse2 },
    { CF_YUY2, OUT_CSP_YUY2,   BIT_8, A,  1,  NONE,                  copy_yuy2 },
#if ENABLE_NV12
    //YUY2 -> nv12(8bit)
    { CF_YUY2, OUT_CSP_NV12,   BIT_8, P,  1,  AVX2|AVX,             convert_yuy2_to_nv12_avx2 },
    { CF_YUY2, OUT_CSP_NV12,   BIT_8, I,  1,  AVX2|AVX,             convert_yuy2_to_nv12_i_avx2 },
    { CF_YUY2, OUT_CSP_NV12,   BIT_8, P,  1,  AVX|SSE2,             convert_yuy2_to_nv12_avx },
    { CF_YUY2, OUT_CSP_NV12,   BIT_8, I,  1,  AVX|SSE2,             convert_yuy2_to_nv12_i_avx },
    { CF_YUY2, OUT_CSP_NV12,   BIT_8, P, 16,  SSE2,                 convert_yuy2_to_nv12_sse2_mod16 },
    { CF_YUY2, OUT_CSP_NV12,   BIT_8, P,  1,  SSE2,                 convert_yuy2_to_nv12_sse2 },
    { CF_YUY2, OUT_CSP_NV12,   BIT_8, P,  1,  NONE,                 convert_yuy2_to_nv12 },
    { CF_YUY2, OUT_CSP_NV12,   BIT_8, I, 16,  SSSE3|SSE2,           convert_yuy2_to_nv12_i_ssse3_mod16 },
    { CF_YUY2, OUT_CSP_NV12,   BIT_8, I,  1,  SSSE3|SSE2,           convert_yuy2_to_nv12_i_ssse3 },
    { CF_YUY2, OUT_CSP_NV12,   BIT_8, I, 16,  SSE2,                 convert_yuy2_to_nv12_i_sse2_mod16 },
    { CF_YUY2, OUT_CSP_NV12,   BIT_8, I,  1,  SSE2,                 convert_yuy2_to_nv12_i_sse2 },
    { CF_YUY2, OUT_CSP_NV12,   BIT_8, I,  1,  NONE,                 convert_yuy2_to_nv12_i },
#else
    //YUY2 -> yv12 (8bit)
    { CF_YUY2, OUT_CSP_YV12,   BIT_8, P,  1,  AVX2|AVX,             convert_yuy2_to_yv12_avx2 },
    { CF_YUY2, OUT_CSP_YV12,   BIT_8, I,  1,  AVX2|AVX,             convert_yuy2_to_yv12_i_avx2 },
    { CF_YUY2, OUT_CSP_YV12,   BIT_8, P,  1,  AVX|SSE2,             convert_yuy2_to_yv12_avx },
    { CF_YUY2, OUT_CSP_YV12,   BIT_8, I,  1,  AVX|SSE2,             convert_yuy2_to_yv12_i_avx },
    { CF_YUY2, OUT_CSP_YV12,   BIT_8, P, 32,  SSE2,                 convert_yuy2_to_yv12_sse2_mod32 },
    { CF_YUY2, OUT_CSP_YV12,   BIT_8, P,  1,  SSE2,                 convert_yuy2_to_yv12_sse2 },
    { CF_YUY2, OUT_CSP_YV12,   BIT_8, P,  1,  NONE,                 convert_yuy2_to_yv12 },
    { CF_YUY2, OUT_CSP_YV12,   BIT_8, I, 32,  SSSE3|SSE2,           convert_yuy2_to_yv12_i_ssse3_mod32 },
    { CF_YUY2, OUT_CSP_YV12,   BIT_8, I,  1,  SSSE3|SSE2,           convert_yuy2_to_yv12_i_ssse3 },
    { CF_YUY2, OUT_CSP_YV12,   BIT_8, I, 32,  SSE2,                 convert_yuy2_to_yv12_i_sse2_mod32 },
    { CF_YUY2, OUT_CSP_YV12,   BIT_8, I,  1,  SSE2,                 convert_yuy2_to_yv12_i_sse2 },
    { CF_YUY2, OUT_CSP_YV12,   BIT_8, I,  1,  NONE,                 convert_yuy2_to_yv12_i },
#endif
#if ENABLE_NV12
    //YC48 -> nv12 (16bit)
    { CF_YC48, OUT_CSP_NV12,   BIT16, P,  1,  AVX2|AVX,             convert_yc48_to_nv12_16bit_avx2 },
    { CF_YC48, OUT_CSP_NV12,   BIT16, P,  1,  AVX|SSE41|SSSE3|SSE2, convert_yc48_to_nv12_16bit_avx },
    { CF_YC48, OUT_CSP_NV12,   BIT16, P,  8,  SSE41|SSSE3|SSE2,     convert_yc48_to_nv12_16bit_sse41_mod8 },
    { CF_YC48, OUT_CSP_NV12,   BIT16, P,  1,  SSE41|SSSE3|SSE2,     convert_yc48_to_nv12_16bit_sse41 },
    { CF_YC48, OUT_CSP_NV12,   BIT16, P,  8,  SSSE3|SSE2,           convert_yc48_to_nv12_16bit_ssse3_mod8 },
    { CF_YC48, OUT_CSP_NV12,   BIT16, P,  1,  SSSE3|SSE2,           convert_yc48_to_nv12_16bit_ssse3 },
    { CF_YC48, OUT_CSP_NV12,   BIT16, P,  8,  SSE2,                 convert_yc48_to_nv12_16bit_sse2_mod8 },
    { CF_YC48, OUT_CSP_NV12,   BIT16, P,  1,  SSE2,                 convert_yc48_to_nv12_16bit_sse2 },
    { CF_YC48, OUT_CSP_NV12,   BIT16, P,  1,  NONE,                 convert_yc48_to_nv12_16bit },

    { CF_YC48, OUT_CSP_NV12,   BIT16, I,  1,  AVX2|AVX,             convert_yc48_to_nv12_i_16bit_avx2 },
    { CF_YC48, OUT_CSP_NV12,   BIT16, I,  1,  AVX|SSE41|SSSE3|SSE2, convert_yc48_to_nv12_i_16bit_avx },
    { CF_YC48, OUT_CSP_NV12,   BIT16, I,  8,  SSE41|SSSE3|SSE2,     convert_yc48_to_nv12_i_16bit_sse41_mod8 },
    { CF_YC48, OUT_CSP_NV12,   BIT16, I,  1,  SSE41|SSSE3|SSE2,     convert_yc48_to_nv12_i_16bit_sse41 },
    { CF_YC48, OUT_CSP_NV12,   BIT16, I,  8,  SSSE3|SSE2,           convert_yc48_to_nv12_i_16bit_ssse3_mod8 },
    { CF_YC48, OUT_CSP_NV12,   BIT16, I,  1,  SSSE3|SSE2,           convert_yc48_to_nv12_i_16bit_ssse3 },
    { CF_YC48, OUT_CSP_NV12,   BIT16, I,  8,  SSE2,                 convert_yc48_to_nv12_i_16bit_sse2_mod8 },
    { CF_YC48, OUT_CSP_NV12,   BIT16, I,  1,  SSE2,                 convert_yc48_to_nv12_i_16bit_sse2 },
    { CF_YC48, OUT_CSP_NV12,   BIT16, I,  1,  NONE,                 convert_yc48_to_nv12_i_16bit },
#else
    //YC48 -> yv12 (16bit)
    { CF_YC48, OUT_CSP_YV12,   BIT16, P,  1,  AVX2|AVX,             convert_yc48_to_yv12_16bit_avx2 },
    { CF_YC48, OUT_CSP_YV12,   BIT16, P,  1,  AVX|SSE41|SSSE3|SSE2, convert_yc48_to_yv12_16bit_avx },
    { CF_YC48, OUT_CSP_YV12,   BIT16, P,  8,  SSE41|SSSE3|SSE2,     convert_yc48_to_yv12_16bit_sse41_mod8 },
    { CF_YC48, OUT_CSP_YV12,   BIT16, P,  1,  SSE41|SSSE3|SSE2,     convert_yc48_to_yv12_16bit_sse41 },
    { CF_YC48, OUT_CSP_YV12,   BIT16, P,  8,  SSSE3|SSE2,           convert_yc48_to_yv12_16bit_ssse3_mod8 },
    { CF_YC48, OUT_CSP_YV12,   BIT16, P,  1,  SSSE3|SSE2,           convert_yc48_to_yv12_16bit_ssse3 },
    { CF_YC48, OUT_CSP_YV12,   BIT16, P,  8,  SSE2,                 convert_yc48_to_yv12_16bit_sse2_mod8 },
    { CF_YC48, OUT_CSP_YV12,   BIT16, P,  1,  SSE2,                 convert_yc48_to_yv12_16bit_sse2 },
    { CF_YC48, OUT_CSP_YV12,   BIT16, P,  1,  NONE,                 convert_yc48_to_yv12_16bit },

    { CF_YC48, OUT_CSP_YV12,   BIT16, I,  1,  AVX2|AVX,             convert_yc48_to_yv12_i_16bit_avx2 },
    { CF_YC48, OUT_CSP_YV12,   BIT16, I,  1,  AVX|SSE41|SSSE3|SSE2, convert_yc48_to_yv12_i_16bit_avx },
    { CF_YC48, OUT_CSP_YV12,   BIT16, I,  8,  SSE41|SSSE3|SSE2,     convert_yc48_to_yv12_i_16bit_sse41_mod8 },
    { CF_YC48, OUT_CSP_YV12,   BIT16, I,  1,  SSE41|SSSE3|SSE2,     convert_yc48_to_yv12_i_16bit_sse41 },
    { CF_YC48, OUT_CSP_YV12,   BIT16, I,  8,  SSSE3|SSE2,           convert_yc48_to_yv12_i_16bit_ssse3_mod8 },
    { CF_YC48, OUT_CSP_YV12,   BIT16, I,  1,  SSSE3|SSE2,           convert_yc48_to_yv12_i_16bit_ssse3 },
    { CF_YC48, OUT_CSP_YV12,   BIT16, I,  8,  SSE2,                 convert_yc48_to_yv12_i_16bit_sse2_mod8 },
    { CF_YC48, OUT_CSP_YV12,   BIT16, I,  1,  SSE2,                 convert_yc48_to_yv12_i_16bit_sse2 },
    { CF_YC48, OUT_CSP_YV12,   BIT16, I,  1,  NONE,                 convert_yc48_to_yv12_i_16bit },

    //YC48 -> yv12 (10bit)
    { CF_YC48, OUT_CSP_YV12,   BIT10, P,  1,  NONE,                 convert_yc48_to_yv12_10bit },
    { CF_YC48, OUT_CSP_YV12,   BIT10, I,  1,  NONE,                 convert_yc48_to_yv12_i_10bit },
#endif
#if ENABLE_NV12
    //YUY2 -> nv16(8bit)
    { CF_YUY2, OUT_CSP_NV16,   BIT_8, A,  1,  AVX2|AVX,             convert_yuy2_to_nv16_avx2 },
    { CF_YUY2, OUT_CSP_NV16,   BIT_8, A,  1,  AVX|SSE2,             convert_yuy2_to_nv16_avx },
    { CF_YUY2, OUT_CSP_NV16,   BIT_8, A, 16,  SSE2,                 convert_yuy2_to_nv16_sse2_mod16 },
    { CF_YUY2, OUT_CSP_NV16,   BIT_8, A,  1,  SSE2,                 convert_yuy2_to_nv16_sse2 },
    { CF_YUY2, OUT_CSP_NV16,   BIT_8, A,  1,  NONE,                 convert_yuy2_to_nv16 },
    //YC48 -> nv16(16bit)
    { CF_YC48, OUT_CSP_NV16,   BIT16, A,  1,  AVX2|AVX,             convert_yc48_to_nv16_16bit_avx2 },
    { CF_YC48, OUT_CSP_NV16,   BIT16, A,  1,  AVX|SSE41|SSSE3|SSE2, convert_yc48_to_nv16_16bit_avx },
    { CF_YC48, OUT_CSP_NV16,   BIT16, A,  8,  SSE41|SSSE3|SSE2,     convert_yc48_to_nv16_16bit_sse41_mod8 },
    { CF_YC48, OUT_CSP_NV16,   BIT16, A,  1,  SSE41|SSSE3|SSE2,     convert_yc48_to_nv16_16bit_sse41 },
    { CF_YC48, OUT_CSP_NV16,   BIT16, A,  8,  SSSE3|SSE2,           convert_yc48_to_nv16_16bit_ssse3_mod8 },
    { CF_YC48, OUT_CSP_NV16,   BIT16, A,  1,  SSSE3|SSE2,           convert_yc48_to_nv16_16bit_ssse3 },
    { CF_YC48, OUT_CSP_NV16,   BIT16, A,  8,  SSE2,                 convert_yc48_to_nv16_16bit_sse2_mod8 },
    { CF_YC48, OUT_CSP_NV16,   BIT16, A,  1,  SSE2,                 convert_yc48_to_nv16_16bit_sse2 },
    { CF_YC48, OUT_CSP_NV16,   BIT16, A,  1,  NONE,                 convert_yc48_to_nv16_16bit },
#else
    //YUY2 -> yuv422(8bit)
    { CF_YUY2, OUT_CSP_YUV422, BIT_8, A,  1,  NONE,                 convert_yuy2_to_yuv422 },
#endif
    //YC48 -> yuv444(8bit)
    { CF_YC48, OUT_CSP_YUV444, BIT_8, A,  1,  AVX2|AVX,             convert_yc48_to_yuv444_avx2 },
    { CF_YC48, OUT_CSP_YUV444, BIT_8, A,  1,  AVX|SSE41|SSSE3|SSE2, convert_yc48_to_yuv444_avx },
    { CF_YC48, OUT_CSP_YUV444, BIT_8, A, 16,  SSE41|SSSE3|SSE2,     convert_yc48_to_yuv444_sse41_mod16 },
    { CF_YC48, OUT_CSP_YUV444, BIT_8, A,  1,  SSE41|SSSE3|SSE2,     convert_yc48_to_yuv444_sse41 },
    { CF_YC48, OUT_CSP_YUV444, BIT_8, A, 16,  SSE2,                 convert_yc48_to_yuv444_sse2_mod16 },
    { CF_YC48, OUT_CSP_YUV444, BIT_8, A,  1,  SSE2,                 convert_yc48_to_yuv444_sse2 },
    { CF_YC48, OUT_CSP_YUV444, BIT_8, A,  1,  NONE,                 convert_yc48_to_yuv444 },

    //YC48 -> yuv444(10bit)
    { CF_YC48, OUT_CSP_YUV444, BIT10, A,  1,  NONE,                 convert_yc48_to_yuv444_10bit },

    //YC48 -> yuv444(16bit)
    { CF_YC48, OUT_CSP_YUV444, BIT16, A,  1,  AVX2|AVX,             convert_yc48_to_yuv444_16bit_avx2 },
    { CF_YC48, OUT_CSP_YUV444, BIT16, A,  1,  AVX|SSE41|SSSE3|SSE2, convert_yc48_to_yuv444_16bit_avx },
    { CF_YC48, OUT_CSP_YUV444, BIT16, A,  8,  SSE41|SSSE3|SSE2,     convert_yc48_to_yuv444_16bit_sse41_mod8 },
    { CF_YC48, OUT_CSP_YUV444, BIT16, A,  1,  SSE41|SSSE3|SSE2,     convert_yc48_to_yuv444_16bit_sse41 },
    { CF_YC48, OUT_CSP_YUV444, BIT16, A,  8,  SSE2,                 convert_yc48_to_yuv444_16bit_sse2_mod8 },
    { CF_YC48, OUT_CSP_YUV444, BIT16, A,  1,  SSE2,                 convert_yc48_to_yuv444_16bit_sse2 },
    { CF_YC48, OUT_CSP_YUV444, BIT16, A,  1,  NONE,                 convert_yc48_to_yuv444_16bit },

    //Copy RGB
    { CF_RGB,  OUT_CSP_RGB,    BIT_8, A,  1,  SSSE3|SSE2,           sort_to_rgb_ssse3 },
    { CF_RGB,  OUT_CSP_RGB,    BIT_8, A,  1,  NONE,                 sort_to_rgb },

    { 0, 0, 0, A, 0, 0, NULL }
};

static void build_simd_info(DWORD simd, char *buf, DWORD nSize) {
    ZeroMemory(buf, nSize);
    if (simd != NONE) {
        strcpy_s(buf, nSize, ", using");
        if (simd & SSE2)  strcat_s(buf, nSize, " SSE2");
        if (simd & SSE3)  strcat_s(buf, nSize, " SSE3");
        if (simd & SSSE3) strcat_s(buf, nSize, " SSSE3");
        if (simd & SSE41) strcat_s(buf, nSize, " SSE4.1");
        if (simd & SSE42) strcat_s(buf, nSize, " SSE4.2");
        if (simd & AVX)   strcat_s(buf, nSize, " AVX");
        if (simd & AVX2)  strcat_s(buf, nSize, " AVX2");
    }
}

static void auo_write_func_info(const COVERT_FUNC_INFO *func_info) {
    char simd_buf[128];
    build_simd_info(func_info->SIMD, simd_buf, _countof(simd_buf));

    if (func_info->output_csp == OUT_CSP_YUY2) {
        write_log_auo_line_fmt(LOG_INFO, "Passing YUY2", simd_buf);
        return;
    }

    if (func_info->output_csp == OUT_CSP_RGB) {
        write_log_auo_line_fmt(LOG_INFO, "Copying RGB%s", simd_buf);
        return;
    }

    const char *interlaced = NULL;
    switch (func_info->for_interlaced) {
        case P: interlaced = "p"; break;
        case I: interlaced = "i"; break;
        case A:
        default:interlaced = ""; break;
    }
    const char *bit_depth = "";
    switch (func_info->bit_depth) {
        case BIT10: bit_depth = "(10bit)"; break;
        case BIT12: bit_depth = "(12bit)"; break;
        case BIT16: bit_depth = "(16bit)"; break;
        default: break;
    }

    write_log_auo_line_fmt(LOG_INFO, "converting %s -> %s%s%s%s",
        CF_NAME[func_info->input_from_aviutl],
        specify_csp[func_info->output_csp],
        interlaced,
        bit_depth,
        simd_buf);
};

//C4189 : ローカル変数が初期化されましたが、参照されていません。
#pragma warning( push )
#pragma warning( disable: 4189 )
//使用する関数を選択する
func_convert_frame get_convert_func(int width, int input_csp, int bit_depth, BOOL interlaced, int output_csp) {
    const DWORD availableSIMD = get_availableSIMD();

    const COVERT_FUNC_INFO *func_info = NULL;
    for (int i = 0; FUNC_TABLE[i].func; i++) {
        if (FUNC_TABLE[i].input_from_aviutl != input_csp)
            continue;
        if (FUNC_TABLE[i].output_csp != output_csp)
            continue;
        if (FUNC_TABLE[i].bit_depth != bit_depth)
            continue;
        if (FUNC_TABLE[i].for_interlaced != A &&
            FUNC_TABLE[i].for_interlaced != (eInterlace)interlaced)
            continue;
        if ((width % FUNC_TABLE[i].mod) != 0)
            continue;
        if ((FUNC_TABLE[i].SIMD & availableSIMD) != FUNC_TABLE[i].SIMD)
            continue;

        func_info = &FUNC_TABLE[i];
        break;
    }

    if (func_info == NULL)
        return NULL;

    auo_write_func_info(func_info);
    return func_info->func;
}
#pragma warning( pop )

BOOL malloc_pixel_data(CONVERT_CF_DATA * const pixel_data, int width, int height, int output_csp, int bit_depth) {
    BOOL ret = TRUE;
    const int to_yv12 = FALSE; // (output_csp == OUT_CSP_YV12);
    const DWORD pixel_size = (bit_depth > 8) ? sizeof(short) : sizeof(BYTE);
    const DWORD simd_check = get_availableSIMD();
    const DWORD align_size = (simd_check & SSE2) ? ((simd_check & AVX2) ? (64<<to_yv12) : (32<<to_yv12)) : 1;
#define ALIGN_NEXT(i, align) (((i) + (align-1)) & (~(align-1))) //alignは2の累乗(1,2,4,8,16,32...)
    const DWORD frame_size = ALIGN_NEXT(width * height * pixel_size + (ALIGN_NEXT(width, align_size / pixel_size) - width) * 2 * pixel_size, align_size);
#undef ALIGN_NEXT

    ZeroMemory(pixel_data->data, sizeof(pixel_data->data));
    switch (output_csp) {
        case OUT_CSP_YUY2: //YUY2であってもコピーフレーム機能をサポートするためにはコピーが必要となる
            if ((pixel_data->data[0] = (BYTE *)_mm_malloc(frame_size * 2, max(align_size, 16))) == NULL)
                ret = FALSE;
            break;
        case OUT_CSP_NV16:
            if ((pixel_data->data[0] = (BYTE *)_mm_malloc(frame_size * 2, max(align_size, 16))) == NULL)
                ret = FALSE;
            pixel_data->data[1] = pixel_data->data[0] + frame_size;
            break;
        case OUT_CSP_YUV444:
            if ((pixel_data->data[0] = (BYTE *)_mm_malloc(frame_size * 3, max(align_size, 16))) == NULL)
                ret = FALSE;
            pixel_data->data[1] = pixel_data->data[0] + frame_size;
            pixel_data->data[2] = pixel_data->data[1] + frame_size;
            break;
        case OUT_CSP_RGB:
            if ((pixel_data->data[0] = (BYTE *)_mm_malloc(frame_size * 3, max(align_size, 16))) == NULL)
                ret = FALSE;
            break;
        //case OUT_CSP_YV12:
        //    if ((pixel_data->data[0] = (BYTE *)_mm_malloc(frame_size * 3 / 2, max(align_size, 16))) == NULL)
        //        ret = FALSE;
        //    pixel_data->data[1] = pixel_data->data[0] + frame_size;
        //    pixel_data->data[2] = pixel_data->data[1] + frame_size / 4;
        //    break;
        //case OUT_CSP_YUV422:
        //    if ((pixel_data->data[0] = (BYTE *)_mm_malloc(frame_size * 2, max(align_size, 16))) == NULL)
        //        ret = FALSE;
        //    pixel_data->data[1] = pixel_data->data[0] + frame_size;
        //    pixel_data->data[2] = pixel_data->data[1] + frame_size / 2;
        //    break;
        case OUT_CSP_NV12:
        default:
            if ((pixel_data->data[0] = (BYTE *)_mm_malloc(frame_size * 3 / 2, max(align_size, 16))) == NULL)
                ret = FALSE;
            pixel_data->data[1] = pixel_data->data[0] + frame_size;
            break;
    }
    return ret;
}

void free_pixel_data(CONVERT_CF_DATA *pixel_data) {
    if (pixel_data->data[0])
        _mm_free(pixel_data->data[0]);
    ZeroMemory(pixel_data, sizeof(CONVERT_CF_DATA));
}

#endif
