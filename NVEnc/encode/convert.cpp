//  -----------------------------------------------------------------------------------------
//    NVEnc by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  -----------------------------------------------------------------------------------------

#include <Windows.h>
#include <stdio.h>
#include <mmintrin.h>  //イントリンシック命令 SSE
#include <emmintrin.h> //イントリンシック命令 SSE2

#include "convert.h"
#include "auo.h"
#include "auo_util.h"

//音声の16bit->8bit変換の選択
func_audio_16to8 get_audio_16to8_func(BOOL split) {
	static const func_audio_16to8 FUNC_CONVERT_AUDIO[][2] = {
		{ convert_audio_16to8,      split_audio_16to8x2      },
		{ convert_audio_16to8_sse2, split_audio_16to8x2_sse2 },
#if (_MSC_VER >= 1700)
		{ convert_audio_16to8_avx2, split_audio_16to8x2_avx2 },
#endif
	};
	int simd = 0;
#if (_MSC_VER >= 1700)
	if (0 == (simd = (!!check_avx2() * 2)))
#endif
		simd = check_sse2();
	return FUNC_CONVERT_AUDIO[simd][!!split];
}

//直前の16byteアライメント
static inline void * get_aligned_next(void *p) {
	return (void *)(((size_t)p + 15) & ~15);
}
//直後の16byteアライメント
static inline void * get_aligned_prev(void *p) {
	return (void *)(((size_t)p) & ~15);
}
//16bit音声 -> 8bit音声
void convert_audio_16to8(BYTE *dst, short *src, int n) {
	BYTE *byte = dst;
	const BYTE *fin = byte + n;
	short *sh = src;
	while (byte < fin) {
		*byte = (*sh >> 8) + 128;
		byte++;
		sh++;
	}
}

void split_audio_16to8x2(BYTE *dst, short *src, int n) {
	BYTE *byte0 = dst;
	BYTE *byte1 = dst + n;
	short *sh = src;
	short *sh_fin = src + n;
	for ( ; sh < sh_fin; sh++, byte0++, byte1++) {
		*byte0 = (*sh >> 8)   + 128;
		*byte1 = (*sh & 0xff) + 128;
	}
}

void split_audio_16to8x2_sse2(BYTE *dst, short *src, int n) {
	BYTE *byte0 = dst;
	BYTE *byte1 = dst + n;
	short *sh = src;
	short *sh_fin = src + (n & ~15);
	__m128i x0, x1, x2, x3;
	__m128i xMask = _mm_srli_epi16(_mm_cmpeq_epi8(_mm_setzero_si128(), _mm_setzero_si128()), 8);
	__m128i xConst = _mm_set1_epi8(-128);
	for ( ; sh < sh_fin; sh += 16, byte0 += 16, byte1 += 16) {
		x0 = _mm_loadu_si128((__m128i*)(sh + 0));
		x1 = _mm_loadu_si128((__m128i*)(sh + 8));
		x2 = _mm_and_si128(x0, xMask); //Lower8bit
		x3 = _mm_and_si128(x1, xMask); //Lower8bit
		x0 = _mm_srli_epi16(x0, 8);    //Upper8bit
		x1 = _mm_srli_epi16(x1, 8);    //Upper8bit
		x2 = _mm_packus_epi16(x2, x3);
		x0 = _mm_packus_epi16(x0, x1);
		x2 = _mm_add_epi8(x2, xConst);
		x0 = _mm_add_epi8(x0, xConst);
		_mm_storeu_si128((__m128i*)byte0, x0);
		_mm_storeu_si128((__m128i*)byte1, x2);
	}
	sh_fin = sh + (n & 15);
	for ( ; sh < sh_fin; sh++, byte0++, byte1++) {
		*byte0 = (*sh >> 8)   + 128;
		*byte1 = (*sh & 0xff) + 128;
	}
}

//上のSSE2版
void convert_audio_16to8_sse2(BYTE *dst, short *src, int n) {
	BYTE *byte = dst;
	short *sh = src;
	BYTE * const loop_start = (BYTE *)get_aligned_next(dst);
	BYTE * const loop_fin   = (BYTE *)get_aligned_prev(dst + n);
	BYTE * const fin = dst + n;
	__m128i xSA, xSB;
	static const __m128i xConst = _mm_set1_epi16(128);
	//アライメント調整
	while (byte < loop_start) {
		*byte = (*sh >> 8) + 128;
		byte++;
		sh++;
	}
	//メインループ
	while (byte < loop_fin) {
		xSA = _mm_loadu_si128((const __m128i *)sh);
		sh += 8;
		xSA = _mm_srai_epi16(xSA, 8);
		xSA = _mm_add_epi16(xSA, xConst);
		xSB = _mm_loadu_si128((const __m128i *)sh);
		sh += 8;
		xSB = _mm_srai_epi16(xSB, 8);
		xSB = _mm_add_epi16(xSB, xConst);
		xSA = _mm_packus_epi16(xSA, xSB);
		_mm_stream_si128((__m128i *)byte, xSA);
		byte += 16;
	}
	//残り
	while (byte < fin) {
		*byte = (*sh >> 8) + 128;
		byte++;
		sh++;
	}
}
#pragma warning( push )
#pragma warning( disable: 4100 )
void copy_yuy2(void *frame, BYTE *dst_Y, BYTE *dst_C, const int width, const int height, const int pitch) {
	int y;
	BYTE *p, *p_fin, *Y;
	__m128i x0, x1, x2, x3;
	int pitch_src = width * 2; 
	//((((size_t)dst_Y | (size_t)pitch) & 0x0F) == 0x00)の条件は常に満たされる->_mm_stream_si128が使用可
	for (y = 0; y < height; y++) {
		p = (BYTE*)frame + y * pitch_src;
		Y = (BYTE*)dst_Y + y * pitch;
		p_fin = p + pitch_src;
		for (; p < p_fin; p += 64, Y += 64) {
			x0 = _mm_loadu_si128((const __m128i *)(p+ 0));
			x1 = _mm_loadu_si128((const __m128i *)(p+16));
			x2 = _mm_loadu_si128((const __m128i *)(p+32));
			x3 = _mm_loadu_si128((const __m128i *)(p+48));
			_mm_stream_si128((__m128i *)(Y +  0), x0);
			_mm_stream_si128((__m128i *)(Y + 16), x1);
			_mm_stream_si128((__m128i *)(Y + 32), x2);
			_mm_stream_si128((__m128i *)(Y + 48), x3);
		}
	}
}
#pragma warning( pop )
void convert_yuy2_to_nv12_sse2_aligned(void *frame, BYTE *dst_Y, BYTE *dst_C, const int width, const int height, const int pitch) {
	int x, y;
	BYTE *p, *pw, *Y, *C;
	__m128i x0, x1, x2, x3;
	for (y = 0; y < height; y += 2) {
		p  = (BYTE *)frame + ((y * width)<<1);
		pw = p + (width<<1);
		Y  = (BYTE *)dst_Y +  (y * pitch);
		C  = (BYTE *)dst_C + ((y * pitch)>>1);
		for (x = 0; x < width; x += 16, p += 32, pw += 32) {
			//-----------1行目---------------
			x0 = _mm_loadu_si128((const __m128i *)(p+ 0));    // VYUYVYUYVYUYVYUY
			x1 = _mm_loadu_si128((const __m128i *)(p+16));    // VYUYVYUYVYUYVYUY

			_mm_prefetch((const char *)pw, _MM_HINT_T1);

			x2 = _mm_unpacklo_epi8(x0, x1); //VVYYUUYYVVYYUUYY
			x1 = _mm_unpackhi_epi8(x0, x1); //VVYYUUYYVVYYUUYY

			x0 = _mm_unpacklo_epi8(x2, x1); //VVVVYYYYUUUUYYYY
			x1 = _mm_unpackhi_epi8(x2, x1); //VVVVYYYYUUUUYYYY

			x2 = _mm_unpacklo_epi8(x0, x1); //UUUUUUUUYYYYYYYY
			x1 = _mm_unpackhi_epi8(x0, x1); //VVVVVVVVYYYYYYYY

			x0 = _mm_unpacklo_epi8(x2, x1); //YYYYYYYYYYYYYYYY
			x3 = _mm_unpackhi_epi8(x2, x1); //VUVUVUVUVUVUVUVU

			_mm_stream_si128((__m128i *)(Y + x), x0);
			//-----------1行目終了---------------

			//-----------2行目---------------
			x0 = _mm_loadu_si128((const __m128i *)(pw+ 0));    // VYUYVYUYVYUYVYUY
			x1 = _mm_loadu_si128((const __m128i *)(pw+16));    // VYUYVYUYVYUYVYUY

			x2 = _mm_unpacklo_epi8(x0, x1); //VVYYUUYYVVYYUUYY
			x1 = _mm_unpackhi_epi8(x0, x1); //VVYYUUYYVVYYUUYY

			x0 = _mm_unpacklo_epi8(x2, x1); //VVVVYYYYUUUUYYYY
			x1 = _mm_unpackhi_epi8(x2, x1); //VVVVYYYYUUUUYYYY

			x2 = _mm_unpacklo_epi8(x0, x1); //UUUUUUUUYYYYYYYY
			x1 = _mm_unpackhi_epi8(x0, x1); //VVVVVVVVYYYYYYYY

			x0 = _mm_unpacklo_epi8(x2, x1); //YYYYYYYYYYYYYYYY
			x1 = _mm_unpackhi_epi8(x2, x1); //VUVUVUVUVUVUVUVU

			_mm_stream_si128((__m128i *)(Y + pitch + x), x0);
			//-----------2行目終了---------------

			x1 = _mm_avg_epu8(x1, x3);  //VUVUVUVUVUVUVUVU
			_mm_stream_si128((__m128i *)(C + x), x1);
		}
	}
}

void convert_yuy2_to_nv12_sse2(void *frame, BYTE *dst_Y, BYTE *dst_C, const int width, const int height, const int pitch) {
	int x, y;
	BYTE *p, *pw, *Y, *C;
	__m128i x0, x1, x2, x3;
	for (y = 0; y < height; y += 2) {
		p  = (BYTE *)frame + ((y * width)<<1);
		pw = p + (width<<1);
		Y  = (BYTE *)dst_Y +  (y * pitch);
		C  = (BYTE *)dst_C + ((y * pitch)>>1);
		for (x = 0; x < width; x += 16, p += 32, pw += 32) {
			//-----------1行目---------------
			x0 = _mm_loadu_si128((const __m128i *)(p+ 0));    // VYUYVYUYVYUYVYUY
			x1 = _mm_loadu_si128((const __m128i *)(p+16));    // VYUYVYUYVYUYVYUY

			_mm_prefetch((const char *)pw, _MM_HINT_T1);

			x2 = _mm_unpacklo_epi8(x0, x1); //VVYYUUYYVVYYUUYY
			x1 = _mm_unpackhi_epi8(x0, x1); //VVYYUUYYVVYYUUYY

			x0 = _mm_unpacklo_epi8(x2, x1); //VVVVYYYYUUUUYYYY
			x1 = _mm_unpackhi_epi8(x2, x1); //VVVVYYYYUUUUYYYY

			x2 = _mm_unpacklo_epi8(x0, x1); //UUUUUUUUYYYYYYYY
			x1 = _mm_unpackhi_epi8(x0, x1); //VVVVVVVVYYYYYYYY

			x0 = _mm_unpacklo_epi8(x2, x1); //YYYYYYYYYYYYYYYY
			x3 = _mm_unpackhi_epi8(x2, x1); //VUVUVUVUVUVUVUVU

			_mm_storeu_si128((__m128i *)(Y + x), x0);
			//-----------1行目終了---------------

			//-----------2行目---------------
			x0 = _mm_loadu_si128((const __m128i *)(pw+ 0));    // VYUYVYUYVYUYVYUY
			x1 = _mm_loadu_si128((const __m128i *)(pw+16));    // VYUYVYUYVYUYVYUY

			x2 = _mm_unpacklo_epi8(x0, x1); //VVYYUUYYVVYYUUYY
			x1 = _mm_unpackhi_epi8(x0, x1); //VVYYUUYYVVYYUUYY

			x0 = _mm_unpacklo_epi8(x2, x1); //VVVVYYYYUUUUYYYY
			x1 = _mm_unpackhi_epi8(x2, x1); //VVVVYYYYUUUUYYYY

			x2 = _mm_unpacklo_epi8(x0, x1); //UUUUUUUUYYYYYYYY
			x1 = _mm_unpackhi_epi8(x0, x1); //VVVVVVVVYYYYYYYY

			x0 = _mm_unpacklo_epi8(x2, x1); //YYYYYYYYYYYYYYYY
			x1 = _mm_unpackhi_epi8(x2, x1); //VUVUVUVUVUVUVUVU

			_mm_storeu_si128((__m128i *)(Y + pitch + x), x0);
			//-----------2行目終了---------------

			x1 = _mm_avg_epu8(x1, x3);  //VUVUVUVUVUVUVUVU
			_mm_storeu_si128((__m128i *)(C + x), x1);
		}
	}
}

void convert_yuy2_to_nv12_i_sse2_aligned(void *frame, BYTE *dst_Y, BYTE *dst_C, const int width, const int height, const int pitch) {
	int x, y, i;
	BYTE *p, *pw, *Y, *C;
	__m128i x0, x1, x2, x3, xC[2];
	for (y = 0; y < height; y += 4) {
		for (i = 0; i < 2; i++) {
			p  = (BYTE *)frame + (((y + i) * width)<<1);
			pw  = p   + (width<<2);
			Y  = (BYTE *)dst_Y +  ((y + i) * pitch);
			C  = (BYTE *)dst_C + ((((y + i) * pitch)+pitch*i)>>1);
			for (x = 0; x < width; x += 16, p += 32, pw += 32) {
				//-----------    1+i行目   ---------------
				x0 = _mm_loadu_si128((const __m128i *)(p+ 0));    // VYUYVYUYVYUYVYUY
				x1 = _mm_loadu_si128((const __m128i *)(p+16));    // VYUYVYUYVYUYVYUY

				_mm_prefetch((const char *)pw, _MM_HINT_T1);

				x2 = _mm_unpacklo_epi8(x0, x1); //VVYYUUYYVVYYUUYY
				x1 = _mm_unpackhi_epi8(x0, x1); //VVYYUUYYVVYYUUYY

				x0 = _mm_unpacklo_epi8(x2, x1); //VVVVYYYYUUUUYYYY
				x1 = _mm_unpackhi_epi8(x2, x1); //VVVVYYYYUUUUYYYY

				x2 = _mm_unpacklo_epi8(x0, x1); //UUUUUUUUYYYYYYYY
				x1 = _mm_unpackhi_epi8(x0, x1); //VVVVVVVVYYYYYYYY

				x0 = _mm_unpacklo_epi8(x2, x1); //YYYYYYYYYYYYYYYY
				xC[0] = _mm_unpackhi_epi8(x2, x1); //VUVUVUVUVUVUVUVU

				_mm_stream_si128((__m128i *)(Y + x), x0);
				//-----------1+i行目終了---------------

				//-----------3+i行目---------------
				x0 = _mm_loadu_si128((const __m128i *)(pw+ 0));    // VYUYVYUYVYUYVYUY
				x1 = _mm_loadu_si128((const __m128i *)(pw+16));    // VYUYVYUYVYUYVYUY

				x2 = _mm_unpacklo_epi8(x0, x1); //VVYYUUYYVVYYUUYY
				x1 = _mm_unpackhi_epi8(x0, x1); //VVYYUUYYVVYYUUYY

				x0 = _mm_unpacklo_epi8(x2, x1); //VVVVYYYYUUUUYYYY
				x1 = _mm_unpackhi_epi8(x2, x1); //VVVVYYYYUUUUYYYY

				x2 = _mm_unpacklo_epi8(x0, x1); //UUUUUUUUYYYYYYYY
				x1 = _mm_unpackhi_epi8(x0, x1); //VVVVVVVVYYYYYYYY

				x0 = _mm_unpacklo_epi8(x2, x1); //YYYYYYYYYYYYYYYY
				xC[1] = _mm_unpackhi_epi8(x2, x1); //VUVUVUVUVUVUVUVU

				_mm_stream_si128((__m128i *)(Y + (pitch<<1) + x), x0);
				//-----------3+i行目終了---------------

				x0 = _mm_unpacklo_epi8(xC[i], _mm_setzero_si128());
				x1 = _mm_unpackhi_epi8(xC[i], _mm_setzero_si128());
				x0 = _mm_mullo_epi16(x0, _mm_set1_epi16(3));
				x1 = _mm_mullo_epi16(x1, _mm_set1_epi16(3));
				x2 = _mm_unpacklo_epi8(xC[(i+1)&0x01], _mm_setzero_si128());
				x3 = _mm_unpackhi_epi8(xC[(i+1)&0x01], _mm_setzero_si128());
				x0 = _mm_add_epi16(x0, x2);
				x1 = _mm_add_epi16(x1, x3);
				x0 = _mm_add_epi16(x0, _mm_set1_epi16(2));
				x1 = _mm_add_epi16(x1, _mm_set1_epi16(2));
				x0 = _mm_srai_epi16(x0, 2);
				x1 = _mm_srai_epi16(x1, 2);
				x0 = _mm_packus_epi16(x0, x1); //VUVUVUVUVUVUVUVU
				_mm_stream_si128((__m128i *)(C + x), x0);
			}
		}
	}
}

void convert_yuy2_to_nv12_i_sse2(void *frame, BYTE *dst_Y, BYTE *dst_C, const int width, const int height, const int pitch) {
	int x, y, i;
	BYTE *p, *pw, *Y, *C;
	__m128i x0, x1, x2, x3, xC[2];
	for (y = 0; y < height; y += 4) {
		for (i = 0; i < 2; i++) {
			p  = (BYTE *)frame + (((y + i) * width)<<1);
			pw  = p   + (width<<2);
			Y  = (BYTE *)dst_Y +  ((y + i) * pitch);
			C  = (BYTE *)dst_C + ((((y + i) * pitch)+pitch*i)>>1);
			for (x = 0; x < width; x += 16, p += 32, pw += 32) {
				//-----------    1行目   ---------------
				x0 = _mm_loadu_si128((const __m128i *)(p+ 0));    // VYUYVYUYVYUYVYUY
				x1 = _mm_loadu_si128((const __m128i *)(p+16));    // VYUYVYUYVYUYVYUY

				_mm_prefetch((const char *)pw, _MM_HINT_T1);

				x2 = _mm_unpacklo_epi8(x0, x1); //VVYYUUYYVVYYUUYY
				x1 = _mm_unpackhi_epi8(x0, x1); //VVYYUUYYVVYYUUYY

				x0 = _mm_unpacklo_epi8(x2, x1); //VVVVYYYYUUUUYYYY
				x1 = _mm_unpackhi_epi8(x2, x1); //VVVVYYYYUUUUYYYY

				x2 = _mm_unpacklo_epi8(x0, x1); //UUUUUUUUYYYYYYYY
				x1 = _mm_unpackhi_epi8(x0, x1); //VVVVVVVVYYYYYYYY

				x0 = _mm_unpacklo_epi8(x2, x1); //YYYYYYYYYYYYYYYY
				xC[0] = _mm_unpackhi_epi8(x2, x1); //VUVUVUVUVUVUVUVU

				_mm_storeu_si128((__m128i *)(Y + x), x0);
				//-----------1行目終了---------------

				//-----------3行目---------------
				x0 = _mm_loadu_si128((const __m128i *)(pw+ 0));    // VYUYVYUYVYUYVYUY
				x1 = _mm_loadu_si128((const __m128i *)(pw+16));    // VYUYVYUYVYUYVYUY

				x2 = _mm_unpacklo_epi8(x0, x1); //VVYYUUYYVVYYUUYY
				x1 = _mm_unpackhi_epi8(x0, x1); //VVYYUUYYVVYYUUYY

				x0 = _mm_unpacklo_epi8(x2, x1); //VVVVYYYYUUUUYYYY
				x1 = _mm_unpackhi_epi8(x2, x1); //VVVVYYYYUUUUYYYY

				x2 = _mm_unpacklo_epi8(x0, x1); //UUUUUUUUYYYYYYYY
				x1 = _mm_unpackhi_epi8(x0, x1); //VVVVVVVVYYYYYYYY

				x0 = _mm_unpacklo_epi8(x2, x1); //YYYYYYYYYYYYYYYY
				xC[1] = _mm_unpackhi_epi8(x2, x1); //VUVUVUVUVUVUVUVU

				_mm_storeu_si128((__m128i *)(Y + (pitch<<1) + x), x0);
				//-----------3行目終了---------------

				x0 = _mm_unpacklo_epi8(xC[i], _mm_setzero_si128());
				x1 = _mm_unpackhi_epi8(xC[i], _mm_setzero_si128());
				x0 = _mm_mullo_epi16(x0, _mm_set1_epi16(3));
				x1 = _mm_mullo_epi16(x1, _mm_set1_epi16(3));
				x2 = _mm_unpacklo_epi8(xC[(i+1)&0x01], _mm_setzero_si128());
				x3 = _mm_unpackhi_epi8(xC[(i+1)&0x01], _mm_setzero_si128());
				x0 = _mm_add_epi16(x0, x2);
				x1 = _mm_add_epi16(x1, x3);
				x0 = _mm_add_epi16(x0, _mm_set1_epi16(2));
				x1 = _mm_add_epi16(x1, _mm_set1_epi16(2));
				x0 = _mm_srai_epi16(x0, 2);
				x1 = _mm_srai_epi16(x1, 2);
				x0 = _mm_packus_epi16(x0, x1); //VUVUVUVUVUVUVUVU
				_mm_storeu_si128((__m128i *)(C + x), x0);
			}
		}
	}
}

void convert_yuy2_to_nv12_i_ssse3_aligned(void *frame, BYTE *dst_Y, BYTE *dst_C, const int width, const int height, const int pitch) {
	int x, y, i;
	BYTE *p, *pw, *Y, *C;
	__m128i x0, x1, x2, x3;
	for (y = 0; y < height; y += 4) {
		for (i = 0; i < 2; i++) {
			p  = (BYTE *)frame + (((y + i) * width)<<1);
			pw  = p   + (width<<2);
			Y  = (BYTE *)dst_Y +  ((y + i) * pitch);
			C  = (BYTE *)dst_C + ((((y + i) * pitch)+pitch*i)>>1);
			for (x = 0; x < width; x += 16, p += 32, pw += 32) {
				//-----------    1行目   ---------------
				x0 = _mm_loadu_si128((__m128i *)(p+ 0));    // VYUYVYUYVYUYVYUY
				x1 = _mm_loadu_si128((__m128i *)(p+16));    // VYUYVYUYVYUYVYUY

				_mm_prefetch((const char *)pw, _MM_HINT_T1);

				x2 = _mm_unpacklo_epi8(x0, x1); //VVYYUUYYVVYYUUYY
				x1 = _mm_unpackhi_epi8(x0, x1); //VVYYUUYYVVYYUUYY

				x0 = _mm_unpacklo_epi8(x2, x1); //VVVVYYYYUUUUYYYY
				x1 = _mm_unpackhi_epi8(x2, x1); //VVVVYYYYUUUUYYYY

				x2 = _mm_unpacklo_epi8(x0, x1); //UUUUUUUUYYYYYYYY
				x1 = _mm_unpackhi_epi8(x0, x1); //VVVVVVVVYYYYYYYY

				x0 = _mm_unpacklo_epi8(x2, x1); //YYYYYYYYYYYYYYYY
				x3 = _mm_unpackhi_epi8(x2, x1); //VUVUVUVUVUVUVUVU

				_mm_stream_si128((__m128i *)(Y + x), x0);
				//-----------1行目終了---------------

				//-----------3行目---------------
				x0 = _mm_loadu_si128((__m128i *)(pw+ 0));    // VYUYVYUYVYUYVYUY
				x1 = _mm_loadu_si128((__m128i *)(pw+16));    // VYUYVYUYVYUYVYUY

				x2 = _mm_unpacklo_epi8(x0, x1); //VVYYUUYYVVYYUUYY
				x1 = _mm_unpackhi_epi8(x0, x1); //VVYYUUYYVVYYUUYY

				x0 = _mm_unpacklo_epi8(x2, x1); //VVVVYYYYUUUUYYYY
				x1 = _mm_unpackhi_epi8(x2, x1); //VVVVYYYYUUUUYYYY

				x2 = _mm_unpacklo_epi8(x0, x1); //UUUUUUUUYYYYYYYY
				x1 = _mm_unpackhi_epi8(x0, x1); //VVVVVVVVYYYYYYYY

				x0 = _mm_unpacklo_epi8(x2, x1); //YYYYYYYYYYYYYYYY
				x1 = _mm_unpackhi_epi8(x2, x1); //VUVUVUVUVUVUVUVU

				_mm_stream_si128((__m128i *)(Y + (pitch<<1) + x), x0);
				//-----------3+i行目終了---------------

				x0 = _mm_unpacklo_epi8(x1, x3);
				x1 = _mm_unpackhi_epi8(x1, x3);
				x0 = _mm_maddubs_epi16(x0, xC_INTERLACE_WEIGHT(i));
				x1 = _mm_maddubs_epi16(x1, xC_INTERLACE_WEIGHT(i));
				x0 = _mm_add_epi16(x0, _mm_set1_epi16(2));
				x1 = _mm_add_epi16(x1, _mm_set1_epi16(2));
				x0 = _mm_srai_epi16(x0, 2);
				x1 = _mm_srai_epi16(x1, 2);
				x0 = _mm_packus_epi16(x0, x1); //VUVUVUVUVUVUVUVU
				_mm_stream_si128((__m128i *)(C + x), x0);
			}
		}
	}
}

void convert_yuy2_to_nv12_i_ssse3(void *frame, BYTE *dst_Y, BYTE *dst_C, const int width, const int height, const int pitch) {
	int x, y, i;
	BYTE *p, *pw, *Y, *C;
	__m128i x0, x1, x2, x3;
	for (y = 0; y < height; y += 4) {
		for (i = 0; i < 2; i++) {
			p  = (BYTE *)frame + (((y + i) * width)<<1);
			pw  = p   + (width<<2);
			Y  = (BYTE *)dst_Y +  ((y + i) * pitch);
			C  = (BYTE *)dst_C + ((((y + i) * pitch)+pitch*i)>>1);
			for (x = 0; x < width; x += 16, p += 32, pw += 32) {
				//-----------    1行目   ---------------
				x0 = _mm_loadu_si128((__m128i *)(p+ 0));    // VYUYVYUYVYUYVYUY
				x1 = _mm_loadu_si128((__m128i *)(p+16));    // VYUYVYUYVYUYVYUY

				_mm_prefetch((const char *)pw, _MM_HINT_T1);

				x2 = _mm_unpacklo_epi8(x0, x1); //VVYYUUYYVVYYUUYY
				x1 = _mm_unpackhi_epi8(x0, x1); //VVYYUUYYVVYYUUYY

				x0 = _mm_unpacklo_epi8(x2, x1); //VVVVYYYYUUUUYYYY
				x1 = _mm_unpackhi_epi8(x2, x1); //VVVVYYYYUUUUYYYY

				x2 = _mm_unpacklo_epi8(x0, x1); //UUUUUUUUYYYYYYYY
				x1 = _mm_unpackhi_epi8(x0, x1); //VVVVVVVVYYYYYYYY

				x0 = _mm_unpacklo_epi8(x2, x1); //YYYYYYYYYYYYYYYY
				x3 = _mm_unpackhi_epi8(x2, x1); //VUVUVUVUVUVUVUVU

				_mm_storeu_si128((__m128i *)(Y + x), x0);
				//-----------1行目終了---------------

				//-----------3行目---------------
				x0 = _mm_loadu_si128((__m128i *)(pw+ 0));    // VYUYVYUYVYUYVYUY
				x1 = _mm_loadu_si128((__m128i *)(pw+16));    // VYUYVYUYVYUYVYUY

				x2 = _mm_unpacklo_epi8(x0, x1); //VVYYUUYYVVYYUUYY
				x1 = _mm_unpackhi_epi8(x0, x1); //VVYYUUYYVVYYUUYY

				x0 = _mm_unpacklo_epi8(x2, x1); //VVVVYYYYUUUUYYYY
				x1 = _mm_unpackhi_epi8(x2, x1); //VVVVYYYYUUUUYYYY

				x2 = _mm_unpacklo_epi8(x0, x1); //UUUUUUUUYYYYYYYY
				x1 = _mm_unpackhi_epi8(x0, x1); //VVVVVVVVYYYYYYYY

				x0 = _mm_unpacklo_epi8(x2, x1); //YYYYYYYYYYYYYYYY
				x1 = _mm_unpackhi_epi8(x2, x1); //VUVUVUVUVUVUVUVU

				_mm_storeu_si128((__m128i *)(Y + (pitch<<1) + x), x0);
				//-----------3行目終了---------------

				x0 = _mm_unpacklo_epi8(x1, x3);
				x1 = _mm_unpackhi_epi8(x1, x3);
				x0 = _mm_maddubs_epi16(x0, xC_INTERLACE_WEIGHT(i));
				x1 = _mm_maddubs_epi16(x1, xC_INTERLACE_WEIGHT(i));
				x0 = _mm_add_epi16(x0, _mm_set1_epi16(2));
				x1 = _mm_add_epi16(x1, _mm_set1_epi16(2));
				x0 = _mm_srai_epi16(x0, 2);
				x1 = _mm_srai_epi16(x1, 2);
				x0 = _mm_packus_epi16(x0, x1); //VUVUVUVUVUVUVUVU
				_mm_storeu_si128((__m128i *)(C + x), x0);
			}
		}
	}
}
