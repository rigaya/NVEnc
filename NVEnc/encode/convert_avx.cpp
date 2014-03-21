#include <Windows.h>
#include <emmintrin.h>
#include "convert.h"

#if (_MSC_VER >= 1600)
#include <immintrin.h>

void convert_yuy2_to_nv12_avx_aligned(void *frame, BYTE *dst_Y, BYTE *dst_C, const int width, const int height, const int pitch) {
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

void convert_yuy2_to_nv12_avx(void *frame, BYTE *dst_Y, BYTE *dst_C, const int width, const int height, const int pitch) {
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

void convert_yuy2_to_nv12_i_avx_aligned(void *frame, BYTE *dst_Y, BYTE *dst_C, const int width, const int height, const int pitch) {
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

void convert_yuy2_to_nv12_i_avx(void *frame, BYTE *dst_Y, BYTE *dst_C, const int width, const int height, const int pitch) {
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

#endif //(_MSC_VER >= 1600)

#if (_MSC_VER >= 1700)

#undef xC_INTERLACE_WEIGHT
#define xC_INTERLACE_WEIGHT(i) _mm256_load_si256((__m256i*)Array_INTERLACE_WEIGHT[i])


void convert_audio_16to8_avx2(BYTE *dst, short *src, int n) {
	BYTE *byte = dst;
	short *sh = src;
	BYTE * const loop_start = (BYTE *)(((size_t)dst + 31) & ~31);
	BYTE * const loop_fin   = (BYTE *)(((size_t)dst + n) & ~31);
	BYTE * const fin = dst + n;
	__m256i ySA, ySB;
	static const __m256i yConst = _mm256_set1_epi16(-128);
	//アライメント調整
	while (byte < loop_start) {
		*byte = (*sh >> 8) + 128;
		byte++;
		sh++;
	}
	//メインループ
	while (byte < loop_fin) {
		ySA = _mm256_loadu_si256((const __m256i *)sh);
		sh += 16;
		ySA = _mm256_srai_epi16(ySA, 8);
		ySA = _mm256_add_epi16(ySA, yConst);
		ySB = _mm256_loadu_si256((const __m256i *)sh);
		sh += 16;
		ySB = _mm256_srai_epi16(ySB, 8);
		ySB = _mm256_add_epi16(ySB, yConst);
		ySA = _mm256_packus_epi16(ySA, ySB);
		_mm256_stream_si256((__m256i *)byte, ySA);
		byte += 32;
	}
	//残り
	while (byte < fin) {
		*byte = (*sh >> 8) + 128;
		byte++;
		sh++;
	}
}

void split_audio_16to8x2_avx2(BYTE *dst, short *src, int n) {
	BYTE *byte0 = dst;
	BYTE *byte1 = dst + n;
	short *sh = src;
	short *sh_fin = src + (n & ~15);
	__m256i y0, y1, y2, y3;
	__m256i xMask = _mm256_srli_epi16(_mm256_cmpeq_epi8(_mm256_setzero_si256(), _mm256_setzero_si256()), 8);
	__m256i xConst = _mm256_set1_epi8(-128);
	for ( ; sh < sh_fin; sh += 16, byte0 += 16, byte1 += 16) {
		y0 = _mm256_loadu_si256((__m256i*)(sh + 0));
		y1 = _mm256_loadu_si256((__m256i*)(sh + 8));
		y2 = _mm256_and_si256(y0, xMask); //Lower8bit
		y3 = _mm256_and_si256(y1, xMask); //Lower8bit
		y0 = _mm256_srli_epi16(y0, 8);    //Upper8bit
		y1 = _mm256_srli_epi16(y1, 8);    //Upper8bit
		y2 = _mm256_packus_epi16(y2, y3);
		y0 = _mm256_packus_epi16(y0, y1);
		y2 = _mm256_add_epi8(y2, xConst);
		y0 = _mm256_add_epi8(y0, xConst);
		_mm256_storeu_si256((__m256i*)byte0, y0);
		_mm256_storeu_si256((__m256i*)byte1, y2);
	}
	sh_fin = sh + (n & 15);
	for ( ; sh < sh_fin; sh++, byte0++, byte1++) {
		*byte0 = (*sh >> 8)   + 128;
		*byte1 = (*sh & 0xff) + 128;
	}
}

void convert_yuy2_to_nv12_avx2_aligned(void *frame, BYTE *dst_Y, BYTE *dst_C, const int width, const int height, const int pitch) {
	int x, y;
	BYTE *p, *pw, *Y, *C;
	__m256i y0, y1, y2, y3;
	for (y = 0; y < height; y += 2) {
		p  = (BYTE *)frame + ((y * width)<<1);
		pw = p + (width<<1);
		Y  = (BYTE *)dst_Y +  (y * pitch);
		C  = (BYTE *)dst_C + ((y * pitch)>>1);
		for (x = 0; x < width; x += 32, p += 64, pw += 64) {
			//-----------1行目---------------
			y0 = _mm256_loadu_si256((const __m256i *)(p+ 0));    // VYUYVYUYVYUYVYUY
			y1 = _mm256_loadu_si256((const __m256i *)(p+32));    // VYUYVYUYVYUYVYUY

			_mm_prefetch((const char *)pw, _MM_HINT_T1);

			y2 = _mm256_unpacklo_epi8(y0, y1); //VVYYUUYYVVYYUUYY
			y1 = _mm256_unpackhi_epi8(y0, y1); //VVYYUUYYVVYYUUYY

			y0 = _mm256_unpacklo_epi8(y2, y1); //VVVVYYYYUUUUYYYY
			y1 = _mm256_unpackhi_epi8(y2, y1); //VVVVYYYYUUUUYYYY

			y2 = _mm256_unpacklo_epi8(y0, y1); //UUUUUUUUYYYYYYYY
			y1 = _mm256_unpackhi_epi8(y0, y1); //VVVVVVVVYYYYYYYY

			y0 = _mm256_unpacklo_epi8(y2, y1); //YYYYYYYYYYYYYYYY
			y3 = _mm256_unpackhi_epi8(y2, y1); //VUVUVUVUVUVUVUVU

			_mm256_stream_si256((__m256i *)(Y + x), y0);
			//-----------1行目終了---------------

			//-----------2行目---------------
			y0 = _mm256_loadu_si256((const __m256i *)(pw+ 0));    // VYUYVYUYVYUYVYUY
			y1 = _mm256_loadu_si256((const __m256i *)(pw+32));    // VYUYVYUYVYUYVYUY

			y2 = _mm256_unpacklo_epi8(y0, y1); //VVYYUUYYVVYYUUYY
			y1 = _mm256_unpackhi_epi8(y0, y1); //VVYYUUYYVVYYUUYY

			y0 = _mm256_unpacklo_epi8(y2, y1); //VVVVYYYYUUUUYYYY
			y1 = _mm256_unpackhi_epi8(y2, y1); //VVVVYYYYUUUUYYYY

			y2 = _mm256_unpacklo_epi8(y0, y1); //UUUUUUUUYYYYYYYY
			y1 = _mm256_unpackhi_epi8(y0, y1); //VVVVVVVVYYYYYYYY

			y0 = _mm256_unpacklo_epi8(y2, y1); //YYYYYYYYYYYYYYYY
			y1 = _mm256_unpackhi_epi8(y2, y1); //VUVUVUVUVUVUVUVU

			_mm256_stream_si256((__m256i *)(Y + pitch + x), y0);
			//-----------2行目終了---------------

			y1 = _mm256_avg_epu8(y1, y3);  //VUVUVUVUVUVUVUVU
			_mm256_stream_si256((__m256i *)(C + x), y1);
		}
	}
}

void convert_yuy2_to_nv12_avx2(void *frame, BYTE *dst_Y, BYTE *dst_C, const int width, const int height, const int pitch) {
	int x, y;
	BYTE *p, *pw, *Y, *C;
	__m256i y0, y1, y2, y3;
	for (y = 0; y < height; y += 2) {
		p  = (BYTE *)frame + ((y * width)<<1);
		pw = p + (width<<1);
		Y  = (BYTE *)dst_Y +  (y * pitch);
		C  = (BYTE *)dst_C + ((y * pitch)>>1);
		for (x = 0; x < width; x += 32, p += 64, pw += 64) {
			//-----------1行目---------------
			y0 = _mm256_loadu_si256((const __m256i *)(p+ 0));    // VYUYVYUYVYUYVYUY
			y1 = _mm256_loadu_si256((const __m256i *)(p+32));    // VYUYVYUYVYUYVYUY

			_mm_prefetch((const char *)pw, _MM_HINT_T1);

			y2 = _mm256_unpacklo_epi8(y0, y1); //VVYYUUYYVVYYUUYY
			y1 = _mm256_unpackhi_epi8(y0, y1); //VVYYUUYYVVYYUUYY

			y0 = _mm256_unpacklo_epi8(y2, y1); //VVVVYYYYUUUUYYYY
			y1 = _mm256_unpackhi_epi8(y2, y1); //VVVVYYYYUUUUYYYY

			y2 = _mm256_unpacklo_epi8(y0, y1); //UUUUUUUUYYYYYYYY
			y1 = _mm256_unpackhi_epi8(y0, y1); //VVVVVVVVYYYYYYYY

			y0 = _mm256_unpacklo_epi8(y2, y1); //YYYYYYYYYYYYYYYY
			y3 = _mm256_unpackhi_epi8(y2, y1); //VUVUVUVUVUVUVUVU

			_mm256_storeu_si256((__m256i *)(Y + x), y0);
			//-----------1行目終了---------------

			//-----------2行目---------------
			y0 = _mm256_loadu_si256((const __m256i *)(pw+ 0));    // VYUYVYUYVYUYVYUY
			y1 = _mm256_loadu_si256((const __m256i *)(pw+32));    // VYUYVYUYVYUYVYUY

			y2 = _mm256_unpacklo_epi8(y0, y1); //VVYYUUYYVVYYUUYY
			y1 = _mm256_unpackhi_epi8(y0, y1); //VVYYUUYYVVYYUUYY

			y0 = _mm256_unpacklo_epi8(y2, y1); //VVVVYYYYUUUUYYYY
			y1 = _mm256_unpackhi_epi8(y2, y1); //VVVVYYYYUUUUYYYY

			y2 = _mm256_unpacklo_epi8(y0, y1); //UUUUUUUUYYYYYYYY
			y1 = _mm256_unpackhi_epi8(y0, y1); //VVVVVVVVYYYYYYYY

			y0 = _mm256_unpacklo_epi8(y2, y1); //YYYYYYYYYYYYYYYY
			y1 = _mm256_unpackhi_epi8(y2, y1); //VUVUVUVUVUVUVUVU

			_mm256_storeu_si256((__m256i *)(Y + pitch + x), y0);
			//-----------2行目終了---------------

			y1 = _mm256_avg_epu8(y1, y3);  //VUVUVUVUVUVUVUVU
			_mm256_storeu_si256((__m256i *)(C + x), y1);
		}
	}
}

void convert_yuy2_to_nv12_i_avx2_aligned(void *frame, BYTE *dst_Y, BYTE *dst_C, const int width, const int height, const int pitch) {
	int x, y, i;
	BYTE *p, *pw, *Y, *C;
	__m256i y0, y1, y2, y3;
	for (y = 0; y < height; y += 4) {
		for (i = 0; i < 2; i++) {
			p  = (BYTE *)frame + (((y + i) * width)<<1);
			pw  = p   + (width<<2);
			Y  = (BYTE *)dst_Y +  ((y + i) * pitch);
			C  = (BYTE *)dst_C + ((((y + i) * pitch)+pitch*i)>>1);
			for (x = 0; x < width; x += 32, p += 64, pw += 64) {
				//-----------    1行目   ---------------
				y0 = _mm256_loadu_si256((__m256i *)(p+ 0));    // VYUYVYUYVYUYVYUY
				y1 = _mm256_loadu_si256((__m256i *)(p+32));    // VYUYVYUYVYUYVYUY

				_mm_prefetch((const char *)pw, _MM_HINT_T1);

				y2 = _mm256_unpacklo_epi8(y0, y1); //VVYYUUYYVVYYUUYY
				y1 = _mm256_unpackhi_epi8(y0, y1); //VVYYUUYYVVYYUUYY

				y0 = _mm256_unpacklo_epi8(y2, y1); //VVVVYYYYUUUUYYYY
				y1 = _mm256_unpackhi_epi8(y2, y1); //VVVVYYYYUUUUYYYY

				y2 = _mm256_unpacklo_epi8(y0, y1); //UUUUUUUUYYYYYYYY
				y1 = _mm256_unpackhi_epi8(y0, y1); //VVVVVVVVYYYYYYYY

				y0 = _mm256_unpacklo_epi8(y2, y1); //YYYYYYYYYYYYYYYY
				y3 = _mm256_unpackhi_epi8(y2, y1); //VUVUVUVUVUVUVUVU

				_mm256_stream_si256((__m256i *)(Y + x), y0);
				//-----------1行目終了---------------

				//-----------3行目---------------
				y0 = _mm256_loadu_si256((__m256i *)(pw+ 0));    // VYUYVYUYVYUYVYUY
				y1 = _mm256_loadu_si256((__m256i *)(pw+32));    // VYUYVYUYVYUYVYUY

				y2 = _mm256_unpacklo_epi8(y0, y1); //VVYYUUYYVVYYUUYY
				y1 = _mm256_unpackhi_epi8(y0, y1); //VVYYUUYYVVYYUUYY

				y0 = _mm256_unpacklo_epi8(y2, y1); //VVVVYYYYUUUUYYYY
				y1 = _mm256_unpackhi_epi8(y2, y1); //VVVVYYYYUUUUYYYY

				y2 = _mm256_unpacklo_epi8(y0, y1); //UUUUUUUUYYYYYYYY
				y1 = _mm256_unpackhi_epi8(y0, y1); //VVVVVVVVYYYYYYYY

				y0 = _mm256_unpacklo_epi8(y2, y1); //YYYYYYYYYYYYYYYY
				y1 = _mm256_unpackhi_epi8(y2, y1); //VUVUVUVUVUVUVUVU

				_mm256_stream_si256((__m256i *)(Y + (pitch<<1) + x), y0);
				//-----------3+i行目終了---------------

				y0 = _mm256_unpacklo_epi8(y1, y3);
				y1 = _mm256_unpackhi_epi8(y1, y3);
				y0 = _mm256_maddubs_epi16(y0, xC_INTERLACE_WEIGHT(i));
				y1 = _mm256_maddubs_epi16(y1, xC_INTERLACE_WEIGHT(i));
				y0 = _mm256_add_epi16(y0, _mm256_set1_epi16(2));
				y1 = _mm256_add_epi16(y1, _mm256_set1_epi16(2));
				y0 = _mm256_srai_epi16(y0, 2);
				y1 = _mm256_srai_epi16(y1, 2);
				y0 = _mm256_packus_epi16(y0, y1); //VUVUVUVUVUVUVUVU
				_mm256_stream_si256((__m256i *)(C + x), y0);
			}
		}
	}
}

void convert_yuy2_to_nv12_i_avx2(void *frame, BYTE *dst_Y, BYTE *dst_C, const int width, const int height, const int pitch) {
	int x, y, i;
	BYTE *p, *pw, *Y, *C;
	__m256i y0, y1, y2, y3;
	for (y = 0; y < height; y += 4) {
		for (i = 0; i < 2; i++) {
			p  = (BYTE *)frame + (((y + i) * width)<<1);
			pw  = p   + (width<<2);
			Y  = (BYTE *)dst_Y +  ((y + i) * pitch);
			C  = (BYTE *)dst_C + ((((y + i) * pitch)+pitch*i)>>1);
			for (x = 0; x < width; x += 32, p += 64, pw += 64) {
				//-----------    1行目   ---------------
				y0 = _mm256_loadu_si256((__m256i *)(p+ 0));    // VYUYVYUYVYUYVYUY
				y1 = _mm256_loadu_si256((__m256i *)(p+32));    // VYUYVYUYVYUYVYUY

				_mm_prefetch((const char *)pw, _MM_HINT_T1);

				y2 = _mm256_unpacklo_epi8(y0, y1); //VVYYUUYYVVYYUUYY
				y1 = _mm256_unpackhi_epi8(y0, y1); //VVYYUUYYVVYYUUYY

				y0 = _mm256_unpacklo_epi8(y2, y1); //VVVVYYYYUUUUYYYY
				y1 = _mm256_unpackhi_epi8(y2, y1); //VVVVYYYYUUUUYYYY

				y2 = _mm256_unpacklo_epi8(y0, y1); //UUUUUUUUYYYYYYYY
				y1 = _mm256_unpackhi_epi8(y0, y1); //VVVVVVVVYYYYYYYY

				y0 = _mm256_unpacklo_epi8(y2, y1); //YYYYYYYYYYYYYYYY
				y3 = _mm256_unpackhi_epi8(y2, y1); //VUVUVUVUVUVUVUVU

				_mm256_storeu_si256((__m256i *)(Y + x), y0);
				//-----------1行目終了---------------

				//-----------3行目---------------
				y0 = _mm256_loadu_si256((__m256i *)(pw+ 0));    // VYUYVYUYVYUYVYUY
				y1 = _mm256_loadu_si256((__m256i *)(pw+32));    // VYUYVYUYVYUYVYUY

				y2 = _mm256_unpacklo_epi8(y0, y1); //VVYYUUYYVVYYUUYY
				y1 = _mm256_unpackhi_epi8(y0, y1); //VVYYUUYYVVYYUUYY

				y0 = _mm256_unpacklo_epi8(y2, y1); //VVVVYYYYUUUUYYYY
				y1 = _mm256_unpackhi_epi8(y2, y1); //VVVVYYYYUUUUYYYY

				y2 = _mm256_unpacklo_epi8(y0, y1); //UUUUUUUUYYYYYYYY
				y1 = _mm256_unpackhi_epi8(y0, y1); //VVVVVVVVYYYYYYYY

				y0 = _mm256_unpacklo_epi8(y2, y1); //YYYYYYYYYYYYYYYY
				y1 = _mm256_unpackhi_epi8(y2, y1); //VUVUVUVUVUVUVUVU

				_mm256_storeu_si256((__m256i *)(Y + (pitch<<1) + x), y0);
				//-----------3行目終了---------------

				y0 = _mm256_unpacklo_epi8(y1, y3);
				y1 = _mm256_unpackhi_epi8(y1, y3);
				y0 = _mm256_maddubs_epi16(y0, xC_INTERLACE_WEIGHT(i));
				y1 = _mm256_maddubs_epi16(y1, xC_INTERLACE_WEIGHT(i));
				y0 = _mm256_add_epi16(y0, _mm256_set1_epi16(2));
				y1 = _mm256_add_epi16(y1, _mm256_set1_epi16(2));
				y0 = _mm256_srai_epi16(y0, 2);
				y1 = _mm256_srai_epi16(y1, 2);
				y0 = _mm256_packus_epi16(y0, y1); //VUVUVUVUVUVUVUVU
				_mm256_storeu_si256((__m256i *)(C + x), y0);
			}
		}
	}
}

#endif
