//  -----------------------------------------------------------------------------------------
//    QSVEnc by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  ----------------------------------------------------------------------------------------

#include <Windows.h>
#include <emmintrin.h>
#include <immintrin.h>

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
		ySA = _mm256_set_m128i(_mm_loadu_si128((__m128i*)(sh + 32)), _mm_loadu_si128((__m128i*)(sh +  0)));
		ySB = _mm256_set_m128i(_mm_loadu_si128((__m128i*)(sh + 48)), _mm_loadu_si128((__m128i*)(sh + 16)));
		ySA = _mm256_srai_epi16(ySA, 8);
		ySB = _mm256_srai_epi16(ySB, 8);
		ySA = _mm256_add_epi16(ySA, yConst);
		ySB = _mm256_add_epi16(ySB, yConst);
		ySA = _mm256_packus_epi16(ySA, ySB);
		_mm256_stream_si256((__m256i *)byte, ySA);
		sh += 32;
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
	__m256i yMask = _mm256_srli_epi16(_mm256_cmpeq_epi8(_mm256_setzero_si256(), _mm256_setzero_si256()), 8);
	__m256i yConst = _mm256_set1_epi8(-128);
	for ( ; sh < sh_fin; sh += 16, byte0 += 16, byte1 += 16) {
		y0 = _mm256_set_m128i(_mm_loadu_si128((__m128i*)(sh + 16)), _mm_loadu_si128((__m128i*)(sh + 0)));
		y1 = _mm256_set_m128i(_mm_loadu_si128((__m128i*)(sh + 24)), _mm_loadu_si128((__m128i*)(sh + 8)));
		y2 = _mm256_and_si256(y0, yMask); //Lower8bit
		y3 = _mm256_and_si256(y1, yMask); //Lower8bit
		y0 = _mm256_srli_epi16(y0, 8);    //Upper8bit
		y1 = _mm256_srli_epi16(y1, 8);    //Upper8bit
		y2 = _mm256_packus_epi16(y2, y3);
		y0 = _mm256_packus_epi16(y0, y1);
		y2 = _mm256_add_epi8(y2, yConst);
		y0 = _mm256_add_epi8(y0, yConst);
		_mm256_storeu_si256((__m256i*)byte0, y0);
		_mm256_storeu_si256((__m256i*)byte1, y2);
	}
	sh_fin = sh + (n & 15);
	for ( ; sh < sh_fin; sh++, byte0++, byte1++) {
		*byte0 = (*sh >> 8)   + 128;
		*byte1 = (*sh & 0xff) + 128;
	}
}
