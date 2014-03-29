//  -----------------------------------------------------------------------------------------
//    NVEnc by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  -----------------------------------------------------------------------------------------

#include <stdint.h>
#include <immintrin.h>

//本来の256bit alignr
#define MM_ABS(x) (((x) < 0) ? -(x) : (x))
#define _mm256_alignr256_epi8(a, b, i) ((i<=16) ? _mm256_alignr_epi8(_mm256_permute2x128_si256(a, b, (0x00<<4) + 0x03), b, i) : _mm256_alignr_epi8(a, _mm256_permute2x128_si256(a, b, (0x00<<4) + 0x03), MM_ABS(i-16)))

//_mm256_srli_si256, _mm256_slli_si256は
//単に128bitシフト×2をするだけの命令である
#define _mm256_bsrli_epi128 _mm256_srli_si256
#define _mm256_bslli_epi128 _mm256_slli_si256
//本当の256bitシフト
#define _mm256_srli256_si256(a, i) ((i<=16) ? _mm256_alignr_epi8(_mm256_permute2x128_si256(a, a, (0x08<<4) + 0x03), a, i) : _mm256_bsrli_epi128(_mm256_permute2x128_si256(a, a, (0x08<<4) + 0x03), MM_ABS(i-16)))
#define _mm256_slli256_si256(a, i) ((i<=16) ? _mm256_alignr_epi8(a, _mm256_permute2x128_si256(a, a, (0x00<<4) + 0x08), MM_ABS(16-i)) : _mm256_bslli_epi128(_mm256_permute2x128_si256(a, a, (0x00<<4) + 0x08), MM_ABS(i-16)))

static const _declspec(align(32)) uint8_t  Array_INTERLACE_WEIGHT[2][32] = {
	{1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3},
	{3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1}
};
#define yC_INTERLACE_WEIGHT(i) _mm256_load_si256((__m256i*)Array_INTERLACE_WEIGHT[i])

static __forceinline void separate_low_up(__m256i& x0_return_lower, __m256i& x1_return_upper) {
	__m256i x4, x5;
	const __m256i xMaskLowByte = _mm256_srli_epi16(_mm256_cmpeq_epi8(_mm256_setzero_si256(), _mm256_setzero_si256()), 8);
	x4 = _mm256_srli_epi16(x0_return_lower, 8);
	x5 = _mm256_srli_epi16(x1_return_upper, 8);

	x0_return_lower = _mm256_and_si256(x0_return_lower, xMaskLowByte);
	x1_return_upper = _mm256_and_si256(x1_return_upper, xMaskLowByte);

	x0_return_lower = _mm256_packus_epi16(x0_return_lower, x1_return_upper);
	x1_return_upper = _mm256_packus_epi16(x4, x5);
}

void convert_yuy2_to_nv12_avx2(void *dst, void *src, int width, int src_y_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop) {
	const int crop_left   = crop[0];
	const int crop_up     = crop[1];
	const int crop_right  = crop[2];
	const int crop_bottom = crop[3];
	uint8_t *srcLine = (uint8_t *)src + src_y_pitch_byte * crop_up + crop_left;
	uint8_t *dstYLine = (uint8_t *)dst;
	uint8_t *dstCLine = dstYLine + dst_y_pitch_byte * dst_height;
	const int y_fin = height - crop_bottom - crop_up;
	for (int y = 0; y < y_fin; y += 2) {
		uint8_t *p = srcLine;
		uint8_t *pw = p + src_y_pitch_byte;
		const int x_fin = width - crop_right - crop_left;
		__m256i y0, y1, y3;
		for (int x = 0; x < x_fin; x += 32, p += 64, pw += 64) {
			//-----------1行目---------------
			y0 = _mm256_set_m128i(_mm_loadu_si128((__m128i*)(p+32)), _mm_loadu_si128((__m128i*)(p+ 0)));
			y1 = _mm256_set_m128i(_mm_loadu_si128((__m128i*)(p+48)), _mm_loadu_si128((__m128i*)(p+16)));

			separate_low_up(y0, y1);
			y3 = y1;

			_mm256_stream_si256((__m256i *)(dstYLine + x), y0);
			//-----------1行目終了---------------

			//-----------2行目---------------
			y0 = _mm256_set_m128i(_mm_loadu_si128((__m128i*)(pw+32)), _mm_loadu_si128((__m128i*)(pw+ 0)));
			y1 = _mm256_set_m128i(_mm_loadu_si128((__m128i*)(pw+48)), _mm_loadu_si128((__m128i*)(pw+16)));

			separate_low_up(y0, y1);

			_mm256_stream_si256((__m256i *)(dstYLine + dst_y_pitch_byte + x), y0);
			//-----------2行目終了---------------

			y1 = _mm256_avg_epu8(y1, y3);  //VUVUVUVUVUVUVUVU
			_mm256_stream_si256((__m256i *)(dstCLine + x), y1);
		}
		srcLine  += src_y_pitch_byte << 1;
		dstYLine += dst_y_pitch_byte << 1;
		dstCLine += dst_y_pitch_byte;
	}
	_mm256_zeroupper();
}

static __forceinline __m256i yuv422_to_420_i_interpolate(__m256i y_up, __m256i y_down, int i) {
	__m256i y0, y1;
	y0 = _mm256_unpacklo_epi8(y_down, y_up);
	y1 = _mm256_unpackhi_epi8(y_down, y_up);
	y0 = _mm256_maddubs_epi16(y0, yC_INTERLACE_WEIGHT(i));
	y1 = _mm256_maddubs_epi16(y1, yC_INTERLACE_WEIGHT(i));
	y0 = _mm256_add_epi16(y0, _mm256_set1_epi16(2));
	y1 = _mm256_add_epi16(y1, _mm256_set1_epi16(2));
	y0 = _mm256_srai_epi16(y0, 2);
	y1 = _mm256_srai_epi16(y1, 2);
	y0 = _mm256_packus_epi16(y0, y1);
	return y0;
}

void convert_yuy2_to_nv12_i_avx2(void *dst, void *src, int width, int src_y_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop) {
	const int crop_left   = crop[0];
	const int crop_up     = crop[1];
	const int crop_right  = crop[2];
	const int crop_bottom = crop[3];
	uint8_t *srcLine = (uint8_t *)src + src_y_pitch_byte * crop_up + crop_left;
	uint8_t *dstYLine = (uint8_t *)dst;
	uint8_t *dstCLine = dstYLine + dst_y_pitch_byte * dst_height;
	const int y_fin = height - crop_bottom - crop_up;
	for (int y = 0; y < y_fin; y += 4) {
		for (int i = 0; i < 2; i++) {
			uint8_t *p = srcLine;
			uint8_t *pw = p + (src_y_pitch_byte<<1);
			__m256i y0, y1, y3;
			const int x_fin = width - crop_right - crop_left;
			for (int x = 0; x < x_fin; x += 32, p += 64, pw += 64) {
				//-----------    1+i行目   ---------------
				y0 = _mm256_set_m128i(_mm_loadu_si128((__m128i*)(p+32)), _mm_loadu_si128((__m128i*)(p+ 0)));
				y1 = _mm256_set_m128i(_mm_loadu_si128((__m128i*)(p+48)), _mm_loadu_si128((__m128i*)(p+16)));

				separate_low_up(y0, y1);
				y3 = y1;

				_mm256_stream_si256((__m256i *)(dstYLine + x), y0);
				//-----------1+i行目終了---------------

				//-----------3+i行目---------------
				y0 = _mm256_set_m128i(_mm_loadu_si128((__m128i*)(pw+32)), _mm_loadu_si128((__m128i*)(pw+ 0)));
				y1 = _mm256_set_m128i(_mm_loadu_si128((__m128i*)(pw+48)), _mm_loadu_si128((__m128i*)(pw+16)));

				separate_low_up(y0, y1);

				_mm256_stream_si256((__m256i *)(dstYLine + (dst_y_pitch_byte<<1) + x), y0);
				//-----------3+i行目終了---------------
				y0 = yuv422_to_420_i_interpolate(y3, y1, i);

				_mm256_stream_si256((__m256i *)(dstCLine + x), y0);
			}
			srcLine  += src_y_pitch_byte;
			dstYLine += dst_y_pitch_byte;
			dstCLine += dst_y_pitch_byte;
		}
		srcLine  += src_y_pitch_byte << 1;
		dstYLine += dst_y_pitch_byte << 1;
	}
	_mm256_zeroupper();
}
