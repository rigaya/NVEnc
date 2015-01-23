//  -----------------------------------------------------------------------------------------
//    NVEnc by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  -----------------------------------------------------------------------------------------

#ifndef _CONVERT_CSP_H_
#define _CONVERT_CSP_H_

#include <cstdint>
#include <emmintrin.h> //イントリンシック命令 SSE2
#if USE_SSSE3
#include <tmmintrin.h> //イントリンシック命令 SSSE3
#endif
#if USE_SSE41
#include <smmintrin.h> //イントリンシック命令 SSE4.1
#endif

static void __forceinline memcpy_sse(uint8_t *dst, const uint8_t *src, int size) {
	if (size < 64) {
		for (int i = 0; i < size; i++)
			dst[i] = src[i];
		return;
	}
	uint8_t *dst_fin = dst + size;
	uint8_t *dst_aligned_fin = (uint8_t *)(((size_t)(dst_fin + 15) & ~15) - 64);
	__m128 x0, x1, x2, x3;
	const int start_align_diff = (int)((size_t)dst & 15);
	if (start_align_diff) {
		x0 = _mm_loadu_ps((float*)src);
		_mm_storeu_ps((float*)dst, x0);
		dst += 16 - start_align_diff;
		src += 16 - start_align_diff;
	}
	for ( ; dst < dst_aligned_fin; dst += 64, src += 64) {
		x0 = _mm_loadu_ps((float*)(src +  0));
		x1 = _mm_loadu_ps((float*)(src + 16));
		x2 = _mm_loadu_ps((float*)(src + 32));
		x3 = _mm_loadu_ps((float*)(src + 48));
		_mm_store_ps((float*)(dst +  0), x0);
		_mm_store_ps((float*)(dst + 16), x1);
		_mm_store_ps((float*)(dst + 32), x2);
		_mm_store_ps((float*)(dst + 48), x3);
	}
	uint8_t *dst_tmp = dst_fin - 64;
	src -= (dst - dst_tmp);
	x0 = _mm_loadu_ps((float*)(src +  0));
	x1 = _mm_loadu_ps((float*)(src + 16));
	x2 = _mm_loadu_ps((float*)(src + 32));
	x3 = _mm_loadu_ps((float*)(src + 48));
	_mm_storeu_ps((float*)(dst_tmp +  0), x0);
	_mm_storeu_ps((float*)(dst_tmp + 16), x1);
	_mm_storeu_ps((float*)(dst_tmp + 32), x2);
	_mm_storeu_ps((float*)(dst_tmp + 48), x3);
}

#define _mm_store_switch_si128(ptr, xmm) ((aligned_store) ? _mm_store_si128(ptr, xmm) : _mm_storeu_si128(ptr, xmm))

#if USE_SSSE3
#define _mm_alignr_epi8_simd(a,b,i) _mm_alignr_epi8(a,b,i)
#else
#define _mm_alignr_epi8_simd(a,b,i) _mm_or_si128( _mm_slli_si128(a, 16-i), _mm_srli_si128(b, i) )
#endif

static __forceinline __m128i select_by_mask(__m128i a, __m128i b, __m128i mask) {
#if USE_SSE41
	return _mm_blendv_epi8(a, b, mask);
#else
	return _mm_or_si128( _mm_andnot_si128(mask,a), _mm_and_si128(b,mask) );
#endif
}

static __forceinline __m128i _mm_packus_epi32_simd(__m128i a, __m128i b) {
#if USE_SSE41
	return _mm_packus_epi32(a, b);
#else
	static const _declspec(align(64)) uint32_t VAL[2][4] = {
		{ 0x00008000, 0x00008000, 0x00008000, 0x00008000 },
		{ 0x80008000, 0x80008000, 0x80008000, 0x80008000 }
	};
#define LOAD_32BIT_0x8000 _mm_load_si128((__m128i *)VAL[0])
#define LOAD_16BIT_0x8000 _mm_load_si128((__m128i *)VAL[1])
	a = _mm_sub_epi32(a, LOAD_32BIT_0x8000);
	b = _mm_sub_epi32(b, LOAD_32BIT_0x8000);
	a = _mm_packs_epi32(a, b);
	return _mm_add_epi16(a, LOAD_16BIT_0x8000);
#undef LOAD_32BIT_0x8000
#undef LOAD_16BIT_0x8000
#endif
}

#if USE_SSSE3
static const _declspec(align(32)) uint8_t  Array_INTERLACE_WEIGHT[2][32] = { 
	{1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3},
	{3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1}
};
#define xC_INTERLACE_WEIGHT(i) _mm_load_si128((__m128i*)Array_INTERLACE_WEIGHT[i])
#endif

static __forceinline void separate_low_up(__m128i& x0_return_lower, __m128i& x1_return_upper) {
	__m128i x4, x5;
	const __m128i xMaskLowByte = _mm_srli_epi16(_mm_cmpeq_epi8(_mm_setzero_si128(), _mm_setzero_si128()), 8);
	x4 = _mm_srli_epi16(x0_return_lower, 8);
	x5 = _mm_srli_epi16(x1_return_upper, 8);

	x0_return_lower = _mm_and_si128(x0_return_lower, xMaskLowByte);
	x1_return_upper = _mm_and_si128(x1_return_upper, xMaskLowByte);

	x0_return_lower = _mm_packus_epi16(x0_return_lower, x1_return_upper);
	x1_return_upper = _mm_packus_epi16(x4, x5);
}

static void __forceinline convert_yuy2_to_nv12_simd(void *dst, const void *src, int width, int src_y_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop) {
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
		__m128i x0, x1, x3;
		for (int x = 0; x < x_fin; x += 16, p += 32, pw += 32) {
			//-----------1行目---------------
			x0 = _mm_loadu_si128((const __m128i *)(p+ 0));
			x1 = _mm_loadu_si128((const __m128i *)(p+16));
			
			separate_low_up(x0, x1);
			x3 = x1;

			_mm_stream_si128((__m128i *)(dstYLine + x), x0);
			//-----------1行目終了---------------

			//-----------2行目---------------
			x0 = _mm_loadu_si128((const __m128i *)(pw+ 0));
			x1 = _mm_loadu_si128((const __m128i *)(pw+16));
			
			separate_low_up(x0, x1);

			_mm_stream_si128((__m128i *)(dstYLine + dst_y_pitch_byte + x), x0);
			//-----------2行目終了---------------

			x1 = _mm_avg_epu8(x1, x3);
			_mm_stream_si128((__m128i *)(dstCLine + x), x1);
		}
		srcLine  += src_y_pitch_byte << 1;
		dstYLine += dst_y_pitch_byte << 1;
		dstCLine += dst_y_pitch_byte;
	}
}

static __forceinline __m128i yuv422_to_420_i_interpolate(__m128i y_up, __m128i y_down, int i) {
	__m128i x0, x1;
#if USE_SSSE3
	x0 = _mm_unpacklo_epi8(y_down, y_up);
	x1 = _mm_unpackhi_epi8(y_down, y_up);
	x0 = _mm_maddubs_epi16(x0, xC_INTERLACE_WEIGHT(i));
	x1 = _mm_maddubs_epi16(x1, xC_INTERLACE_WEIGHT(i));
#else
	__m128i x2, x3, xC[2];
	xC[0] = y_up;
	xC[1] = y_down;
	x0 = _mm_unpacklo_epi8(xC[i], _mm_setzero_si128());
	x1 = _mm_unpackhi_epi8(xC[i], _mm_setzero_si128());
	x0 = _mm_mullo_epi16(x0, _mm_set1_epi16(3));
	x1 = _mm_mullo_epi16(x1, _mm_set1_epi16(3));
	x2 = _mm_unpacklo_epi8(xC[(i+1)&0x01], _mm_setzero_si128());
	x3 = _mm_unpackhi_epi8(xC[(i+1)&0x01], _mm_setzero_si128());
	x0 = _mm_add_epi16(x0, x2);
	x1 = _mm_add_epi16(x1, x3);
#endif
	x0 = _mm_add_epi16(x0, _mm_set1_epi16(2));
	x1 = _mm_add_epi16(x1, _mm_set1_epi16(2));
	x0 = _mm_srai_epi16(x0, 2);
	x1 = _mm_srai_epi16(x1, 2);
	x0 = _mm_packus_epi16(x0, x1);
	return x0;
}

static void __forceinline convert_yuy2_to_nv12_i_simd(void *dst, const void *src, int width, int src_y_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop) {
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
			__m128i x0, x1, x3;
			const int x_fin = width - crop_right - crop_left;
			for (int x = 0; x < x_fin; x += 16, p += 32, pw += 32) {
				//-----------    1+i行目   ---------------
				x0 = _mm_loadu_si128((const __m128i *)(p+ 0));
				x1 = _mm_loadu_si128((const __m128i *)(p+16));

				separate_low_up(x0, x1);
				x3 = x1;

				_mm_stream_si128((__m128i *)(dstYLine + x), x0);
				//-----------1+i行目終了---------------

				//-----------3+i行目---------------
				x0 = _mm_loadu_si128((const __m128i *)(pw+ 0));
				x1 = _mm_loadu_si128((const __m128i *)(pw+16));

				separate_low_up(x0, x1);

				_mm_stream_si128((__m128i *)(dstYLine + (dst_y_pitch_byte<<1) + x), x0);
				//-----------3+i行目終了---------------
				x0 = yuv422_to_420_i_interpolate(x3, x1, i);

				_mm_stream_si128((__m128i *)(dstCLine + x), x0);
			}
			srcLine  += src_y_pitch_byte;
			dstYLine += dst_y_pitch_byte;
			dstCLine += dst_y_pitch_byte;
		}
		srcLine  += src_y_pitch_byte << 1;
		dstYLine += dst_y_pitch_byte << 1;
	}
}

#pragma warning (push)
#pragma warning (disable: 4100)
#pragma warning (disable: 4127)
template<bool uv_only>
static void __forceinline convert_yv12_to_nv12_simd(void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop) {
	const int crop_left   = crop[0];
	const int crop_up     = crop[1];
	const int crop_right  = crop[2];
	const int crop_bottom = crop[3];
	//Y成分のコピー
	if (!uv_only) {
		uint8_t *srcYLine = (uint8_t *)src[0] + src_y_pitch_byte * crop_up + crop_left;
		uint8_t *dstLine = (uint8_t *)dst[0];
		const int y_fin = height - crop_bottom;
		const int y_width = width - crop_right - crop_left;
		for (int y = crop_up; y < y_fin; y++, srcYLine += src_y_pitch_byte, dstLine += dst_y_pitch_byte) {
			memcpy_sse(dstLine, srcYLine, y_width);
		}
	}
	//UV成分のコピー
	uint8_t *srcULine = (uint8_t *)src[1] + (((src_uv_pitch_byte * crop_up) + crop_left) >> 1);
	uint8_t *srcVLine = (uint8_t *)src[2] + (((src_uv_pitch_byte * crop_up) + crop_left) >> 1);
	uint8_t *dstLine = (uint8_t *)dst[1];
	const int uv_fin = (height - crop_bottom) >> 1;
	for (int y = crop_up >> 1; y < uv_fin; y++, srcULine += src_uv_pitch_byte, srcVLine += src_uv_pitch_byte, dstLine += dst_y_pitch_byte) {
		const int x_fin = width - crop_right;
		uint8_t *src_u_ptr = srcULine;
		uint8_t *src_v_ptr = srcVLine;
		uint8_t *dst_ptr = dstLine;
		__m128i x0, x1, x2;
		for (int x = crop_left; x < x_fin; x += 32, src_u_ptr += 16, src_v_ptr += 16, dst_ptr += 32) {
			x0 = _mm_loadu_si128((const __m128i *)src_u_ptr);
			x1 = _mm_loadu_si128((const __m128i *)src_v_ptr);

			x2 = _mm_unpackhi_epi8(x0, x1);
			x0 = _mm_unpacklo_epi8(x0, x1);

			_mm_storeu_si128((__m128i *)(dst_ptr +  0), x0);
			_mm_storeu_si128((__m128i *)(dst_ptr + 16), x2);
		}
	}
}
#pragma warning (pop)

#endif //_CONVERT_CSP_H_
