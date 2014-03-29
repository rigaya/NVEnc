#include <cstdint>

#pragma warning(push)
#pragma warning(disable:4100)
void convert_yv12_to_nv12(void *dst, void *src, int width, int src_y_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop) {

}
#pragma warning(pop)
//適当。
void convert_yuy2_to_nv12(void *dst, void *src, int width, int src_y_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop) {
	int crop_left   = crop[0];
	int crop_up     = crop[1];
	int crop_right  = crop[2];
	int crop_bottom = crop[3];
	uint8_t *srcFrame = (uint8_t *)src;
	uint8_t *dstYFrame = (uint8_t *)dst;
	uint8_t *dstCFrame = dstYFrame + dst_y_pitch_byte * dst_height;
	const int y_fin = height - crop_bottom - crop_up;
	for (int y = 0; y < y_fin; y += 2) {
		uint8_t *dstY = dstYFrame +   dst_y_pitch_byte * y;
		uint8_t *dstC = dstCFrame + ((dst_y_pitch_byte * y) >> 1);
		uint8_t *srcP = srcFrame  +   src_y_pitch_byte * (y + crop_up) + crop_left;
		const int x_fin = width - crop_right - crop_left;
		for (int x = 0; x < x_fin; x += 2, dstY += 2, dstC += 2, srcP += 4) {
			dstY[0*dst_y_pitch_byte  + 0] = srcP[0*src_y_pitch_byte + 0];
			dstY[0*dst_y_pitch_byte  + 1] = srcP[0*src_y_pitch_byte + 2];
			dstY[1*dst_y_pitch_byte  + 0] = srcP[1*src_y_pitch_byte + 0];
			dstY[1*dst_y_pitch_byte  + 1] = srcP[1*src_y_pitch_byte + 2];
			dstC[0*dst_y_pitch_byte/2+ 0] =(srcP[0*src_y_pitch_byte + 1] + srcP[1*src_y_pitch_byte + 1] + 1)/2;
			dstC[0*dst_y_pitch_byte/2+ 1] =(srcP[0*src_y_pitch_byte + 3] + srcP[1*src_y_pitch_byte + 3] + 1)/2;
		}
	}
}

//これも適当。
void convert_yuy2_to_nv12_i(void *dst, void *src, int width, int src_y_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop) {
	int crop_left   = crop[0];
	int crop_up     = crop[1];
	int crop_right  = crop[2];
	int crop_bottom = crop[3];
	uint8_t *srcFrame = (uint8_t *)src;
	uint8_t *dstYFrame = (uint8_t *)dst;
	uint8_t *dstCFrame = dstYFrame + dst_y_pitch_byte * dst_height;
	const int y_fin = height - crop_bottom - crop_up;
	for (int y = 0; y < y_fin; y += 4) {
		uint8_t *dstY = dstYFrame +   dst_y_pitch_byte * y;
		uint8_t *dstC = dstCFrame + ((dst_y_pitch_byte * y) >> 1);
		uint8_t *srcP = srcFrame  +   src_y_pitch_byte * (y + crop_up);
		const int x_fin = width - crop_right - crop_left;
		for (int x = 0; x < x_fin; x += 2, dstY += 2, dstC += 2, srcP += 4) {
			dstY[0*dst_y_pitch_byte   + 0] = srcP[0*src_y_pitch_byte + 0];
			dstY[0*dst_y_pitch_byte   + 1] = srcP[0*src_y_pitch_byte + 2];
			dstY[1*dst_y_pitch_byte   + 0] = srcP[1*src_y_pitch_byte + 0];
			dstY[1*dst_y_pitch_byte   + 1] = srcP[1*src_y_pitch_byte + 2];
			dstY[2*dst_y_pitch_byte   + 0] = srcP[2*src_y_pitch_byte + 0];
			dstY[2*dst_y_pitch_byte   + 1] = srcP[2*src_y_pitch_byte + 2];
			dstY[3*dst_y_pitch_byte   + 0] = srcP[3*src_y_pitch_byte + 0];
			dstY[3*dst_y_pitch_byte   + 1] = srcP[3*src_y_pitch_byte + 2];
			dstC[0*dst_y_pitch_byte/2 + 0] =(srcP[0*src_y_pitch_byte + 1] * 3 + srcP[2*src_y_pitch_byte + 1] * 1 + 2)>>2;
			dstC[0*dst_y_pitch_byte/2 + 1] =(srcP[0*src_y_pitch_byte + 3] * 3 + srcP[2*src_y_pitch_byte + 3] * 1 + 2)>>2;
			dstC[1*dst_y_pitch_byte/2 + 0] =(srcP[1*src_y_pitch_byte + 1] * 1 + srcP[3*src_y_pitch_byte + 1] * 3 + 2)>>2;
			dstC[1*dst_y_pitch_byte/2 + 1] =(srcP[1*src_y_pitch_byte + 3] * 1 + srcP[3*src_y_pitch_byte + 3] * 3 + 2)>>2;
		}
	}
}