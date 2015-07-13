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

typedef void (*funcConvertCSP) (void **dst, const void **src, int width, int src_y_pitch_byte, int src_uv_pitch_byte, int dst_y_pitch_byte, int height, int dst_height, int *crop);

enum NV_ENC_CSP {
    NV_ENC_CSP_NA,
    NV_ENC_CSP_NV12,
    NV_ENC_CSP_YV12,
    NV_ENC_CSP_YUY2,
};

static const TCHAR *NV_ENC_CSP_NAMES[] = { _T("Invalid"), _T("nv12"), _T("yv12"), _T("yuy2") };

typedef struct ConvertCSP {
    NV_ENC_CSP csp_from, csp_to;
    bool uv_only;
    funcConvertCSP func[2];
    unsigned int simd;
} ConvertCSP;

const ConvertCSP *get_convert_csp_func(NV_ENC_CSP csp_from, NV_ENC_CSP csp_to, bool uv_only);
const TCHAR *get_simd_str(unsigned int simd);

#endif //_CONVERT_CSP_H_
