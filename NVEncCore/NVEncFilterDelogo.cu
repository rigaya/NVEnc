// -----------------------------------------------------------------------------------------
// NVEnc by rigaya
// -----------------------------------------------------------------------------------------
//
// The MIT License
//
// Copyright (c) 2014-2016 rigaya
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
// ------------------------------------------------------------------------------------------

#include <map>
#include <array>
#include <cstdint>
#include "NVEncFilterDelogo.h"
#include "cuda_runtime.h"
#include "device_functions.hpp"
#include "device_launch_parameters.h"

typedef struct {
    int16_t x, y;
} int16x2_t;

template<typename Type, int bit_depth, bool target_y>
__global__ void kernel_delogo(
    uint8_t *__restrict__ pFrame, const int framePitch, const int width, const int height,
    uint8_t *__restrict__ pLogo, const int logo_pitch, const int logo_x, const int logo_y, const int logo_width, const int logo_height, const float logo_depth_mul_fade) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    const float nv12_2_yc48_mul = (target_y) ? 1197.0f / (1<<(bit_depth-2)) : 4681.0f / (1<<bit_depth);
    const float nv12_2_yc48_sub = (target_y) ? 299.0f : 599332.0f / 256.0f;
    const float yc48_2_nv12_mul = (target_y) ?   219.0f / (1<<(20-bit_depth)) :    14.0f / (1<<(16-bit_depth));
    const float yc48_2_nv12_add = (target_y) ? 65919.0f / (1<<(20-bit_depth)) : 32900.0f / (1<<(16-bit_depth));
    if (x < logo_width && y < logo_height && (x + logo_x) < width && (y + logo_y) < height) {
        //ロゴ情報取り出し
        const int16x2_t logo_data = *(int16x2_t *)(&pLogo[y * logo_pitch + x * sizeof(int16x2_t)]);
        float logo_dp = (float)logo_data.x;
        float logo    = (float)logo_data.y;

        logo_dp = (logo_dp * logo_depth_mul_fade) * (1.0f / (float)(128 * LOGO_FADE_MAX));
        //0での除算回避
        if (logo_dp == LOGO_MAX_DP) {
            logo_dp -= 1.0f;
        }

        //画素データ取り出し
        pFrame += (y + logo_y) * framePitch + (x + logo_x) * sizeof(Type);
        Type pixel_yuv = *(Type *)pFrame;

        //nv12->yc48
        float pixel_yc48 = (float)pixel_yuv * nv12_2_yc48_mul - nv12_2_yc48_sub;

        //ロゴ除去
        float yc = (pixel_yc48 * (float)LOGO_MAX_DP - logo * logo_dp + ((float)LOGO_MAX_DP - logo_dp) * 0.5f) * __frcp_rn((float)LOGO_MAX_DP - logo_dp);

        *(Type *)pFrame = (Type)clamp((yc * yc48_2_nv12_mul + yc48_2_nv12_add + 0.5f), 0.0f, (float)(1<<bit_depth)-0.1f);
    }
}

template<typename Type, int bit_depth, bool target_y>
__global__ void kernel_delogo_add(
    uint8_t *__restrict__ pFrame, const int framePitch, const int width, const int height,
    uint8_t *__restrict__ pLogo, const int logo_pitch, const int logo_x, const int logo_y, const int logo_width, const int logo_height, const float logo_depth_mul_fade) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    const float nv12_2_yc48_mul = (target_y) ? 1197.0f / (1<<(bit_depth-2)) : 4681.0f / (1<<bit_depth);
    const float nv12_2_yc48_sub = (target_y) ? 299.0f : 599332.0f / 256.0f;
    const float yc48_2_nv12_mul = (target_y) ?   219.0f / (1<<(20-bit_depth)) :    14.0f / (1<<(16-bit_depth));
    const float yc48_2_nv12_add = (target_y) ? 65919.0f / (1<<(20-bit_depth)) : 32900.0f / (1<<(16-bit_depth));
    if (x < logo_width && y < logo_height && (x + logo_x) < width && (y + logo_y) < height) {
        //ロゴ情報取り出し
        const int16x2_t logo_data = *(int16x2_t *)(&pLogo[y * logo_pitch + x * sizeof(int16x2_t)]);
        float logo_dp = (float)logo_data.x;
        float logo    = (float)logo_data.y;

        logo_dp = (logo_dp * logo_depth_mul_fade) * (1.0f / (float)(128 * LOGO_FADE_MAX));

        //画素データ取り出し
        pFrame += (y + logo_y) * framePitch + (x + logo_x) * sizeof(Type);
        Type pixel_yuv = *(Type *)pFrame;

        //nv12->yc48
        float pixel_yc48 = (float)pixel_yuv * nv12_2_yc48_mul - nv12_2_yc48_sub;

        //ロゴ付加
        float yc = (pixel_yc48 * ((float)LOGO_MAX_DP - logo_dp) + logo * logo_dp) * (1.0f / (float)LOGO_MAX_DP);

        *(Type *)pFrame = (Type)clamp((yc * yc48_2_nv12_mul + yc48_2_nv12_add + 0.5f), 0.0f, (float)(1<<bit_depth)-0.1f);
    }
}

template<typename Type, int bit_depth, bool target_y>
void run_delogo(FrameInfo *pFrame, const ProcessDataDelogo *pDelego, int target_yuv, int mode) {
    dim3 blockSize(32, 4);
    dim3 gridSize(divCeil(pDelego->width, blockSize.x), divCeil(pDelego->height, blockSize.y));
    uint8_t *dptr = (uint8_t *)pFrame->ptr;
    switch (target_yuv) {
    case LOGO__U:
    case LOGO_UV:
        dptr += pFrame->pitch * pFrame->height;
        break;
    case LOGO__V:
        dptr += pFrame->pitch * pFrame->height * 3 / 2;
        break;
    case LOGO__Y:
    default:
        break;
    }
    if (mode == DELOGO_MODE_ADD) {
        kernel_delogo_add<Type, bit_depth, target_y><<<gridSize, blockSize>>>(
            dptr,
            pFrame->pitch,
            pFrame->width,
            (target_y) ? pFrame->height : (pFrame->height>>1),
            (uint8_t *)pDelego->pDevLogo->frame.ptr, pDelego->pDevLogo->frame.pitch,
            pDelego->i_start, pDelego->j_start, pDelego->width, pDelego->height, (float)pDelego->depth * pDelego->fade);
    } else {
        kernel_delogo<Type, bit_depth, target_y><<<gridSize, blockSize>>>(
            dptr,
            pFrame->pitch,
            pFrame->width,
            (target_y) ? pFrame->height : (pFrame->height>>1),
            (uint8_t *)pDelego->pDevLogo->frame.ptr, pDelego->pDevLogo->frame.pitch,
            pDelego->i_start, pDelego->j_start, pDelego->width, pDelego->height, (float)pDelego->depth * pDelego->fade);
    }
}

NVENCSTATUS NVEncFilterDelogo::delogoY(FrameInfo *pFrame) {
    //Y
    static const std::map<RGY_CSP, void(*)(FrameInfo *pFrame, const ProcessDataDelogo *pDelego, int target_yuv, int mode)> delogo_y_list = {
        { RGY_CSP_YV12,      run_delogo<uint8_t,   8, true> },
        { RGY_CSP_YV12_16,   run_delogo<uint16_t, 16, true> },
        { RGY_CSP_YV12_14,   run_delogo<uint16_t, 14, true> },
        { RGY_CSP_YV12_12,   run_delogo<uint16_t, 12, true> },
        { RGY_CSP_YV12_10,   run_delogo<uint16_t, 10, true> },
        { RGY_CSP_YV12_09,   run_delogo<uint16_t,  9, true> },
        { RGY_CSP_NV12,      run_delogo<uint8_t,   8, true> },
        { RGY_CSP_P010,      run_delogo<uint16_t, 16, true> },
        { RGY_CSP_YUV444,    run_delogo<uint8_t,   8, true> },
        { RGY_CSP_YUV444_16, run_delogo<uint16_t, 16, true> },
        { RGY_CSP_YUV444_14, run_delogo<uint16_t, 14, true> },
        { RGY_CSP_YUV444_12, run_delogo<uint16_t, 12, true> },
        { RGY_CSP_YUV444_10, run_delogo<uint16_t, 10, true> },
        { RGY_CSP_YUV444_09, run_delogo<uint16_t,  9, true> },
    };
    auto pDelogoParam = std::dynamic_pointer_cast<NVEncFilterParamDelogo>(m_pParam);
    if (!pDelogoParam) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return NV_ENC_ERR_INVALID_PARAM;
    }
    delogo_y_list.at(pFrame->csp)(pFrame, &m_sProcessData[LOGO__Y], LOGO__Y, pDelogoParam->mode);
    auto cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) {
        AddMessage(RGY_LOG_ERROR, _T("error at delogo_uv_list(%s): %s.\n"),
            RGY_CSP_NAMES[pFrame->csp],
            char_to_tstring(cudaGetErrorString(cudaerr)).c_str());
        return NV_ENC_ERR_INVALID_CALL;
    }
    return NV_ENC_SUCCESS;
}

NVENCSTATUS NVEncFilterDelogo::delogoUV(FrameInfo *pFrame) {
    const auto supportedCspYV12   = make_array<RGY_CSP>(RGY_CSP_YV12, RGY_CSP_YV12_09, RGY_CSP_YV12_10, RGY_CSP_YV12_12, RGY_CSP_YV12_14, RGY_CSP_YV12_16);
    //const auto supportedCspYUV444 = make_array<RGY_CSP>(RGY_CSP_YUV444, RGY_CSP_YUV444_09, RGY_CSP_YUV444_10, RGY_CSP_YUV444_12, RGY_CSP_YUV444_14, RGY_CSP_YUV444_16);
    //UV
    static const std::map<RGY_CSP, void (*)(FrameInfo *pFrame, const ProcessDataDelogo *pDelego, int target_yuv, int mode)> delogo_uv_list = {
        { RGY_CSP_YV12,    run_delogo<uint8_t,   8, false> },
        { RGY_CSP_YV12_16, run_delogo<uint16_t, 16, false> },
        { RGY_CSP_YV12_14, run_delogo<uint16_t, 14, false> },
        { RGY_CSP_YV12_12, run_delogo<uint16_t, 12, false> },
        { RGY_CSP_YV12_10, run_delogo<uint16_t, 10, false> },
        { RGY_CSP_YV12_09, run_delogo<uint16_t,  9, false> },
        { RGY_CSP_NV12,    run_delogo<uint8_t,   8, false> },
        { RGY_CSP_P010,    run_delogo<uint16_t, 16, false> },
    };
    auto pDelogoParam = std::dynamic_pointer_cast<NVEncFilterParamDelogo>(m_pParam);
    if (!pDelogoParam) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return NV_ENC_ERR_INVALID_PARAM;
    }
    if (delogo_uv_list.count(pFrame->csp) == 0) {
        AddMessage(RGY_LOG_ERROR, _T("unsupported csp for delogo: %s.\n"), RGY_CSP_NAMES[pFrame->csp]);
        return NV_ENC_ERR_UNIMPLEMENTED;
    }
    if (std::find(supportedCspYV12.begin(), supportedCspYV12.end(), pFrame->csp) != supportedCspYV12.end()) {
        //YV12
        delogo_uv_list.at(pFrame->csp)(pFrame, &m_sProcessData[LOGO__U], LOGO__U, pDelogoParam->mode);
        auto cudaerr = cudaGetLastError();
        if (cudaerr != cudaSuccess) {
            AddMessage(RGY_LOG_ERROR, _T("error at delogo_uv_list(%s): %s.\n"),
                RGY_CSP_NAMES[pFrame->csp],
                char_to_tstring(cudaGetErrorString(cudaerr)).c_str());
            return NV_ENC_ERR_INVALID_CALL;
        }
        delogo_uv_list.at(pFrame->csp)(pFrame, &m_sProcessData[LOGO__V], LOGO__V, pDelogoParam->mode);
        cudaerr = cudaGetLastError();
        if (cudaerr != cudaSuccess) {
            AddMessage(RGY_LOG_ERROR, _T("error at delogo_uv_list(%s): %s.\n"),
                RGY_CSP_NAMES[pFrame->csp],
                char_to_tstring(cudaGetErrorString(cudaerr)).c_str());
            return NV_ENC_ERR_INVALID_CALL;
        }
    } else {
        //NV12
        delogo_uv_list.at(pFrame->csp)(pFrame, &m_sProcessData[LOGO_UV], LOGO_UV, pDelogoParam->mode);
        auto cudaerr = cudaGetLastError();
        if (cudaerr != cudaSuccess) {
            AddMessage(RGY_LOG_ERROR, _T("error at delogo_uv_list(%s): %s.\n"),
                RGY_CSP_NAMES[pFrame->csp],
                char_to_tstring(cudaGetErrorString(cudaerr)).c_str());
            return NV_ENC_ERR_INVALID_CALL;
        }
    }
    return NV_ENC_SUCCESS;

}

NVEncFilterDelogo::NVEncFilterDelogo() {
    m_sFilterName = _T("delogo");
}

NVEncFilterDelogo::~NVEncFilterDelogo() {
    close();
}

int NVEncFilterDelogo::readLogoFile() {
    int sts = 0;

    auto pDelogoParam = std::dynamic_pointer_cast<NVEncFilterParamDelogo>(m_pParam);
    if (!pDelogoParam) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return NV_ENC_ERR_INVALID_PARAM;
    }
    if (pDelogoParam->logoFilePath == nullptr) {
        return 1;
    }
    auto file_deleter = [](FILE *fp) {
        fclose(fp);
    };
    unique_ptr<FILE, decltype(file_deleter)> fp(_tfopen(pDelogoParam->logoFilePath, _T("rb")), file_deleter);
    if (fp.get() == NULL) {
        AddMessage(RGY_LOG_ERROR, _T("could not open logo file \"%s\".\n"), pDelogoParam->logoFilePath);
        return 1;
    }
    // ファイルヘッダ取得
    int logo_header_ver = 0;
    LOGO_FILE_HEADER logo_file_header ={ 0 };
    if (sizeof(logo_file_header) != fread(&logo_file_header, 1, sizeof(logo_file_header), fp.get())) {
        AddMessage(RGY_LOG_ERROR, _T("invalid logo file.\n"));
        sts = 1;
    } else if (0 == (logo_header_ver = get_logo_file_header_ver(&logo_file_header))) {
        AddMessage(RGY_LOG_ERROR, _T("invalid logo file.\n"));
        sts = 1;
    } else {
        const size_t logo_header_size = (logo_header_ver == 2) ? sizeof(LOGO_HEADER) : sizeof(LOGO_HEADER_OLD);
        const int logonum = SWAP_ENDIAN(logo_file_header.logonum.l);
        m_sLogoDataList.resize(logonum);

        for (int i = 0; i < logonum; i++) {
            memset(&m_sLogoDataList[i], 0, sizeof(m_sLogoDataList[i]));
            if (logo_header_size != fread(&m_sLogoDataList[i].header, 1, logo_header_size, fp.get())) {
                AddMessage(RGY_LOG_ERROR, _T("invalid logo file.\n"));
                sts = 1;
                break;
            }
            if (logo_header_ver == 1) {
                convert_logo_header_v1_to_v2(&m_sLogoDataList[i].header);
            }

            const auto logoPixelBytes = logo_pixel_size(&m_sLogoDataList[i].header);

            // メモリ確保
            m_sLogoDataList[i].logoPixel.resize(logoPixelBytes / sizeof(m_sLogoDataList[i].logoPixel[0]), { 0 });

            if (logoPixelBytes != (int)fread(m_sLogoDataList[i].logoPixel.data(), 1, logoPixelBytes, fp.get())) {
                AddMessage(RGY_LOG_ERROR, _T("invalid logo file.\n"));
                sts = 1;
                break;
            }
        }
    }
    return sts;
}

std::string NVEncFilterDelogo::logoNameList() {
    std::string strlist;
    for (int i = 0; i < (int)m_sLogoDataList.size(); i++) {
        strlist += strsprintf("%3d: %s\n", i+1, m_sLogoDataList[i].header.name);
    }
    return strlist;
}

int NVEncFilterDelogo::getLogoIdx(const std::string& logoName) {
    int idx = LOGO_AUTO_SELECT_INVALID;
    for (int i = 0; i < (int)m_sLogoDataList.size(); i++) {
        if (0 == strcmp(m_sLogoDataList[i].header.name, logoName.c_str())) {
            idx = i;
            break;
        }
    }
    return idx;
}

int NVEncFilterDelogo::selectLogo(const TCHAR *selectStr) {
    if (selectStr == nullptr) {
        if (m_sLogoDataList.size() > 1) {
            AddMessage(RGY_LOG_ERROR, _T("--vpp-delogo-select option is required to select logo from logo pack.\n"));
            AddMessage(RGY_LOG_ERROR, char_to_tstring(logoNameList()));
            return LOGO_AUTO_SELECT_INVALID;
        }
        return 0;
    }

    //ロゴ名として扱い、インデックスを取得
    {
        int idx = getLogoIdx(tchar_to_string(selectStr));
        if (idx != LOGO_AUTO_SELECT_INVALID) {
            return idx;
        }
    }
    //数字として扱い、インデックスを取得
    {
        TCHAR *eptr = nullptr;
        long j = _tcstol(selectStr, &eptr, 10);
        if (j != 0
            && (eptr == nullptr || eptr == selectStr + _tcslen(selectStr))
            && 0 < j && j <= (int)m_sLogoDataList.size())
            return j-1;
    }

    //自動ロゴ選択ファイルか?
    std::string logoName = GetFullPath(tchar_to_string(selectStr).c_str());
    if (!PathFileExists(selectStr)) {
        AddMessage(RGY_LOG_ERROR,
            _T("--vpp-delogo-select option has invalid param.\n")
            _T("Please set logo name or logo index (starting from 1),\n")
            _T("or auto select file.\n"));
        return LOGO_AUTO_SELECT_INVALID;
    }
    //自動選択キー
    int count = 0;
    for (;; count++) {
        char buf[512] ={ 0 };
        GetPrivateProfileStringA("LOGO_AUTO_SELECT", strsprintf("logo%d", count+1).c_str(), "", buf, sizeof(buf), logoName.c_str());
        if (strlen(buf) == 0)
            break;
    }
    if (count == 0) {
        AddMessage(RGY_LOG_ERROR, _T("could not find any key to auto select from \"%s\".\n"), selectStr);
        return LOGO_AUTO_SELECT_INVALID;
    }
    std::vector<LOGO_SELECT_KEY> logoAutoSelectKeys;
    logoAutoSelectKeys.reserve(count);
    for (int i = 0; i < count; i++) {
        LOGO_SELECT_KEY selectKey;
        char buf[512] ={ 0 };
        GetPrivateProfileStringA("LOGO_AUTO_SELECT", strsprintf("logo%d", i+1).c_str(), "", buf, sizeof(buf), logoName.c_str());
        char *ptr = strchr(buf, ',');
        if (ptr != NULL) {
            ptr[0] = '\0';
            selectKey.key = buf;
            strcpy_s(selectKey.logoname, ptr+1);
            logoAutoSelectKeys.push_back(std::move(selectKey));
        }
    }
    auto pDelogoParam = std::dynamic_pointer_cast<NVEncFilterParamDelogo>(m_pParam);
    if (!pDelogoParam) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return NV_ENC_ERR_INVALID_PARAM;
    }
    for (const auto& selectKey : logoAutoSelectKeys) {
        if (NULL != _tcsstr(pDelogoParam->inputFileName, char_to_tstring(selectKey.key.c_str()).c_str())) {
            logoName = selectKey.logoname;
            return getLogoIdx(logoName);
        }
    }
    return LOGO_AUTO_SELECT_NOHIT;
}

NVENCSTATUS NVEncFilterDelogo::init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    NVENCSTATUS sts = NV_ENC_SUCCESS;
    m_pPrintMes = pPrintMes;
    auto pDelogoParam = std::dynamic_pointer_cast<NVEncFilterParamDelogo>(pParam);
    if (!pDelogoParam) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return NV_ENC_ERR_INVALID_PARAM;
    }
    //delogoは常に元のフレームを書き換え
    if (!pDelogoParam->bOutOverwrite) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid param, delogo will overwrite input frame.\n"));
        return NV_ENC_ERR_INVALID_PARAM;
    }
    pDelogoParam->frameOut = pDelogoParam->frameIn;
    //コピーを保存
    m_pParam = pDelogoParam;

    //パラメータチェック
    if (readLogoFile()) {
        return NV_ENC_ERR_INVALID_PARAM;
    }
    if (0 > (m_nLogoIdx = selectLogo(pDelogoParam->logoSelect))) {
        if (m_nLogoIdx == LOGO_AUTO_SELECT_NOHIT) {
            AddMessage(RGY_LOG_ERROR, _T("no logo was selected by auto select \"%s\".\n"), pDelogoParam->logoSelect);
            return NV_ENC_ERR_INVALID_PARAM;
        } else {
            AddMessage(RGY_LOG_ERROR, _T("could not select logo by \"%s\".\n"), pDelogoParam->logoSelect);
            AddMessage(RGY_LOG_ERROR, char_to_tstring(logoNameList()));
            return NV_ENC_ERR_INVALID_PARAM;
        }
    }

    auto& logoData = m_sLogoDataList[m_nLogoIdx];
    if (pDelogoParam->posX || pDelogoParam->posY) {
        LogoData origData;
        origData.header = logoData.header;
        origData.logoPixel = logoData.logoPixel;

        logoData.logoPixel = std::vector<LOGO_PIXEL>((logoData.header.w + 1) * (logoData.header.h + 1), { 0 });

        create_adj_exdata(logoData.logoPixel.data(), &logoData.header, origData.logoPixel.data(), &origData.header, pDelogoParam->posX, pDelogoParam->posY);
    }
    const int frameWidth  = pDelogoParam->frameIn.width;
    const int frameHeight = pDelogoParam->frameIn.height;

    m_sProcessData[LOGO__Y].offset[0] = pDelogoParam->Y  << 4;
    m_sProcessData[LOGO__Y].offset[1] = pDelogoParam->Y  << 4;
    m_sProcessData[LOGO_UV].offset[0] = pDelogoParam->Cb << 4;
    m_sProcessData[LOGO_UV].offset[1] = pDelogoParam->Cr << 4;
    m_sProcessData[LOGO__U].offset[0] = pDelogoParam->Cb << 4;
    m_sProcessData[LOGO__U].offset[1] = pDelogoParam->Cb << 4;
    m_sProcessData[LOGO__V].offset[0] = pDelogoParam->Cr << 4;
    m_sProcessData[LOGO__V].offset[1] = pDelogoParam->Cr << 4;

    m_sProcessData[LOGO__Y].fade = 256;
    m_sProcessData[LOGO_UV].fade = 256;
    m_sProcessData[LOGO__U].fade = 256;
    m_sProcessData[LOGO__V].fade = 256;

    m_sProcessData[LOGO__Y].depth = pDelogoParam->depth;
    m_sProcessData[LOGO_UV].depth = pDelogoParam->depth;
    m_sProcessData[LOGO__U].depth = pDelogoParam->depth;
    m_sProcessData[LOGO__V].depth = pDelogoParam->depth;

    m_sProcessData[LOGO__Y].i_start = (std::min)(logoData.header.x & ~63, frameWidth);
    m_sProcessData[LOGO__Y].width   = (((std::min)(logoData.header.x + logoData.header.w, frameWidth) + 63) & ~63) - m_sProcessData[LOGO__Y].i_start;
    m_sProcessData[LOGO_UV].i_start = m_sProcessData[LOGO__Y].i_start;
    m_sProcessData[LOGO_UV].width   = m_sProcessData[LOGO__Y].width;
    m_sProcessData[LOGO__U].i_start = m_sProcessData[LOGO__Y].i_start >> 1;
    m_sProcessData[LOGO__U].width   = m_sProcessData[LOGO__Y].width >> 1;
    m_sProcessData[LOGO__V].i_start = m_sProcessData[LOGO__U].i_start;
    m_sProcessData[LOGO__V].width   = m_sProcessData[LOGO__U].width;
    const int yWidthOffset = logoData.header.x - m_sProcessData[LOGO__Y].i_start;

    m_sProcessData[LOGO__Y].j_start = (std::min)((int)logoData.header.y, frameHeight);
    m_sProcessData[LOGO__Y].height  = (std::min)(logoData.header.y + logoData.header.h, frameHeight) - m_sProcessData[LOGO__Y].j_start;
    m_sProcessData[LOGO_UV].j_start = logoData.header.y >> 1;
    m_sProcessData[LOGO_UV].height  = (((logoData.header.y + logoData.header.h + 1) & ~1) - (m_sProcessData[LOGO_UV].j_start << 1)) >> 1;
    m_sProcessData[LOGO__U].j_start = m_sProcessData[LOGO_UV].j_start;
    m_sProcessData[LOGO__U].height  = m_sProcessData[LOGO_UV].height;
    m_sProcessData[LOGO__V].j_start = m_sProcessData[LOGO__U].j_start;
    m_sProcessData[LOGO__V].height  = m_sProcessData[LOGO__U].height;

    if (logoData.header.x >= frameWidth || logoData.header.y >= frameHeight) {
        AddMessage(RGY_LOG_ERROR, _T("\"%s\" was not included in frame size %dx%d.\ndelogo disabled.\n"), pDelogoParam->logoSelect, frameWidth, frameHeight);
        AddMessage(RGY_LOG_ERROR, _T("logo pos x=%d, y=%d, including pos offset value %d:%d.\n"), logoData.header.x, logoData.header.y, pDelogoParam->posX, pDelogoParam->posY);
        return NV_ENC_ERR_INVALID_PARAM;
    }

    m_sProcessData[LOGO__Y].pLogoPtr.reset((int16_t *)_aligned_malloc(sizeof(int16_t) * 2 * m_sProcessData[LOGO__Y].width * m_sProcessData[LOGO__Y].height, 32));
    m_sProcessData[LOGO_UV].pLogoPtr.reset((int16_t *)_aligned_malloc(sizeof(int16_t) * 2 * m_sProcessData[LOGO_UV].width * m_sProcessData[LOGO_UV].height, 32));
    m_sProcessData[LOGO__U].pLogoPtr.reset((int16_t *)_aligned_malloc(sizeof(int16_t) * 2 * m_sProcessData[LOGO__U].width * m_sProcessData[LOGO__U].height, 32));
    m_sProcessData[LOGO__V].pLogoPtr.reset((int16_t *)_aligned_malloc(sizeof(int16_t) * 2 * m_sProcessData[LOGO__V].width * m_sProcessData[LOGO__V].height, 32));

    memset(m_sProcessData[LOGO__Y].pLogoPtr.get(), 0, sizeof(int16_t) * 2 * m_sProcessData[LOGO__Y].width * m_sProcessData[LOGO__Y].height);
    memset(m_sProcessData[LOGO_UV].pLogoPtr.get(), 0, sizeof(int16_t) * 2 * m_sProcessData[LOGO_UV].width * m_sProcessData[LOGO_UV].height);
    memset(m_sProcessData[LOGO__U].pLogoPtr.get(), 0, sizeof(int16_t) * 2 * m_sProcessData[LOGO__U].width * m_sProcessData[LOGO__U].height);
    memset(m_sProcessData[LOGO__V].pLogoPtr.get(), 0, sizeof(int16_t) * 2 * m_sProcessData[LOGO__V].width * m_sProcessData[LOGO__V].height);

    //まず輝度成分をコピーしてしまう
    for (int j = 0; j < m_sProcessData[LOGO__Y].height; j++) {
        //輝度成分はそのままコピーするだけ
        for (int i = 0; i < logoData.header.w; i++) {
            int16x2_t logoY = *(int16x2_t *)&logoData.logoPixel[j * logoData.header.w + i].dp_y;
            ((int16x2_t *)m_sProcessData[LOGO__Y].pLogoPtr.get())[j * m_sProcessData[LOGO__Y].width + i + yWidthOffset] = logoY;
        }
    }
    //まずは4:4:4->4:2:0処理時に端を気にしなくていいよう、縦横ともに2の倍数となるよう拡張する
    //CbCrの順番に並べていく
    //0で初期化しておく
    std::vector<int16x2_t> bufferCbCr444ForShrink(2 * m_sProcessData[LOGO_UV].height * 2 * m_sProcessData[LOGO__Y].width, { 0, 0 });
    int j_src = 0; //読み込み側の行
    int j_dst = 0; //書き込み側の行
    auto copyUVLineForShrink = [&]() {
        for (int i = 0; i < logoData.header.w; i++) {
            int16x2_t logoCb = *(int16x2_t *)&logoData.logoPixel[j_src * logoData.header.w + i].dp_cb;
            int16x2_t logoCr = *(int16x2_t *)&logoData.logoPixel[j_src * logoData.header.w + i].dp_cr;
            bufferCbCr444ForShrink[(j_dst * m_sProcessData[LOGO_UV].width + i + yWidthOffset) * 2 + 0] = logoCb;
            bufferCbCr444ForShrink[(j_dst * m_sProcessData[LOGO_UV].width + i + yWidthOffset) * 2 + 1] = logoCr;
        }
        if (yWidthOffset & 1) {
            //奇数列はじまりなら、それをその前の偶数列に拡張する
            int16x2_t logoCb = *(int16x2_t *)&bufferCbCr444ForShrink[(j_dst * m_sProcessData[LOGO_UV].width + 0 + yWidthOffset) * 2 + 0];
            int16x2_t logoCr = *(int16x2_t *)&bufferCbCr444ForShrink[(j_dst * m_sProcessData[LOGO_UV].width + 0 + yWidthOffset) * 2 + 1];
            bufferCbCr444ForShrink[(j_dst * m_sProcessData[LOGO_UV].width + 0 + yWidthOffset - 1) * 2 + 0] = logoCb;
            bufferCbCr444ForShrink[(j_dst * m_sProcessData[LOGO_UV].width + 0 + yWidthOffset - 1) * 2 + 1] = logoCr;
        }
        if ((yWidthOffset + logoData.header.w) & 1) {
            //偶数列おわりなら、それをその次の奇数列に拡張する
            int16x2_t logoCb = *(int16x2_t *)&bufferCbCr444ForShrink[(j_dst * m_sProcessData[LOGO_UV].width + logoData.header.w + yWidthOffset) * 2 + 0];
            int16x2_t logoCr = *(int16x2_t *)&bufferCbCr444ForShrink[(j_dst * m_sProcessData[LOGO_UV].width + logoData.header.w + yWidthOffset) * 2 + 1];
            bufferCbCr444ForShrink[(j_dst * m_sProcessData[LOGO_UV].width + logoData.header.w + yWidthOffset + 1) * 2 + 0] = logoCb;
            bufferCbCr444ForShrink[(j_dst * m_sProcessData[LOGO_UV].width + logoData.header.w + yWidthOffset + 1) * 2 + 1] = logoCr;
        }
    };
    if (logoData.header.y & 1) {
        copyUVLineForShrink();
        j_dst++; //書き込み側は1行進める
    }
    for (; j_src < logoData.header.h; j_src++, j_dst++) {
        copyUVLineForShrink();
    }
    if ((logoData.header.y + logoData.header.h) & 1) {
        j_src--; //読み込み側は1行戻る
        copyUVLineForShrink();
    }

    //実際に縮小処理を行う
    //2x2->1x1の処理なのでインクリメントはそれぞれ2ずつ
    for (int j = 0; j < m_sProcessData[LOGO__Y].height; j += 2) {
        for (int i = 0; i < m_sProcessData[LOGO_UV].width; i += 2) {
            int16x2_t logoCb0 = bufferCbCr444ForShrink[((j + 0) * m_sProcessData[LOGO_UV].width + i + 0) * 2 + 0];
            int16x2_t logoCr0 = bufferCbCr444ForShrink[((j + 0) * m_sProcessData[LOGO_UV].width + i + 0) * 2 + 1];
            int16x2_t logoCb1 = bufferCbCr444ForShrink[((j + 0) * m_sProcessData[LOGO_UV].width + i + 1) * 2 + 0];
            int16x2_t logoCr1 = bufferCbCr444ForShrink[((j + 0) * m_sProcessData[LOGO_UV].width + i + 1) * 2 + 1];
            int16x2_t logoCb2 = bufferCbCr444ForShrink[((j + 1) * m_sProcessData[LOGO_UV].width + i + 0) * 2 + 0];
            int16x2_t logoCr2 = bufferCbCr444ForShrink[((j + 1) * m_sProcessData[LOGO_UV].width + i + 0) * 2 + 1];
            int16x2_t logoCb3 = bufferCbCr444ForShrink[((j + 1) * m_sProcessData[LOGO_UV].width + i + 1) * 2 + 0];
            int16x2_t logoCr3 = bufferCbCr444ForShrink[((j + 1) * m_sProcessData[LOGO_UV].width + i + 1) * 2 + 1];

            int16x2_t logoCb, logoCr;
            logoCb.x = (logoCb0.x + logoCb1.x + logoCb2.x + logoCb3.x + 2) >> 2;
            logoCb.y = (logoCb0.y + logoCb1.y + logoCb2.y + logoCb3.y + 2) >> 2;
            logoCr.x = (logoCr0.x + logoCr1.x + logoCr2.x + logoCr3.x + 2) >> 2;
            logoCr.y = (logoCr0.y + logoCr1.y + logoCr2.y + logoCr3.y + 2) >> 2;

            //単純平均により4:4:4->4:2:0に
            ((int16x2_t *)m_sProcessData[LOGO_UV].pLogoPtr.get())[(j >> 1) * m_sProcessData[LOGO_UV].width * 1 + (i >> 1) * 2 + 0] = logoCb;
            ((int16x2_t *)m_sProcessData[LOGO_UV].pLogoPtr.get())[(j >> 1) * m_sProcessData[LOGO_UV].width * 1 + (i >> 1) * 2 + 1] = logoCr;
            ((int16x2_t *)m_sProcessData[LOGO__U].pLogoPtr.get())[(j >> 1) * m_sProcessData[LOGO__U].width * 1 + (i >> 1) * 1    ] = logoCb;
            ((int16x2_t *)m_sProcessData[LOGO__V].pLogoPtr.get())[(j >> 1) * m_sProcessData[LOGO__V].width * 1 + (i >> 1) * 1    ] = logoCr;
        }
    }

    for (uint32_t i = 0; i < _countof(m_sProcessData); i++) {
        unique_ptr<CUFrameBuf> uptr(new CUFrameBuf(m_sProcessData[i].width * sizeof(int16x2_t), m_sProcessData[i].height));
        auto cudaerr = uptr->alloc();
        if (cudaerr != cudaSuccess) {
            m_pFrameBuf.clear();
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate memory for logo data %d: %s.\n"),
                i, char_to_tstring(cudaGetErrorString(cudaerr)).c_str());
            return NV_ENC_ERR_OUT_OF_MEMORY;
        }
        m_sProcessData[i].pDevLogo = std::move(uptr);
        //ロゴデータをGPUに転送
        cudaerr = cudaMemcpy2DAsync(m_sProcessData[i].pDevLogo->frame.ptr, m_sProcessData[i].pDevLogo->frame.pitch,
            (void *)m_sProcessData[i].pLogoPtr.get(), m_sProcessData[i].width * sizeof(int16x2_t),
            m_sProcessData[i].width * sizeof(int16x2_t), m_sProcessData[i].height, cudaMemcpyHostToDevice);
        if (cudaerr != cudaSuccess) {
            AddMessage(RGY_LOG_ERROR, _T("error at sending logo data %d cudaMemcpy2DAsync(%s): %s.\n"),
                i,
                getCudaMemcpyKindStr(cudaMemcpyHostToDevice),
                char_to_tstring(cudaGetErrorString(cudaerr)).c_str());
        }
    }

    //フィルタ情報の調整
    std::string str = "";
    switch (pDelogoParam->mode) {
    case DELOGO_MODE_ADD:
        str += ", add";
        break;
    case DELOGO_MODE_REMOVE:
    default:
        break;
    }
    if (pDelogoParam->posX || pDelogoParam->posY) {
        str += strsprintf(", pos=%d:%d", pDelogoParam->posX, pDelogoParam->posY);
    }
    if (pDelogoParam->depth != FILTER_DEFAULT_DELOGO_DEPTH) {
        str += strsprintf(", dpth=%d", pDelogoParam->depth);
    }
    if (pDelogoParam->Y || pDelogoParam->Cb || pDelogoParam->Cr) {
        str += strsprintf(", YCbCr=%d:%d:%d", pDelogoParam->Y, pDelogoParam->Cb, pDelogoParam->Cr);
    }
    m_sFilterInfo = char_to_tstring("delogo: " + std::string(logoData.header.name) + str);

    return sts;
}

NVENCSTATUS NVEncFilterDelogo::run_filter(const FrameInfo *pInputFrame, FrameInfo **ppOutputFrames, int *pOutputFrameNum) {
    NVENCSTATUS sts = NV_ENC_SUCCESS;

    if (pInputFrame->ptr == nullptr) {
        return sts;
    }

    *pOutputFrameNum = 1;
    if (ppOutputFrames[0] == nullptr) {
        AddMessage(RGY_LOG_ERROR, _T("ppOutputFrames[0] must be set.\n"));
        return NV_ENC_ERR_UNSUPPORTED_PARAM;
    }
    if (!ppOutputFrames[0]->deivce_mem) {
        AddMessage(RGY_LOG_ERROR, _T("only supported on device memory.\n"));
        return NV_ENC_ERR_UNSUPPORTED_PARAM;
    }
    const auto memcpyKind = getCudaMemcpyKind(ppOutputFrames[0]->deivce_mem, ppOutputFrames[0]->deivce_mem);
    if (memcpyKind != cudaMemcpyDeviceToDevice) {
        AddMessage(RGY_LOG_ERROR, _T("only supported on device memory.\n"));
        return NV_ENC_ERR_UNSUPPORTED_PARAM;
    }
    if (m_pParam->frameOut.csp != m_pParam->frameIn.csp) {
        AddMessage(RGY_LOG_ERROR, _T("csp does not match.\n"));
        return NV_ENC_ERR_UNSUPPORTED_PARAM;
    }

    if (NV_ENC_SUCCESS != (sts = delogoY(ppOutputFrames[0]))) {
        return sts;
    }

    if (NV_ENC_SUCCESS != (sts = delogoUV(ppOutputFrames[0]))) {
        return sts;
    }

    return sts;
}

void NVEncFilterDelogo::close() {
    m_pFrameBuf.clear();
    m_sLogoDataList.clear();
}
