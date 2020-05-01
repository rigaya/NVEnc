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
#include <algorithm>
#include "convert_csp.h"
#include "NVEncFilterSubburn.h"
#include "NVEncParam.h"
#pragma warning (push)
#pragma warning (disable: 4819)
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#pragma warning (pop)
#include "rgy_cuda_util.h"

#if ENABLE_AVSW_READER

static __device__ float lerpf(float a, float b, float c) {
    return a + (b - a) * c;
}

template<typename TypePixel, int bit_depth>
__inline__ __device__
TypePixel blend(TypePixel pix, uint8_t alpha, uint8_t val, float transparency_offset, float pix_offset, float contrast) {
    //alpha値は 0が透明, 255が不透明
    float subval = val * (1.0f / (float)(1 << 8));
    subval = contrast * (subval - 0.5f) + 0.5f + pix_offset;
    float ret = lerpf((float)pix, subval * (float)(1<<bit_depth), alpha * (1.0f / 255.0f) * (1.0f - transparency_offset));
    return (TypePixel)clamp(ret, 0.0f, (1<<bit_depth)-0.5f);
}

template<typename TypePixel2, int bit_depth>
__inline__ __device__
void blend(void *pix, const void *alpha, const void *val, float transparency_offset, float pix_offset, float contrast) {
    uchar2 a = *(uchar2 *)alpha;
    uchar2 v = *(uchar2 *)val;
    TypePixel2 p = *(TypePixel2 *)pix;
    p.x = blend<decltype(TypePixel2::x), bit_depth>(p.x, a.x, v.x, transparency_offset, pix_offset, contrast);
    p.y = blend<decltype(TypePixel2::x), bit_depth>(p.y, a.y, v.y, transparency_offset, pix_offset, contrast);
    *(TypePixel2 *)pix = p;
}

template<typename TypePixel, int bit_depth, bool yuv420>
__global__ void kernel_subburn(uint8_t *__restrict__ pPlaneY, uint8_t *__restrict__ pPlaneU, uint8_t *__restrict__ pPlaneV,
    const int pitchFrame,
    const uint8_t *__restrict__ pSubY, const uint8_t *__restrict__ pSubU, const uint8_t *__restrict__ pSubV, const uint8_t *__restrict__ pSubA,
    const int pitchSub,
    const int width, const int height, bool interlaced, float transparency_offset, float brightness, float contrast) {
    //縦横2x2pixelを1スレッドで処理する
    const int ix = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    const int iy = (blockIdx.y * blockDim.y + threadIdx.y) * 2;

    struct __align__(sizeof(TypePixel) * 2) TypePixel2 {
        TypePixel x, y;
    };
    if (ix < width && iy < height) {
        pPlaneY += iy * pitchFrame + ix * sizeof(TypePixel);
        pSubY   += iy * pitchSub + ix;
        pSubU   += iy * pitchSub + ix;
        pSubV   += iy * pitchSub + ix;
        pSubA   += iy * pitchSub + ix;

        blend<TypePixel2, bit_depth>(pPlaneY,              pSubA,            pSubY,            transparency_offset, brightness, contrast);
        blend<TypePixel2, bit_depth>(pPlaneY + pitchFrame, pSubA + pitchSub, pSubY + pitchSub, transparency_offset, brightness, contrast);

        if (yuv420) {
            pPlaneU += (iy>>1) * pitchFrame + (ix>>1) * sizeof(TypePixel);
            pPlaneV += (iy>>1) * pitchFrame + (ix>>1) * sizeof(TypePixel);
            uint8_t subU, subV, subA;
            if (interlaced) {
                if (((iy>>1) & 1) == 0) {
                    const int offset_y1 = (iy+2<height) ? pitchSub*2 : 0;
                    subU = (pSubU[0] * 3 + pSubU[offset_y1] + 2) >> 2;
                    subV = (pSubV[0] * 3 + pSubV[offset_y1] + 2) >> 2;
                    subA = (pSubA[0] * 3 + pSubA[offset_y1] + 2) >> 2;
                } else {
                    subU = (pSubU[-pitchSub] + pSubU[pitchSub] * 3 + 2) >> 2;
                    subV = (pSubV[-pitchSub] + pSubV[pitchSub] * 3 + 2) >> 2;
                    subA = (pSubA[-pitchSub] + pSubA[pitchSub] * 3 + 2) >> 2;
                }
            } else {
                subU = (pSubU[0] + pSubU[pitchSub] + 1) >> 1;
                subV = (pSubV[0] + pSubV[pitchSub] + 1) >> 1;
                subA = (pSubA[0] + pSubA[pitchSub] + 1) >> 1;
            }
            *(TypePixel *)pPlaneU = blend<TypePixel, bit_depth>(*(TypePixel *)pPlaneU, subA, subU, transparency_offset, 0.0f, 1.0f);
            *(TypePixel *)pPlaneV = blend<TypePixel, bit_depth>(*(TypePixel *)pPlaneV, subA, subV, transparency_offset, 0.0f, 1.0f);
        } else {
            pPlaneU += iy * pitchFrame + ix * sizeof(TypePixel);
            pPlaneV += iy * pitchFrame + ix * sizeof(TypePixel);
            blend<TypePixel2, bit_depth>(pPlaneU,              pSubA,            pSubU,            transparency_offset, 0.0f, 1.0f);
            blend<TypePixel2, bit_depth>(pPlaneU + pitchFrame, pSubA + pitchSub, pSubU + pitchSub, transparency_offset, 0.0f, 1.0f);
            blend<TypePixel2, bit_depth>(pPlaneV,              pSubA,            pSubV,            transparency_offset, 0.0f, 1.0f);
            blend<TypePixel2, bit_depth>(pPlaneV + pitchFrame, pSubA + pitchSub, pSubV + pitchSub, transparency_offset, 0.0f, 1.0f);
        }
    }
}

template<typename TypePixel, int bit_depth>
cudaError_t proc_frame(FrameInfo *pFrame,
    const FrameInfo *pSubImg,
    int pos_x, int pos_y,
    float transparency_offset, float brightness, float contrast,
    cudaStream_t stream) {
    //焼きこみフレームの範囲内に収まるようチェック
    const int burnWidth  = std::min((pos_x & ~1) + pSubImg->width,  pFrame->width)  - (pos_x & ~1);
    const int burnHeight = std::min((pos_y & ~1) + pSubImg->height, pFrame->height) - (pos_y & ~1);
    if (burnWidth < 0 || burnHeight < 0) {
        return cudaSuccess;
    }

    dim3 blockSize(32, 8);
    dim3 gridSize(divCeil(burnWidth, blockSize.x * 2), divCeil(burnHeight, blockSize.y * 2)); // 2x2pixel/thread
    auto planeFrameY = getPlane(pFrame, RGY_PLANE_Y);
    auto planeFrameU = getPlane(pFrame, RGY_PLANE_U);
    auto planeFrameV = getPlane(pFrame, RGY_PLANE_V);
    auto planeSubY = getPlane(pSubImg, RGY_PLANE_Y);
    auto planeSubU = getPlane(pSubImg, RGY_PLANE_U);
    auto planeSubV = getPlane(pSubImg, RGY_PLANE_V);
    auto planeSubA = getPlane(pSubImg, RGY_PLANE_A);

    const int frameOffsetByte = (pos_y & ~1) * pFrame->pitch + (pos_x & ~1) * sizeof(TypePixel);

    cudaError_t cudaerr = cudaSuccess;
    if (RGY_CSP_CHROMA_FORMAT[pFrame->csp] == RGY_CHROMAFMT_YUV420) {
        const int frameOffsetByteUV = (pos_y >> 1) * pFrame->pitch + (pos_x >> 1) * sizeof(TypePixel);
        kernel_subburn<TypePixel, bit_depth, true><<<gridSize, blockSize, 0, stream>>>(
            planeFrameY.ptr + frameOffsetByte,
            planeFrameU.ptr + frameOffsetByteUV,
            planeFrameV.ptr + frameOffsetByteUV,
            planeFrameY.pitch,
            planeSubY.ptr, planeSubU.ptr, planeSubV.ptr, planeSubA.ptr, planeSubY.pitch,
            burnWidth, burnHeight, interlaced(*pFrame), transparency_offset, brightness, contrast);
    } else {
        kernel_subburn<TypePixel, bit_depth, false><<<gridSize, blockSize, 0, stream>>>(
            planeFrameY.ptr + frameOffsetByte,
            planeFrameU.ptr + frameOffsetByte,
            planeFrameV.ptr + frameOffsetByte,
            planeFrameY.pitch,
            planeSubY.ptr, planeSubU.ptr, planeSubV.ptr, planeSubA.ptr, planeSubY.pitch,
            burnWidth, burnHeight, interlaced(*pFrame), transparency_offset, brightness, contrast);
    }
    cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess) {
        return cudaerr;
    }
    return cudaerr;
}

SubImageData NVEncFilterSubburn::textRectToImage(const ASS_Image *image, cudaStream_t stream) {
    //YUV420の関係で縦横2pixelずつ処理するので、2で割り切れている必要がある
    const int x_offset = ((image->dst_x % 2) != 0) ? 1 : 0;
    const int y_offset = ((image->dst_y % 2) != 0) ? 1 : 0;
    FrameInfo img;
    img.csp = RGY_CSP_YUVA444;
    img.width  = ALIGN(image->w + x_offset, 2);
    img.height = ALIGN(image->h + y_offset, 2);
    img.deivce_mem = false;
    img.picstruct = RGY_PICSTRUCT_FRAME;
    auto imgInfoEx = getFrameInfoExtra(&img);
    img.pitch = ALIGN(imgInfoEx.width_byte, 64);
    imgInfoEx = getFrameInfoExtra(&img);
    cudaMallocHost(&img.ptr, imgInfoEx.frame_size);
    unique_ptr<void, decltype(&cudaFreeHost)> bufCPU(img.ptr, cudaFreeHost);

    //とりあえずすべて0で初期化しておく
    //Alpha=0で透明なので都合がよい
    memset(img.ptr, 0, imgInfoEx.frame_size);

    auto planeY = getPlane(&img, RGY_PLANE_Y);
    auto planeU = getPlane(&img, RGY_PLANE_U);
    auto planeV = getPlane(&img, RGY_PLANE_V);
    auto planeA = getPlane(&img, RGY_PLANE_A);

    for (int j = 0; j < img.height; j++) {
        for (int i = 0; i < img.width; i++) {
            const int idx = j * img.pitch + i;
            planeU.ptr[idx] = 128;
            planeV.ptr[idx] = 128;
        }
    }

    const uint32_t subColor = image->color;
    const uint8_t subR = (uint8_t) (subColor >> 24);
    const uint8_t subG = (uint8_t)((subColor >> 16) & 0xff);
    const uint8_t subB = (uint8_t)((subColor >>  8) & 0xff);
    const uint8_t subA = (uint8_t)(255 - (subColor        & 0xff));

    const uint8_t subY = (uint8_t)clamp((( 66 * subR + 129 * subG +  25 * subB + 128) >> 8) +  16, 0, 255);
    const uint8_t subU = (uint8_t)clamp(((-38 * subR -  74 * subG + 112 * subB + 128) >> 8) + 128, 0, 255);
    const uint8_t subV = (uint8_t)clamp(((112 * subR -  94 * subG -  18 * subB + 128) >> 8) + 128, 0, 255);

    //YUVで字幕の画像データを構築
    for (int j = 0; j < image->h; j++) {
        for (int i = 0; i < image->w; i++) {
            const int src_idx = j * image->stride + i;
            const uint8_t alpha = image->bitmap[src_idx];

            const int dst_idx = (j+y_offset) * img.pitch + (i+x_offset);
            planeY.ptr[dst_idx] = subY;
            planeU.ptr[dst_idx] = subU;
            planeV.ptr[dst_idx] = subV;
            planeA.ptr[dst_idx] = (uint8_t)clamp(((int)subA * alpha) >> 8, 0, 255);
        }
    }
    //GPUへ転送
    auto frame = std::make_unique<CUFrameBuf>(img.width, img.height, img.csp);
    frame->copyFrameAsync(&img, stream);
    return SubImageData(std::move(frame), std::unique_ptr<CUFrameBuf>(), std::move(bufCPU), image->dst_x, image->dst_y);
}


RGY_ERR NVEncFilterSubburn::procFrameText(FrameInfo *pOutputFrame, int64_t frameTimeMs, cudaStream_t stream) {
    int nDetectChange = 0;
    const auto frameImages = ass_render_frame(m_assRenderer.get(), m_assTrack.get(), frameTimeMs, &nDetectChange);

    if (!frameImages) {
        m_subImages.clear();
    } else if (nDetectChange) {
        m_subImages.clear();
        for (auto image = frameImages; image; image = image->next) {
            m_subImages.push_back(textRectToImage(image, stream));
        }
    }
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamSubburn>(m_pParam);
    if (!prm) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (m_subImages.size()) {
        static const std::map<RGY_CSP, decltype(proc_frame<uint8_t, 8>) *> func_list ={
            { RGY_CSP_YV12,      proc_frame<uint8_t,   8> },
            { RGY_CSP_YV12_16,   proc_frame<uint16_t, 16> },
            { RGY_CSP_YUV444,    proc_frame<uint8_t,   8> },
            { RGY_CSP_YUV444_16, proc_frame<uint16_t, 16> }
        };
        if (func_list.count(pOutputFrame->csp) == 0) {
            AddMessage(RGY_LOG_ERROR, _T("unsupported csp %s.\n"), RGY_CSP_NAMES[pOutputFrame->csp]);
            return RGY_ERR_UNSUPPORTED;
        }
        for (uint32_t irect = 0; irect < m_subImages.size(); irect++) {
            const FrameInfo *pSubImg = &m_subImages[irect].image->frame;
            auto cudaerr = func_list.at(pOutputFrame->csp)(pOutputFrame, pSubImg, m_subImages[irect].x, m_subImages[irect].y,
                prm->subburn.transparency_offset, prm->subburn.brightness, prm->subburn.contrast, stream);
            if (cudaerr != cudaSuccess) {
                AddMessage(RGY_LOG_ERROR, _T("error at subburn(%s): %s.\n"),
                    RGY_CSP_NAMES[pOutputFrame->csp],
                    char_to_tstring(cudaGetErrorString(cudaerr)).c_str());
                return RGY_ERR_CUDA;
            }
        }
    }
    return RGY_ERR_NONE;
}

SubImageData NVEncFilterSubburn::bitmapRectToImage(const AVSubtitleRect *rect, const FrameInfo *outputFrame, const sInputCrop &crop, cudaStream_t stream) {
    //YUV420の関係で縦横2pixelずつ処理するので、2で割り切れている必要がある
    const int x_offset = ((rect->x % 2) != 0) ? 1 : 0;
    const int y_offset = ((rect->y % 2) != 0) ? 1 : 0;
    FrameInfo img;
    img.csp = RGY_CSP_YUVA444;
    img.width  = ALIGN(rect->w + x_offset, 2);
    img.height = ALIGN(rect->h + y_offset, 2);
    img.deivce_mem = false;
    img.picstruct = RGY_PICSTRUCT_FRAME;
    auto imgInfoEx = getFrameInfoExtra(&img);
    img.pitch = ALIGN(imgInfoEx.width_byte, 64);
    imgInfoEx = getFrameInfoExtra(&img);
    cudaMallocHost(&img.ptr, imgInfoEx.frame_size);
    unique_ptr<void, decltype(&cudaFreeHost)> bufCPU(img.ptr, cudaFreeHost);

    //とりあえずすべて0で初期化しておく
    //Alpha=0で透明なので都合がよい
    memset(img.ptr, 0, imgInfoEx.frame_size);

    auto planeY = getPlane(&img, RGY_PLANE_Y);
    auto planeU = getPlane(&img, RGY_PLANE_U);
    auto planeV = getPlane(&img, RGY_PLANE_V);
    auto planeA = getPlane(&img, RGY_PLANE_A);

    for (int j = 0; j < img.height; j++) {
        for (int i = 0; i < img.width; i++) {
            const int idx = j * img.pitch + i;
            planeU.ptr[idx] = 128;
            planeV.ptr[idx] = 128;
        }
    }

    //色テーブルをRGBA->YUVAに変換
    const uint32_t *pColorARGB = (uint32_t *)rect->data[1];
    alignas(32) uint32_t colorTableYUVA[256];
    memset(colorTableYUVA, 0, sizeof(colorTableYUVA));

    const uint32_t nColorTableSize = rect->nb_colors;
    assert(nColorTableSize <= _countof(colorTableYUVA));
    for (uint32_t ic = 0; ic < nColorTableSize; ic++) {
        const uint32_t subColor = pColorARGB[ic];
        const uint8_t subA = (uint8_t)(subColor >> 24);
        const uint8_t subR = (uint8_t)((subColor >> 16) & 0xff);
        const uint8_t subG = (uint8_t)((subColor >>  8) & 0xff);
        const uint8_t subB = (uint8_t)(subColor        & 0xff);

        const uint8_t subY = (uint8_t)clamp((( 66 * subR + 129 * subG +  25 * subB + 128) >> 8) +  16, 0, 255);
        const uint8_t subU = (uint8_t)clamp(((-38 * subR -  74 * subG + 112 * subB + 128) >> 8) + 128, 0, 255);
        const uint8_t subV = (uint8_t)clamp(((112 * subR -  94 * subG -  18 * subB + 128) >> 8) + 128, 0, 255);

        colorTableYUVA[ic] = ((subA << 24) | (subV << 16) | (subU << 8) | subY);
    }

    //YUVで字幕の画像データを構築
    for (int j = 0; j < rect->h; j++) {
        for (int i = 0; i < rect->w; i++) {
            const int src_idx = j * rect->linesize[0] + i;
            const int ic = rect->data[0][src_idx];

            const uint32_t subColor = colorTableYUVA[ic];
            const uint8_t subA = (uint8_t)(subColor >> 24);
            const uint8_t subV = (uint8_t)((subColor >> 16) & 0xff);
            const uint8_t subU = (uint8_t)((subColor >>  8) & 0xff);
            const uint8_t subY = (uint8_t)(subColor        & 0xff);

            const int dst_idx = (j+y_offset) * img.pitch + (i+x_offset);
            planeY.ptr[dst_idx] = subY;
            planeU.ptr[dst_idx] = subU;
            planeV.ptr[dst_idx] = subV;
            planeA.ptr[dst_idx] = subA;
        }
    }

    //GPUへ転送
    auto frameTemp = std::make_unique<CUFrameBuf>(img.width, img.height, img.csp);
    frameTemp->copyFrameAsync(&img, stream);
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamSubburn>(m_pParam);

    decltype(frameTemp) frame;
    if (prm->subburn.scale == 1.0f) {
        frame = std::move(frameTemp);
    } else {
#if 0
        FrameInfo tempframe = img;
        std::vector<uint8_t> temp(imgInfoEx.frame_size);
        memcpy(temp.data(), img.ptr, temp.size());
        tempframe.ptr = temp.data();
        auto tmpY = getPlane(&tempframe, RGY_PLANE_Y);
        auto tmpU = getPlane(&tempframe, RGY_PLANE_U);
        auto tmpV = getPlane(&tempframe, RGY_PLANE_V);
        for (int j = 0; j < rect->h; j++) {
            for (int i = 0; i < rect->w; i++) {
                #define IDX(x, y) ((clamp(y,0,rect->h)+y_offset) * img.pitch + (clamp(x,0,rect->w)+x_offset))
                const int dst_idx = IDX(i,j);
                if (planeA.ptr[dst_idx] == 0) {
                    int minidx = -1;
                    uint8_t minval = 255;
                    for (int jy = -1; jy <= 1; jy++) {
                        for (int ix = -1; ix <= 1; ix++) {
                            int idx = IDX(i+ix, j+jy);
                            if (planeA.ptr[idx] != 0) {
                                auto val = tmpY.ptr[idx];
                                if (val < minval) {
                                    minidx = idx;
                                    minval = val;
                                }
                            }
                        }
                    }
                    if (minidx >= 0) {
                        planeY.ptr[dst_idx] = tmpY.ptr[minidx];
                        planeU.ptr[dst_idx] = tmpU.ptr[minidx];
                        planeV.ptr[dst_idx] = tmpV.ptr[minidx];
                    }
                }
                #undef IDX
            }
        }
#endif

        frame = std::make_unique<CUFrameBuf>(
            ALIGN((int)(img.width  * prm->subburn.scale + 0.5f), 4),
            ALIGN((int)(img.height * prm->subburn.scale + 0.5f), 4), img.csp);
        frame->alloc();
        unique_ptr<NVEncFilterResize> filterResize(new NVEncFilterResize());
        shared_ptr<NVEncFilterParamResize> paramResize(new NVEncFilterParamResize());
        paramResize->frameIn = frameTemp->frame;
        paramResize->frameOut = frame->frame;
        paramResize->baseFps = prm->baseFps;
        paramResize->frameOut.deivce_mem = true;
        paramResize->bOutOverwrite = false;
        paramResize->interp = RESIZE_CUDA_TEXTURE_BILINEAR;
        filterResize->init(paramResize, m_pPrintMes);
        m_resize = std::move(filterResize);

        int filterOutputNum = 0;
        FrameInfo *filterOutput[1] = { &frame->frame };
        m_resize->filter(&frameTemp->frame, (FrameInfo **)&filterOutput, &filterOutputNum, stream);
    }
    int x_pos = ALIGN((int)(prm->subburn.scale * rect->x + 0.5f) - ((crop.e.left + crop.e.right) / 2), 2);
    int y_pos = ALIGN((int)(prm->subburn.scale * rect->y + 0.5f) - crop.e.up - crop.e.bottom, 2);
    if (m_outCodecDecodeCtx->height > 0) {
        const double y_factor = rect->y / (double)m_outCodecDecodeCtx->height;
        y_pos = ALIGN((int)(outputFrame->height * y_factor + 0.5f), 2);
        y_pos = std::min(y_pos, outputFrame->height - rect->h);
    }
    return SubImageData(std::move(frame), std::move(frameTemp), std::move(bufCPU), x_pos, y_pos);
}


RGY_ERR NVEncFilterSubburn::procFrameBitmap(FrameInfo *pOutputFrame, const sInputCrop &crop, cudaStream_t stream) {
    if (m_subData) {
        if (m_subData->num_rects != m_subImages.size()) {
            for (uint32_t irect = 0; irect < m_subData->num_rects; irect++) {
                const AVSubtitleRect *rect = m_subData->rects[irect];
                m_subImages.push_back(bitmapRectToImage(rect, pOutputFrame, crop, stream));
            }
        }
        if ((m_subData->num_rects != m_subImages.size())) {
            AddMessage(RGY_LOG_ERROR, _T("unexpected error.\n"));
            return RGY_ERR_UNKNOWN;
        }
        auto prm = std::dynamic_pointer_cast<NVEncFilterParamSubburn>(m_pParam);
        if (!prm) {
            AddMessage(RGY_LOG_ERROR, _T("Invalid parameter type.\n"));
            return RGY_ERR_INVALID_PARAM;
        }
        static const std::map<RGY_CSP, decltype(proc_frame<uint8_t, 8>) *> func_list = {
            { RGY_CSP_YV12,      proc_frame<uint8_t,   8> },
            { RGY_CSP_YV12_16,   proc_frame<uint16_t, 16> },
            { RGY_CSP_YUV444,    proc_frame<uint8_t,   8> },
            { RGY_CSP_YUV444_16, proc_frame<uint16_t, 16> }
        };
        if (func_list.count(pOutputFrame->csp) == 0) {
            AddMessage(RGY_LOG_ERROR, _T("unsupported csp %s.\n"), RGY_CSP_NAMES[pOutputFrame->csp]);
            return RGY_ERR_UNSUPPORTED;
        }
        for (uint32_t irect = 0; irect < m_subImages.size(); irect++) {
            const FrameInfo *pSubImg = &m_subImages[irect].image->frame;
            auto cudaerr = func_list.at(pOutputFrame->csp)(pOutputFrame, pSubImg, m_subImages[irect].x, m_subImages[irect].y,
                prm->subburn.transparency_offset, prm->subburn.brightness, prm->subburn.contrast, stream);
            if (cudaerr != cudaSuccess) {
                AddMessage(RGY_LOG_ERROR, _T("error at subburn(%s): %s.\n"),
                    RGY_CSP_NAMES[pOutputFrame->csp],
                    char_to_tstring(cudaGetErrorString(cudaerr)).c_str());
                return RGY_ERR_CUDA;
            }
        }
    }
    return RGY_ERR_NONE;
}

#endif //#if ENABLE_AVSW_READER
