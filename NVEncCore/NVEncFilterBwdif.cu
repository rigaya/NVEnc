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
#include "convert_csp.h"
#include "NVEncFilterBwdif.h"
#include "NVEncParam.h"

static const int BWDIF_BLOCK_X = 32;
static const int BWDIF_BLOCK_Y = 8;

// BBC PH-2071 w3fdif coefficients (13-bit fixed point).
#define W3F_LF0   4309
#define W3F_LF1    213
#define W3F_HF0   5570
#define W3F_HF1   3801
#define W3F_HF2   1016
#define W3F_SP0   5077
#define W3F_SP1    981
#define W3F_SHIFT   13

template<typename TypePixel>
__inline__ __device__ int readPix(cudaTextureObject_t tex, const int x, const int y) {
    return (int)tex2D<TypePixel>(tex, (float)x + 0.5f, (float)y + 0.5f);
}

template<typename TypePixel, int bit_depth>
__global__ void kernel_bwdif_frame(
    TypePixel *__restrict__ pDst, const int dstPitch,
    cudaTextureObject_t texPrev2,
    cudaTextureObject_t texPrev,
    cudaTextureObject_t texCur,
    cudaTextureObject_t texNext,
    cudaTextureObject_t texNext2,
    const int width,
    const int height,
    const int preserveTopField,
    const int thr
) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= width || iy >= height) {
        return;
    }

    const int preservedParity = preserveTopField ? 0 : 1;
    const int needsInterp = ((iy & 1) != preservedParity);
    TypePixel *dstPix = (TypePixel *)((uint8_t *)pDst + iy * dstPitch + ix * sizeof(TypePixel));

    if (!needsInterp) {
        dstPix[0] = (TypePixel)readPix<TypePixel>(texCur, ix, iy);
        return;
    }

    const int rowU = readPix<TypePixel>(texCur,   ix, iy - 1);
    const int rowL = readPix<TypePixel>(texCur,   ix, iy + 1);
    const int p2_0 = readPix<TypePixel>(texPrev2, ix, iy);
    const int n2_0 = readPix<TypePixel>(texNext2, ix, iy);
    const int tAvg = (p2_0 + n2_0) >> 1;

    const int pUp = readPix<TypePixel>(texPrev, ix, iy - 1);
    const int pDn = readPix<TypePixel>(texPrev, ix, iy + 1);
    const int nUp = readPix<TypePixel>(texNext, ix, iy - 1);
    const int nDn = readPix<TypePixel>(texNext, ix, iy + 1);

    const int motA = abs(p2_0 - n2_0);
    const int motB = (abs(pUp - rowU) + abs(pDn - rowL)) >> 1;
    const int motC = (abs(nUp - rowU) + abs(nDn - rowL)) >> 1;
    int motion = max(motA >> 1, max(motB, motC));

    const int max_val = (1 << bit_depth) - 1;
    if (motion <= thr) {
        dstPix[0] = (TypePixel)clamp(tAvg, 0, max_val);
        return;
    }

    const int hasSpatBounds = (iy >= 2) && (iy < height - 2);
    const int hasFullCtx    = (iy >= 4) && (iy < height - 4);

    int localMotion = motion;
    if (hasSpatBounds) {
        const int p2_m2 = readPix<TypePixel>(texPrev2, ix, iy - 2);
        const int p2_p2 = readPix<TypePixel>(texPrev2, ix, iy + 2);
        const int n2_m2 = readPix<TypePixel>(texNext2, ix, iy - 2);
        const int n2_p2 = readPix<TypePixel>(texNext2, ix, iy + 2);
        const int spreadU = ((p2_m2 + n2_m2) >> 1) - rowU;
        const int spreadL = ((p2_p2 + n2_p2) >> 1) - rowL;
        const int dU      = tAvg - rowU;
        const int dL      = tAvg - rowL;
        const int hiSet   = max(dL, max(dU, min(spreadU, spreadL)));
        const int loSet   = min(dL, min(dU, max(spreadU, spreadL)));
        localMotion = max(localMotion, max(loSet, -hiSet));
    }

    int interp;
    if (hasFullCtx) {
        const int p2_m4 = readPix<TypePixel>(texPrev2, ix, iy - 4);
        const int p2_p4 = readPix<TypePixel>(texPrev2, ix, iy + 4);
        const int n2_m4 = readPix<TypePixel>(texNext2, ix, iy - 4);
        const int n2_p4 = readPix<TypePixel>(texNext2, ix, iy + 4);
        const int p2_m2 = readPix<TypePixel>(texPrev2, ix, iy - 2);
        const int p2_p2 = readPix<TypePixel>(texPrev2, ix, iy + 2);
        const int n2_m2 = readPix<TypePixel>(texNext2, ix, iy - 2);
        const int n2_p2 = readPix<TypePixel>(texNext2, ix, iy + 2);
        const int curU3 = readPix<TypePixel>(texCur,   ix, iy - 3);
        const int curD3 = readPix<TypePixel>(texCur,   ix, iy + 3);

        const int verticalEdge = abs(rowU - rowL);
        if (verticalEdge > motA) {
            const int hf = ( W3F_HF0 * (p2_0 + n2_0)
                           - W3F_HF1 * (p2_m2 + n2_m2 + p2_p2 + n2_p2)
                           + W3F_HF2 * (p2_m4 + n2_m4 + p2_p4 + n2_p4)) >> 2;
            interp = (hf + W3F_LF0 * (rowU + rowL) - W3F_LF1 * (curU3 + curD3)) >> W3F_SHIFT;
        } else {
            interp = (W3F_SP0 * (rowU + rowL) - W3F_SP1 * (curU3 + curD3)) >> W3F_SHIFT;
        }
    } else {
        interp = (rowU + rowL) >> 1;
    }

    interp = clamp(interp, tAvg - localMotion, tAvg + localMotion);
    interp = clamp(interp, 0, max_val);
    dstPix[0] = (TypePixel)interp;
}

template<typename TypePixel>
cudaError_t setTexFieldBwdif(cudaTextureObject_t& texSrc, const RGYFrameInfo *pFrame) {
    texSrc = 0;

    cudaResourceDesc resDescSrc;
    memset(&resDescSrc, 0, sizeof(resDescSrc));
    resDescSrc.resType = cudaResourceTypePitch2D;
    resDescSrc.res.pitch2D.desc = cudaCreateChannelDesc<TypePixel>();
    resDescSrc.res.pitch2D.pitchInBytes = pFrame->pitch[0];
    resDescSrc.res.pitch2D.width = pFrame->width;
    resDescSrc.res.pitch2D.height = pFrame->height;
    resDescSrc.res.pitch2D.devPtr = (uint8_t *)pFrame->ptr[0];

    cudaTextureDesc texDescSrc;
    memset(&texDescSrc, 0, sizeof(texDescSrc));
    texDescSrc.addressMode[0] = cudaAddressModeClamp;
    texDescSrc.addressMode[1] = cudaAddressModeClamp;
    texDescSrc.filterMode = cudaFilterModePoint;
    texDescSrc.readMode = cudaReadModeElementType;
    texDescSrc.normalizedCoords = 0;

    return cudaCreateTextureObject(&texSrc, &resDescSrc, &texDescSrc, nullptr);
}

template<typename TypePixel, int bit_depth>
RGY_ERR run_bwdif_plane(RGYFrameInfo *pOutputPlane,
    const RGYFrameInfo *pPrev2,
    const RGYFrameInfo *pPrev,
    const RGYFrameInfo *pCur,
    const RGYFrameInfo *pNext,
    const RGYFrameInfo *pNext2,
    const int preserveTopField,
    const int thr,
    cudaStream_t stream) {
    cudaTextureObject_t texPrev2 = 0;
    cudaTextureObject_t texPrev  = 0;
    cudaTextureObject_t texCur   = 0;
    cudaTextureObject_t texNext  = 0;
    cudaTextureObject_t texNext2 = 0;

    auto cudaerr = cudaSuccess;
    if (   (cudaerr = setTexFieldBwdif<TypePixel>(texPrev2, pPrev2)) != cudaSuccess
        || (cudaerr = setTexFieldBwdif<TypePixel>(texPrev,  pPrev )) != cudaSuccess
        || (cudaerr = setTexFieldBwdif<TypePixel>(texCur,   pCur  )) != cudaSuccess
        || (cudaerr = setTexFieldBwdif<TypePixel>(texNext,  pNext )) != cudaSuccess
        || (cudaerr = setTexFieldBwdif<TypePixel>(texNext2, pNext2)) != cudaSuccess) {
        if (texPrev2) cudaDestroyTextureObject(texPrev2);
        if (texPrev)  cudaDestroyTextureObject(texPrev);
        if (texCur)   cudaDestroyTextureObject(texCur);
        if (texNext)  cudaDestroyTextureObject(texNext);
        if (texNext2) cudaDestroyTextureObject(texNext2);
        return err_to_rgy(cudaerr);
    }

    dim3 blockSize(BWDIF_BLOCK_X, BWDIF_BLOCK_Y);
    dim3 gridSize(divCeil(pOutputPlane->width, blockSize.x), divCeil(pOutputPlane->height, blockSize.y));

    kernel_bwdif_frame<TypePixel, bit_depth><<<gridSize, blockSize, 0, stream>>>(
        (TypePixel *)pOutputPlane->ptr[0],
        pOutputPlane->pitch[0],
        texPrev2, texPrev, texCur, texNext, texNext2,
        pOutputPlane->width, pOutputPlane->height,
        preserveTopField,
        thr);

    cudaerr = cudaGetLastError();
    cudaDestroyTextureObject(texPrev2);
    cudaDestroyTextureObject(texPrev);
    cudaDestroyTextureObject(texCur);
    cudaDestroyTextureObject(texNext);
    cudaDestroyTextureObject(texNext2);
    return err_to_rgy(cudaerr);
}

template<typename TypePixel, int bit_depth>
RGY_ERR run_bwdif_frame_typed(RGYFrameInfo *pOutputFrame,
    const RGYFrameInfo *pPrev2,
    const RGYFrameInfo *pPrev,
    const RGYFrameInfo *pCur,
    const RGYFrameInfo *pNext,
    const RGYFrameInfo *pNext2,
    const int preserveTopField,
    const int thr,
    cudaStream_t stream) {
    for (int iplane = 0; iplane < RGY_CSP_PLANES[pOutputFrame->csp]; iplane++) {
        const auto plane = (RGY_PLANE)iplane;
        auto planeOutput = getPlane(pOutputFrame, plane);
        const auto planePrev2 = getPlane(pPrev2, plane);
        const auto planePrev  = getPlane(pPrev,  plane);
        const auto planeCur   = getPlane(pCur,   plane);
        const auto planeNext  = getPlane(pNext,  plane);
        const auto planeNext2 = getPlane(pNext2, plane);
        auto sts = run_bwdif_plane<TypePixel, bit_depth>(&planeOutput,
            &planePrev2, &planePrev, &planeCur, &planeNext, &planeNext2,
            preserveTopField, thr, stream);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
    }
    return RGY_ERR_NONE;
}

RGY_ERR run_bwdif_frame(RGYFrameInfo *pOutputFrame,
    const RGYFrameInfo *pPrev2,
    const RGYFrameInfo *pPrev,
    const RGYFrameInfo *pCur,
    const RGYFrameInfo *pNext,
    const RGYFrameInfo *pNext2,
    const int preserveTopField,
    const int thr,
    cudaStream_t stream) {
    static const std::map<RGY_DATA_TYPE, decltype(run_bwdif_frame_typed<uint8_t, 8>)*> func_list = {
        { RGY_DATA_TYPE_U8,  run_bwdif_frame_typed<uint8_t,   8> },
        { RGY_DATA_TYPE_U16, run_bwdif_frame_typed<uint16_t, 16> }
    };
    if (func_list.count(RGY_CSP_DATA_TYPE[pOutputFrame->csp]) == 0) {
        return RGY_ERR_UNSUPPORTED;
    }
    return func_list.at(RGY_CSP_DATA_TYPE[pOutputFrame->csp])(pOutputFrame, pPrev2, pPrev, pCur, pNext, pNext2, preserveTopField, thr, stream);
}
