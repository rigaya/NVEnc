// -----------------------------------------------------------------------------------------
// NVEnc by rigaya
// -----------------------------------------------------------------------------------------
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

#pragma once
#ifndef __NVENC_UTIL_H__
#define __NVENC_UTIL_H__

#include <utility>
#include <array>
#include "nvEncodeAPI.h"
#include "cuviddec.h"
#include "ConvertCsp.h"
#include "rgy_util.h"
#include "rgy_err.h"

#ifndef cudaVideoCodec_VP8
#define cudaVideoCodec_VP8 (cudaVideoCodec)(cudaVideoCodec_HEVC+1)
#endif
#ifndef cudaVideoCodec_VP9
#define cudaVideoCodec_VP9 (cudaVideoCodec)(cudaVideoCodec_HEVC+2)
#endif

MAP_PAIR_0_1_PROTO(codec, rgy, RGY_CODEC, enc, cudaVideoCodec);
MAP_PAIR_0_1_PROTO(chromafmt, rgy, RGY_CHROMAFMT, enc, cudaVideoChromaFormat);
MAP_PAIR_0_1_PROTO(csp, rgy, RGY_CSP, enc, NV_ENC_BUFFER_FORMAT);
MAP_PAIR_0_1_PROTO(codec_guid, rgy, RGY_CODEC, enc, GUID);
MAP_PAIR_0_1_PROTO(codec_guid_profile, rgy, RGY_CODEC_DATA, enc, GUID);

NV_ENC_PIC_STRUCT picstruct_rgy_to_enc(RGY_PICSTRUCT picstruct);
RGY_PICSTRUCT picstruct_enc_to_rgy(NV_ENC_PIC_STRUCT picstruct);

RGY_CSP getEncCsp(NV_ENC_BUFFER_FORMAT enc_buffer_format);

VideoInfo videooutputinfo(
    const GUID& encCodecGUID,
    NV_ENC_BUFFER_FORMAT buffer_fmt,
    int nEncWidth,
    int nEncHeight,
    const NV_ENC_CONFIG *pEncConfig,
    NV_ENC_PIC_STRUCT nPicStruct,
    std::pair<int, int> sar,
    std::pair<int, int> outFps);


struct RGYBitstream {
private:
    uint8_t *dataptr;
    uint32_t dataLength;
    uint32_t dataOffset;
    uint32_t maxLength;
    int64_t  dataDts;
    int64_t  dataPts;
    uint32_t dataFlag;
    uint32_t dataAvgQP;
    RGY_FRAMETYPE dataFrametype;
    RGY_PICSTRUCT dataPicstruct;
    int dataFrameIdx;
    int64_t dataDuration;

public:
    uint8_t *bufptr() const {
        return dataptr;
    }

    const uint8_t *data() const {
        return dataptr + dataOffset;
    }

    uint32_t dataflag() const {
        return dataFlag;
    }

    void setDataflag(uint32_t flag) {
        dataFlag = flag;
    }

    RGY_FRAMETYPE frametype() const {
        return dataFrametype;
    }

    void setFrametype(RGY_FRAMETYPE type) {
        dataFrametype = type;
    }

    RGY_PICSTRUCT picstruct() const {
        return dataPicstruct;
    }

    void setPicstruct(RGY_PICSTRUCT picstruct) {
        dataPicstruct = picstruct;
    }

    int64_t duration() {
        return dataDuration;
    }

    void setDuration(int64_t duration) {
        dataDuration = duration;
    }

    int frameIdx() {
        return dataFrameIdx;
    }

    void setFrameIdx(int frameIdx) {
        dataFrameIdx = frameIdx;
    }

    uint32_t size() const {
        return dataLength;
    }

    void setSize(uint32_t size) {
        dataLength = size;
    }

    uint32_t offset() const {
        return dataOffset;
    }

    void addOffset(uint32_t add) {
        dataOffset += add;
    }

    void setOffset(uint32_t offset) {
        dataOffset = offset;
    }

    uint32_t bufsize() const {
        return maxLength;
    }

    int64_t pts() const {
        return dataPts;
    }

    void setPts(int64_t pts) {
        dataPts = pts;
    }

    int64_t dts() const {
        return dataDts;
    }

    void setDts(int64_t dts) {
        dataDts = dts;
    }

    uint32_t avgQP() {
        return dataAvgQP;
    }

    void setAvgQP(uint32_t avgQP) {
        dataAvgQP = avgQP;
    }

    void clear() {
        if (dataptr) {
            _aligned_free(dataptr);
        }
        dataptr = nullptr;
        dataLength = 0;
        dataOffset = 0;
        maxLength = 0;
    }

    RGY_ERR init(uint32_t nSize) {
        clear();

        if (nSize > 0) {
            if (nullptr == (dataptr = (uint8_t *)_aligned_malloc(nSize, 32))) {
                return RGY_ERR_NULL_PTR;
            }

            maxLength = nSize;
        }
        return RGY_ERR_NONE;
    }

    RGY_ERR set(const uint8_t *setData, uint32_t setSize) {
        if (setData == nullptr || setSize == 0) {
            return RGY_ERR_MORE_BITSTREAM;
        }
        if (maxLength < setSize) {
            clear();
            auto sts = init(setSize);
            if (sts != RGY_ERR_NONE) {
                return sts;
            }
        }
        dataLength = setSize;
        dataOffset = 0;
        memcpy(dataptr, setData, setSize);
        return RGY_ERR_NONE;
    }

    RGY_ERR set(const uint8_t *setData, uint32_t setSize, int64_t dts, int64_t pts) {
        auto sts = set(setData, setSize);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        dataDts = dts;
        dataPts = pts;
        return RGY_ERR_NONE;
    }

    RGY_ERR copy(const RGYBitstream *pBitstream) {
        auto sts = set(pBitstream->data(), pBitstream->size());
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        return RGY_ERR_NONE;
    }

    RGY_ERR changeSize(uint32_t nNewSize) {
        uint8_t *pData = (uint8_t *)_aligned_malloc(nNewSize, 32);
        if (pData == nullptr) {
            return RGY_ERR_NULL_PTR;
        }

        auto nDataLen = dataLength;
        if (dataLength) {
            memcpy(pData, dataptr + dataOffset, (std::min)(nDataLen, nNewSize));
        }
        clear();

        dataptr       = pData;
        dataOffset = 0;
        dataLength = nDataLen;
        maxLength  = nNewSize;

        return RGY_ERR_NONE;
    }

    RGY_ERR append(const uint8_t *appendData, uint32_t appendSize) {
        if (appendData && appendSize > 0) {
            const uint32_t new_data_length = appendSize + dataLength;
            if (maxLength < new_data_length) {
                auto sts = changeSize(new_data_length);
                if (sts != RGY_ERR_NONE) {
                    return sts;
                }
            }

            if (maxLength < new_data_length + dataOffset) {
                memmove(dataptr, dataptr + dataOffset, dataLength);
                dataOffset = 0;
            }
            memcpy(dataptr + dataLength + dataOffset, appendData, appendSize);
            dataLength = new_data_length;
        }
        return RGY_ERR_NONE;
    }

    RGY_ERR append(RGYBitstream *pBitstream) {
        return append(pBitstream->data(), pBitstream->size());
    }
};

static inline RGYBitstream RGYBitstreamInit() {
    RGYBitstream bitstream;
    memset(&bitstream, 0, sizeof(bitstream));
    return bitstream;
}

#ifndef __CUDACC__
static_assert(std::is_pod<RGYBitstream>::value == true, "RGYBitstream should be POD type.");
#endif

struct RGYFrame {
private:
    FrameInfo info;
public:
    void set(const FrameInfo& frameinfo) {
        info = frameinfo;
    }
    void set(uint8_t *ptr, int width, int height, int pitch, RGY_CSP csp, int64_t timestamp = 0) {
        info.ptr = ptr;
        info.width = width;
        info.height = height;
        info.pitch = pitch;
        info.csp = csp;
        info.timestamp = timestamp;
    }
    void ptrArray(void *array[3]) {
        array[0] = info.ptr;
        array[1] = info.ptr + info.pitch * info.height;
        array[2] = info.ptr + info.pitch * info.height * 2;
    }
    uint8_t *ptrY() {
        return info.ptr;
    }
    uint8_t *ptrUV() {
        return info.ptr + info.pitch * info.height;
    }
    uint8_t *ptrU() {
        return info.ptr + info.pitch * info.height;
    }
    uint8_t *ptrV() {
        return info.ptr + info.pitch * info.height * 2;
    }
    uint8_t *ptrRGB() {
        return info.ptr;
    }
    uint32_t pitch() {
        return info.pitch;
    }
    uint64_t timestamp() {
        return info.timestamp;
    }
    void setTimestamp(uint64_t frame_timestamp) {
        info.timestamp = frame_timestamp;
    }
};

static inline RGYFrame RGYFrameInit() {
    RGYFrame frame;
    memset(&frame, 0, sizeof(frame));
    return frame;
}

static inline RGYFrame RGYFrameInit(const FrameInfo& frameinfo) {
    RGYFrame frame;
    frame.set(frameinfo);
    return frame;
}

#ifndef __CUDACC__
static_assert(std::is_pod<RGYFrame>::value == true, "RGYFrame should be POD type.");
#endif

static inline RGY_FRAMETYPE frametype_enc_to_rgy(const NV_ENC_PIC_TYPE frametype) {
    RGY_FRAMETYPE type = RGY_FRAMETYPE_UNKNOWN;
    type |=  (NV_ENC_PIC_TYPE_IDR == frametype) ? RGY_FRAMETYPE_IDR : RGY_FRAMETYPE_UNKNOWN;
    type |=  (NV_ENC_PIC_TYPE_I   == frametype) ? RGY_FRAMETYPE_I   : RGY_FRAMETYPE_UNKNOWN;
    type |=  (NV_ENC_PIC_TYPE_P   == frametype) ? RGY_FRAMETYPE_P   : RGY_FRAMETYPE_UNKNOWN;
    type |=  (NV_ENC_PIC_TYPE_B   == frametype) ? RGY_FRAMETYPE_B   : RGY_FRAMETYPE_UNKNOWN;
    return type;
}

static inline RGYBitstream RGYBitstreamInit(const NV_ENC_LOCK_BITSTREAM& nv_bitstream) {
    RGYBitstream bitstream;
    memset(&bitstream, 0, sizeof(bitstream));
    bitstream.set((const uint8_t *)nv_bitstream.bitstreamBufferPtr, nv_bitstream.bitstreamSizeInBytes, (int64_t)nv_bitstream.outputTimeStamp, (int64_t)0);
    bitstream.setAvgQP(nv_bitstream.frameAvgQP);
    bitstream.setFrametype(frametype_enc_to_rgy(nv_bitstream.pictureType));
    bitstream.setPicstruct(picstruct_enc_to_rgy(nv_bitstream.pictureStruct));
    bitstream.setFrameIdx(nv_bitstream.frameIdx);
    bitstream.setDuration(nv_bitstream.outputDuration);
    return bitstream;
}

#endif //__NVENC_UTIL_H__
