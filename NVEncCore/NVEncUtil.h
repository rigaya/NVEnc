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
#include "rgy_osdep.h"

#pragma warning (push)
#pragma warning (disable: 4819)
#include "nvEncodeAPI.h"
#pragma warning (pop)

#pragma warning (push)
#pragma warning (disable: 4201)
#include "dynlink_cuviddec.h"
#pragma warning (pop)

#include "rgy_util.h"
#include "rgy_def.h"
#pragma warning (push)
#pragma warning (disable: 4819)
RGY_DISABLE_WARNING_PUSH
RGY_DISABLE_WARNING_STR("-Wswitch")
#include "helper_cuda.h"
#include "helper_nvenc.h"
RGY_DISABLE_WARNING_POP
#pragma warning (pop)
#include "convert_csp.h"
#include "rgy_frame.h"
#include "rgy_err.h"

#define NVENCAPI_VERSION (NVENCAPI_MAJOR_VERSION | (NVENCAPI_MINOR_VERSION << 24))

static constexpr uint32_t nvenc_api_ver(uint32_t major, uint8_t minor) {
    return major | ((uint32_t)minor << 24);
}
static constexpr uint32_t nvenc_api_ver_major(uint32_t ver) {
    return ver & 0x00ffffff;
}
static constexpr uint32_t nvenc_api_ver_minor(uint32_t ver) {
    return ver >> 24;
}
static constexpr uint32_t nvenc_api_struct_ver(uint32_t apiver, uint32_t structver) {
    return ((uint32_t)apiver | (structver << 16) | (0x7 << 28));
}

static_assert(nvenc_api_ver(NVENCAPI_MAJOR_VERSION, NVENCAPI_MINOR_VERSION) == NVENCAPI_VERSION, "API ver check!");
//static_assert(NVENCAPI_STRUCT_VERSION(ver) ((uint32_t)NVENCAPI_VERSION | ((ver) << 16) | (0x7 << 28)); ,"API struct ver check!");

#if defined(_MSC_VER) && _MSC_VER <= 1900
static const bool nvenc_api_ver_check(uint32_t ver, uint32_t required) {
    auto required_major = nvenc_api_ver_major(required);
    auto required_minor = nvenc_api_ver_minor(required);
    auto ver_major = nvenc_api_ver_major(ver);
    auto ver_minor = nvenc_api_ver_minor(ver);
    return required_major < ver_major || (ver_major == required_major && required_minor <= ver_minor);
}
#else
static constexpr bool nvenc_api_ver_check(uint32_t ver, uint32_t required) {
    auto required_major = nvenc_api_ver_major(required);
    auto required_minor = nvenc_api_ver_minor(required);
    auto ver_major = nvenc_api_ver_major(ver);
    auto ver_minor = nvenc_api_ver_minor(ver);
    return required_major < ver_major || (ver_major == required_major && required_minor <= ver_minor);
}
#endif

#if !defined(_MSC_VER)
static bool operator==(const GUID &guid1, const GUID &guid2) {
     if (guid1.Data1    == guid2.Data1 &&
         guid1.Data2    == guid2.Data2 &&
         guid1.Data3    == guid2.Data3 &&
         guid1.Data4[0] == guid2.Data4[0] &&
         guid1.Data4[1] == guid2.Data4[1] &&
         guid1.Data4[2] == guid2.Data4[2] &&
         guid1.Data4[3] == guid2.Data4[3] &&
         guid1.Data4[4] == guid2.Data4[4] &&
         guid1.Data4[5] == guid2.Data4[5] &&
         guid1.Data4[6] == guid2.Data4[6] &&
         guid1.Data4[7] == guid2.Data4[7]) {
        return true;
    }
    return false;
}
static bool operator!=(const GUID &guid1, const GUID &guid2) {
    return !(guid1 == guid2);
}
#endif

MAP_PAIR_0_1_PROTO(codec, rgy, RGY_CODEC, dec, cudaVideoCodec);
MAP_PAIR_0_1_PROTO(chromafmt, rgy, RGY_CHROMAFMT, enc, cudaVideoChromaFormat);
MAP_PAIR_0_1_PROTO(csp, rgy, RGY_CSP, enc, NV_ENC_BUFFER_FORMAT);
MAP_PAIR_0_1_PROTO(codec_guid, rgy, RGY_CODEC, enc, GUID);
MAP_PAIR_0_1_PROTO(codec_guid_profile, rgy, RGY_CODEC_DATA, enc, GUID);
MAP_PAIR_0_1_PROTO(csp, rgy, RGY_CSP, surfacefmt, cudaVideoSurfaceFormat);

NV_ENC_PIC_STRUCT picstruct_rgy_to_enc(RGY_PICSTRUCT picstruct);
RGY_PICSTRUCT picstruct_enc_to_rgy(NV_ENC_PIC_STRUCT picstruct);

RGY_CSP getEncCsp(NV_ENC_BUFFER_FORMAT enc_buffer_format, const bool alphaChannel, const bool yuv444_as_rgb);

VideoInfo videooutputinfo(
    const GUID& encCodecGUID,
    NV_ENC_BUFFER_FORMAT buffer_fmt,
    int nEncWidth,
    int nEncHeight,
    const NV_ENC_CONFIG *pEncConfig,
    NV_ENC_PIC_STRUCT nPicStruct,
    std::pair<int, int> sar,
    rgy_rational<int> outFps);


struct RGYBitstream {
private:
    uint8_t *dataptr;
    size_t dataLength;
    size_t dataOffset;
    size_t maxLength;
    int64_t  dataDts;
    int64_t  dataPts;
    RGY_FRAME_FLAGS dataFlag;
    uint32_t dataAvgQP;
    RGY_FRAMETYPE dataFrametype;
    RGY_PICSTRUCT dataPicstruct;
    int dataFrameIdx;
    int64_t dataDuration;
    RGYFrameData **frameDataList;
    int frameDataNum;
public:
    uint8_t *bufptr() const {
        return dataptr;
    }

    uint8_t *data() const {
        return dataptr + dataOffset;
    }

    uint8_t *release() {
        uint8_t *ptr = dataptr;
        dataptr = nullptr;
        dataOffset = 0;
        dataLength = 0;
        maxLength = 0;
        return ptr;
    }

    RGY_FRAME_FLAGS dataflag() const {
        return dataFlag;
    }

    void setDataflag(RGY_FRAME_FLAGS flag) {
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

    size_t size() const {
        return dataLength;
    }

    void setSize(size_t size) {
        dataLength = size;
    }

    size_t offset() const {
        return dataOffset;
    }

    void addOffset(size_t add) {
        dataOffset += add;
    }

    void setOffset(size_t offset) {
        dataOffset = offset;
    }

    size_t bufsize() const {
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
        if (dataptr && maxLength) {
            _aligned_free(dataptr);
        }
        dataptr = nullptr;
        clearFrameDataList();
        dataLength = 0;
        dataOffset = 0;
        maxLength = 0;
    }

    RGY_ERR init(size_t nSize) {
        clear();

        if (nSize > 0) {
            if (nullptr == (dataptr = (uint8_t *)_aligned_malloc(nSize, 32))) {
                return RGY_ERR_NULL_PTR;
            }

            maxLength = nSize;
        }
        return RGY_ERR_NONE;
    }

    void trim() {
        if (dataOffset > 0 && dataLength > 0) {
            memmove(dataptr, dataptr + dataOffset, dataLength);
            dataOffset = 0;
        }
    }

    RGY_ERR copy(const uint8_t *setData, size_t setSize) {
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

    RGY_ERR copy(const uint8_t *setData, size_t setSize, int64_t dts, int64_t pts) {
        auto sts = copy(setData, setSize);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        dataDts = dts;
        dataPts = pts;
        return RGY_ERR_NONE;
    }

    RGY_ERR ref(uint8_t *refData, size_t dataSize) {
        clear();
        dataptr = refData;
        dataLength = dataSize;
        dataOffset = 0;
        maxLength = 0;
        return RGY_ERR_NONE;
    }

    RGY_ERR ref(uint8_t *refData, size_t dataSize, int64_t dts, int64_t pts) {
        auto sts = ref(refData, dataSize);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        dataDts = dts;
        dataPts = pts;
        return RGY_ERR_NONE;
    }

    RGY_ERR copy(const RGYBitstream *pBitstream) {
        auto sts = copy(pBitstream->data(), pBitstream->size());
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        return RGY_ERR_NONE;
    }

    RGY_ERR changeSize(size_t nNewSize) {
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

    RGY_ERR append(const uint8_t *appendData, size_t appendSize) {
        if (appendData && appendSize > 0) {
            const auto new_data_length = appendSize + dataLength;
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

    RGY_ERR resize(size_t nNewSize) {
        if (nNewSize > maxLength) {
            return changeSize(nNewSize);
        }
        if (nNewSize + dataOffset > maxLength) {
            memmove(dataptr, dataptr + dataOffset, dataLength);
            dataOffset = 0;
        }
        dataLength = nNewSize;
        return RGY_ERR_NONE;
    }

    void addFrameData(RGYFrameData *frameData);
    void clearFrameDataList();
    std::vector<RGYFrameData *> getFrameDataList();
};

static inline RGYBitstream RGYBitstreamInit() {
    RGYBitstream bitstream;
    memset(&bitstream, 0, sizeof(bitstream));
    return bitstream;
}

#ifndef __CUDACC__
static_assert(std::is_pod<RGYBitstream>::value == true, "RGYBitstream should be POD type.");
#endif
#if 0
struct RGYFrame {
private:
    RGYFrameInfo info;
public:
    RGYFrame() : info() {};
    RGYFrame(const RGYFrameInfo& frameinfo) : info(frameinfo) {};

    RGYFrameInfo getInfo() const {
        return info;
    }
    void set(const RGYFrameInfo& frameinfo) {
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
    void ptrArray(void *array[3], bool bRGB) {
        UNREFERENCED_PARAMETER(bRGB);
        array[0] = info.ptr;
        array[1] = info.ptr + info.pitch * info.height;
        array[2] = info.ptr + info.pitch * info.height * 2;
    }
    RGY_CSP csp() const {
        return info.csp;
    }
    sInputCrop crop() const {
        return sInputCrop();
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
    uint32_t pitch() const {
        return info.pitch;
    }
    uint32_t width() const {
        return info.width;
    }
    uint32_t height() const {
        return info.height;
    }
    uint64_t timestamp() const {
        return info.timestamp;
    }
    void setTimestamp(uint64_t frame_timestamp) {
        info.timestamp = frame_timestamp;
    }
    int64_t duration() const {
        return info.duration;
    }
    void setDuration(int64_t frame_duration) {
        info.duration = frame_duration;
    }
    RGY_PICSTRUCT picstruct() const {
        return info.picstruct;
    }
    void setPicstruct(RGY_PICSTRUCT picstruct) {
        info.picstruct = picstruct;
    }
    RGY_FRAME_FLAGS flags() const {
        return info.flags;
    }
    void setFlags(RGY_FRAME_FLAGS flags) {
        info.flags = flags;
    }
    const std::vector<std::shared_ptr<RGYFrameData>> &dataList() const { return info.dataList; };
    std::vector<std::shared_ptr<RGYFrameData>> &dataList() { return info.dataList; };
};
#endif

static inline RGY_FRAMETYPE frametype_enc_to_rgy(const NV_ENC_PIC_TYPE frametype) {
    RGY_FRAMETYPE type = RGY_FRAMETYPE_UNKNOWN;
    type |=  (frametype == NV_ENC_PIC_TYPE_IDR) ? RGY_FRAMETYPE_IDR : RGY_FRAMETYPE_UNKNOWN;
    type |=  (frametype == NV_ENC_PIC_TYPE_I  ) ? RGY_FRAMETYPE_I   : RGY_FRAMETYPE_UNKNOWN;
    type |=  (frametype == NV_ENC_PIC_TYPE_P  ) ? RGY_FRAMETYPE_P   : RGY_FRAMETYPE_UNKNOWN;
    type |=  (frametype == NV_ENC_PIC_TYPE_B  ) ? RGY_FRAMETYPE_B   : RGY_FRAMETYPE_UNKNOWN;
    return type;
}

static inline RGYBitstream RGYBitstreamInit(const NV_ENC_LOCK_BITSTREAM& nv_bitstream) {
    RGYBitstream bitstream;
    memset(&bitstream, 0, sizeof(bitstream));
    bitstream.ref((uint8_t *)nv_bitstream.bitstreamBufferPtr, nv_bitstream.bitstreamSizeInBytes, (int64_t)0, (int64_t)nv_bitstream.outputTimeStamp);
    bitstream.setAvgQP(nv_bitstream.frameAvgQP);
    bitstream.setFrametype(frametype_enc_to_rgy(nv_bitstream.pictureType));
    bitstream.setPicstruct(picstruct_enc_to_rgy(nv_bitstream.pictureStruct));
    bitstream.setFrameIdx(nv_bitstream.frameIdx);
    bitstream.setDuration(nv_bitstream.outputDuration);
    return bitstream;
}

int64_t rational_rescale(int64_t v, rgy_rational<int> from, rgy_rational<int> to);

#endif //__NVENC_UTIL_H__
