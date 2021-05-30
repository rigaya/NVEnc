// -----------------------------------------------------------------------------------------
// QSVEnc/NVEnc by rigaya
// -----------------------------------------------------------------------------------------
// The MIT License
//
// Copyright (c) 2011-2016 rigaya
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
// --------------------------------------------------------------------------------------------

#pragma once
#ifndef __RGY_STREAM_H__
#define __RGY_STREAM_H__

#include <cstdint>
#include <cassert>
#include "rgy_osdep.h"
#include "rgy_err.h"

class rgy_stream {
    uint8_t *bufptr_;
    size_t buf_size_;
    size_t data_length_;
    int64_t offset_;

    uint32_t data_flag_;
    int duration_;
    int64_t pts_;
    int64_t dts_;
public:
    rgy_stream() :
        bufptr_(nullptr),
        buf_size_(0),
        data_length_(0),
        offset_(0),
        data_flag_(0),
        duration_(0),
        pts_(0),
        dts_(0) {
    };
    ~rgy_stream() {
        if (bufptr_) {
            _aligned_free(bufptr_);
        }
        bufptr_ = nullptr;
        buf_size_ = 0;
    }
    uint8_t *bufptr() const {
        return bufptr_;
    }
    uint8_t *data() const {
        return bufptr_ + offset_;
    }
    size_t size() const {
        return data_length_;
    }
    size_t buf_size() const {
        return buf_size_;
    }
    void add_offset(size_t add) {
        if (data_length_ < add) {
            add = data_length_;
        }
        offset_ += add;
        data_length_ -= add;
        assert(offset_ >= 0);
    }

    void clear() {
        data_length_ = 0;
        offset_ = 0;
    }
    RGY_ERR alloc(size_t size) {
        clear();
        if (bufptr_) {
            _aligned_free(bufptr_);
        }
        bufptr_ = nullptr;
        buf_size_ = 0;

        if (size > 0) {
            if (nullptr == (bufptr_ = (uint8_t *)_aligned_malloc(size, 32))) {
                return RGY_ERR_NULL_PTR;
            }
            buf_size_ = size;
        }
        return RGY_ERR_NONE;
    }
    RGY_ERR realloc(size_t size) {
        if (bufptr_ == nullptr || data_length_ == 0) {
            return alloc(size);
        }
        if (size > 0) {
            auto newptr = (uint8_t *)_aligned_malloc(size, 32);
            if (newptr == nullptr) {
                return RGY_ERR_NULL_PTR;
            }
            auto newdatalen = (std::min)(size, data_length_);
            memcpy(newptr, bufptr_ + offset_, newdatalen);
            _aligned_free(bufptr_);
            bufptr_ = newptr;
            buf_size_ = size;
            offset_ = 0;
            data_length_ = newdatalen;
        }
        return RGY_ERR_NONE;
    }
    void init() {
        bufptr_ = nullptr;
        buf_size_ = 0;
        data_length_ = 0;
        offset_ = 0;

        data_flag_ = 0;
        duration_ = 0;
        pts_ = 0;
        dts_ = 0;
    }

    void trim() {
        if (offset_ > 0 && data_length_ > 0) {
            memmove(bufptr_, bufptr_ + offset_, data_length_);
            offset_ = 0;
        }
    }

    RGY_ERR copy(const uint8_t *data, size_t size) {
        if (data == nullptr || size == 0) {
            return RGY_ERR_MORE_BITSTREAM;
        }
        if (buf_size_ < size) {
            clear();
            auto sts = alloc(size);
            if (sts != RGY_ERR_NONE) {
                return sts;
            }
        }
        data_length_ = size;
        offset_ = 0;
        memcpy(bufptr_, data, size);
        return RGY_ERR_NONE;
    }

    RGY_ERR copy(const uint8_t *data, size_t size, int64_t pts) {
        pts_ = pts;
        return copy(data, size);
    }

    RGY_ERR copy(const uint8_t *data, size_t size, int64_t pts, int64_t dts) {
        dts_ = dts;
        return copy(data, size, pts);
    }

    RGY_ERR copy(const uint8_t *data, size_t size, int64_t pts, int64_t dts, int duration) {
        duration_ = duration;
        return copy(data, size, pts, dts);
    }

    RGY_ERR copy(const uint8_t *data, size_t size, int64_t pts, int64_t dts, int duration, uint32_t flag) {
        data_flag_ = flag;
        return copy(data, size, pts, dts, duration);
    }

    RGY_ERR copy(const rgy_stream *pBitstream) {
        auto sts = copy(pBitstream->data(), pBitstream->size());
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        return copy(pBitstream->data(), pBitstream->size(), pBitstream->pts(), pBitstream->dts(), pBitstream->duration(), pBitstream->data_flag());
    }

    RGY_ERR append(const uint8_t *append_data, size_t append_size) {
        if (append_data && append_size > 0) {
            const auto new_data_length = data_length_ + append_size;
            if (buf_size_ < new_data_length) {
                auto sts = realloc(new_data_length + (std::min<size_t>)(new_data_length / 2, 256 * 1024u));
                if (sts != RGY_ERR_NONE) {
                    return sts;
                }
            }

            if (buf_size_ < new_data_length + offset_) {
                memmove(bufptr_, bufptr_ + offset_, data_length_);
                offset_ = 0;
            }
            assert(new_data_length + offset_ <= buf_size_);
            memcpy(bufptr_ + offset_ + data_length_, append_data, append_size);
            data_length_ = new_data_length;
        }
        return RGY_ERR_NONE;
    }

    uint32_t data_flag() const {
        return data_flag_;
    }
    void set_data_flag(uint32_t flag) {
        data_flag_  = flag;
    }
    int duration() const {
        return duration_;
    }
    void set_duration(int duration) {
        duration_ = duration;
    }
    int64_t pts() const {
        return pts_;
    }
    void set_pts(int64_t pts) {
        pts_ = pts;
    }
    int64_t dts() const {
        return dts_;
    }
    void set_dts(int64_t dts) {
        dts_ = dts;
    }
};

#endif //__RGY_STREAM_H__
