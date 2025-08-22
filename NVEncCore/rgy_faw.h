﻿// -----------------------------------------------------------------------------------------
// QSVEnc/NVEnc by rigaya
// -----------------------------------------------------------------------------------------
// The MIT License
//
// Copyright (c) 2023 rigaya
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

#ifndef __RGY_FAW_H__
#define __RGY_FAW_H__

#include <cstdint>
#include <array>
#include <vector>
#include "rgy_wav_parser.h"
#include "rgy_memmem.h"
#include "rgy_bitstream_aac.h"

static const std::array<uint8_t, 8> fawstart1 = {
    0x72, 0xF8, 0x1F, 0x4E, 0x07, 0x01, 0x00, 0x00
};
static const std::array<uint8_t, 16> fawstart2 = {
    0x00, 0xF2, 0x00, 0x78, 0x00, 0x9F, 0x00, 0xCE,
    0x00, 0x87, 0x00, 0x81, 0x00, 0x80, 0x00, 0x80
};
static const std::array<uint8_t, 12> fawfin1 = {
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x45, 0x4E, 0x44, 0x00
};
static const std::array<uint8_t, 24> fawfin2 = {
    0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80,
    0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80,
    0x00, 0xC5, 0x00, 0xCE, 0x00, 0xC4, 0x00, 0x80
};

size_t rgy_memmem_fawstart1_c(const void *data_, const size_t data_size);
size_t rgy_memmem_fawstart1_avx2(const void *data_, const size_t data_size);
size_t rgy_memmem_fawstart1_avx512bw(const void *data_, const size_t data_size);

void rgy_convert_audio_16to8(uint8_t *dst, const short *src, const size_t n);
void rgy_convert_audio_16to8_avx2(uint8_t *dst, const short *src, const size_t n);

void rgy_split_audio_16to8x2(uint8_t *dst0, uint8_t *dst1, const short *src, const size_t n);
void rgy_split_audio_16to8x2_avx2(uint8_t *dst0, uint8_t *dst1, const short *src, const size_t n);

using RGYFAWDecoderOutput = std::array<std::vector<uint8_t>, 2>;

enum class RGYFAWMode {
    Unknown,
    Full,
    Half,
    Mix
};

class RGYFAWBitstream {
private:
    std::vector<uint8_t> buffer;
    size_t bufferOffset;
    size_t bufferLength;

    int bytePerWholeSample; // channels * bits per sample
    uint64_t inputLengthByte;
    uint64_t outSamples;

    RGYAACHeader aacHeader;
public:
    RGYFAWBitstream();
    ~RGYFAWBitstream();

    void setBytePerSample(const int val);

    uint8_t *data() { return buffer.data() + bufferOffset; }
    const uint8_t *data() const { return buffer.data() + bufferOffset; }
    size_t size() const { return bufferLength; }
    uint64_t inputLength() const { return inputLengthByte; }
    uint64_t inputSampleStart() const { return (inputLengthByte - bufferLength) / bytePerWholeSample; }
    uint64_t inputSampleFin() const { return inputLengthByte / bytePerWholeSample; }
    uint64_t outputSamples() const { return outSamples; }
    int bytePerSample() const { return bytePerWholeSample; }

    void addOffset(size_t offset);
    void addOutputSamples(size_t samples);

    void append(const uint8_t *input, const size_t inputLength);

    void clear();

    void parseAACHeader(const uint8_t *buffer);
    uint32_t aacChannels() const;
    uint32_t aacFrameSize() const;
};

class RGYFAWDecoder {
private:
    RGYWAVHeader wavheader;
    RGYFAWMode fawmode;

    RGYFAWBitstream bufferIn;

    RGYFAWBitstream bufferHalf0;
    RGYFAWBitstream bufferHalf1;

    decltype(rgy_memmem_c)* funcMemMem;
    decltype(rgy_memmem_fawstart1_c)* funcMemMemFAWStart1;
    decltype(rgy_convert_audio_16to8)* funcAudio16to8;
    decltype(rgy_split_audio_16to8x2)* funcSplitAudio16to8x2;
public:
    RGYFAWDecoder();
    ~RGYFAWDecoder();

    RGYFAWMode mode() const { return fawmode; }
    int init(const uint8_t *data);
    int init(const RGYWAVHeader *data);
    int decode(RGYFAWDecoderOutput& output, const uint8_t *data, const size_t dataLength);
    void fin(RGYFAWDecoderOutput& output);
private:
    void appendFAWHalf(const uint8_t *data, const size_t dataLength);
    void appendFAWMix(const uint8_t *data, const size_t dataLength);

    void setWavInfo();
    int decode(std::vector<uint8_t>& output, RGYFAWBitstream& input);
    int decodeBlock(std::vector<uint8_t>& output, RGYFAWBitstream& input);
    void addSilent(std::vector<uint8_t>& output, RGYFAWBitstream& input);
    void fin(std::vector<uint8_t>& output, RGYFAWBitstream& input);
};

class RGYFAWEncoder {
private:
    RGYWAVHeader wavheader;
    RGYFAWMode fawmode;
    int delaySamples;

    int64_t inputAACPosByte;
    int64_t outputFAWPosByte;
    RGYFAWBitstream bufferIn;
    RGYFAWBitstream bufferTmp;
public:
    RGYFAWEncoder();
    ~RGYFAWEncoder();

    int init(const RGYWAVHeader *data, const RGYFAWMode mode, const int delayMillisec);
    int encode(std::vector<uint8_t>& output, const uint8_t *data, const size_t dataLength);
    int fin(std::vector<uint8_t>& output);
private:
    int encode(std::vector<uint8_t>& output);
    void encodeBlock(const uint8_t *data, const size_t dataLength);
};

#endif //__RGY_FAW_H__