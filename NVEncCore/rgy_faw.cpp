// -----------------------------------------------------------------------------------------
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

#include <vector>
#include <array>
#include "rgy_faw.h"
#include "rgy_simd.h"

size_t rgy_memmem_fawstart1_c(const void *data_, const size_t data_size) {
    return rgy_memmem_c(data_, data_size, fawstart1.data(), fawstart1.size());
}

decltype(rgy_memmem_fawstart1_c)* get_memmem_fawstart1_func() {
#if defined(_M_IX86) || defined(_M_X64) || defined(__x86_64)
    const auto simd = get_availableSIMD();
#if defined(_M_X64) || defined(__x86_64)
    if ((simd & RGY_SIMD::AVX512BW) == RGY_SIMD::AVX512BW) return rgy_memmem_fawstart1_avx512bw;
#endif
    if ((simd & RGY_SIMD::AVX2) == RGY_SIMD::AVX2) return rgy_memmem_fawstart1_avx2;
#endif
    return rgy_memmem_fawstart1_c;
}

static const std::array<uint8_t, 2> AACSYNC_BYTES = { 0xff, 0xf0 };

static size_t rgy_find_aacsync_c(const void *data_, const size_t data_size) {
    const uint16_t target = *(const uint16_t *)AACSYNC_BYTES.data();
    const size_t target_size = AACSYNC_BYTES.size();
    const uint8_t *data = (const uint8_t *)data_;
    if (data_size < target_size) {
        return RGY_MEMMEM_NOT_FOUND;
    }
    for (size_t i = 0; i <= data_size - target_size; i++) {
        if ((*(const uint16_t *)(data + i) & target) == target) {
            return i;
        }
    }
    return RGY_MEMMEM_NOT_FOUND;
}

//16bit音声 -> 8bit音声
void rgy_convert_audio_16to8(uint8_t *dst, const short *src, const size_t n) {
    uint8_t *byte = dst;
    const uint8_t *fin = byte + n;
    const short *sh = src;
    while (byte < fin) {
        *byte = (*sh >> 8) + 128;
        byte++;
        sh++;
    }
}

void rgy_split_audio_16to8x2(uint8_t *dst0, uint8_t *dst1, const short *src, const size_t n) {
    const short *sh = src;
    const short *sh_fin = src + n;
    for (; sh < sh_fin; sh++, dst0++, dst1++) {
        *dst0 = (*sh >> 8) + 128;
        *dst1 = (*sh & 0xff) + 128;
    }
}

decltype(rgy_convert_audio_16to8)* get_convert_audio_16to8_func() {
#if defined(_M_IX86) || defined(_M_X64) || defined(__x86_64)
    const auto simd = get_availableSIMD();
    if ((simd & RGY_SIMD::AVX2) == RGY_SIMD::AVX2) return rgy_convert_audio_16to8_avx2;
#endif
    return rgy_convert_audio_16to8;
}

decltype(rgy_split_audio_16to8x2)* get_split_audio_16to8x2_func() {
#if defined(_M_IX86) || defined(_M_X64) || defined(__x86_64)
    const auto simd = get_availableSIMD();
    if ((simd & RGY_SIMD::AVX2) == RGY_SIMD::AVX2) return rgy_split_audio_16to8x2_avx2;
#endif
    return rgy_split_audio_16to8x2;
}

template<bool upperhalf>
static uint8_t faw_read_half(const uint16_t v) {
    uint8_t i = (upperhalf) ? (v & 0xff00) >> 8 : (v & 0xff);
    return i - 0x80;
}

template<bool ishalf, bool upperhalf>
void faw_read(uint8_t *dst, const uint8_t *src, const size_t outlen) {
    if (!ishalf) {
        memcpy(dst, src, outlen);
        return;
    }
    const uint16_t *srcPtr = (const uint16_t *)src;
    for (size_t i = 0; i < outlen; i++) {
        dst[i] = faw_read_half<upperhalf>(srcPtr[i]);
    }
}

static uint32_t faw_checksum_calc(const uint8_t *buf, const size_t len) {
    uint32_t _v4288 = 0;
    uint32_t _v48 = 0;
    const size_t fin_mod2 = (len & (~1));
    for (size_t i = 0; i < fin_mod2; i += 2) {
        uint32_t _v132 = *(uint16_t *)(buf + i);
        _v4288 += _v132;
        _v48 ^= _v132;
    }
    if ((len & 1) != 0) {
        uint32_t _v132 = *(uint8_t *)(buf + len - 1);
        _v4288 += _v132;
        _v48 ^= _v132;
    }
    uint32_t res = (_v4288 & 0xffff) | ((_v48 & 0xffff) << 16);
    return res;
}

static uint32_t faw_checksum_read(const uint8_t *buf) {
    uint32_t v;
    memcpy(&v, buf, sizeof(v));
    return v;
}

RGYFAWBitstream::RGYFAWBitstream() :
    buffer(),
    bufferOffset(0),
    bufferLength(0),
    bytePerWholeSample(0),
    inputLengthByte(0),
    outSamples(0),
    aacHeader() {

}

RGYFAWBitstream::~RGYFAWBitstream() {};

void RGYFAWBitstream::setBytePerSample(const int val) {
    bytePerWholeSample = val;
}

void RGYFAWBitstream::parseAACHeader(const uint8_t *buf) {
    aacHeader.parse(buf, RGYAACHeader::HEADER_BYTE_SIZE);
}

uint32_t RGYFAWBitstream::aacChannels() const {
    return aacHeader.channel;
}
uint32_t RGYFAWBitstream::aacFrameSize() const {
    return aacHeader.aac_frame_length;
}

void RGYFAWBitstream::addOffset(size_t offset) {
    if (bufferLength < offset) {
        bufferLength = 0;
    } else {
        bufferLength -= offset;
    }
    if (bufferLength == 0) {
        bufferOffset = 0;
    } else {
        bufferOffset += offset;
    }
}

void RGYFAWBitstream::addOutputSamples(size_t samples) {
    outSamples += samples;
}


void RGYFAWBitstream::append(const uint8_t *input, const size_t inputLength) {
    if (buffer.size() < bufferLength + inputLength) {
        buffer.resize(std::max(bufferLength + inputLength, buffer.size() * 2));
        if (bufferLength == 0) {
            bufferOffset = 0;
        }
        if (bufferOffset > 0) {
            memmove(buffer.data(), buffer.data() + bufferOffset, bufferLength);
            bufferOffset = 0;
        }
    } else if (buffer.size() < bufferOffset + bufferLength + inputLength) {
        if (bufferLength == 0) {
            bufferOffset = 0;
        }
        if (bufferOffset > 0) {
            memmove(buffer.data(), buffer.data() + bufferOffset, bufferLength);
            bufferOffset = 0;
        }
    }
    if (input != nullptr) {
        memcpy(buffer.data() + bufferOffset + bufferLength, input, inputLength);
    }
    bufferLength += inputLength;
    inputLengthByte += inputLength;
}

void RGYFAWBitstream::clear() {
    bufferLength = 0;
    bufferOffset = 0;
    inputLengthByte = 0;
    outSamples = 0;
}

static const std::array<uint8_t, 16> aac_silent0 = {
    0xFF, 0xF9, 0x4C, 0x00, 0x02, 0x1F, 0xFC, 0x21,
    0x00, 0x49, 0x90, 0x02, 0x19, 0x00, 0x23, 0x80
};
static const std::array<uint8_t, 13> aac_silent1 = {
    0xFF, 0xF9, 0x4C, 0x40, 0x01, 0xBF, 0xFC, 0x00,
    0xC8, 0x40, 0x80, 0x23, 0x80
};
static const std::array<uint8_t, 16> aac_silent2 = {
    0xFF, 0xF9, 0x4C, 0x80, 0x02, 0x1F, 0xFC, 0x21,
    0x00, 0x49, 0x90, 0x02, 0x19, 0x00, 0x23, 0x80
};
static const std::array<uint8_t, 33> aac_silent6 = {
    0xFF, 0xF9, 0x4D, 0x80, 0x04, 0x3F, 0xFC, 0x00,
    0xC8, 0x00, 0x80, 0x20, 0x84, 0x01, 0x26, 0x40,
    0x08, 0x64, 0x00, 0x82, 0x30, 0x04, 0x99, 0x00,
    0x21, 0x90, 0x02, 0x18, 0x32, 0x00, 0x20, 0x08,
    0xE0
};

RGYFAWDecoder::RGYFAWDecoder() :
    wavheader(),
    fawmode(RGYFAWMode::Unknown),
    bufferIn(),
    bufferHalf0(),
    bufferHalf1(),
    funcMemMem(get_memmem_func()),
    funcMemMemFAWStart1(get_memmem_fawstart1_func()),
    funcAudio16to8(get_convert_audio_16to8_func()),
    funcSplitAudio16to8x2(get_split_audio_16to8x2_func()) {
}
RGYFAWDecoder::~RGYFAWDecoder() {

}

void RGYFAWDecoder::setWavInfo() {
    bufferIn.setBytePerSample(wavheader.number_of_channels * wavheader.bits_per_sample / 8);
    if (wavheader.bits_per_sample > 8) {
        bufferHalf0.setBytePerSample(wavheader.number_of_channels * wavheader.bits_per_sample / 16);
        bufferHalf1.setBytePerSample(wavheader.number_of_channels * wavheader.bits_per_sample / 16);
    }
}

int RGYFAWDecoder::init(const uint8_t *data) {
    int headerSize = wavheader.parseHeader(data);
    setWavInfo();
    return headerSize;
}

int RGYFAWDecoder::init(const RGYWAVHeader *data) {
    wavheader = *data;
    setWavInfo();
    return 0;
}

void RGYFAWDecoder::appendFAWHalf(const uint8_t *data, const size_t dataLength) {
    const auto prevSize = bufferHalf0.size();
    bufferHalf0.append(nullptr, dataLength / sizeof(short));
    funcAudio16to8(bufferHalf0.data() + prevSize, (const short *)data, dataLength / sizeof(short));
}

void RGYFAWDecoder::appendFAWMix(const uint8_t *data, const size_t dataLength) {
    const auto prevSize0 = bufferHalf0.size();
    const auto prevSize1 = bufferHalf1.size();
    bufferHalf0.append(nullptr, dataLength / sizeof(short));
    bufferHalf1.append(nullptr, dataLength / sizeof(short));
    funcSplitAudio16to8x2(bufferHalf0.data() + prevSize0, bufferHalf1.data() + prevSize1, (const short *)data, dataLength / sizeof(short));
}

int RGYFAWDecoder::decode(RGYFAWDecoderOutput& output, const uint8_t *input, const size_t inputLength) {
    for (auto& b : output) {
        b.clear();
    }

    bool inputDataAppended = false;

    // FAWの種類を判別
    if (fawmode == RGYFAWMode::Unknown) {
        bufferIn.append(input, inputLength);
        inputDataAppended = true;

        decltype(funcMemMemFAWStart1(nullptr, 0)) ret0 = 0, ret1 = 0;
        if ((ret0 = funcMemMemFAWStart1(bufferIn.data(), bufferIn.size())) != RGY_MEMMEM_NOT_FOUND) {
            fawmode = RGYFAWMode::Full;
        } else if ((ret0 = funcMemMem(bufferIn.data(), bufferIn.size(), fawstart2.data(), fawstart2.size())) != RGY_MEMMEM_NOT_FOUND) {
            fawmode = RGYFAWMode::Half;
            appendFAWHalf(bufferIn.data(), bufferIn.size());
            bufferIn.clear();
        } else {
            appendFAWMix(bufferIn.data(), bufferIn.size());
            if (   (ret0 = funcMemMemFAWStart1(bufferHalf0.data(), bufferHalf0.size())) != RGY_MEMMEM_NOT_FOUND
                && (ret1 = funcMemMemFAWStart1(bufferHalf1.data(), bufferHalf1.size())) != RGY_MEMMEM_NOT_FOUND) {
                fawmode = RGYFAWMode::Mix;
                bufferIn.clear();
            } else {
                bufferHalf0.clear();
                bufferHalf1.clear();
            }
        }
    }
    if (fawmode == RGYFAWMode::Unknown) {
        return -1;
    }
    if (!inputDataAppended) {
        if (fawmode == RGYFAWMode::Full) {
            bufferIn.append(input, inputLength);
        } else if (fawmode == RGYFAWMode::Half) {
            appendFAWHalf(input, inputLength);
        } else if (fawmode == RGYFAWMode::Mix) {
            appendFAWMix(input, inputLength);
        }
        inputDataAppended = true;
    }

    // デコード
    if (fawmode == RGYFAWMode::Full) {
        decode(output[0], bufferIn);
    } else if (fawmode == RGYFAWMode::Half) {
        decode(output[0], bufferHalf0);
    } else if (fawmode == RGYFAWMode::Mix) {
        decode(output[0], bufferHalf0);
        decode(output[1], bufferHalf1);
    }
    return 0;
}

int RGYFAWDecoder::decode(std::vector<uint8_t>& output, RGYFAWBitstream& input) {
    while (input.size() > 0) {
        auto ret = decodeBlock(output, input);
        if (ret == 0) {
            break;
        }
    }
    return 0;
}

int RGYFAWDecoder::decodeBlock(std::vector<uint8_t>& output, RGYFAWBitstream& input) {
    auto posStart = funcMemMemFAWStart1(input.data(), input.size());
    if (posStart == RGY_MEMMEM_NOT_FOUND) {
        return 0;
    }
    input.parseAACHeader(input.data() + posStart + fawstart1.size());

    auto posFin = funcMemMem(input.data() + posStart + fawstart1.size(), input.size() - posStart - fawstart1.size(), fawfin1.data(), fawfin1.size());
    if (posFin == RGY_MEMMEM_NOT_FOUND) {
        return 0;
    }
    posFin += posStart + fawstart1.size(); // データの先頭からの位置に変更

    // pos_start から pos_fin までの間に、別のfawstart1がないか探索する
    while (posStart + fawstart1.size() < posFin) {
        auto ret = funcMemMemFAWStart1(input.data() + posStart + fawstart1.size(), posFin - posStart - fawstart1.size());
        if (ret == RGY_MEMMEM_NOT_FOUND) {
            break;
        }
        posStart += ret + fawstart1.size();
        input.parseAACHeader(input.data() + posStart + fawstart1.size());
    }

    if (posStart + fawstart1.size() + 4 >= posFin) {
        // 無効なブロックなので破棄
        input.addOffset(posFin + fawfin1.size());
        return 1;
    }
    const size_t blockSize = posFin - posStart - fawstart1.size() - 4 /*checksum*/;
    const uint32_t checksumCalc = faw_checksum_calc(input.data() + posStart + fawstart1.size(), blockSize);
    const uint32_t checksumRead = faw_checksum_read(input.data() + posFin - 4);
    // checksumとフレーム長が一致しない場合、そのデータは破棄
    if (checksumCalc != checksumRead || blockSize != input.aacFrameSize()) {
        input.addOffset(posFin + fawfin1.size());
        return 1;
    }

    // pos_start -> sample start
    const auto posStartSample = input.inputSampleStart() + posStart / input.bytePerSample();
    //fprintf(stderr, "Found block: %lld\n", posStartSample);

    // 出力が先行していたらdrop
    if (posStartSample + (AAC_BLOCK_SAMPLES / 2) < input.outputSamples()) {
        input.addOffset(posFin + fawfin1.size());
        return 1;
    }

    // 時刻ずれを無音データで補正
    while (input.outputSamples() + (AAC_BLOCK_SAMPLES/2) < posStartSample) {
        //fprintf(stderr, "Insert silence: %lld: %lld -> %lld\n", posStartSample, input.outputSamples(), input.outputSamples() + AAC_BLOCK_SAMPLES);
        addSilent(output, input);
    }

    // ブロックを出力に追加
    const auto orig_size = output.size();
    output.resize(orig_size + blockSize);
    memcpy(output.data() + orig_size, input.data() + posStart + fawstart1.size(), blockSize);
    //fprintf(stderr, "Set block: %lld: %lld -> %lld\n", posStartSample, input.outputSamples(), input.outputSamples() + AAC_BLOCK_SAMPLES);

    input.addOutputSamples(AAC_BLOCK_SAMPLES);
    input.addOffset(posFin + fawfin1.size());
    return 1;
}

void RGYFAWDecoder::addSilent(std::vector<uint8_t>& output, RGYFAWBitstream& input) {
    auto ptrSilent = aac_silent0.data();
    auto dataSize = aac_silent0.size();
    switch (input.aacChannels()) {
    case 0:
        break;
    case 1:
        ptrSilent = aac_silent1.data();
        dataSize = aac_silent1.size();
        break;
    case 6:
        ptrSilent = aac_silent6.data();
        dataSize = aac_silent6.size();
        break;
    case 2:
    default:
        ptrSilent = aac_silent2.data();
        dataSize = aac_silent2.size();
        break;
    }
    const auto orig_size = output.size();
    output.resize(orig_size + dataSize);
    memcpy(output.data() + orig_size, ptrSilent, dataSize);
    input.addOutputSamples(AAC_BLOCK_SAMPLES);
}

void RGYFAWDecoder::fin(RGYFAWDecoderOutput& output) {
    for (auto& b : output) {
        b.clear();
    }
    if (fawmode == RGYFAWMode::Full) {
        fin(output[0], bufferIn);
    } else if (fawmode == RGYFAWMode::Half) {
        fin(output[0], bufferHalf0);
    } else if (fawmode == RGYFAWMode::Mix) {
        fin(output[0], bufferHalf0);
        fin(output[1], bufferHalf1);
    }
}

void RGYFAWDecoder::fin(std::vector<uint8_t>& output, RGYFAWBitstream& input) {
    //fprintf(stderr, "Fin sample: %lld\n", input.inputSampleFin());
    while (input.outputSamples() + (AAC_BLOCK_SAMPLES / 2) < input.inputSampleFin()) {
        //fprintf(stderr, "Insert silence: %lld -> %lld\n", input.outputSamples(), input.outputSamples() + AAC_BLOCK_SAMPLES);
        addSilent(output, input);
    }
}

RGYFAWEncoder::RGYFAWEncoder() :
    wavheader(),
    fawmode(),
    delaySamples(0),
    inputAACPosByte(0),
    outputFAWPosByte(0),
    bufferIn(),
    bufferTmp() {

}

RGYFAWEncoder::~RGYFAWEncoder() {

}

int RGYFAWEncoder::init(const RGYWAVHeader *data, const RGYFAWMode mode, const int delayMillisec) {
    wavheader = *data;
    fawmode = mode;
    bufferTmp.setBytePerSample(wavheader.number_of_channels * wavheader.bits_per_sample / 8);
    delaySamples = delayMillisec * (int)wavheader.sample_rate / 1000;
    inputAACPosByte += delaySamples * bufferTmp.bytePerSample();
    return 0;
}

int RGYFAWEncoder::encode(std::vector<uint8_t>& output, const uint8_t *input, const size_t inputLength) {
    output.clear();
    bufferTmp.clear();

    if (fawmode == RGYFAWMode::Unknown) {
        return -1;
    }

    bufferIn.append(input, inputLength);

    const auto ret = rgy_find_aacsync_c(bufferIn.data(), bufferIn.size());
    if (ret == RGY_MEMMEM_NOT_FOUND) {
        return 0;
    }
    bufferIn.addOffset(ret);
    return encode(output);
}

int RGYFAWEncoder::encode(std::vector<uint8_t>& output) {
    if (bufferIn.size() < AAC_HEADER_MIN_SIZE) {
        return 0;
    }
    bufferIn.parseAACHeader(bufferIn.data());
    auto aacBlockSize = bufferIn.aacFrameSize();
    if (aacBlockSize > bufferIn.size()) {
        return 0;
    }
    auto ret0 = rgy_find_aacsync_c(bufferIn.data() + aacBlockSize, bufferIn.size() - aacBlockSize);
    while (ret0 != RGY_MEMMEM_NOT_FOUND) {
        ret0 += aacBlockSize;
        if (inputAACPosByte < outputFAWPosByte) {
            ; // このブロックを破棄
        } else {
            if (outputFAWPosByte < inputAACPosByte) {
                const auto offsetBytes = inputAACPosByte - outputFAWPosByte;
                const auto origSize = bufferTmp.size();
                bufferTmp.append(nullptr, (size_t)offsetBytes);
                memset(bufferTmp.data() + origSize, 0, (size_t)offsetBytes);
                outputFAWPosByte = inputAACPosByte;
            }
            // outputWavPosSample == inputAACPosSample
            encodeBlock(bufferIn.data(), aacBlockSize);
        }
        inputAACPosByte += AAC_BLOCK_SAMPLES * bufferTmp.bytePerSample();

        bufferIn.addOffset(ret0);
        if (bufferIn.size() < AAC_HEADER_MIN_SIZE) {
            break;
        }
        bufferIn.parseAACHeader(bufferIn.data());
        aacBlockSize = bufferIn.aacFrameSize();
        if (aacBlockSize > bufferIn.size()) {
            break;
        }
        ret0 = rgy_find_aacsync_c(bufferIn.data() + aacBlockSize, bufferIn.size() - aacBlockSize);
    }

    output.resize(bufferTmp.size());
    memcpy(output.data(), bufferTmp.data(), bufferTmp.size());
    bufferTmp.clear();
    return 0;
}

void RGYFAWEncoder::encodeBlock(const uint8_t *data, const size_t dataLength) {
    const uint32_t checksumCalc = faw_checksum_calc(data, dataLength);

    bufferTmp.append(fawstart1.data(), fawstart1.size());
    outputFAWPosByte += fawstart1.size();

    bufferTmp.append(data, dataLength);
    outputFAWPosByte += dataLength;

    bufferTmp.append((const uint8_t *)&checksumCalc, sizeof(checksumCalc));
    outputFAWPosByte += sizeof(checksumCalc);

    bufferTmp.append(fawfin1.data(), fawfin1.size());
    outputFAWPosByte += fawfin1.size();
}

int RGYFAWEncoder::fin(std::vector<uint8_t>& output) {
    output.clear();
    bufferIn.append(AACSYNC_BYTES.data(), AACSYNC_BYTES.size());
    auto ret = encode(output);
    if (outputFAWPosByte < inputAACPosByte) {
        // 残りのbyteを0で調整
        const auto offsetBytes = inputAACPosByte - outputFAWPosByte;
        output.resize(output.size() + (size_t)offsetBytes, 0);
    }
    if (delaySamples < 0) {
        // 負のdelayの場合、wavの長さを合わせるために0で埋める
        const auto offsetBytes = -1 * delaySamples * bufferTmp.bytePerSample();
        output.resize(output.size() + offsetBytes, 0);
    }
    //最終出力は4byte少ない (先頭に4byte入れたためと思われる)
    if (output.size() > 4) {
        output.resize(output.size() - 4);
    }
    return ret;
}
