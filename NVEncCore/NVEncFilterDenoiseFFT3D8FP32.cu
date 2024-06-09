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

#include <array>
#include <map>
#include "convert_csp.h"
#include "NVEncFilterDenoiseFFT3D.h"
#include "NVEncFilterDenoiseFFT3D.cuh"

// 並列コンパイルによりコンパイルを高速化するため、ファイルをTypePixelとFFTのprecisionで分割する
std::unique_ptr<DenoiseFFT3DBase> getDenoiseFFT3DFunc8FP32(const int block_size) {
    switch (block_size) {
    case  8: return std::unique_ptr<DenoiseFFT3DBase>(new DenoiseFFT3DFuncs<uint8_t, 8, float2, 8, 8>());
    case 16: return std::unique_ptr<DenoiseFFT3DBase>(new DenoiseFFT3DFuncs<uint8_t, 8, float2, 16, 4>());
    case 32: return std::unique_ptr<DenoiseFFT3DBase>(new DenoiseFFT3DFuncs<uint8_t, 8, float2, 32, 2>());
    case 64: return std::unique_ptr<DenoiseFFT3DBase>(new DenoiseFFT3DFuncs<uint8_t, 8, float2, 64, 1>());
    default: return nullptr;
    }
}
