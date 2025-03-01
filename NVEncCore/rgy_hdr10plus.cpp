// -----------------------------------------------------------------------------------------
// QSVEnc/NVEnc by rigaya
// -----------------------------------------------------------------------------------------
// The MIT License
//
// Copyright (c) 2019 rigaya
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

#include "rgy_osdep.h"
#include "rgy_hdr10plus.h"
#include "rgy_filesystem.h"
#include "rgy_util.h"
#if ENABLE_LIBHDR10PLUS
#include <libhdr10plus-rs/hdr10plus.h>

#if defined(_WIN32) || defined(_WIN64)
#pragma comment(lib, "hdr10plus-rs.lib")
#pragma comment(lib, "ws2_32.lib")
#pragma comment(lib, "ntdll.lib")
#pragma comment(lib, "Userenv.lib")
#endif

#endif

RGYHDR10Plus::RGYHDR10Plus() :
    m_hdr10plusJson(std::unique_ptr<Hdr10PlusRsJsonOpaque, funcHdr10PlusRsJsonOpaqueDelete>(nullptr, nullptr)),
    m_inputJson() {
}

RGYHDR10Plus::~RGYHDR10Plus() {
}

RGY_ERR RGYHDR10Plus::init(const tstring &inputJson) {
#if ENABLE_LIBHDR10PLUS
    if (!(rgy_file_exists(inputJson))) {
        return RGY_ERR_NOT_FOUND;
    }
    m_inputJson = inputJson;

    auto inputJsonStr = tchar_to_string(inputJson);
    m_hdr10plusJson = std::unique_ptr<Hdr10PlusRsJsonOpaque, funcHdr10PlusRsJsonOpaqueDelete>(
        hdr10plus_rs_parse_json(inputJsonStr.c_str()), hdr10plus_rs_json_free);
    if (!m_hdr10plusJson) {
        return RGY_ERR_INVALID_FORMAT;
    }
    return RGY_ERR_NONE;
#else
    return RGY_ERR_UNSUPPORTED;
#endif
}

tstring RGYHDR10Plus::getError() {
#if ENABLE_LIBHDR10PLUS
    return (m_hdr10plusJson) ? char_to_tstring(hdr10plus_rs_json_get_error(m_hdr10plusJson.get())) : tstring();
#else
    return tstring();
#endif
}

const std::vector<uint8_t> RGYHDR10Plus::getData(int64_t iframe) {
#if ENABLE_LIBHDR10PLUS
    std::unique_ptr<const Hdr10PlusRsData, decltype(&hdr10plus_rs_data_free)> av1_metadata(
        hdr10plus_rs_write_av1_metadata_obu_t35_complete(m_hdr10plusJson.get(), iframe), hdr10plus_rs_data_free);
    if (!av1_metadata) {
        return std::vector<uint8_t>();
    }
    std::vector<uint8_t> buffer(av1_metadata->len);
    memcpy(buffer.data(), av1_metadata->data, av1_metadata->len);
    return buffer;
#else
    return std::vector<uint8_t>();
#endif
}
