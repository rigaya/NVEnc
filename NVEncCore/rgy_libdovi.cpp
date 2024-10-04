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

#include "rgy_libdovi.h"
#include "rgy_tchar.h"

#if 0
#if defined(_WIN32) || defined(_WIN64)
static const TCHAR * LIBDOVI_MODULENAME = _T("dovi.dll");
#else
static const TCHAR * LIBDOVI_MODULENAME = _T("dovi.so");
#endif

RGYDoviFuncs::RGYDoviFuncs() :
    m_hModule(nullptr),
    parse_rpu(nullptr),
    parse_itu_t35_dovi_metadata_obu(nullptr),
    parse_unspec62_nalu(nullptr),
    rpu_free(nullptr),
    rpu_get_error(nullptr),
    data_free(nullptr),
    write_rpu(nullptr),
    write_unspec62_nalu(nullptr),
    convert_rpu_with_mode(nullptr),
    rpu_get_header(nullptr),
    rpu_free_header(nullptr),
    rpu_get_data_mapping(nullptr),
    rpu_free_data_mapping(nullptr),
    rpu_get_vdr_dm_data(nullptr),
    rpu_free_vdr_dm_data(nullptr),
    parse_rpu_bin_file(nullptr),
    rpu_list_free(nullptr),
    rpu_set_active_area_offsets(nullptr),
    rpu_remove_mapping(nullptr),
    write_av1_rpu_metadata_obu_t35_payload(nullptr),
    write_av1_rpu_metadata_obu_t35_complete(nullptr) {
}

RGYDoviFuncs::~RGYDoviFuncs() {
    close();
}

bool RGYDoviFuncs::load() {
    if (m_hModule) return true;

    if ((m_hModule = RGY_LOAD_LIBRARY(LIBDOVI_MODULENAME)) == nullptr) {
        return false;
    }

#define LOAD_FUNC(x) \
if (nullptr == (x = (decltype(x))RGY_GET_PROC_ADDRESS(m_hModule, #x))) { \
    return false; \
}

    LOAD_FUNC(parse_rpu);
    LOAD_FUNC(parse_itu_t35_dovi_metadata_obu);
    LOAD_FUNC(parse_unspec62_nalu);
    LOAD_FUNC(rpu_free);
    LOAD_FUNC(rpu_get_error);
    LOAD_FUNC(data_free);
    LOAD_FUNC(write_rpu);
    LOAD_FUNC(write_unspec62_nalu);
    LOAD_FUNC(convert_rpu_with_mode);
    LOAD_FUNC(rpu_get_header);
    LOAD_FUNC(rpu_free_header);
    LOAD_FUNC(rpu_get_data_mapping);
    LOAD_FUNC(rpu_free_data_mapping);
    LOAD_FUNC(rpu_get_vdr_dm_data);
    LOAD_FUNC(rpu_free_vdr_dm_data);
    LOAD_FUNC(parse_rpu_bin_file);
    LOAD_FUNC(rpu_list_free);
    LOAD_FUNC(rpu_set_active_area_offsets);
    LOAD_FUNC(rpu_remove_mapping);
    LOAD_FUNC(write_av1_rpu_metadata_obu_t35_payload);
    LOAD_FUNC(write_av1_rpu_metadata_obu_t35_complete);

#undef LOAD_FUNC

    return true;
}

void RGYDoviFuncs::close() {
    if (m_hModule) {
        RGY_FREE_LIBRARY(m_hModule);
        m_hModule = nullptr;
    }
}
#endif
