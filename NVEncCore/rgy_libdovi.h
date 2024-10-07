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

#ifndef __RGY_LIBDOVI_H__
#define __RGY_LIBDOVI_H__

#if ENABLE_LIBDOVI

extern "C"
#include "libdovi/rpu_parser.h"

#if 1
#if defined(_WIN32) || defined(_WIN64)
#pragma comment(lib, "dovi.lib")
#pragma comment(lib, "ws2_32.lib")
#pragma comment(lib, "ntdll.lib")
#pragma comment(lib, "Userenv.lib")
#endif
#else

#include "rgy_osdep.h"

class RGYDoviFuncs {
private:
    HMODULE m_hModule;

public:
    decltype(&dovi_parse_rpu) parse_rpu;
    decltype(&dovi_parse_itu_t35_dovi_metadata_obu) parse_itu_t35_dovi_metadata_obu;
    decltype(&dovi_parse_unspec62_nalu) parse_unspec62_nalu;
    decltype(&dovi_rpu_free) rpu_free;
    decltype(&dovi_rpu_get_error) rpu_get_error;
    decltype(&dovi_data_free) data_free;
    decltype(&dovi_write_rpu) write_rpu;
    decltype(&dovi_write_unspec62_nalu) write_unspec62_nalu;
    decltype(&dovi_convert_rpu_with_mode) convert_rpu_with_mode;
    decltype(&dovi_rpu_get_header) rpu_get_header;
    decltype(&dovi_rpu_free_header) rpu_free_header;
    decltype(&dovi_rpu_get_data_mapping) rpu_get_data_mapping;
    decltype(&dovi_rpu_free_data_mapping) rpu_free_data_mapping;
    decltype(&dovi_rpu_get_vdr_dm_data) rpu_get_vdr_dm_data;
    decltype(&dovi_rpu_free_vdr_dm_data) rpu_free_vdr_dm_data;
    decltype(&dovi_parse_rpu_bin_file) parse_rpu_bin_file;
    decltype(&dovi_rpu_list_free) rpu_list_free;
    decltype(&dovi_rpu_set_active_area_offsets) rpu_set_active_area_offsets;
    decltype(&dovi_rpu_remove_mapping) rpu_remove_mapping;
    decltype(&dovi_write_av1_rpu_metadata_obu_t35_payload) write_av1_rpu_metadata_obu_t35_payload;
    decltype(&dovi_write_av1_rpu_metadata_obu_t35_complete) write_av1_rpu_metadata_obu_t35_complete;

    RGYDoviFuncs();
    ~RGYDoviFuncs();

    bool load();

    void close();
};
#endif

#endif // ENABLE_LIBDOVI

#endif // __RGY_LIBDOVI_H__
