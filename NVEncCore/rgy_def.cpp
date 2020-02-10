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

#include "rgy_def.h"
#include "rgy_log.h"

const CX_DESC list_log_level[7] = {
    { _T("trace"), RGY_LOG_TRACE },
    { _T("debug"), RGY_LOG_DEBUG },
    { _T("more"),  RGY_LOG_MORE  },
    { _T("info"),  RGY_LOG_INFO  },
    { _T("warn"),  RGY_LOG_WARN  },
    { _T("error"), RGY_LOG_ERROR },
    { NULL, 0 }
};

tstring VideoVUIInfo::print_main() const {
    return tstring(_T("matrix:")) + get_cx_desc(list_colormatrix, matrix) + _T(",")
        + tstring(_T("colorprim:")) + get_cx_desc(list_colorprim, colorprim) + _T(",")
        + tstring(_T("transfer:")) + get_cx_desc(list_transfer, transfer);
}

tstring VideoVUIInfo::print_all(bool write_all) const {
    tstring str;
    if (write_all || matrix != get_cx_value(list_colormatrix, _T("undef"))) {
        str += tstring(_T(",matrix:")) + get_cx_desc(list_colormatrix, matrix);
    }
    if (write_all || colorprim != get_cx_value(list_colorprim, _T("undef"))) {
        str += tstring(_T(",colorprim:")) + get_cx_desc(list_colorprim, colorprim);
    }
    if (write_all || transfer != get_cx_value(list_transfer, _T("undef"))) {
        str += tstring(_T(",transfer:")) + get_cx_desc(list_transfer, transfer);
    }
    if (write_all || format != get_cx_value(list_videoformat, _T("undef"))) {
        str += tstring(_T(",videoformat:")) + get_cx_desc(list_videoformat, format);
    }
    if (write_all || colorrange != get_cx_value(list_colorrange, _T("undef"))) {
        str += tstring(_T(",range:")) + get_cx_desc(list_colorrange, colorrange);
    }
    if (write_all || chromaloc != get_cx_value(list_chromaloc_str, _T("undef"))) {
        str += tstring(_T(",chromaloc:")) + get_cx_desc(list_chromaloc_str, chromaloc);
    }
    return (str.length() > 1) ? str.substr(1) : str;
}
