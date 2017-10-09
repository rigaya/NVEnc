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


#include <windows.h>
#include <process.h>
#include <algorithm>
#include <vector>
#include <cstdint>

#define DEFINE_GLOBAL
#include "filter.h"
#include "cufilters_version.h"
#include "NVEncParam.h"
#include "cufilters.h"


/*====================================================================
* 	add_res_dlg()		コールバックプロシージャ
*===================================================================*/
BOOL CALLBACK add_res_dlg(HWND hdlg, UINT msg, WPARAM wParam, LPARAM lParam) {

    switch (msg) {
    case WM_INITDIALOG:
        SetDlgItemText(hdlg, ID_TX_RESIZE_RES_ADD, (const char*)"");
        return TRUE;
    case WM_COMMAND:
        switch (LOWORD(wParam)) {
        case IDOK: {
            char str[128] = { 0 };
            uint32_t w = 0, h = 0;
            GetDlgItemText(hdlg, ID_TX_RESIZE_RES_ADD, str, sizeof(str));
            if (2 != sscanf_s(str, "%dx%d", &w, &h)) {
                MessageBox(hdlg, "誤った書式で指定されています。\n1280x720など、<幅>x<高さ>の書式で指定してください。", MB_OK, MB_ICONEXCLAMATION);
                return FALSE;
            }
            if (w >= (1<<15) || (h >= (1<<15))) {
                MessageBox(hdlg, "解像度が大きすぎます。32767以下の値で指定してください。", MB_OK, MB_ICONEXCLAMATION);
                return FALSE;
            }
            EndDialog(hdlg, ((w & 0xffff) << 16) | (h & 0xffff));
            return TRUE;
        }
        case IDCANCEL:
            EndDialog(hdlg, 0);
            return TRUE;
        }
        break;
    }

    return FALSE;
}
