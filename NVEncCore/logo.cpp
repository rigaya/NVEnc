/*********************************************************************
*     透過性ロゴ（BSマークとか）除去フィルタ
*                                 ver 0.13
* *********************************************************************/

#include <stdlib.h>
#include <string.h>
#include "logo.h"

#define Abs(x) ((x>0)? x:-x)

/*--------------------------------------------------------------------
*    get_logo_file_header_ver() LOGO_HEADERのバージョンを取得
*-------------------------------------------------------------------*/
int get_logo_file_header_ver(const LOGO_FILE_HEADER *logo_file_header) {
    int logo_header_ver = 0;
    if (0 == strcmp(logo_file_header->str, LOGO_FILE_HEADER_STR)) {
        logo_header_ver = 2;
    } else if (0 == strcmp(logo_file_header->str, LOGO_FILE_HEADER_STR_OLD)) {
        logo_header_ver = 1;
    }
    return logo_header_ver;
}

/*--------------------------------------------------------------------
*    convert_logo_header_v1_to_v2() LOGO_HEADERをv1からv2に変換
*-------------------------------------------------------------------*/
void convert_logo_header_v1_to_v2(LOGO_HEADER *logo_header) {
    LOGO_HEADER_OLD old_header;
    memcpy(&old_header,       logo_header,      sizeof(old_header));
    memset(logo_header,       0,                sizeof(logo_header[0]));
    memcpy(logo_header->name, &old_header.name, sizeof(old_header.name));
    memcpy(&logo_header->x,   &old_header.x,    sizeof(short) * 8);
}

/*--------------------------------------------------------------------
*    create_adj_exdata()        位置調整ロゴデータ作成
*-------------------------------------------------------------------*/
#pragma warning (push)
#pragma warning (disable: 4244) //C4244: '=' : 'int' から 'short' への変換です。データが失われる可能性があります。
bool create_adj_exdata(LOGO_PIXEL *ex, LOGO_HEADER *adjdata, const LOGO_PIXEL *df, const LOGO_HEADER *data, int pos_x, int pos_y) {
    int i, j;

    if (data == NULL)
        return false;

    // ロゴ名コピー
    memcpy(adjdata->name, data->name, LOGO_MAX_NAME);

    // 左上座標設定（位置調整後）
    const int LOGO_XY_MIN = -4096;
    adjdata->x = data->x + (int)(pos_x-LOGO_XY_MIN)/4 + LOGO_XY_MIN/4;
    adjdata->y = data->y + (int)(pos_y-LOGO_XY_MIN)/4 + LOGO_XY_MIN/4;

    const int w = data->w + 1; // 1/4単位調整するため
    const int h = data->h + 1; // 幅、高さを１増やす
    adjdata->w = w; 
    adjdata->h = h;

    const int adjx = (pos_x - LOGO_XY_MIN) % 4; // 位置端数
    const int adjy = (pos_y - LOGO_XY_MIN) % 4;

    //----------------------------------------------------- 一番上の列
    ex[0].dp_y  = df[0].dp_y *(4-adjx)*(4-adjy)/16; // 左端
    ex[0].dp_cb = df[0].dp_cb*(4-adjx)*(4-adjy)/16;
    ex[0].dp_cr = df[0].dp_cr*(4-adjx)*(4-adjy)/16;
    ex[0].y  = df[0].y;
    ex[0].cb = df[0].cb;
    ex[0].cr = df[0].cr;
    for (i = 1; i < w-1; ++i) { //中
        // Y
        ex[i].dp_y = (df[i-1].dp_y*adjx*(4-adjy)
                            + df[i].dp_y*(4-adjx)*(4-adjy)) /16;
        if (ex[i].dp_y)
            ex[i].y  = (df[i-1].y*Abs(df[i-1].dp_y)*adjx*(4-adjy)
                    + df[i].y * Abs(df[i].dp_y)*(4-adjx)*(4-adjy))
                /(Abs(df[i-1].dp_y)*adjx*(4-adjy) + Abs(df[i].dp_y)*(4-adjx)*(4-adjy));
        // Cb
        ex[i].dp_cb = (df[i-1].dp_cb*adjx*(4-adjy)
                            + df[i].dp_cb*(4-adjx)*(4-adjy)) /16;
        if (ex[i].dp_cb)
            ex[i].cb = (df[i-1].cb*Abs(df[i-1].dp_cb)*adjx*(4-adjy)
                    + df[i].cb * Abs(df[i].dp_cb)*(4-adjx)*(4-adjy))
                / (Abs(df[i-1].dp_cb)*adjx*(4-adjy)+Abs(df[i].dp_cb)*(4-adjx)*(4-adjy));
        // Cr
        ex[i].dp_cr = (df[i-1].dp_cr*adjx*(4-adjy)
                            + df[i].dp_cr*(4-adjx)*(4-adjy)) /16;
        if (ex[i].dp_cr)
            ex[i].cr = (df[i-1].cr*Abs(df[i-1].dp_cr)*adjx*(4-adjy)
                    + df[i].cr * Abs(df[i].dp_cr)*(4-adjx)*(4-adjy))
                / (Abs(df[i-1].dp_cr)*adjx*(4-adjy)+Abs(df[i].dp_cr)*(4-adjx)*(4-adjy));
    }
    ex[i].dp_y  = df[i-1].dp_y * adjx *(4-adjy)/16; // 右端
    ex[i].dp_cb = df[i-1].dp_cb* adjx *(4-adjy)/16;
    ex[i].dp_cr = df[i-1].dp_cr* adjx *(4-adjy)/16;
    ex[i].y  = df[i-1].y;
    ex[i].cb = df[i-1].cb;
    ex[i].cr = df[i-1].cr;

    //----------------------------------------------------------- 中
    for (j = 1; j < h-1; ++j) {
        // 輝度Y  //---------------------- 左端
        ex[j*w].dp_y = (df[(j-1)*data->w].dp_y*(4-adjx)*adjy
                        + df[j*data->w].dp_y*(4-adjx)*(4-adjy)) /16;
        if (ex[j*w].dp_y)
            ex[j*w].y = (df[(j-1)*data->w].y*Abs(df[(j-1)*data->w].dp_y)*(4-adjx)*adjy
                        + df[j*data->w].y*Abs(df[j*data->w].dp_y)*(4-adjx)*(4-adjy))
                / (Abs(df[(j-1)*data->w].dp_y)*(4-adjx)*adjy+Abs(df[j*data->w].dp_y)*(4-adjx)*(4-adjy));
        // 色差(青)Cb
        ex[j*w].dp_cb = (df[(j-1)*data->w].dp_cb*(4-adjx)*adjy
                        + df[j*data->w].dp_cb*(4-adjx)*(4-adjy)) / 16;
        if (ex[j*w].dp_cb)
            ex[j*w].cb = (df[(j-1)*data->w].cb*Abs(df[(j-1)*data->w].dp_cb)*(4-adjx)*adjy
                        + df[j*data->w].cb*Abs(df[j*data->w].dp_cb)*(4-adjx)*(4-adjy))
                / (Abs(df[(j-1)*data->w].dp_cb)*(4-adjx)*adjy+Abs(df[j*data->w].dp_cb)*(4-adjx)*(4-adjy));
        // 色差(赤)Cr
        ex[j*w].dp_cr = (df[(j-1)*data->w].dp_cr*(4-adjx)*adjy
                        + df[j*data->w].dp_cr*(4-adjx)*(4-adjy)) / 16;
        if (ex[j*w].dp_cr)
            ex[j*w].cr = (df[(j-1)*data->w].cr*Abs(df[(j-1)*data->w].dp_cr)*(4-adjx)*adjy
                        + df[j*data->w].cr*Abs(df[j*data->w].dp_cr)*(4-adjx)*(4-adjy))
                / (Abs(df[(j-1)*data->w].dp_cr)*(4-adjx)*adjy+Abs(df[j*data->w].dp_cr)*(4-adjx)*(4-adjy));
        for (i = 1; i < w-1; ++i) { //------------------ 中
            // Y
            ex[j*w+i].dp_y = (df[(j-1)*data->w+i-1].dp_y*adjx*adjy
                            + df[(j-1)*data->w+i].dp_y*(4-adjx)*adjy
                            + df[j*data->w+i-1].dp_y*adjx*(4-adjy)
                            + df[j*data->w+i].dp_y*(4-adjx)*(4-adjy) ) /16;
            if (ex[j*w+i].dp_y)
                ex[j*w+i].y = (df[(j-1)*data->w+i-1].y*Abs(df[(j-1)*data->w+i-1].dp_y)*adjx*adjy
                            + df[(j-1)*data->w+i].y*Abs(df[(j-1)*data->w+i].dp_y)*(4-adjx)*adjy
                            + df[j*data->w+i-1].y*Abs(df[j*data->w+i-1].dp_y)*adjx*(4-adjy)
                            + df[j*data->w+i].y*Abs(df[j*data->w+i].dp_y)*(4-adjx)*(4-adjy) )
                    / (Abs(df[(j-1)*data->w+i-1].dp_y)*adjx*adjy + Abs(df[(j-1)*data->w+i].dp_y)*(4-adjx)*adjy
                        + Abs(df[j*data->w+i-1].dp_y)*adjx*(4-adjy)+Abs(df[j*data->w+i].dp_y)*(4-adjx)*(4-adjy));
            // Cb
            ex[j*w+i].dp_cb = (df[(j-1)*data->w+i-1].dp_cb*adjx*adjy
                            + df[(j-1)*data->w+i].dp_cb*(4-adjx)*adjy
                            + df[j*data->w+i-1].dp_cb*adjx*(4-adjy)
                            + df[j*data->w+i].dp_cb*(4-adjx)*(4-adjy) ) /16;
            if (ex[j*w+i].dp_cb)
                ex[j*w+i].cb = (df[(j-1)*data->w+i-1].cb*Abs(df[(j-1)*data->w+i-1].dp_cb)*adjx*adjy
                            + df[(j-1)*data->w+i].cb*Abs(df[(j-1)*data->w+i].dp_cb)*(4-adjx)*adjy
                            + df[j*data->w+i-1].cb*Abs(df[j*data->w+i-1].dp_cb)*adjx*(4-adjy)
                            + df[j*data->w+i].cb*Abs(df[j*data->w+i].dp_cb)*(4-adjx)*(4-adjy) )
                    / (Abs(df[(j-1)*data->w+i-1].dp_cb)*adjx*adjy + Abs(df[(j-1)*data->w+i].dp_cb)*(4-adjx)*adjy
                        + Abs(df[j*data->w+i-1].dp_cb)*adjx*(4-adjy) + Abs(df[j*data->w+i].dp_cb)*(4-adjx)*(4-adjy));
            // Cr
            ex[j*w+i].dp_cr = (df[(j-1)*data->w+i-1].dp_cr*adjx*adjy
                            + df[(j-1)*data->w+i].dp_cr*(4-adjx)*adjy
                            + df[j*data->w+i-1].dp_cr*adjx*(4-adjy)
                            + df[j*data->w+i].dp_cr*(4-adjx)*(4-adjy) ) /16;
            if (ex[j*w+i].dp_cr)
                ex[j*w+i].cr = (df[(j-1)*data->w+i-1].cr*Abs(df[(j-1)*data->w+i-1].dp_cr)*adjx*adjy
                            + df[(j-1)*data->w+i].cr*Abs(df[(j-1)*data->w+i].dp_cr)*(4-adjx)*adjy
                            + df[j*data->w+i-1].cr*Abs(df[j*data->w+i-1].dp_cr)*adjx*(4-adjy)
                            + df[j*data->w+i].cr*Abs(df[j*data->w+i].dp_cr)*(4-adjx)*(4-adjy) )
                    / (Abs(df[(j-1)*data->w+i-1].dp_cr)*adjx*adjy +Abs(df[(j-1)*data->w+i].dp_cr)*(4-adjx)*adjy
                        + Abs(df[j*data->w+i-1].dp_cr)*adjx*(4-adjy)+Abs(df[j*data->w+i].dp_cr)*(4-adjx)*(4-adjy));
        }
        // Y //----------------------- 右端
        ex[j*w+i].dp_y = (df[(j-1)*data->w+i-1].dp_y*adjx*adjy
                        + df[j*data->w+i-1].dp_y*adjx*(4-adjy)) / 16;
        if (ex[j*w+i].dp_y)
            ex[j*w+i].y = (df[(j-1)*data->w+i-1].y*Abs(df[(j-1)*data->w+i-1].dp_y)*adjx*adjy
                        + df[j*data->w+i-1].y*Abs(df[j*data->w+i-1].dp_y)*adjx*(4-adjy))
                / (Abs(df[(j-1)*data->w+i-1].dp_y)*adjx*adjy+Abs(df[j*data->w+i-1].dp_y)*adjx*(4-adjy));
        // Cb
        ex[j*w+i].dp_cb = (df[(j-1)*data->w+i-1].dp_cb*adjx*adjy
                        + df[j*data->w+i-1].dp_cb*adjx*(4-adjy)) / 16;
        if (ex[j*w+i].dp_cb)
            ex[j*w+i].cb = (df[(j-1)*data->w+i-1].cb*Abs(df[(j-1)*data->w+i-1].dp_cb)*adjx*adjy
                        + df[j*data->w+i-1].cb*Abs(df[j*data->w+i-1].dp_cb)*adjx*(4-adjy))
                / (Abs(df[(j-1)*data->w+i-1].dp_cb)*adjx*adjy+Abs(df[j*data->w+i-1].dp_cb)*adjx*(4-adjy));
        // Cr
        ex[j*w+i].dp_cr = (df[(j-1)*data->w+i-1].dp_cr*adjx*adjy
                        + df[j*data->w+i-1].dp_cr*adjx*(4-adjy)) / 16;
        if (ex[j*w+i].dp_cr)
            ex[j*w+i].cr = (df[(j-1)*data->w+i-1].cr*Abs(df[(j-1)*data->w+i-1].dp_cr)*adjx*adjy
                        + df[j*data->w+i-1].cr*Abs(df[j*data->w+i-1].dp_cr)*adjx*(4-adjy))
                / (Abs(df[(j-1)*data->w+i-1].dp_cr)*adjx*adjy+Abs(df[j*data->w+i-1].dp_cr)*adjx*(4-adjy));
    }
    //--------------------------------------------------------- 一番下
    ex[j*w].dp_y  = df[(j-1)*data->w].dp_y *(4-adjx)*adjy /16; // 左端
    ex[j*w].dp_cb = df[(j-1)*data->w].dp_cb*(4-adjx)*adjy /16;
    ex[j*w].dp_cr = df[(j-1)*data->w].dp_cr*(4-adjx)*adjy /16;
    ex[j*w].y  = df[(j-1)*data->w].y;
    ex[j*w].cb = df[(j-1)*data->w].cb;
    ex[j*w].cr = df[(j-1)*data->w].cr;
    for (i = 1; i < w-1; ++i) { // 中
        // Y
        ex[j*w+i].dp_y = (df[(j-1)*data->w+i-1].dp_y * adjx * adjy
                                + df[(j-1)*data->w+i].dp_y * (4-adjx) *adjy) /16;
        if (ex[j*w+i].dp_y)
            ex[j*w+i].y = (df[(j-1)*data->w+i-1].y*Abs(df[(j-1)*data->w+i-1].dp_y)*adjx*adjy
                        + df[(j-1)*data->w+i].y*Abs(df[(j-1)*data->w+i].dp_y)*(4-adjx)*adjy)
                / (Abs(df[(j-1)*data->w+i-1].dp_y)*adjx*adjy +Abs(df[(j-1)*data->w+i].dp_y)*(4-adjx)*adjy);
        // Cb
        ex[j*w+i].dp_cb = (df[(j-1)*data->w+i-1].dp_cb * adjx * adjy
                                + df[(j-1)*data->w+i].dp_cb * (4-adjx) *adjy) /16;
        if (ex[j*w+i].dp_cb)
            ex[j*w+i].cb = (df[(j-1)*data->w+i-1].cb*Abs(df[(j-1)*data->w+i-1].dp_cb)*adjx*adjy
                        + df[(j-1)*data->w+i].cb*Abs(df[(j-1)*data->w+i].dp_cb)*(4-adjx)*adjy )
                / (Abs(df[(j-1)*data->w+i-1].dp_cb)*adjx*adjy +Abs(df[(j-1)*data->w+i].dp_cb)*(4-adjx)*adjy);
        // Cr
        ex[j*w+i].dp_cr = (df[(j-1)*data->w+i-1].dp_cr * adjx * adjy
                                + df[(j-1)*data->w+i].dp_cr * (4-adjx) *adjy) /16;
        if (ex[j*w+i].dp_cr)
            ex[j*w+i].cr = (df[(j-1)*data->w+i-1].cr*Abs(df[(j-1)*data->w+i-1].dp_cr)*adjx*adjy
                        + df[(j-1)*data->w+i].cr*Abs(df[(j-1)*data->w+i].dp_cr)*(4-adjx)*adjy)
                / (Abs(df[(j-1)*data->w+i-1].dp_cr)*adjx*adjy +Abs(df[(j-1)*data->w+i].dp_cr)*(4-adjx)*adjy);
    }
    ex[j*w+i].dp_y  = df[(j-1)*data->w+i-1].dp_y *adjx*adjy /16; // 右端
    ex[j*w+i].dp_cb = df[(j-1)*data->w+i-1].dp_cb*adjx*adjy /16;
    ex[j*w+i].dp_cr = df[(j-1)*data->w+i-1].dp_cr*adjx*adjy /16;
    ex[j*w+i].y  = df[(j-1)*data->w+i-1].y;
    ex[j*w+i].cb = df[(j-1)*data->w+i-1].cb;
    ex[j*w+i].cr = df[(j-1)*data->w+i-1].cr;

    return true;
}
#pragma warning (pop)
