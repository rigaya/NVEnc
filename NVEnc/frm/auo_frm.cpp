// -----------------------------------------------------------------------------------------
// NVEnc by rigaya
// -----------------------------------------------------------------------------------------
// The MIT License
//
// Copyright (c) 1999-2016 rigaya
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

#include <stdlib.h>
#include <string.h>
#include "auo_frm.h"
#include "auo_util.h"

const int NEW_LINE_THRESHOLD = 125;
const int MAKE_NEW_LINE_THRESHOLD = 140;

static inline int check_log_type(char *mes) {
    if (strstr(mes, "warning")) return LOG_WARNING;
    if (strstr(mes, "error")) return LOG_ERROR;
    return LOG_INFO;
}

static inline void add_line_to_cache(LOG_CACHE *cache_line, const char *mes) {
    if (cache_line->idx >= cache_line->max_line) {
        //メモリ不足なら再確保
        if (NULL != (cache_line->lines = (char **)realloc(cache_line->lines, sizeof(cache_line->lines[0]) * cache_line->max_line * 2))) {
            memset(&cache_line->lines[cache_line->max_line], 0, sizeof(cache_line->lines[0]) * cache_line->max_line);
            cache_line->max_line *= 2;
        }
    }
    if (cache_line->lines) {
        //一行のデータを格納
        const int line_len = strlen(mes) + 1;
        char *line_ptr = (char *)malloc(line_len * sizeof(line_ptr[0]));
        memcpy(line_ptr, mes, line_len);
        cache_line->lines[cache_line->idx] = line_ptr;
        cache_line->idx++;
    }
}

void release_log_cache(LOG_CACHE *log_cache) {
    if (log_cache && log_cache->lines) {
        for (int i = 0; i < log_cache->idx; i++)
            if (log_cache->lines[i]) free(log_cache->lines[i]);
        free(log_cache->lines);
        log_cache->lines = NULL;
        log_cache->idx = 0;
    }
}

//LOG_CACHEの初期化、成功->0, 失敗->1
int init_log_cache(LOG_CACHE *log_cache) {
    release_log_cache(log_cache);
    log_cache->idx = 0;
    log_cache->max_line = 64;
    return NULL == (log_cache->lines = (char **)calloc(log_cache->max_line, sizeof(log_cache->lines[0])));
}

//長すぎたら適当に折り返す
static int write_log_enc_mes_line(char *const mes, LOG_CACHE *cache_line) {
    const int mes_len = strlen(mes);
    const int mes_type = check_log_type(mes);
    char *const fin = mes + mes_len;
    char *const prefix_ptr = strstr(mes, "]: ");
    const int prefix_len = (prefix_ptr) ? prefix_ptr - mes + strlen("]: ") : 0;
    char *p = mes, *q = NULL;
    BOOL flag_continue = FALSE;
    do {
        const int threshold = NEW_LINE_THRESHOLD - (p != mes) * prefix_len;
        flag_continue = mes_len >= MAKE_NEW_LINE_THRESHOLD
            && (p + threshold) < fin
            && (q = strrchr(p, ' ', threshold)) != NULL;
        if (flag_continue) *q = '\0';
        if (p != mes)
            for (char *const prefix_adjust = p - prefix_len; p > prefix_adjust; p--)
                *(p-1) = ' ';
        (cache_line) ? add_line_to_cache(cache_line, p) : write_log_line(mes_type, p, true);
        p=q+1;
    } while (flag_continue);
    return mes_len;
}

void write_log_enc_mes(char *const msg, DWORD *log_len, int total_drop, int current_frames, LOG_CACHE *cache_line) {
    char *a, *b, *mes = msg;
    char * const fin = mes + *log_len; //null文字の位置
    *fin = '\0';
    while ((a = strchr(mes, '\n')) != NULL) {
        if ((b = strrchr(mes, '\r', a - mes - 2)) != NULL)
            mes = b + 1;
        *a = '\0';
        write_log_enc_mes_line(mes, cache_line);
        mes = a + 1;
    }
    if ((a = strrchr(mes, '\r', fin - mes - 1)) != NULL) {
        b = a - 1;
        while (*b == ' ' || *b == '\r')
            b--;
        *(b+1) = '\0';
        if ((b = strrchr(mes, '\r', b - mes - 2)) != NULL)
            mes = b + 1;
        set_window_title_enc_mes(mes, total_drop, current_frames);
        mes = a + 1;
    }
    if (mes == msg && *log_len)
        mes += write_log_enc_mes_line(mes, cache_line);
    memmove(msg, mes, ((*log_len = fin - mes) + 1) * sizeof(msg[0]));
}

void write_args(const char *args) {
    size_t len = strlen(args);
    char *const c = (char *)malloc((len+1)*sizeof(c[0]));
    char *const fin = c + len;
    memcpy(c, args, (len+1)*sizeof(c[0]));
    char *p = c;
    for (char *q = NULL; p + NEW_LINE_THRESHOLD < fin && (q = strrchr(p, ' ', NEW_LINE_THRESHOLD)) != NULL; p = q+1) {
        *q = '\0';
        write_log_line(LOG_INFO, p);
    }
    write_log_line(LOG_INFO, p);
    free(c);
}

void write_log_exe_mes(char *const msg, DWORD *log_len, const char *exename, LOG_CACHE *cache_line) {
    char *a, *b, *mes = msg;
    char * const fin = mes + *log_len; //null文字の位置
    char * buffer = NULL;
    DWORD buffer_len = 0;
    *fin = '\0';
    while ((a = strchr(mes, '\n')) != NULL) {
        if ((b = strrchr(mes, '\r', a - mes - 2)) != NULL)
            mes = b + 1;
        *a = '\0';
        write_log_enc_mes_line(mes, cache_line);
        mes = a + 1;
    }
    if ((a = strrchr(mes, '\r', fin - mes - 1)) != NULL) {
        b = a - 1;
        while (*b == ' ' || *b == '\r')
            b--;
        *(b+1) = '\0';
        if ((b = strrchr(mes, '\r', b - mes - 2)) != NULL)
            mes = b + 1;
        if (exename) {
            if (buffer_len == 0) buffer_len = *log_len + strlen(exename) + 3;
            if (buffer != NULL || NULL != (buffer = (char*)malloc(buffer_len * sizeof(buffer[0])))) {
                sprintf_s(buffer, buffer_len, "%s: %s", exename, mes);
                set_window_title(buffer);
            }
        }
        mes = a + 1;
    }
    if (mes == msg && *log_len)
        mes += write_log_enc_mes_line(mes, cache_line);
    memmove(msg, mes, ((*log_len = fin - mes) + 1) * sizeof(msg[0]));
    if (buffer) free(buffer);
}

