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

#pragma once
#ifndef __RGY_QUEUE_H__
#define __RGY_QUEUE_H__

#include <cstdint>
#include <cstring>
#include <atomic>
#include <climits>
#include <memory>
#include <mutex>
#include "rgy_arch.h"
#include "rgy_osdep.h"
#include "rgy_util.h"
#include "rgy_event.h"

#ifndef clamp
#define clamp(x, low, high) (((x) <= (high)) ? (((x) >= (low)) ? (x) : (low)) : (high))
#endif

class RGYQueueLock {
public:
    RGYQueueLock(std::atomic<bool>& lock) : m_lock(lock) {
        while (!tryLock()) {
            rgy_yield();
        }
    };
    ~RGYQueueLock() { m_lock = false; }
private:
    bool tryLock() {
        bool expected = false;
        return m_lock.compare_exchange_weak(expected, true);
    }
    std::atomic<bool>& m_lock;
};

#pragma warning (push)
#pragma warning (disable: 4324) //アラインメント指定子のために構造体がパッドされました
template<typename Type, size_t align_byte = sizeof(Type)>
class RGYQueueMPMP {
    union queueData {
        Type data;
        char pad[((sizeof(Type) + (align_byte - 1)) & (~(align_byte - 1)))];
    };
public:
    //並列で押し込みと取り出しが可能なキューを作成する
    //スレッド並列対応のため、データにはパディングをつけてアライメントをとることが可能 (align_byte)
    //どこまで効果があるかは不明だが、align_byte=64としてfalse sharingを回避できる
    RGYQueueMPMP() :
        m_nPushRestartExtra(0),
        m_heEventPoped(NULL),
        m_heEventPushed(NULL),
        m_nMallocAlign(32),
        m_nMaxCapacity(SIZE_MAX),
        m_nKeepLength(0),
        m_pBufIn(nullptr), m_pBufOut(nullptr),
        m_pBufStart(), m_pBufFin(nullptr),
        m_bUsingData(false), m_bPush(false) {
        //実際のメモリのアライメントに適切な2の倍数であるか確認する
        //そうでない場合は32をデフォルトとして使用
        for (uint32_t i = 4; i < sizeof(i) * 8; i++) {
            int test = 1 << i;
            if (test == align_byte) {
                m_nMallocAlign = test;
                break;
            }
        }
    }
    ~RGYQueueMPMP() {
        close();
    }
    //indexの位置への参照を返す
    // !! push側のスレッドからのみ有効 !!
    queueData& operator[](uint32_t index) {
        return m_pBufOut[index];
    }
    //データの先頭へのポインタを返す
    // !! push側のスレッドからのみ有効 !!
    queueData *get() {
        return m_pBufOut;
    }
    //indexの位置へのポインタを返す
    // !! push側のスレッドからのみ有効 !!
    queueData *get(uint32_t index) {
        return m_pBufOut + index;
    }
    //キューが一定の長さに達しないとfront_copy/popできないように設定する
    void set_keep_length(size_t keepLength) {
        m_nKeepLength = keepLength;
    }
    size_t get_keep_length() {
        return m_nKeepLength;
    }
    //キューを初期化する
    //bufSizeはキューの内部データバッファサイズ maxCapacityを超えてもかまわない
    //maxCapacityはキューに格納できる最大のデータ数
    void init(size_t bufSize = 1024, size_t maxCapacity = SIZE_MAX, int nPushRestart = 1) {
        close();
        alloc(bufSize);
        m_heEventPoped = CreateEvent(NULL, TRUE, TRUE, NULL);
        m_heEventPushed = CreateEvent(NULL, TRUE, TRUE, NULL);
        m_nMaxCapacity = maxCapacity;
        m_nKeepLength = 0;
        m_nPushRestartExtra = clamp(nPushRestart - 1, 0, (int)std::min<size_t>(INT_MAX, maxCapacity) - 4);
    }
    //キューのデータをクリアする
    void clear() {
        const auto bufSize = m_pBufFin - m_pBufStart.get();
        m_pBufFin = m_pBufStart.get() + bufSize;
        m_pBufIn = m_pBufStart.get();
        m_pBufOut = m_pBufStart.get();
    }
    //キューのデータをクリアする際に、指定した関数で内部データを開放してから、データをクリアする
    template<typename Func>
    void clear(Func deleter) {
        queueData *ptrFin = m_pBufIn;
        for (queueData *ptr = m_pBufOut; ptr < ptrFin; ptr++) {
            deleter(&ptr->data);
        }
        clear();
    }
    //キューのデータをクリアし、リソースを破棄する
    void close() {
        if (m_heEventPoped) {
            CloseEvent(m_heEventPoped);
            m_heEventPoped = nullptr;
        }
        if (m_heEventPushed) {
            CloseEvent(m_heEventPushed);
            m_heEventPushed = nullptr;
        }
        m_pBufStart.reset();
        m_pBufFin = nullptr;
        m_pBufIn = nullptr;
        m_pBufOut = nullptr;
        m_bUsingData = false;
        m_bPush = false;
    }
    //キューのデータをクリアする際に、指定した関数で内部データを開放してから、リソースを破棄する
    template<typename Func>
    void close(Func deleter) {
        clear(deleter);
        close();
    }
    //データをキューにコピーし押し込む
    //キューのデータ量があらかじめ設定した上限に達した場合は、キューに空きができるまで待機する
    bool push(const Type& in) {
        //最初に決めた容量分までキューにデータがたまっていたら、キューに空きができるまで待機する
        while (size() >= m_nMaxCapacity) {
            ResetEvent(m_heEventPoped);
            WaitForSingleObject(m_heEventPoped, 16);
        }
        // pushするスレッド同士が競合しないよう、下記領域にロックをかける
        RGYQueueLock pushLock(m_bPush);
        if (m_pBufIn >= m_pBufFin) {
            //現時点でのm_pBufOut (この後別スレッドによって書き換わるかもしれない)
            queueData *pBufOutOld = m_pBufOut.load();
            //現在キューにあるデータサイズ
            const size_t dataSize = m_pBufFin - pBufOutOld;
            //新たに確保するバッファのデータサイズ
            const size_t bufSize = (std::max)((size_t)(m_pBufFin - m_pBufStart.get()), dataSize * 2);
            //新たなバッファ
            auto newBuf = std::unique_ptr<queueData, aligned_malloc_deleter>(
                (queueData *)_aligned_malloc(sizeof(queueData) * bufSize, m_nMallocAlign), aligned_malloc_deleter());
            if (!newBuf) {
                return false;
            }
            if (std::is_trivially_copyable<Type>::value) {
                memcpy(newBuf.get(), pBufOutOld, sizeof(queueData) * dataSize);
            } else {
                queueData *pBufOutNew = newBuf.get();
                for (size_t i = 0; i < dataSize; i++) {
                    pBufOutNew[i].data = pBufOutOld[i].data;
                }
            }
            queueData *pBufOutNew = newBuf.get();
            queueData *pBufOutExpected = pBufOutOld;
            //更新前にnullptrをセット
            m_pBufIn = nullptr;
            //m_pBufOutが変更されていなければ、pBufOutNewをm_pBufOutに代入
            //変更されていれば、pBufOutNewを修正して再度代入
            while (!std::atomic_compare_exchange_weak(&m_pBufOut, &pBufOutExpected, pBufOutNew)) {
                pBufOutNew += (pBufOutExpected - pBufOutOld);
                pBufOutOld = pBufOutExpected;
            }
            //新しいバッファ用にデータを書き換え
            m_pBufIn = newBuf.get() + dataSize;
            m_pBufFin = newBuf.get() + bufSize;
            //取り出し側のコピー終了を待機
            //一度falseになったことが確認できれば、
            //その次の取り出しは新しいバッファから行われていることになるので、
            //古いバッファは破棄してよい
            RGYQueueLock lockUsing(m_bUsingData);
            //古いバッファを破棄
            m_pBufStart = std::move(newBuf);
        }
        m_pBufIn.load()->data = in;
        m_pBufIn++;
        SetEvent(m_heEventPushed);
        return true;
    }
    //キューのsizeを取得する
    size_t size() const {
        if (!m_pBufStart)
            return 0;
        //バッファはあるが、m_pBufInがnullptrの場合は、
        //押し込み処理で書き換え中なので待機する
        queueData *ptr = nullptr;
        while ((ptr = m_pBufIn.load()) == nullptr) {
            rgy_yield();
        }
        return ptr - m_pBufOut;
    }
    //キューが空ならtrueを返す
    bool empty() const {
        return size() == 0;
    }
    //キューの最大サイズを取得する
    size_t capacity() const {
        return m_nMaxCapacity;
    }
    //キューの最大サイズを設定する
    void set_capacity(size_t capacity) {
        m_nMaxCapacity = capacity;
        m_nPushRestartExtra = (std::min)(m_nPushRestartExtra, (int)std::min<size_t>(INT_MAX, m_nMaxCapacity) - 1);
    }
    //indexの位置のコピーを取得する
    bool copy(Type *out, uint32_t index, size_t *pnSize = nullptr) {
        bool bCopy; decltype(size()) nSize;
        { // 同時に更新しないよう、ロックする
            RGYQueueLock lockUsing(m_bUsingData);
            nSize = size();
            bCopy = index < nSize;
            if (bCopy) {
                auto ptr = m_pBufOut + index;
                *out = ptr->data;
            }
        }
        if (!bCopy) {
            ResetEvent(m_heEventPushed);
        }
        if (pnSize) {
            *pnSize = nSize;
        }
        return bCopy;
    }
    //キューの先頭のデータを取り出す (outにコピーする)
    //キューが空ならなにもせずfalseを返す
    bool front_copy_no_lock(Type *out, size_t *pnSize = nullptr) {
        bool bCopy; decltype(size()) nSize;
        { // 同時に更新しないよう、ロックする
            RGYQueueLock lockUsing(m_bUsingData);
            nSize = size();
            bCopy = nSize > m_nKeepLength;
            if (bCopy) {
                *out = m_pBufOut.load()->data;
            }
        }
        if (!bCopy) {
            ResetEvent(m_heEventPushed);
        }
        if (pnSize) {
            *pnSize = nSize;
        }
        return bCopy;
    }
    //キューの先頭のデータを取り出しながら(outにコピーする)、キューから取り除く
    //キューが空ならなにもせずfalseを返す
    bool front_copy_and_pop_no_lock(Type *out, size_t *pnSize = nullptr) {
        bool bCopy; decltype(size()) nSize;
        { // 同時に更新しないよう、ロックする
            RGYQueueLock lockUsing(m_bUsingData);
            nSize = size();
            bCopy = nSize > m_nKeepLength;
            if (bCopy) {
                *out = m_pBufOut.load()->data;
                m_pBufOut++;
                if (nSize <= m_nMaxCapacity - m_nPushRestartExtra) {
                    SetEvent(m_heEventPoped);
                }
            }
        }
        if (!bCopy) {
            ResetEvent(m_heEventPushed);
        }
        if (pnSize) {
            *pnSize = nSize;
        }
        return bCopy;
    }
    //キューの先頭のデータを取り除く
    //キューが空ならfalseを返す
    bool pop() {
        bool bCopy; decltype(size()) nSize;
        { // 同時に更新しないよう、ロックする
            RGYQueueLock lockUsing(m_bUsingData);
            nSize = size();
            bCopy = nSize > m_nKeepLength;
            if (bCopy) {
                m_pBufOut++;
                if (nSize <= m_nMaxCapacity - m_nPushRestartExtra) {
                    SetEvent(m_heEventPoped);
                }
            }
        }
        if (!bCopy) {
            ResetEvent(m_heEventPushed);
        }
        return bCopy;
    }
    //要素が追加されるまで待機する
    void wait_for_push() {
        WaitForSingleObject(m_heEventPushed, 16);
    }
    //要素が追加されるまで待機するイベントを取得
    HANDLE get_push_event() {
        return m_heEventPushed;
    }
protected:
    //bufSize分の内部領域を確保する
    //m_nMaxCapacity以上確保してもかまわない
    //基本的には大きいほうがパフォーマンスは向上する
    void alloc(size_t bufSize) {
        m_pBufStart = std::unique_ptr<queueData, aligned_malloc_deleter>(
            (queueData *)_aligned_malloc(sizeof(queueData) * bufSize, (std::max)(16, m_nMallocAlign)), aligned_malloc_deleter());
        m_pBufFin = m_pBufStart.get() + bufSize;
        m_pBufIn = m_pBufStart.get();
        m_pBufOut = m_pBufStart.get();
    }

    int m_nPushRestartExtra; //キューに空きがこのぶんだけ余剰にないと空き通知を行わない (0 = ひとつあけば通知を行う)
    HANDLE m_heEventPoped; //キューからデータを取り出したときセットする
    HANDLE m_heEventPushed; //キューにデータが追加されたときセットする
    int m_nMallocAlign; //メモリのアライメント
    size_t m_nMaxCapacity; //キューに詰められる有効なデータの最大数
    size_t m_nKeepLength; //ある一定の長さを常にキュー内に保持するようにする
    alignas(64) std::atomic<queueData*> m_pBufIn; //キューにデータを格納する位置へのポインタ
                std::atomic<queueData*> m_pBufOut; //キューから取り出すべき先頭のデータへのポインタ
                std::unique_ptr<queueData, aligned_malloc_deleter> m_pBufStart; //確保しているメモリ領域の先頭へのポインタ
                queueData *m_pBufFin; //確保しているメモリ領域の終端
    alignas(64) std::atomic<bool> m_bUsingData; //キューから読み出し中のスレッドの数
    alignas(64) std::atomic<bool> m_bPush; //push用のロックの数
};
#pragma warning (pop)

class RGYQueueBuffer {
public:
    RGYQueueBuffer() :
        m_ptr(nullptr),
        m_size(0),
        m_capacity(0),
        m_maxCapacity(1 * 1024 * 1024),
        m_offset(0),
        m_heEventPushed(nullptr),
        m_EOF(false),
        m_mutex() {
        m_heEventPoped = CreateEvent(NULL, TRUE, TRUE, NULL);
        m_heEventPushed = CreateEvent(NULL, TRUE, TRUE, NULL);
    }
    ~RGYQueueBuffer() {
        close();
        if (m_heEventPoped) CloseEvent(m_heEventPoped);
        if (m_heEventPushed) CloseEvent(m_heEventPushed);
    }
    void clear() {
        m_size = 0;
        m_capacity = 0;
        m_offset = 0;
    }
    void close() {
        clear();
        if (m_ptr) {
            free(m_ptr);
            m_ptr = nullptr;
        }
    }
    void init(int64_t bufSize = 1 * 1024 * 1024) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_ptr = (uint8_t *)malloc((size_t)bufSize);
        m_capacity = bufSize;
    }
    int64_t size() const {
        return m_size;
    }
    bool empty() const {
        return m_size == 0;
    }
    bool pushData(const uint8_t *data, int64_t addSize, int timeout) {
        //最初に決めた容量分までキューにデータがたまっていたら、キューに空きができるまで待機する
        int wait = 0;
        while (m_size >= m_maxCapacity) {
            ResetEvent(m_heEventPoped);
            const int wait_time = 16;
            WaitForSingleObject(m_heEventPoped, wait_time);
            wait += wait_time;
            if (wait > timeout) {
                return false;
            }
        }
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_size + addSize > m_capacity) {
            m_capacity = (std::max)(m_capacity * 2, m_size + addSize);
            auto tmp = (uint8_t *)malloc((size_t)m_capacity);
            if (m_size > 0) {
                memcpy(tmp, m_ptr + m_offset, (size_t)m_size);
            }
            free(m_ptr);
            m_ptr = tmp;
            m_offset = 0;
        } else if (m_size + m_offset + addSize > m_capacity) {
            if (m_size > 0) {
                memmove(m_ptr, m_ptr + m_offset, (size_t)m_size);
            }
            m_offset = 0;
        }
        memcpy(m_ptr + m_offset + m_size, data, (size_t)addSize);
        m_size += addSize;
        SetEvent(m_heEventPushed);
        return true;
    }
    int64_t popDataBlock(uint8_t *data, const int64_t maxSize) {
        int64_t copy_size = popData(data, maxSize);
        while (copy_size == 0) {
            WaitForSingleObject(m_heEventPushed, 16);
            copy_size = popData(data, maxSize);
        }
        return copy_size;
    }
    void setEOF() {
        m_EOF = true;
    }
    void setMaxCapacity(int64_t maxCapacity) {
        m_maxCapacity = maxCapacity;
    }
    int64_t getMaxCapacity() const {
        return m_maxCapacity;
    }
protected:
    int64_t popData(uint8_t *data, const int64_t maxSize) {
        if (m_size == 0) return (m_EOF) ? -1 : 0;

        std::lock_guard<std::mutex> lock(m_mutex);
        int64_t copy_size = (std::min)(maxSize, m_size);
        memcpy(data, m_ptr + m_offset, (size_t)copy_size);
        m_offset += copy_size;
        m_size -= copy_size;
        if (m_size == 0) {
            m_offset = 0;
        }
        SetEvent(m_heEventPoped);
        ResetEvent(m_heEventPushed);
        return copy_size;
    }
    uint8_t *m_ptr;
    int64_t m_size;
    int64_t m_capacity;
    int64_t m_maxCapacity;
    int64_t m_offset;
    HANDLE m_heEventPoped; //キューからデータを取り出したときセットする
    HANDLE m_heEventPushed; //キューにデータが追加されたときセットする
    bool m_EOF;
    std::mutex m_mutex;
};

#endif //__RGY_QUEUE_H__
