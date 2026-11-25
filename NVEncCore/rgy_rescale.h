#pragma once

#include <cstdint>

// 遅延ロードされるライブラリがなくても、同じ規則で時刻を換算できるようにする。
inline int64_t rgy_rescale(int64_t value, int fromNum, int fromDen, int toNum, int toDen) {
    const int64_t numerator = static_cast<int64_t>(fromNum) * toDen;
    const int64_t denominator = static_cast<int64_t>(fromDen) * toNum;
    if (numerator < 0 || denominator <= 0) {
        return INT64_MIN;
    }

    // 比較元と同じく、負の最小値も絶対値を符号付き64ビットで表せる範囲に丸める。
    const bool negative = value < 0;
    const uint64_t magnitude = value == INT64_MIN ? INT64_MAX : (negative ? -value : value);
    const uint64_t multiplier = numerator;
    const uint64_t divisor = denominator;
    const uint64_t half = divisor / 2;
    uint64_t quotient;
    if (multiplier == 0) {
        return 0;
    }
    if (magnitude <= (UINT64_MAX - half) / multiplier) {
        quotient = (magnitude * multiplier + half) / divisor;
    } else {
        // 32ビットずつに分けて積を求め、符号に関係なく中間値を正確に保持する。
        constexpr uint64_t lowMask = UINT32_MAX;
        const uint64_t lowProduct = (magnitude & lowMask) * (multiplier & lowMask);
        const uint64_t crossProduct = (magnitude >> 32) * (multiplier & lowMask)
                                    + (magnitude & lowMask) * (multiplier >> 32);
        const uint64_t middle = (lowProduct >> 32) + (crossProduct & lowMask);
        uint64_t high = (magnitude >> 32) * (multiplier >> 32) + (crossProduct >> 32) + (middle >> 32);
        uint64_t low = (middle << 32) | (lowProduct & lowMask);

        // 除算前に分母の半分を加え、ちょうど半分なら絶対値の大きい方に丸める。
        low += half;
        high += low < half;
        if (high >= divisor) {
            return INT64_MIN;
        }

        // 上位語を余りの初期値として筆算する。分母は最大でも2の62乗なので、
        // 分母未満の余りに下位語の次のビットを足しても桁あふれしない。
        quotient = 0;
        for (int bit = 63; bit >= 0; bit--) {
            high = (high << 1) | ((low >> bit) & 1);
            if (high >= divisor) {
                high -= divisor;
                quotient |= uint64_t{1} << bit;
            }
        }
    }
    if (quotient > INT64_MAX) {
        return INT64_MIN;
    }
    return negative ? -static_cast<int64_t>(quotient) : static_cast<int64_t>(quotient);
}
