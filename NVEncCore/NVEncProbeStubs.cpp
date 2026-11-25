#include "NVEncFilterSsim.h"

// 診断構成では品質比較フィルタを作らないが、通常版と共通の呼び出し側のリンク先は必要になる。
RGY_ERR NVEncFilterSsim::initDecode(const RGYBitstream *) {
    return RGY_ERR_UNSUPPORTED;
}

RGY_ERR NVEncFilterSsim::addBitstream(const RGYBitstream *) {
    return RGY_ERR_UNSUPPORTED;
}
