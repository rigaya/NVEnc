// -----------------------------------------------------------------------------------------
// NVEnc by rigaya
// -----------------------------------------------------------------------------------------

#include "NVEncFilterRtgmcSearchPrefilter.cuh"

const NVEncRtgmcSearchPrefilterLaunchFuncs *getNVEncRtgmcSearchPrefilterU8TR0() {
    return getNVEncRtgmcSearchPrefilterLaunchFuncs<uint8_t, 0>();
}
