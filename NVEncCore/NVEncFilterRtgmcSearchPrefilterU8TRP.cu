// -----------------------------------------------------------------------------------------
// NVEnc by rigaya
// -----------------------------------------------------------------------------------------

#include "NVEncFilterRtgmcSearchPrefilter.cuh"

const NVEncRtgmcSearchPrefilterLaunchFuncs *getNVEncRtgmcSearchPrefilterU8TRP() {
    return getNVEncRtgmcSearchPrefilterLaunchFuncs<uint8_t>();
}
