// -----------------------------------------------------------------------------------------
// NVEnc by rigaya
// -----------------------------------------------------------------------------------------

#include "NVEncFilterRtgmcSearchPrefilter.cuh"

const NVEncRtgmcSearchPrefilterLaunchFuncs *getNVEncRtgmcSearchPrefilterU16TRP() {
    return getNVEncRtgmcSearchPrefilterLaunchFuncs<uint16_t>();
}
