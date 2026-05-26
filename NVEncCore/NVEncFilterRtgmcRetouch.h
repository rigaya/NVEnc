// -----------------------------------------------------------------------------------------
// NVEnc by rigaya
// -----------------------------------------------------------------------------------------
//
// The MIT License
//
// Copyright (c) 2026 rigaya
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

#pragma once

#include <fstream>
#include <string>
#include <vector>

#include "NVEncFilter.h"
#include "NVEncFilterDegrainMV.h"
#include "rgy_frame_info.h"
#include "rgy_prm.h"

struct RGYRtgmcRetouchTemporalLimitFrames {
    const RGYFrameInfo *ref;
    const RGYFrameInfo *motionBack;
    const RGYFrameInfo *motionForw;

    RGYRtgmcRetouchTemporalLimitFrames() :
        ref(nullptr),
        motionBack(nullptr),
        motionForw(nullptr) {
    }

    bool any() const {
        return ref || motionBack || motionForw;
    }

    bool valid() const {
        return ref != nullptr && motionBack != nullptr && motionForw != nullptr;
    }
};

class NVEncFilterParamRtgmcRetouch : public NVEncFilterParam {
public:
    VppRtgmcRetouch rtgmc_retouch;
    RGYRtgmcRetouchTemporalLimitFrames temporalLimit;
    bool skipPostTR2LimitModes;

    NVEncFilterParamRtgmcRetouch() : rtgmc_retouch(), temporalLimit(), skipPostTR2LimitModes(false) {}
    virtual ~NVEncFilterParamRtgmcRetouch() {}
    virtual tstring print() const override { return rtgmc_retouch.print(); }
};

class NVEncFilterRtgmcRetouch : public NVEncFilter {
public:
    NVEncFilterRtgmcRetouch();
    virtual ~NVEncFilterRtgmcRetouch();
    virtual RGY_ERR init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) override;
    void setSpatialLimitBaseFrame(const RGYFrameInfo *frame);
    void clearSpatialLimitBaseFrame();
    void setTemporalLimitFrames(const RGYRtgmcRetouchTemporalLimitFrames &frames);
    void clearTemporalLimitFrames();

protected:
    virtual RGY_ERR run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
        cudaStream_t stream) override;
    RGY_ERR run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
        cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events, RGYCudaEvent *event);
    virtual void close() override;

    RGY_ERR checkParam(const std::shared_ptr<NVEncFilterParamRtgmcRetouch> &prm);
    RGY_ERR buildKernels(const std::shared_ptr<NVEncFilterParamRtgmcRetouch> &prm);
    RGY_ERR processFrame(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame,
        const NVEncFilterParamRtgmcRetouch &prm,
        cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events, RGYCudaEvent *event);
    bool temporalLimitFramesReady(const RGYFrameInfo *srcFrame) const;
    bool temporalLimitFramesCompatible(const RGYFrameInfo *srcFrame) const;

    enum class RtgmcRetouchNodeKind {
        DetailBoost,
        EdgeNarrowCorrection,
        PreLimitRollback,
        SpatialOvershootGuard,
        TemporalOvershootGuard,
        PostLimitRollback,
    };

    struct RtgmcRetouchNode {
        RtgmcRetouchNodeKind kind;
        int inputSlot;
        int outputSlot;
        int workSlot0;
        int workSlot1;
        const char *dumpStage;
    };

    struct RtgmcRetouchPlan {
        std::vector<RtgmcRetouchNode> nodes;
    };

    RtgmcRetouchPlan buildRtgmcRetouchPlan(const VppRtgmcRetouch &retouch, bool chromaPlane, bool skipPostTR2LimitModes, float detailGain) const;
    std::string describeRtgmcRetouchPlan(const RtgmcRetouchPlan &plan) const;

    RGY_ERR setupDetailRollbackGaussFilter(const NVEncFilterParamRtgmcRetouch &prm);
    RGY_ERR initLumaDump(const RGYFrameInfo &frameInfo, const NVEncFilterParamRtgmcRetouch &prm);
    RGY_ERR dumpLumaFrame(const RGYFrameInfo *frame, cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events, bool dumpChroma);
    RGY_ERR dumpStageFrame(const char *stage, const RGYFrameInfo *frame, const char *target,
        cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events);

    std::string m_buildOptions;
    std::ofstream m_lumaDump;
    std::string m_lumaDumpPath;
    std::string m_lumaDumpStage;
    std::string m_lumaDumpTarget;
    int m_lumaDumpMaxFrames;
    int m_lumaDumpFrameCount;
    bool m_lumaDumpEnabled;
    bool m_lumaDumpHeaderWritten;
    bool m_lumaDumpChroma;
    bool m_useKernel;
    RGYRtgmcRetouchTemporalLimitFrames m_temporalLimitFrames;
    const RGYFrameInfo *m_spatialLimitBaseFrame;
    bool m_loggedTemporalFallback;
};
