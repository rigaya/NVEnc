#pragma once

#include <array>
#include <memory>
#include <vector>

#include "NVEncFilterDegrainMV.h"
#include "NVEncFilterNnedi.h"
#include "rgy_prm.h"

class NVEncFilterParamRtgmcEdi : public NVEncFilterParam {
public:
    VppRtgmcEdiMode mode;
    VppRtgmcChromaEdiMode chromaEdi;
    int nnsize;
    int nneurons;
    int ediqual;
    VppRtgmcBobOrder order;
    RGYFrameInfo sourceFrameIn;
    rgy_rational<int> sourceBaseFps;
    rgy_rational<int> sourceTimebase;
    HMODULE hModule;

    NVEncFilterParamRtgmcEdi()
        : mode(VppRtgmcEdiMode::BobChromaMerge),
          chromaEdi(VppRtgmcChromaEdiMode::None),
          nnsize(1),
          nneurons(1),
          ediqual(1),
          order(VppRtgmcBobOrder::Auto),
          sourceFrameIn(),
          sourceBaseFps(),
          sourceTimebase(),
          hModule(NULL) {};
    virtual ~NVEncFilterParamRtgmcEdi() {};
    virtual tstring print() const override;
};

class NVEncFilterRtgmcEdi : public NVEncFilter {
public:
    NVEncFilterRtgmcEdi();
    virtual ~NVEncFilterRtgmcEdi();
    virtual RGY_ERR init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) override;

    RGY_ERR run_filter(const RGYFrameInfo *pBobInputFrame, const RGYFrameInfo *pEdiInputFrame,
        RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
        cudaStream_t stream);
    RGY_ERR run_filter(const RGYFrameInfo *pBobInputFrame, const RGYFrameInfo *pEdiInputFrame, const RGYFrameInfo *pSourceInputFrame,
        RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
        cudaStream_t stream);

protected:
    virtual RGY_ERR run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
        cudaStream_t stream) override;
    virtual void close() override;
public:
    virtual void resetTemporalState() override;
protected:

    class FrameSource {
    public:
        FrameSource();
        RGY_ERR alloc(const RGYFrameInfo& frameInfo);
        RGY_ERR add(const RGYFrameInfo *pInputFrame, cudaStream_t stream, bool copyChroma = true);
        CUFrameBuf *get(int iframe);
        int findIndexByInputFrameId(int inputFrameId) const;
        int inframe() const { return m_nFramesInput; }
        void clear();
        // Logical reset for resetTemporalState(): rewind the ring buffer without freeing the
        // pooled GPU allocations (clear() would memfree them, leaving pitch=0 and breaking the
        // next add() with cudaErrorInvalidPitchValue).
        void resetFrames();
    private:
        int m_nFramesInput;
        std::array<CUFrameBuf, 4> m_buf;
    };

    struct FrameKey {
        int inputFrameId;
        int64_t timestamp;
        int64_t duration;

        FrameKey() : inputFrameId(-1), timestamp(0), duration(0) {}
        explicit FrameKey(const RGYFrameInfo *frame)
            : inputFrameId(frame ? frame->inputFrameId : -1),
              timestamp(frame ? frame->timestamp : 0),
              duration(frame ? frame->duration : 0) {}
        bool matches(const RGYFrameInfo *frame) const {
            return frame
                && inputFrameId == frame->inputFrameId
                && timestamp == frame->timestamp
                && duration == frame->duration;
        }
    };

    struct NnediAdapterState {
        std::unique_ptr<NVEncFilterNnedi> filter;
        std::unique_ptr<NVEncFilterCspCrop> outputCsp;
        std::array<RGYFrameInfo *, 2> cachedFrames;
        FrameKey cachedKey;
        RGYCudaEvent cachedEvent;
        bool cacheValid;

        NnediAdapterState();
        void clear();
    };

    RGY_ERR checkParam(const std::shared_ptr<NVEncFilterParamRtgmcEdi> &prm);
    RGY_ERR checkInputs(const RGYFrameInfo *pBobInputFrame, const RGYFrameInfo *pEdiInputFrame);
    RGY_ERR buildKernels(const std::shared_ptr<NVEncFilterParamRtgmcEdi> &prm);
    RGY_ERR run_filter_impl(const RGYFrameInfo *pBobInputFrame, const RGYFrameInfo *pEdiInputFrame, const RGYFrameInfo *pSourceInputFrame,
        RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
        cudaStream_t stream,
        const NVEncFilterParamRtgmcEdi &prm);
    RGY_ERR processFrame(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pBobInputFrame,
        const RGYFrameInfo *pEdiPrevFrame, const RGYFrameInfo *pEdiInputFrame, const RGYFrameInfo *pEdiNextFrame,
        const NVEncFilterParamRtgmcEdi &prm,
        const int targetField,
        cudaStream_t stream);
    RGY_ERR runTemporalYadif(const RGYFrameInfo *pBobInputFrame, const RGYFrameInfo *pEdiInputFrame, const RGYFrameInfo *pSourceInputFrame,
        RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
        cudaStream_t stream,
        const NVEncFilterParamRtgmcEdi &prm);
    RGY_ERR initNnediAdapterState(NnediAdapterState &state, const std::shared_ptr<NVEncFilterParamRtgmcEdi> &prm, const bool chroma);
    RGY_ERR runNnediAdapterState(NnediAdapterState &state, const RGYFrameInfo *pBobInputFrame, const RGYFrameInfo *pSourceInputFrame,
        RGYFrameInfo **ppOutputFrame, const RGYFrameInfo **ppSelectedFrame,
        cudaStream_t stream,
        const NVEncFilterParamRtgmcEdi &prm, const bool chroma);
    RGY_ERR runNnediAdapter(const RGYFrameInfo *pBobInputFrame, const RGYFrameInfo *pSourceInputFrame,
        RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
        cudaStream_t stream,
        const NVEncFilterParamRtgmcEdi &prm);
    int targetField(const RGYFrameInfo *pBobInputFrame, const RGYFrameInfo *pParityFrame = nullptr);

    std::string m_buildOptions;
    FrameSource m_bobSource;
    FrameSource m_ediSource;
    FrameSource m_inputSource;
    std::array<NnediAdapterState, 2> m_nnediStates;
    RGYCudaEvent m_nnediAdapterCopyEvent;
    int m_nFrame;
    int m_lastInputFrameId;
    int m_pairFrameIndex;
    int m_fallbackFrameIndex;
    bool m_useKernel;
};
