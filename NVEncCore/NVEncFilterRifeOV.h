#pragma once
#ifndef __NVENC_FILTER_RIFE_OV_H__
#define __NVENC_FILTER_RIFE_OV_H__

#include "NVEncFilter.h"
#include "NVEncFilterParam.h"
#include "rgy_prm.h"
#include "rgy_onnxrt_cuda.h"
#include <memory>
#include <vector>

class NVEncFilterParamRifeOV : public NVEncFilterParam {
public:
    tstring modelFile;
    tstring modelDir;
    tstring device;
    int multi;
    tstring colormatrix;
    tstring colorrange;
    int deviceID;
    NVEncFilterParamRifeOV() : modelFile(), modelDir(), device(_T("GPU.0")), multi(2), colormatrix(_T("auto")), colorrange(_T("auto")), deviceID(-1) {};
    virtual tstring print() const override;
};

class NVEncFilterRifeOV : public NVEncFilter {
public:
    NVEncFilterRifeOV();
    virtual ~NVEncFilterRifeOV();
    virtual RGY_ERR init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) override;
protected:
    virtual RGY_ERR run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum, cudaStream_t stream) override;
    virtual void close() override;

    void yuvToRGB(const RGYFrameInfo &hin, float *dst);
    void rgbToYUV(const RGYFrameInfo &hout, const float *src);
    void setupColorCoeffs(int matrixSel, bool rangeTV, int pixMax);
    RGY_ERR interpolate(float t);
    RGY_ERR initCudaPath(cudaStream_t stream);
    RGY_ERR runCuda(const RGYFrameInfo *input, RGYFrameInfo **outputs, int *outputCount, cudaStream_t stream);
    RGYFrameInfo rgbFrame(float *ptr) const;

    std::unique_ptr<RGYOnnxRTCUDA> m_ov;
    int m_W, m_H, m_multi;
    float m_maxval;
    float m_yOff, m_yScale, m_yRange, m_cOff, m_cScale, m_cRange;
    float m_matVR, m_matUG, m_matVG, m_matUB;
    float m_matRY, m_matGY, m_matBY, m_matRU, m_matGU, m_matBU, m_matRV, m_matGV, m_matBV;
    bool m_havePrev;
    int64_t m_prevTimestamp, m_prevDuration;
    std::vector<float> m_prevRGB, m_currRGB, m_inBuf, m_outBuf, m_baseGrid, m_multiplier;
    std::unique_ptr<CUFrameBuf> m_inStaging, m_outStaging;
    std::unique_ptr<CUMemBuf> m_inputDevice, m_outputDevice;
    std::unique_ptr<NVEncFilterCspCrop> m_cropToRgb, m_cropFromRgb;
    tstring m_modelPath;
    int m_deviceID;
    bool m_cudaPathTried, m_cudaPath;
};

#endif
