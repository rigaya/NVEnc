//  -----------------------------------------------------------------------------------------
//    NVEnc by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  -----------------------------------------------------------------------------------------


#include <string>
#include "nvEncodeAPI.h"
#include "NVEncStatus.h"
#include "NVEncInput.h"
#include "ConvertCSP.h"
#include "NVEncCore.h"
#include "NVEncInputVpy.h"
#if VPY_READER
#include <algorithm>
#include <sstream>
#include <map>
#include <fstream>

NVEncInputVpy::NVEncInputVpy() {
    m_sVSapi = NULL;
    m_sVSscript = NULL;
    m_sVSnode = NULL;
    m_nAsyncFrames = 0;
    m_nMaxFrame = 0;
    m_nFrame = 0;
    memset(m_pAsyncBuffer, 0, sizeof(m_pAsyncBuffer));
    memset(m_hAsyncEventFrameSetFin,   0, sizeof(m_hAsyncEventFrameSetFin));
    memset(m_hAsyncEventFrameSetStart, 0, sizeof(m_hAsyncEventFrameSetStart));
    
    m_bAbortAsync = false;
    m_nCopyOfInputFrames = 0;
    memset(&m_sVS, 0, sizeof(m_sVS));

    m_strReaderName = _T("vpy");
}

NVEncInputVpy::~NVEncInputVpy() {
    Close();
}

void NVEncInputVpy::release_vapoursynth() {
    if (m_sVS.hVSScriptDLL)
        FreeLibrary(m_sVS.hVSScriptDLL);

    memset(&m_sVS, 0, sizeof(m_sVS));
}

int NVEncInputVpy::load_vapoursynth() {
    release_vapoursynth();
    
    if (NULL == (m_sVS.hVSScriptDLL = LoadLibrary(_T("vsscript.dll")))) {
        AddMessage(NV_LOG_ERROR, _T("Failed to load vsscript.dll.\n"));
        return 1;
    }

    std::map<void **, const char*> vs_func_list = {
        { (void **)&m_sVS.init,           (VPY_X64) ? "vsscript_init"           : "_vsscript_init@0"            },
        { (void **)&m_sVS.finalize,       (VPY_X64) ? "vsscript_finalize"       : "_vsscript_finalize@0",       },
        { (void **)&m_sVS.evaluateScript, (VPY_X64) ? "vsscript_evaluateScript" : "_vsscript_evaluateScript@16" },
        { (void **)&m_sVS.evaluateFile,   (VPY_X64) ? "vsscript_evaluateFile"   : "_vsscript_evaluateFile@12"   },
        { (void **)&m_sVS.freeScript,     (VPY_X64) ? "vsscript_freeScript"     : "_vsscript_freeScript@4"      },
        { (void **)&m_sVS.getError,       (VPY_X64) ? "vsscript_getError"       : "_vsscript_getError@4"        },
        { (void **)&m_sVS.getOutput,      (VPY_X64) ? "vsscript_getOutput"      : "_vsscript_getOutput@8"       },
        { (void **)&m_sVS.clearOutput,    (VPY_X64) ? "vsscript_clearOutput"    : "_vsscript_clearOutput@8"     },
        { (void **)&m_sVS.getCore,        (VPY_X64) ? "vsscript_getCore"        : "_vsscript_getCore@4"         },
        { (void **)&m_sVS.getVSApi,       (VPY_X64) ? "vsscript_getVSApi"       : "_vsscript_getVSApi@0"        },
    };

    for (auto vs_func : vs_func_list) {
        if (NULL == (*(vs_func.first) = GetProcAddress(m_sVS.hVSScriptDLL, vs_func.second))) {
            AddMessage(NV_LOG_ERROR, _T("Failed to load vsscript functions.\n"));
            return 1;
        }
    }
    return 0;
}

int NVEncInputVpy::initAsyncEvents() {
    for (int i = 0; i < _countof(m_hAsyncEventFrameSetFin); i++) {
        if (   NULL == (m_hAsyncEventFrameSetFin[i]   = CreateEvent(NULL, FALSE, FALSE, NULL))
            || NULL == (m_hAsyncEventFrameSetStart[i] = CreateEvent(NULL, FALSE, TRUE,  NULL)))
            return 1;
    }
    return 0;
}
void NVEncInputVpy::closeAsyncEvents() {
    m_bAbortAsync = true;
    for (int i_frame = m_nCopyOfInputFrames; i_frame < m_nAsyncFrames; i_frame++) {
        if (m_hAsyncEventFrameSetFin[i_frame & (ASYNC_BUFFER_SIZE-1)])
            WaitForSingleObject(m_hAsyncEventFrameSetFin[i_frame & (ASYNC_BUFFER_SIZE-1)], INFINITE);
    }
    for (int i = 0; i < _countof(m_hAsyncEventFrameSetFin); i++) {
        if (m_hAsyncEventFrameSetFin[i])
            CloseHandle(m_hAsyncEventFrameSetFin[i]);
        if (m_hAsyncEventFrameSetStart[i])
            CloseHandle(m_hAsyncEventFrameSetStart[i]);
    }
    memset(m_hAsyncEventFrameSetFin,   0, sizeof(m_hAsyncEventFrameSetFin));
    memset(m_hAsyncEventFrameSetStart, 0, sizeof(m_hAsyncEventFrameSetStart));
    m_bAbortAsync = false;
}

#pragma warning(push)
#pragma warning(disable:4100)
void VS_CC frameDoneCallback(void *userData, const VSFrameRef *f, int n, VSNodeRef *, const char *errorMsg) {
    reinterpret_cast<NVEncInputVpy*>(userData)->setFrameToAsyncBuffer(n, f);
}
#pragma warning(pop)

void NVEncInputVpy::setFrameToAsyncBuffer(int n, const VSFrameRef* f) {
    WaitForSingleObject(m_hAsyncEventFrameSetStart[n & (ASYNC_BUFFER_SIZE-1)], INFINITE);
    m_pAsyncBuffer[n & (ASYNC_BUFFER_SIZE-1)] = f;
    SetEvent(m_hAsyncEventFrameSetFin[n & (ASYNC_BUFFER_SIZE-1)]);

    if (m_nAsyncFrames < m_nMaxFrame && !m_bAbortAsync) {
        m_sVSapi->getFrameAsync(m_nAsyncFrames, m_sVSnode, frameDoneCallback, this);
        m_nAsyncFrames++;
    }
}

int NVEncInputVpy::getRevInfo(const char *vsVersionString) {
    char *api_info = NULL;
    char buf[1024];
    strcpy_s(buf, _countof(buf), vsVersionString);
    for (char *p = buf, *q = NULL, *r = NULL; NULL != (q = strtok_s(p, "\n", &r)); ) {
        if (NULL != (api_info = strstr(q, "Core"))) {
            strcpy_s(buf, _countof(buf), api_info);
            for (char *s = buf; *s; s++)
                *s = (char)tolower(*s);
            int rev = 0;
            return (1 == sscanf_s(buf, "core r%d", &rev)) ? rev : 0;
        }
        p = NULL;
    }
    return 0;
}

int NVEncInputVpy::Init(InputVideoInfo *inputPrm, shared_ptr<EncodeStatus> pStatus) {
    Close();

    m_pEncSatusInfo = pStatus;
    InputInfoVpy *info = reinterpret_cast<InputInfoVpy *>(inputPrm->otherPrm);
    m_bInterlaced = info->interlaced;
    
    if (load_vapoursynth()) {
        return 1;
    }
    //ファイルデータ読み込み
    std::ifstream inputFile(inputPrm->filename);
    if (inputFile.bad()) {
        AddMessage(NV_LOG_ERROR, _T("Failed to open vpy file.\n"));
        return 1;
    }
    std::istreambuf_iterator<char> data_begin(inputFile);
    std::istreambuf_iterator<char> data_end;
    std::string script_data = std::string(data_begin, data_end);
    inputFile.close();

    const VSVideoInfo *vsvideoinfo = NULL;
    const VSCoreInfo *vscoreinfo = NULL;
    if (   !m_sVS.init()
        || initAsyncEvents()
        || NULL == (m_sVSapi = m_sVS.getVSApi())
        || m_sVS.evaluateScript(&m_sVSscript, script_data.c_str(), NULL, efSetWorkingDir)
        || NULL == (m_sVSnode = m_sVS.getOutput(m_sVSscript, 0))
        || NULL == (vsvideoinfo = m_sVSapi->getVideoInfo(m_sVSnode))
        || NULL == (vscoreinfo = m_sVSapi->getCoreInfo(m_sVS.getCore(m_sVSscript)))) {
        AddMessage(NV_LOG_ERROR, _T("VapourSynth Initialize Error.\n"));
        if (m_sVSscript) {
            AddMessage(NV_LOG_ERROR, char_to_tstring(m_sVS.getError(m_sVSscript)).c_str());
        }
        return 1;
    }
    if (vscoreinfo->api < 3) {
        AddMessage(NV_LOG_ERROR, _T("VapourSynth API v3 or later is necessary.\n"));
        return 1;
    }

    if (vsvideoinfo->height <= 0 || vsvideoinfo->width <= 0) {
        AddMessage(NV_LOG_ERROR, _T("Variable resolution is not supported.\n"));
        return 1;
    }

    if (vsvideoinfo->numFrames == 0) {
        AddMessage(NV_LOG_ERROR, _T("Length of input video is unknown.\n"));
        return 1;
    }

    if (!vsvideoinfo->format) {
        AddMessage(NV_LOG_ERROR, _T("Variable colorformat is not supported.\n"));
        return 1;
    }

    if (pfNone == vsvideoinfo->format->id) {
        AddMessage(NV_LOG_ERROR, _T("Invalid colorformat.\n"));
        return 1;
    }

    typedef struct CSPMap {
        int fmtID;
        NV_ENC_CSP in, out;
    } CSPMap;

    static const std::vector<CSPMap> valid_csp_list = {
        { pfYUV420P8, NV_ENC_CSP_YV12,   inputPrm->csp },
        { pfYUV422P8, NV_ENC_CSP_YUV422, inputPrm->csp },
        { pfYUV444P8, NV_ENC_CSP_YUV444, inputPrm->csp }
    };

    for (auto csp : valid_csp_list) {
        if (csp.fmtID == vsvideoinfo->format->id) {
            m_pConvCSPInfo = get_convert_csp_func(csp.in, csp.out, false);
            break;
        }
    }

    if (nullptr == m_pConvCSPInfo) {
        AddMessage(NV_LOG_ERROR, _T("invalid colorformat.\n"));
        return 1;
    }

    if (vsvideoinfo->fpsNum <= 0 || vsvideoinfo->fpsDen <= 0) {
        AddMessage(NV_LOG_ERROR, _T("Invalid framerate.\n"));
        return 1;
    }
    
    int64_t fps_gcd = nv_get_gcd(vsvideoinfo->fpsNum, vsvideoinfo->fpsDen);
    pStatus->m_sData.frameTotal = vsvideoinfo->numFrames;
    m_nMaxFrame = vsvideoinfo->numFrames;
    inputPrm->width = vsvideoinfo->width;
    inputPrm->height = vsvideoinfo->height;
    inputPrm->rate = (int)(vsvideoinfo->fpsNum / fps_gcd);
    inputPrm->scale = (int)(vsvideoinfo->fpsDen / fps_gcd);

    m_nAsyncFrames = vsvideoinfo->numFrames;
    m_nAsyncFrames = (std::min)(m_nAsyncFrames, vscoreinfo->numThreads);
    m_nAsyncFrames = (std::min)(m_nAsyncFrames, ASYNC_BUFFER_SIZE-1);
    if (!info->mt)
        m_nAsyncFrames = 1;

    for (int i = 0; i < m_nAsyncFrames; i++)
        m_sVSapi->getFrameAsync(i, m_sVSnode, frameDoneCallback, this);
    
    TCHAR rev_info[128] = { 0 };
    int rev = getRevInfo(vscoreinfo->versionString);
    if (0 != rev)
        _stprintf_s(rev_info, _countof(rev_info), _T("VapourSynth r%d"), rev);

    memcpy(&m_sDecParam, inputPrm, sizeof(m_sDecParam));
    m_sDecParam.src_pitch = 0;
    CreateInputInfo(rev_info, NV_ENC_CSP_NAMES[m_pConvCSPInfo->csp_from], NV_ENC_CSP_NAMES[m_pConvCSPInfo->csp_to], get_simd_str(m_pConvCSPInfo->simd), inputPrm);
    AddMessage(NV_LOG_DEBUG, m_strInputInfo);
    return 0;
}

void NVEncInputVpy::Close() {
    closeAsyncEvents();
    if (m_sVSapi && m_sVSnode)
        m_sVSapi->freeNode(m_sVSnode);
    if (m_sVSscript)
        m_sVS.freeScript(m_sVSscript);
    if (m_sVSapi)
        m_sVS.finalize();

    release_vapoursynth();
    
    m_bAbortAsync = false;
    m_nCopyOfInputFrames = 0;

    m_sVSapi = NULL;
    m_sVSscript = NULL;
    m_sVSnode = NULL;
    m_nAsyncFrames = 0;
    m_pEncSatusInfo.reset();
}

int NVEncInputVpy::LoadNextFrame(void *dst, int dst_pitch) {
    if (m_nFrame >= m_nMaxFrame) {
        return NVENC_THREAD_FINISHED;
    }

    const VSFrameRef *src_frame = getFrameFromAsyncBuffer(m_nFrame);
    if (NULL == src_frame) {
        return NVENC_THREAD_ERROR;
    }

    void *dst_array[3];
    dst_array[0] = dst;
    dst_array[1] = (uint8_t *)dst_array[0] + dst_pitch * (m_sDecParam.height - m_sDecParam.crop.c[1] - m_sDecParam.crop.c[3]);
    dst_array[2] = (uint8_t *)dst_array[1] + dst_pitch * (m_sDecParam.height - m_sDecParam.crop.c[1] - m_sDecParam.crop.c[3]); //YUV444出力時

    const void *src_array[3] = { m_sVSapi->getReadPtr(src_frame, 0), m_sVSapi->getReadPtr(src_frame, 1), m_sVSapi->getReadPtr(src_frame, 2) };
    m_pConvCSPInfo->func[!!m_bInterlaced](dst_array, src_array, m_sDecParam.width, m_sVSapi->getStride(src_frame, 0), m_sVSapi->getStride(src_frame, 1), dst_pitch, m_sDecParam.height, m_sDecParam.height, m_sDecParam.crop.c);
    
    m_sVSapi->freeFrame(src_frame);

    m_nFrame++;
    m_pEncSatusInfo->m_sData.frameIn++;
    m_nCopyOfInputFrames = m_nFrame;
    m_pEncSatusInfo->UpdateDisplay();
    return 0;
}

#endif //VPY_READER
