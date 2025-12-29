// -----------------------------------------------------------------------------------------
// QSVEnc/NVEnc by rigaya
// -----------------------------------------------------------------------------------------
// The MIT License
//
// ------------------------------------------------------------------------------------------

#include "rgy_vapoursynth_wrapper.h"

#include "rgy_osdep.h"
#include "rgy_filesystem.h"
#include "rgy_util.h"
#include "rgy_codepage.h"

#if ENABLE_VAPOURSYNTH_READER

#include <algorithm>

namespace rgy_vapoursynth_wrapper_v4 {

// VapourSynth v4 headers (provided by SDK/include path)
#include "VSScript4.h"

#if _M_IX86
static constexpr bool VPY4_X64 = false;
#else
static constexpr bool VPY4_X64 = true;
#endif

using func_getVSScriptAPI = const VSSCRIPTAPI * (VS_CC *)(int version);

static func_getVSScriptAPI load_getVSScriptAPI(HMODULE h, void *&outFunc) {
    outFunc = nullptr;
    if (!h) return nullptr;
    func_getVSScriptAPI f = nullptr;
#if defined(_WIN32) || defined(_WIN64)
    const char *candidates[] = {
        "getVSScriptAPI",
#if !VPY4_X64
        "_getVSScriptAPI@4",
#endif
        nullptr
    };
    for (int i = 0; candidates[i]; i++) {
        f = reinterpret_cast<func_getVSScriptAPI>(RGY_GET_PROC_ADDRESS(h, candidates[i]));
        if (f) break;
    }
#else
    f = reinterpret_cast<func_getVSScriptAPI>(RGY_GET_PROC_ADDRESS(h, "getVSScriptAPI"));
#endif
    outFunc = reinterpret_cast<void*>(f);
    return f;
}

class VapourSynthWrapperV4 final : public RGYVapourSynthWrapper {
public:
    VapourSynthWrapperV4(const tstring& vsdir, RGYLog *log) : m_vsdir(vsdir), m_log(log) {}
    virtual ~VapourSynthWrapperV4() override { close(); }

    int apiMajor() const override { return 4; }
    const RGYVapourSynthVideoInfo& videoInfo() const override { return m_vi; }

    bool isAvailable() {
        if (loadDll()) return false;
        const auto f = load_getVSScriptAPI(m_hDll, m_getVSScriptAPI);
        if (!f) { releaseDll(); return false; }
        const auto vssapi = f(VSSCRIPT_API_VERSION);
        releaseDll();
        return vssapi != nullptr;
    }

    RGY_ERR openScriptFromBuffer(const std::string& script, const std::string& scriptFilenameUtf8) override {
        if (loadDll()) return RGY_ERR_NULL_PTR;
        auto getApi = load_getVSScriptAPI(m_hDll, m_getVSScriptAPI);
        if (!getApi) return RGY_ERR_NULL_PTR;
        m_vssapi = getApi(VSSCRIPT_API_VERSION);
        if (!m_vssapi) return RGY_ERR_NULL_PTR;

        m_vsapi = m_vssapi->getVSAPI(VAPOURSYNTH_API_VERSION);
        if (!m_vsapi) return RGY_ERR_NULL_PTR;

        auto core = m_vsapi->createCore(0);
        if (!core) return RGY_ERR_NULL_PTR;

        m_script = m_vssapi->createScript(core);
        if (!m_script) return RGY_ERR_NULL_PTR;

        if (m_vssapi->evaluateBuffer(m_script, script.c_str(), scriptFilenameUtf8.c_str())) {
            return RGY_ERR_NULL_PTR;
        }

        m_node = m_vssapi->getOutputNode(m_script, 0);
        if (!m_node) return RGY_ERR_NULL_PTR;

        const auto vi = m_vsapi->getVideoInfo(m_node);
        if (!vi) return RGY_ERR_NULL_PTR;

        VSCoreInfo coreInfo = {};
        m_vsapi->getCoreInfo(m_vssapi->getCore(m_script), &coreInfo);

        m_vi.width = vi->width;
        m_vi.height = vi->height;
        m_vi.numFrames = vi->numFrames;
        m_vi.fpsNum = vi->fpsNum;
        m_vi.fpsDen = vi->fpsDen;
        m_vi.api = coreInfo.api;
        m_vi.numThreads = coreInfo.numThreads;
        m_vi.versionString = (coreInfo.versionString) ? coreInfo.versionString : "";

        // v4: format is struct
        m_vi.isYUV = (vi->format.colorFamily == cfYUV);
        m_vi.isInteger = (vi->format.sampleType == stInteger);
        m_vi.bitsPerSample = vi->format.bitsPerSample;
        m_vi.subSamplingW = vi->format.subSamplingW;
        m_vi.subSamplingH = vi->format.subSamplingH;
        return RGY_ERR_NONE;
    }

    void getFrameAsync(int n, RGYVapourSynthFrameDoneCallback cb, void *userData) override {
        m_cbPair = { cb, userData };
        m_vsapi->getFrameAsync(n, m_node, [](void *ud, const VSFrame *f, int nn, VSNode *, const char *err) {
            auto p = reinterpret_cast<std::pair<RGYVapourSynthFrameDoneCallback, void*>*>(ud);
            p->first(p->second, f, nn, err);
        }, &m_cbPair);
    }

    const uint8_t *getReadPtr(const void *frame, int plane) const override {
        return m_vsapi->getReadPtr(reinterpret_cast<const VSFrame*>(frame), plane);
    }
    ptrdiff_t getStride(const void *frame, int plane) const override {
        return m_vsapi->getStride(reinterpret_cast<const VSFrame*>(frame), plane);
    }
    void freeFrame(const void *frame) const override {
        m_vsapi->freeFrame(reinterpret_cast<const VSFrame*>(frame));
    }

    void close() override {
        if (m_vsapi && m_node) {
            m_vsapi->freeNode(m_node);
        }
        if (m_vssapi && m_script) {
            m_vssapi->freeScript(m_script);
        }
        m_node = nullptr;
        m_script = nullptr;
        m_vsapi = nullptr;
        m_vssapi = nullptr;
        releaseDll();
    }

private:
    int loadDll() {
        releaseDll();
#if defined(_WIN32) || defined(_WIN64)
        if (m_vsdir.length() > 0 && rgy_directory_exists(m_vsdir)) {
            SetDllDirectory(m_vsdir.c_str());
        }
        m_hDll = RGY_LOAD_LIBRARY(_T("vsscript.dll"));
#else
        m_hDll = dlopen(_T("libvapoursynth-script.so"), RTLD_LAZY|RTLD_GLOBAL);
#endif
        return (m_hDll == nullptr) ? 1 : 0;
    }
    void releaseDll() {
        if (m_hDll) {
#if defined(_WIN32) || defined(_WIN64)
            RGY_FREE_LIBRARY(m_hDll);
#else
            dlclose(m_hDll);
#endif
        }
        m_hDll = nullptr;
        m_getVSScriptAPI = nullptr;
    }

    tstring m_vsdir;
    RGYLog *m_log;

    HMODULE m_hDll = nullptr;
    void *m_getVSScriptAPI = nullptr;

    const VSSCRIPTAPI *m_vssapi = nullptr;
    const VSAPI *m_vsapi = nullptr;
    VSScript *m_script = nullptr;
    VSNode *m_node = nullptr;

    RGYVapourSynthVideoInfo m_vi = {};
    mutable std::pair<RGYVapourSynthFrameDoneCallback, void*> m_cbPair = { nullptr, nullptr };
};

} // namespace

std::unique_ptr<RGYVapourSynthWrapper> CreateVapourSynthWrapperV4(const tstring& vsdir, RGYLog *log) {
    auto p = std::make_unique<rgy_vapoursynth_wrapper_v4::VapourSynthWrapperV4>(vsdir, log);
    if (!p->isAvailable()) return nullptr;
    return p;
}

#endif // ENABLE_VAPOURSYNTH_READER


