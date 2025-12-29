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

#if ENABLE_VAPOURSYNTH_READER


#include <algorithm>
#include <utility>

namespace rgy_vapoursynth_wrapper_v3 {

#include "VapourSynth.h"
#include "VSScript.h"

#if _M_IX86
static constexpr bool VPY_X64 = false;
#else
static constexpr bool VPY_X64 = true;
#endif

// import lib を使うと vsscript.dll がない場合に起動できないので動的ロード
typedef int (__stdcall *func_vs_init)(void);
typedef int (__stdcall *func_vs_finalize)(void);
typedef int (__stdcall *func_vs_evaluateScript)(VSScript **handle, const char *script, const char *errorFilename, int flags);
typedef void (__stdcall *func_vs_freeScript)(VSScript *handle);
typedef const char * (__stdcall *func_vs_getError)(VSScript *handle);
typedef VSNodeRef * (__stdcall *func_vs_getOutput)(VSScript *handle, int index);
typedef VSCore * (__stdcall *func_vs_getCore)(VSScript *handle);
typedef const VSAPI * (__stdcall *func_vs_getVSApi)(void);

struct vsscript3_t {
    HMODULE hVSScriptDLL = nullptr;
    func_vs_init init = nullptr;
    func_vs_finalize finalize = nullptr;
    func_vs_evaluateScript evaluateScript = nullptr;
    func_vs_freeScript freeScript = nullptr;
    func_vs_getError getError = nullptr;
    func_vs_getOutput getOutput = nullptr;
    func_vs_getCore getCore = nullptr;
    func_vs_getVSApi getVSApi = nullptr;
};

class VapourSynthWrapperV3 final : public RGYVapourSynthWrapper {
public:
    VapourSynthWrapperV3(const tstring& vsdir, RGYLog *log) : m_vsdir(vsdir), m_log(log) {}
    virtual ~VapourSynthWrapperV3() override { close(); }

    int apiMajor() const override { return 3; }
    const RGYVapourSynthVideoInfo& videoInfo() const override { return m_vi; }

    RGY_ERR openScriptFromBuffer(const std::string& script, const std::string& scriptFilenameUtf8) override {
        (void)scriptFilenameUtf8; // v3 は従来通り errorFilename を nullptr で評価

        if (loadDll()) {
            return RGY_ERR_NULL_PTR;
        }
        if (!m_vs.init()) {
            return RGY_ERR_NULL_PTR;
        }
        m_vsapi = m_vs.getVSApi();
        if (!m_vsapi) {
            return RGY_ERR_NULL_PTR;
        }
        if (m_vs.evaluateScript(&m_script, script.c_str(), nullptr, efSetWorkingDir)) {
            return RGY_ERR_NULL_PTR;
        }
        m_node = m_vs.getOutput(m_script, 0);
        if (!m_node) {
            return RGY_ERR_NULL_PTR;
        }
        const auto vsvideoinfo = m_vsapi->getVideoInfo(m_node);
        if (!vsvideoinfo) {
            return RGY_ERR_NULL_PTR;
        }

        // core info
        VSCoreInfo vscoreinfo = {};
        if (m_vsapi->getCoreInfo2) {
            m_vsapi->getCoreInfo2(m_vs.getCore(m_script), &vscoreinfo);
        } else {
            #pragma warning(push)
            #pragma warning(disable:4996) //warning C4996: 'VSAPI::getCoreInfo': getCoreInfo has been deprecated as of api 3.6, use getCoreInfo2 instead
            RGY_DISABLE_WARNING_PUSH
            RGY_DISABLE_WARNING_STR("-Wdeprecated-declarations")
            auto infoptr = m_vsapi->getCoreInfo(m_vs.getCore(m_script));
            RGY_DISABLE_WARNING_POP
            #pragma warning(pop)
            if (!infoptr) return RGY_ERR_NULL_PTR;
            vscoreinfo = *infoptr;
        }

        m_vi.width = vsvideoinfo->width;
        m_vi.height = vsvideoinfo->height;
        m_vi.numFrames = vsvideoinfo->numFrames;
        m_vi.fpsNum = vsvideoinfo->fpsNum;
        m_vi.fpsDen = vsvideoinfo->fpsDen;
        m_vi.api = vscoreinfo.api;
        m_vi.numThreads = vscoreinfo.numThreads;
        m_vi.versionString = (vscoreinfo.versionString) ? vscoreinfo.versionString : "";

        // format decode
        m_vi.isYUV = false;
        m_vi.isInteger = true;
        m_vi.bitsPerSample = 0;
        m_vi.subSamplingW = 0;
        m_vi.subSamplingH = 0;
        if (vsvideoinfo->format) {
            const int fmtid = vsvideoinfo->format->id;
            if (fmtid == pfNone) {
                // unknown
            } else {
                // v3 の preset ID から必要情報だけ抽出
                auto set420 = [&](int bits) { m_vi.isYUV = true; m_vi.bitsPerSample = bits; m_vi.subSamplingW = 1; m_vi.subSamplingH = 1; };
                auto set422 = [&](int bits) { m_vi.isYUV = true; m_vi.bitsPerSample = bits; m_vi.subSamplingW = 1; m_vi.subSamplingH = 0; };
                auto set444 = [&](int bits) { m_vi.isYUV = true; m_vi.bitsPerSample = bits; m_vi.subSamplingW = 0; m_vi.subSamplingH = 0; };
                switch (fmtid) {
                case pfYUV420P8:  set420(8);  break;
                case pfYUV420P10: set420(10); break;
                case pfYUV420P12: set420(12); break;
                case pfYUV420P14: set420(14); break;
                case pfYUV420P16: set420(16); break;
                case pfYUV422P8:  set422(8);  break;
                case pfYUV422P10: set422(10); break;
                case pfYUV422P12: set422(12); break;
                case pfYUV422P14: set422(14); break;
                case pfYUV422P16: set422(16); break;
                case pfYUV444P8:  set444(8);  break;
                case pfYUV444P10: set444(10); break;
                case pfYUV444P12: set444(12); break;
                case pfYUV444P14: set444(14); break;
                case pfYUV444P16: set444(16); break;
                default:
                    break;
                }
            }
        }
        return RGY_ERR_NONE;
    }

    void getFrameAsync(int n, RGYVapourSynthFrameDoneCallback cb, void *userData) override {
        // v3 callback signature is compatible except VSNodeRef*
        auto thunk = [](void *ud, const VSFrameRef *f, int nn, VSNodeRef *, const char *err) {
            auto p = reinterpret_cast<std::pair<RGYVapourSynthFrameDoneCallback, void*>*>(ud);
            p->first(p->second, f, nn, err);
        };
        m_cbPair = { cb, userData };
        m_vsapi->getFrameAsync(n, m_node, thunk, &m_cbPair);
    }

    const uint8_t *getReadPtr(const void *frame, int plane) const override {
        return m_vsapi->getReadPtr(reinterpret_cast<const VSFrameRef*>(frame), plane);
    }
    ptrdiff_t getStride(const void *frame, int plane) const override {
        return m_vsapi->getStride(reinterpret_cast<const VSFrameRef*>(frame), plane);
    }
    void freeFrame(const void *frame) const override {
        m_vsapi->freeFrame(reinterpret_cast<const VSFrameRef*>(frame));
    }

    void close() override {
        if (m_vsapi && m_node) {
            m_vsapi->freeNode(m_node);
        }
        if (m_script) {
            m_vs.freeScript(m_script);
        }
        if (m_vsapi) {
            m_vs.finalize();
        }
        releaseDll();
        m_vsapi = nullptr;
        m_script = nullptr;
        m_node = nullptr;
    }

    bool isAvailable() {
        if (loadDll()) return false;
        releaseDll();
        return true;
    }

private:
    int loadDll() {
        releaseDll();
#if defined(_WIN32) || defined(_WIN64)
        if (m_vsdir.length() > 0) {
            if (rgy_directory_exists(m_vsdir)) {
                SetDllDirectory(m_vsdir.c_str());
            }
        }
        const TCHAR *dllname = _T("vsscript.dll");
        m_vs.hVSScriptDLL = RGY_LOAD_LIBRARY(dllname);
#else
        const TCHAR *dllname = _T("libvapoursynth-script.so");
        m_vs.hVSScriptDLL = dlopen(dllname, RTLD_LAZY|RTLD_GLOBAL);
#endif
        if (!m_vs.hVSScriptDLL) {
            return 1;
        }
        auto load = [&](void **dst, const char *name) -> bool {
            *dst = RGY_GET_PROC_ADDRESS(m_vs.hVSScriptDLL, name);
            return *dst != nullptr;
        };
        const char *n_init     = (VPY_X64) ? "vsscript_init"     : "_vsscript_init@0";
        const char *n_fin      = (VPY_X64) ? "vsscript_finalize" : "_vsscript_finalize@0";
        const char *n_eval     = (VPY_X64) ? "vsscript_evaluateScript" : "_vsscript_evaluateScript@16";
        const char *n_free     = (VPY_X64) ? "vsscript_freeScript" : "_vsscript_freeScript@4";
        const char *n_err      = (VPY_X64) ? "vsscript_getError" : "_vsscript_getError@4";
        const char *n_out      = (VPY_X64) ? "vsscript_getOutput" : "_vsscript_getOutput@8";
        const char *n_core     = (VPY_X64) ? "vsscript_getCore" : "_vsscript_getCore@4";
        const char *n_vsapi    = (VPY_X64) ? "vsscript_getVSApi" : "_vsscript_getVSApi@0";

        if (!load((void**)&m_vs.init, n_init)
         || !load((void**)&m_vs.finalize, n_fin)
         || !load((void**)&m_vs.evaluateScript, n_eval)
         || !load((void**)&m_vs.freeScript, n_free)
         || !load((void**)&m_vs.getError, n_err)
         || !load((void**)&m_vs.getOutput, n_out)
         || !load((void**)&m_vs.getCore, n_core)
         || !load((void**)&m_vs.getVSApi, n_vsapi)) {
            releaseDll();
            return 1;
        }
        return 0;
    }

    void releaseDll() {
        if (m_vs.hVSScriptDLL) {
#if defined(_WIN32) || defined(_WIN64)
            RGY_FREE_LIBRARY(m_vs.hVSScriptDLL);
#else
            dlclose(m_vs.hVSScriptDLL);
#endif
        }
        m_vs = {};
    }

    tstring m_vsdir;
    RGYLog *m_log;

    vsscript3_t m_vs;
    const VSAPI *m_vsapi = nullptr;
    VSScript *m_script = nullptr;
    VSNodeRef *m_node = nullptr;

    RGYVapourSynthVideoInfo m_vi = {};
    mutable std::pair<RGYVapourSynthFrameDoneCallback, void*> m_cbPair = { nullptr, nullptr };
};

} // namespace

std::unique_ptr<RGYVapourSynthWrapper> CreateVapourSynthWrapperV3(const tstring& vsdir, RGYLog *log) {
    auto p = std::make_unique<rgy_vapoursynth_wrapper_v3::VapourSynthWrapperV3>(vsdir, log);
    if (!p->isAvailable()) return nullptr;
    return p;
}

#endif // ENABLE_VAPOURSYNTH_READER


