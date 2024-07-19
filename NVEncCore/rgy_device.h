// -----------------------------------------------------------------------------------------
// QSVEnc/NVEnc by rigaya
// -----------------------------------------------------------------------------------------
// The MIT License
//
// Copyright (c) 2011-2016 rigaya
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
// -------------------------------------------------------------------------------------------

#pragma once
#ifndef __RGY_DEVICE_H__
#define __RGY_DEVICE_H__

#include "rgy_osdep.h"
#include "rgy_version.h"
#include <string>
#include <memory>
#include <future>
#include <thread>

#if ENABLE_D3D9 || ENABLE_D3D11
#include <SDKDDKVer.h>
#include <atlbase.h>
#endif
#if ENABLE_D3D9
#include <d3d9.h>
#endif
#if ENABLE_D3D11
#include <d3d11.h>
#if ENABLE_D3D11_DEVINFO_WMI
#include "rgy_device_info_wmi.h"
#endif
#endif
#include "rgy_err.h"
#include "rgy_log.h"

#if ENABLE_D3D9
class DeviceDX9 {
public:
    DeviceDX9();
    virtual ~DeviceDX9();

    RGY_ERR Init(bool dx9ex, int adapterID, bool bFullScreen, int width, int height, std::shared_ptr<RGYLog> log);
    RGY_ERR Terminate();

    bool isValid() const;
    ATL::CComPtr<IDirect3DDevice9> GetDevice();
    LUID getLUID() const { return m_devLUID; };
    std::wstring GetDisplayDeviceName() const { return m_displayDeviceName; }
    static int adapterCount();
protected:
    void AddMessage(RGYLogLevel log_level, const tstring &str);
    void AddMessage(RGYLogLevel log_level, const TCHAR *format, ...);
private:
    RGY_ERR EnumerateAdapters();

    tstring                             m_name;
    ATL::CComPtr<IDirect3D9>            m_pD3D;
    ATL::CComPtr<IDirect3DDevice9>      m_pD3DDevice;
    LUID                                m_devLUID;

    static const int             MAXADAPTERS = 128;
    int                          m_adaptersCount;
    int                          m_adaptersIndexes[MAXADAPTERS];
    std::wstring                 m_displayDeviceName;
    std::shared_ptr<RGYLog>      m_log;
};
#endif

#if ENABLE_D3D11

class DX11AdapterManager {
private:
    static DX11AdapterManager *m_instance;
    std::vector<int> m_adaptersIndexes;
    bool m_onlyWithOutputs;
    bool m_initialized;

    DX11AdapterManager() : m_adaptersIndexes(), m_onlyWithOutputs(false), m_initialized(false) { }
public:
    static DX11AdapterManager *getInstance() {
        if (!m_instance) {
            m_instance = new DX11AdapterManager();
            m_instance->EnumerateAdapters();
        }
        return m_instance;
    }

    DX11AdapterManager(const DX11AdapterManager&) = delete;
    DX11AdapterManager& operator=(const DX11AdapterManager&) = delete;

    int adapterCount() { return (int)m_adaptersIndexes.size(); }
    const std::vector<int>& getAdapterIndexes() { return m_adaptersIndexes; }
private:
    void EnumerateAdapters();
};


class DeviceDX11 {
public:
    DeviceDX11();
    virtual ~DeviceDX11();

    RGY_ERR Init(int adapterID, std::shared_ptr<RGYLog> log);
    RGY_ERR Terminate();

    bool isValid() const;
    ID3D11Device *GetDevice();
    ID3D11DeviceContext *GetDeviceContext();
    IDXGIAdapter *GetAdaptor();
    LUID getLUID() const { return m_devLUID; };
    int getVendorID() const { return m_vendorID; };
    int getDeviceID() const { return m_deviceID; };
    std::wstring GetDisplayDeviceName() const { return m_displayDeviceName; }
    tstring getDriverVersion();
#if ENABLE_D3D11_DEVINFO_WMI
    const RGYDeviceInfoWMI *getDeviceInfo();
#endif
    static int adapterCount();
protected:
    void AddMessage(RGYLogLevel log_level, const tstring &str);
    void AddMessage(RGYLogLevel log_level, const TCHAR *format, ...);
private:

    tstring                      m_name;
    ATL::CComPtr<IDXGIAdapter>   m_pAdapter;
    ATL::CComPtr<ID3D11Device>   m_pD3DDevice;
    ATL::CComPtr<ID3D11DeviceContext> m_pD3DDeviceCtx;
    LUID                         m_devLUID;
    int                          m_vendorID;
    int                          m_deviceID;

#if ENABLE_D3D11_DEVINFO_WMI
    std::unique_ptr<RGYDeviceInfoWMI> m_deviceInfo;
    std::future<std::tuple<RGY_ERR, std::unique_ptr<RGYDeviceInfoWMI>>> m_fDeviceInfo;
#endif
    std::wstring                 m_displayDeviceName;
    std::shared_ptr<RGYLog>      m_log;
};
#endif

#endif //__RGY_DEVICE_H__
