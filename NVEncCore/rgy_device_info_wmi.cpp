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

#if defined(_WIN32) || defined(_WIN64)
#include <regex>
#include "rgy_device_info_wmi.h"
#include "rgy_osdep.h"
#include <comdef.h>
#include <Wbemidl.h>

RGYDeviceInfoWMI::RGYDeviceInfoWMI() :
    vendorID(0),
    deviceID(0),
    name(),
    PNPDeviceID(),
    DriverVersion(),
    DriverDate() {

}

std::tuple<RGY_ERR, std::unique_ptr<RGYDeviceInfoWMI>> GetDeviceInfoWMI(const int vendorID, const int deviceID) {
    HRESULT hres = CoInitializeEx(NULL, COINIT_MULTITHREADED);
    if (FAILED(hres)) {
        return { RGY_ERR_UNKNOWN, nullptr };
    }

    hres = CoInitializeSecurity(
        NULL,
        -1,                          // COM authentication
        NULL,                        // Authentication services
        NULL,                        // Reserved
        RPC_C_AUTHN_LEVEL_DEFAULT,   // Default authentication 
        RPC_C_IMP_LEVEL_IMPERSONATE, // Default Impersonation  
        NULL,                        // Authentication info
        EOAC_NONE,                   // Additional capabilities 
        NULL                         // Reserved
    );
    // 2度目の呼び出しでは、RPC_E_TOO_LATEが生じる場合がある
    if (hres != RPC_E_TOO_LATE && FAILED(hres)) {
        CoUninitialize();
        return { RGY_ERR_UNKNOWN, nullptr };
    }

    IWbemLocator *pLoc = NULL;

    hres = CoCreateInstance(
        CLSID_WbemLocator,
        0,
        CLSCTX_INPROC_SERVER,
        IID_IWbemLocator, (LPVOID *)&pLoc);
    if (FAILED(hres)) {
        CoUninitialize();
        return { RGY_ERR_UNKNOWN, nullptr };
    }

    IWbemServices *pSvc = NULL;
    hres = pLoc->ConnectServer(
        _bstr_t(L"ROOT\\CIMV2"), // Object path of WMI namespace
        NULL,                    // User name. NULL = current user
        NULL,                    // User password. NULL = current
        0,                       // Locale. NULL indicates current
        NULL,                    // Security flags.
        0,                       // Authority (for example, Kerberos)
        0,                       // Context object 
        &pSvc                    // pointer to IWbemServices proxy
    );
    if (FAILED(hres)) {
        CoUninitialize();
        return { RGY_ERR_UNKNOWN, nullptr };
    }

    hres = CoSetProxyBlanket(
        pSvc,                        // Indicates the proxy to set
        RPC_C_AUTHN_WINNT,           // RPC_C_AUTHN_xxx
        RPC_C_AUTHZ_NONE,            // RPC_C_AUTHZ_xxx
        NULL,                        // Server principal name 
        RPC_C_AUTHN_LEVEL_CALL,      // RPC_C_AUTHN_LEVEL_xxx 
        RPC_C_IMP_LEVEL_IMPERSONATE, // RPC_C_IMP_LEVEL_xxx
        NULL,                        // client identity
        EOAC_NONE                    // proxy capabilities 
    );
    if (FAILED(hres)) {
        CoUninitialize();
        return { RGY_ERR_UNKNOWN, nullptr };
    }

    IEnumWbemClassObject* pEnumerator = NULL;
    hres = pSvc->ExecQuery(
        bstr_t("WQL"),
        bstr_t("SELECT * FROM Win32_VideoController"),
        WBEM_FLAG_FORWARD_ONLY | WBEM_FLAG_RETURN_IMMEDIATELY,
        NULL,
        &pEnumerator);
    if (FAILED(hres)) {
        pSvc->Release();
        pLoc->Release();
        CoUninitialize();
        return { RGY_ERR_UNKNOWN, nullptr };
    }

    auto info = std::make_unique<RGYDeviceInfoWMI>();
    IWbemClassObject *pclsObj = NULL;
    while (pEnumerator)  {
        ULONG uReturn = 0;
        HRESULT hr = pEnumerator->Next(WBEM_INFINITE, 1,
            &pclsObj, &uReturn);
        if (uReturn == 0) {
            break;
        }

        VARIANT vtProp;
        VariantInit(&vtProp);
        hr = pclsObj->Get(L"PNPDeviceID", 0, &vtProp, 0, 0);
        if (SUCCEEDED(hr)) {
            info->PNPDeviceID = std::wstring(vtProp.bstrVal);
            auto venpos = info->PNPDeviceID.find(L"VEN");
            if (venpos != std::wstring::npos) {
                std::wstring target = info->PNPDeviceID.substr(venpos);
                std::wsmatch results;
                if (std::regex_match(target, results, std::wregex(LR"(VEN_([0-9a-fA-F]+)&DEV_([0-9a-fA-F]+)&.+)"))) {
                    try {
                        info->vendorID = std::stoi(results[1].str(), nullptr, 16);
                        info->deviceID = std::stoi(results[2].str(), nullptr, 16);
                    } catch (...) {
                        info->vendorID = 0;
                        info->deviceID = 0;
                    }
                }
            }
        }
        VariantClear(&vtProp);

        if (info->vendorID == vendorID && info->deviceID == deviceID) {
            VariantInit(&vtProp);
            hr = pclsObj->Get(L"Name", 0, &vtProp, 0, 0);
            if (SUCCEEDED(hr)) {
                info->name = std::wstring(vtProp.bstrVal);
            }
            VariantClear(&vtProp);


            VariantInit(&vtProp);
            hr = pclsObj->Get(L"DriverVersion", 0, &vtProp, 0, 0);
            if (SUCCEEDED(hr)) {
                info->DriverVersion = std::wstring(vtProp.bstrVal);
            }
            VariantClear(&vtProp);

            VariantInit(&vtProp);
            hr = pclsObj->Get(L"DriverDate", 0, &vtProp, 0, 0);
            if (SUCCEEDED(hr)) {
                info->DriverDate = std::wstring(vtProp.bstrVal);
            }
            VariantClear(&vtProp);
        }

        pclsObj->Release();

        if (info->vendorID == vendorID && info->deviceID == deviceID) {
            break;
        }
    }

    pSvc->Release();
    pLoc->Release();
    pEnumerator->Release();
    CoUninitialize();

    if (info->vendorID != vendorID || info->deviceID != deviceID) {
        return { RGY_ERR_NOT_FOUND, nullptr };
    }
    return { RGY_ERR_NONE, std::move(info) };
}


#endif //#if defined(_WIN32) || defined(_WIN64)
