// -----------------------------------------------------------------------------------------
// NVEnc by rigaya
// -----------------------------------------------------------------------------------------
// The MIT License
//
// Copyright (c) 1999-2016 rigaya
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

using namespace System;
using namespace System::Data;
using namespace System::Threading;
using namespace System::IO;
using namespace System::Collections::Generic;

#define HIDE_MPEG2

namespace NVEnc {

    ref class LocalSettings 
    {
    public:
        List<String^>^ audEncName;
        List<String^>^ audEncExeName;
        List<String^>^ audEncPath;
        String^ MP4MuxerExeName;
        String^ MP4MuxerPath;
        String^ MKVMuxerExeName;
        String^ MKVMuxerPath;
        String^ TC2MP4ExeName;
        String^ TC2MP4Path;
        String^ MPGMuxerExeName;
        String^ MPGMuxerPath;
        String^ MP4RawExeName;
        String^ MP4RawPath;
        String^ CustomTmpDir;
        String^ CustomAudTmpDir;
        String^ CustomMP4TmpDir;
        String^ LastAppDir;
        String^ LastBatDir;

        LocalSettings() {
            audEncName = gcnew List<String^>();
            audEncExeName = gcnew List<String^>();
            audEncPath = gcnew List<String^>();
        }
        ~LocalSettings() {
            delete audEncName;
            delete audEncExeName;
            delete audEncPath;
        }
    };

    value struct ExeControls
    {
        String^ Name;
        String^ Path;
        const char* args;
    };
    /*
    ref class NVEncParamCache
    {
    public:
        DataTable^ dataTableNVEncFeatures;
        std::vector<NV_ENC_CONFIG> presetConfigs;
        std::vector<NVEncCap> nvencCapabilities;
        Thread^ thPresetConfig;
        Thread^ thFeatures;
        bool presetConfigCacheAvailable;
        bool featuresCacheAvaialble;

        NVEncParamCache() {
            dataTableNVEncFeatures = gcnew DataTable();
            dataTableNVEncFeatures->Columns->Add(L"機能");
            dataTableNVEncFeatures->Columns->Add(L"サポート");

            presetConfigCacheAvailable = false;
            featuresCacheAvaialble = false;

            thPresetConfig = gcnew Thread(gcnew ThreadStart(this, &NVEncParamCache::createPresetConfigCache));
            thFeatures = gcnew Thread(gcnew ThreadStart(this, &NVEncParamCache::createFeaturesCache));
            thPresetConfig->Start();
            thFeatures->Start();
        }
        ~NVEncParamCache() {
            delete dataTableNVEncFeatures;
            delete thFeatures;
            delete thPresetConfig;
        }

        void createPresetConfigCache() {
            NVEncParam param;
            presetConfig = param.GetNVEncH264Preset(0);
        }

        void createFeaturesCache() {
            NVEncParam param;
            auto nvencCapabilities = param.GetNVEncCapability(0);
            for (auto cap : nvencCapabilities) {
                DataRow^ drb = dataTableQsvFeatures->NewRow();
                drb[0] = String(cap.name).ToString();
                drb[1] = String(cap.value).ToString();
                dataTableQsvFeatures->Rows->Add(drb);
            }
        }

        std::vector<NV_ENC_CONFIG> getPresetConfigCache() {
            if (!presetConfigCacheAvailable) {
                thPresetConfig->Join();
                presetConfigCacheAvailable = true;
            }
            return presetConfig;
        }

        std::vector<NV_ENC_CONFIG> getFeaturesCache() {
            if (!featuresCacheAvaialble) {
                thFeatures->Join();
                featuresCacheAvaialble = true;
            }
            return nvencCapabilities;
        }

        DataTable^ getFeatureTable() {
            return dataTableNVEncFeatures;
        }
    };*/
};

const int fcgTBQualityTimerLatency = 600;
const int fcgTBQualityTimerPeriod = 40;
const int fcgTXCmdfulloffset = 57;
const int fcgCXAudioEncModeSmallWidth = 189;
const int fcgCXAudioEncModeLargeWidth = 237;


static const WCHAR * const list_aspect_ratio[] = {
    L"SAR(PAR, 画素比)で指定",
    L"DAR(画面比)で指定",
    NULL
};

static const WCHAR * const list_tempdir[] = {
    L"出力先と同じフォルダ (デフォルト)",
    L"システムの一時フォルダ",
    L"カスタム",
    NULL
};

static const WCHAR * const list_audtempdir[] = {
    L"変更しない",
    L"カスタム",
    NULL
};

static const WCHAR * const list_mp4boxtempdir[] = {
    L"指定しない",
    L"カスタム",
    NULL
};

const WCHAR * const audio_enc_timing_desc[] = {
    L"後",
    L"前",
    L"同時",
    NULL
};

//メモ表示用 RGB
const int StgNotesColor[][3] = {
    {  80,  72,  92 },
    { 120, 120, 120 }
};

const WCHAR * const DefaultStgNotes = L"メモ...";