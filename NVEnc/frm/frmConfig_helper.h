//  -----------------------------------------------------------------------------------------
//    NVEnc by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  -----------------------------------------------------------------------------------------

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

const WCHAR * const audio_delay_cut_desc[] = {
	L"補正なし",
	L"音声カット",
	NULL
};

//メモ表示用 RGB
const int StgNotesColor[][3] = {
	{  80,  72,  92 },
	{ 120, 120, 120 }
};

const WCHAR * const DefaultStgNotes = L"メモ...";