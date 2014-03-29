//  -----------------------------------------------------------------------------------------
//    NVEnc by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  -----------------------------------------------------------------------------------------

#pragma once

#include <Windows.h>

#include "auo.h"
#include "auo_pipe.h"
#include "auo_conf.h"
#include "auo_settings.h"
#include "auo_system.h"
#include "auo_util.h"
#include "auo_clrutil.h"

#include "frmConfig_helper.h"

#include "NVEncParam.h"

using namespace System;
using namespace System::ComponentModel;
using namespace System::Collections;
using namespace System::Windows::Forms;
using namespace System::Data;
using namespace System::Drawing;
using namespace System::IO;


namespace NVEnc {

	/// <summary>
	/// frmConfig の概要
	///
	/// 警告: このクラスの名前を変更する場合、このクラスが依存するすべての .resx ファイルに関連付けられた
	///          マネージ リソース コンパイラ ツールに対して 'Resource File Name' プロパティを
	///          変更する必要があります。この変更を行わないと、
	///          デザイナと、このフォームに関連付けられたローカライズ済みリソースとが、
	///          正しく相互に利用できなくなります。
	/// </summary>
	public ref class frmConfig : public System::Windows::Forms::Form
	{
	public:
		frmConfig(CONF_GUIEX *_conf, const SYSTEM_DATA *_sys_dat)
		{
			paramCache = new NVEncParam();
			paramCache->createCacheAsync(0);
			InitData(_conf, _sys_dat);
			cnf_stgSelected = (CONF_GUIEX *)calloc(1, sizeof(CONF_GUIEX));
			InitializeComponent();
			//
			//TODO: ここにコンストラクタ コードを追加します
			//
		}

	protected:
		/// <summary>
		/// 使用中のリソースをすべてクリーンアップします。
		/// </summary>
		~frmConfig()
		{
			if (components)
			{
				delete components;
			}
			CloseBitrateCalc();
			if (cnf_stgSelected) free(cnf_stgSelected); cnf_stgSelected = NULL;
		}




	private: System::Windows::Forms::ToolStrip^  fcgtoolStripSettings;
	private: System::Windows::Forms::GroupBox^  fcggroupBoxAudio;
	private: System::Windows::Forms::TabControl^  fcgtabControlMux;
	private: System::Windows::Forms::TabPage^  fcgtabPageMP4;
	private: System::Windows::Forms::TabPage^  fcgtabPageMKV;
	private: System::Windows::Forms::TextBox^  fcgTXCmd;

	private: System::Windows::Forms::Button^  fcgBTCancel;

	private: System::Windows::Forms::Button^  fcgBTOK;
	private: System::Windows::Forms::Button^  fcgBTDefault;




	private: System::Windows::Forms::ToolStripButton^  fcgTSBSave;

	private: System::Windows::Forms::ToolStripButton^  fcgTSBSaveNew;

	private: System::Windows::Forms::ToolStripButton^  fcgTSBDelete;

	private: System::Windows::Forms::ToolStripSeparator^  fcgtoolStripSeparator1;
	private: System::Windows::Forms::ToolStripDropDownButton^  fcgTSSettings;


















































































































































































































































	private: System::Windows::Forms::ComboBox^  fcgCXAudioEncoder;
	private: System::Windows::Forms::Label^  fcgLBAudioEncoder;







	private: System::Windows::Forms::ComboBox^  fcgCXAudioPriority;
	private: System::Windows::Forms::Label^  fcgLBAudioPriority;
	private: System::Windows::Forms::TabPage^  fcgtabPageMux;

	private: System::Windows::Forms::Label^  fcgLBAudioEncoderPath;


	private: System::Windows::Forms::CheckBox^  fcgCBAudioOnly;
	private: System::Windows::Forms::CheckBox^  fcgCBFAWCheck;
	private: System::Windows::Forms::CheckBox^  fcgCBAudio2pass;

	private: System::Windows::Forms::ComboBox^  fcgCXAudioEncMode;
	private: System::Windows::Forms::Label^  fcgLBAudioEncMode;
	private: System::Windows::Forms::Button^  fcgBTAudioEncoderPath;
	private: System::Windows::Forms::TextBox^  fcgTXAudioEncoderPath;
	private: System::Windows::Forms::ComboBox^  fcgCXMP4CmdEx;

	private: System::Windows::Forms::Label^  fcgLBMP4CmdEx;
	private: System::Windows::Forms::CheckBox^  fcgCBMP4MuxerExt;
	private: System::Windows::Forms::Button^  fcgBTMP4BoxTempDir;
	private: System::Windows::Forms::TextBox^  fcgTXMP4BoxTempDir;


	private: System::Windows::Forms::ComboBox^  fcgCXMP4BoxTempDir;
	private: System::Windows::Forms::Label^  fcgLBMP4BoxTempDir;
	private: System::Windows::Forms::Button^  fcgBTTC2MP4Path;
	private: System::Windows::Forms::TextBox^  fcgTXTC2MP4Path;
	private: System::Windows::Forms::Button^  fcgBTMP4MuxerPath;
	private: System::Windows::Forms::TextBox^  fcgTXMP4MuxerPath;

	private: System::Windows::Forms::Label^  fcgLBTC2MP4Path;
	private: System::Windows::Forms::Label^  fcgLBMP4MuxerPath;


	private: System::Windows::Forms::Button^  fcgBTMKVMuxerPath;

	private: System::Windows::Forms::TextBox^  fcgTXMKVMuxerPath;

	private: System::Windows::Forms::Label^  fcgLBMKVMuxerPath;
	private: System::Windows::Forms::ComboBox^  fcgCXMKVCmdEx;
	private: System::Windows::Forms::Label^  fcgLBMKVMuxerCmdEx;
	private: System::Windows::Forms::CheckBox^  fcgCBMKVMuxerExt;
	private: System::Windows::Forms::ComboBox^  fcgCXMuxPriority;
	private: System::Windows::Forms::Label^  fcgLBMuxPriority;
	private: System::Windows::Forms::CheckBox^  fcgCBMuxMinimize;
	private: System::Windows::Forms::Label^  fcgLBVersionDate;


	private: System::Windows::Forms::Label^  fcgLBVersion;
	private: System::Windows::Forms::FolderBrowserDialog^  fcgfolderBrowserTemp;
	private: System::Windows::Forms::OpenFileDialog^  fcgOpenFileDialog;







	private: System::Windows::Forms::NumericUpDown^  fcgNUAudioBitrate;
	private: System::Windows::Forms::Label^  fcgLBAudioBitrate;

private: System::Windows::Forms::ToolTip^  fcgTTEx;






private: System::Windows::Forms::ToolStripSeparator^  toolStripSeparator2;
private: System::Windows::Forms::ToolStripButton^  fcgTSBOtherSettings;































































private: System::Windows::Forms::CheckBox^  fcgCBAudioUsePipe;





































private: System::Windows::Forms::Label^  fcgLBAudioTemp;
private: System::Windows::Forms::ComboBox^  fcgCXAudioTempDir;
private: System::Windows::Forms::TextBox^  fcgTXCustomAudioTempDir;
private: System::Windows::Forms::Button^  fcgBTCustomAudioTempDir;






private: System::Windows::Forms::ToolStripButton^  fcgTSBBitrateCalc;
private: System::Windows::Forms::TabControl^  fcgtabControlNVEnc;

private: System::Windows::Forms::TabPage^  tabPageVideoEnc;












private: System::Windows::Forms::GroupBox^  fcgGroupBoxAspectRatio;
private: System::Windows::Forms::Label^  fcgLBAspectRatio;
private: System::Windows::Forms::NumericUpDown^  fcgNUAspectRatioY;
private: System::Windows::Forms::NumericUpDown^  fcgNUAspectRatioX;
private: System::Windows::Forms::ComboBox^  fcgCXAspectRatio;
private: System::Windows::Forms::ComboBox^  fcgCXInterlaced;
private: System::Windows::Forms::Label^  fcgLBInterlaced;
private: System::Windows::Forms::NumericUpDown^  fcgNUGopLength;
private: System::Windows::Forms::Label^  fcgLBGOPLength;
private: System::Windows::Forms::ComboBox^  fcgCXCodecLevel;
private: System::Windows::Forms::ComboBox^  fcgCXCodecProfile;
private: System::Windows::Forms::Label^  fcgLBCodecLevel;
private: System::Windows::Forms::Label^  fcgLBCodecProfile;




private: System::Windows::Forms::Label^  fcgLBMaxBitrate2;
private: System::Windows::Forms::Label^  fcgLBEncMode;
private: System::Windows::Forms::ComboBox^  fcgCXEncMode;
private: System::Windows::Forms::Label^  fcgLBMaxkbps;
private: System::Windows::Forms::NumericUpDown^  fcgNUMaxkbps;
private: System::Windows::Forms::Label^  fcgLBQPB;
private: System::Windows::Forms::Label^  fcgLBQPP;
private: System::Windows::Forms::Label^  fcgLBQPI;

private: System::Windows::Forms::NumericUpDown^  fcgNUQPB;
private: System::Windows::Forms::NumericUpDown^  fcgNUQPP;
private: System::Windows::Forms::NumericUpDown^  fcgNUQPI;



private: System::Windows::Forms::Label^  fcgLBBitrate2;
private: System::Windows::Forms::NumericUpDown^  fcgNUBitrate;
private: System::Windows::Forms::Label^  fcgLBBitrate;

























private: System::Windows::Forms::TabPage^  tabPageExOpt;
private: System::Windows::Forms::Label^  fcgLBTempDir;

private: System::Windows::Forms::Button^  fcgBTCustomTempDir;
private: System::Windows::Forms::TextBox^  fcgTXCustomTempDir;

private: System::Windows::Forms::ComboBox^  fcgCXTempDir;


private: System::Windows::Forms::Label^  fcgLBGOPLengthAuto;






























private: System::Windows::Forms::NumericUpDown^  fcgNUSlices;

private: System::Windows::Forms::Label^  fcgLBSlices;

private: System::Windows::Forms::ToolStripLabel^  fcgTSLSettingsNotes;
private: System::Windows::Forms::ToolStripTextBox^  fcgTSTSettingsNotes;
private: System::Windows::Forms::TabPage^  fcgtabPageBat;
private: System::Windows::Forms::Button^  fcgBTBatAfterPath;

private: System::Windows::Forms::TextBox^  fcgTXBatAfterPath;

private: System::Windows::Forms::Label^  fcgLBBatAfterPath;

private: System::Windows::Forms::CheckBox^  fcgCBWaitForBatAfter;

private: System::Windows::Forms::CheckBox^  fcgCBRunBatAfter;

private: System::Windows::Forms::CheckBox^  fcgCBMP4MuxApple;











private: System::Windows::Forms::Panel^  fcgPNBitrate;
private: System::Windows::Forms::Panel^  fcgPNQP;



private: System::Windows::Forms::CheckBox^  fcgCBCABAC;









private: System::Windows::Forms::Button^  fcgBTMP4RawPath;

private: System::Windows::Forms::TextBox^  fcgTXMP4RawPath;
private: System::Windows::Forms::Label^  fcgLBMP4RawPath;





private: System::Windows::Forms::TabPage^  fcgtabPageMPG;
private: System::Windows::Forms::Button^  fcgBTMPGMuxerPath;

private: System::Windows::Forms::TextBox^  fcgTXMPGMuxerPath;

private: System::Windows::Forms::Label^  fcgLBMPGMuxerPath;

private: System::Windows::Forms::ComboBox^  fcgCXMPGCmdEx;
private: System::Windows::Forms::Label^  fcgLBMPGMuxerCmdEx;


private: System::Windows::Forms::CheckBox^  fcgCBMPGMuxerExt;

private: System::Windows::Forms::ContextMenuStrip^  fcgCSExeFiles;
private: System::Windows::Forms::ToolStripMenuItem^  fcgTSExeFileshelp;
private: System::Windows::Forms::Label^  fcgLBAudioEncTiming;
private: System::Windows::Forms::ComboBox^  fcgCXAudioEncTiming;


private: System::Windows::Forms::Label^  fcgLBBatAfterString;

private: System::Windows::Forms::Label^  fcgLBBatBeforeString;
private: System::Windows::Forms::Panel^  fcgPNSeparator;
private: System::Windows::Forms::Button^  fcgBTBatBeforePath;
private: System::Windows::Forms::TextBox^  fcgTXBatBeforePath;
private: System::Windows::Forms::Label^  fcgLBBatBeforePath;
private: System::Windows::Forms::CheckBox^  fcgCBWaitForBatBefore;
private: System::Windows::Forms::CheckBox^  fcgCBRunBatBefore;
private: System::Windows::Forms::LinkLabel^  fcgLBguiExBlog;



























private: System::Windows::Forms::Label^  fcgLBCABAC;


private: System::Windows::Forms::CheckBox^  fcgCBAuoTcfileout;
private: System::Windows::Forms::CheckBox^  fcgCBAFS;
private: System::Windows::Forms::CheckBox^  fcgCBDeblock;
private: System::Windows::Forms::ComboBox^  fcgCXAudioDelayCut;
private: System::Windows::Forms::Label^  fcgLBAudioDelayCut;
private: System::Windows::Forms::GroupBox^  fcgGroupBoxQulaityStg;
private: System::Windows::Forms::Button^  fcgBTQualityStg;
private: System::Windows::Forms::ComboBox^  fcgCXQualityPreset;

private: System::Windows::Forms::NumericUpDown^  fcgNUBframes;
private: System::Windows::Forms::Label^  fcgLBBframes;
private: System::Windows::Forms::ComboBox^  fcgCXAdaptiveTransform;
private: System::Windows::Forms::Label^  fcgLBAdaptiveTransform;
private: System::Windows::Forms::Label^  fcgLBFullrange;
private: System::Windows::Forms::CheckBox^  fcgCBFullrange;
private: System::Windows::Forms::ComboBox^  fcgCXVideoFormat;
private: System::Windows::Forms::Label^  fcgLBVideoFormat;
private: System::Windows::Forms::GroupBox^  fcggroupBoxColor;
private: System::Windows::Forms::ComboBox^  fcgCXTransfer;
private: System::Windows::Forms::ComboBox^  fcgCXColorPrim;
private: System::Windows::Forms::ComboBox^  fcgCXColorMatrix;
private: System::Windows::Forms::Label^  fcgLBTransfer;
private: System::Windows::Forms::Label^  fcgLBColorPrim;
private: System::Windows::Forms::Label^  fcgLBColorMatrix;
private: System::Windows::Forms::NumericUpDown^  fcgNURefFrames;
private: System::Windows::Forms::Label^  fcgLBRefFrames;
private: System::Windows::Forms::ComboBox^  fcgCXBDirectMode;
private: System::Windows::Forms::Label^  fcgLBBDirectMode;
private: System::Windows::Forms::Label^  fcgLBMVPRecision;
private: System::Windows::Forms::ComboBox^  fcgCXMVPrecision;
private: System::Windows::Forms::TabPage^  tabPageNVEncFeatures;
private: System::Windows::Forms::DataGridView^  fcgDGVFeatures;
private: System::Windows::Forms::Label^  fcgLBOSInfo;
private: System::Windows::Forms::Label^  fcgLBOSInfoLabel;
private: System::Windows::Forms::Label^  fcgLBCPUInfoOnFeatureTab;
private: System::Windows::Forms::Label^  fcgLBCPUInfoLabelOnFeatureTab;
private: System::Windows::Forms::Label^  label2;
private: System::Windows::Forms::Label^  fcgLBGPUInfoOnFeatureTab;
private: System::Windows::Forms::Label^  fcgLBGPUInfoLabelOnFeatureTab;
private: System::Windows::Forms::PictureBox^  fcgPBNVEncLogoDisabled;

private: System::Windows::Forms::PictureBox^  fcgPBNVEncLogoEnabled;



































































































	private: System::ComponentModel::IContainer^  components;


	

	private:
		/// <summary>
		/// 必要なデザイナ変数です。
		/// </summary>


#pragma region Windows Form Designer generated code
		/// <summary>
		/// デザイナ サポートに必要なメソッドです。このメソッドの内容を
		/// コード エディタで変更しないでください。
		/// </summary>
		void InitializeComponent(void)
		{
			this->components = (gcnew System::ComponentModel::Container());
			System::ComponentModel::ComponentResourceManager^  resources = (gcnew System::ComponentModel::ComponentResourceManager(frmConfig::typeid));
			this->fcgtoolStripSettings = (gcnew System::Windows::Forms::ToolStrip());
			this->fcgTSBSave = (gcnew System::Windows::Forms::ToolStripButton());
			this->fcgTSBSaveNew = (gcnew System::Windows::Forms::ToolStripButton());
			this->fcgTSBDelete = (gcnew System::Windows::Forms::ToolStripButton());
			this->fcgtoolStripSeparator1 = (gcnew System::Windows::Forms::ToolStripSeparator());
			this->fcgTSSettings = (gcnew System::Windows::Forms::ToolStripDropDownButton());
			this->fcgTSBBitrateCalc = (gcnew System::Windows::Forms::ToolStripButton());
			this->toolStripSeparator2 = (gcnew System::Windows::Forms::ToolStripSeparator());
			this->fcgTSBOtherSettings = (gcnew System::Windows::Forms::ToolStripButton());
			this->fcgTSLSettingsNotes = (gcnew System::Windows::Forms::ToolStripLabel());
			this->fcgTSTSettingsNotes = (gcnew System::Windows::Forms::ToolStripTextBox());
			this->fcggroupBoxAudio = (gcnew System::Windows::Forms::GroupBox());
			this->fcgCXAudioDelayCut = (gcnew System::Windows::Forms::ComboBox());
			this->fcgLBAudioDelayCut = (gcnew System::Windows::Forms::Label());
			this->fcgLBAudioEncTiming = (gcnew System::Windows::Forms::Label());
			this->fcgCXAudioEncTiming = (gcnew System::Windows::Forms::ComboBox());
			this->fcgLBAudioTemp = (gcnew System::Windows::Forms::Label());
			this->fcgCXAudioTempDir = (gcnew System::Windows::Forms::ComboBox());
			this->fcgTXCustomAudioTempDir = (gcnew System::Windows::Forms::TextBox());
			this->fcgBTCustomAudioTempDir = (gcnew System::Windows::Forms::Button());
			this->fcgCBAudioUsePipe = (gcnew System::Windows::Forms::CheckBox());
			this->fcgLBAudioBitrate = (gcnew System::Windows::Forms::Label());
			this->fcgNUAudioBitrate = (gcnew System::Windows::Forms::NumericUpDown());
			this->fcgCBAudio2pass = (gcnew System::Windows::Forms::CheckBox());
			this->fcgCXAudioEncMode = (gcnew System::Windows::Forms::ComboBox());
			this->fcgLBAudioEncMode = (gcnew System::Windows::Forms::Label());
			this->fcgBTAudioEncoderPath = (gcnew System::Windows::Forms::Button());
			this->fcgTXAudioEncoderPath = (gcnew System::Windows::Forms::TextBox());
			this->fcgLBAudioEncoderPath = (gcnew System::Windows::Forms::Label());
			this->fcgCBAudioOnly = (gcnew System::Windows::Forms::CheckBox());
			this->fcgCBFAWCheck = (gcnew System::Windows::Forms::CheckBox());
			this->fcgCXAudioPriority = (gcnew System::Windows::Forms::ComboBox());
			this->fcgLBAudioPriority = (gcnew System::Windows::Forms::Label());
			this->fcgCXAudioEncoder = (gcnew System::Windows::Forms::ComboBox());
			this->fcgLBAudioEncoder = (gcnew System::Windows::Forms::Label());
			this->fcgtabControlMux = (gcnew System::Windows::Forms::TabControl());
			this->fcgtabPageMP4 = (gcnew System::Windows::Forms::TabPage());
			this->fcgBTMP4RawPath = (gcnew System::Windows::Forms::Button());
			this->fcgTXMP4RawPath = (gcnew System::Windows::Forms::TextBox());
			this->fcgLBMP4RawPath = (gcnew System::Windows::Forms::Label());
			this->fcgCBMP4MuxApple = (gcnew System::Windows::Forms::CheckBox());
			this->fcgBTMP4BoxTempDir = (gcnew System::Windows::Forms::Button());
			this->fcgTXMP4BoxTempDir = (gcnew System::Windows::Forms::TextBox());
			this->fcgCXMP4BoxTempDir = (gcnew System::Windows::Forms::ComboBox());
			this->fcgLBMP4BoxTempDir = (gcnew System::Windows::Forms::Label());
			this->fcgBTTC2MP4Path = (gcnew System::Windows::Forms::Button());
			this->fcgTXTC2MP4Path = (gcnew System::Windows::Forms::TextBox());
			this->fcgBTMP4MuxerPath = (gcnew System::Windows::Forms::Button());
			this->fcgTXMP4MuxerPath = (gcnew System::Windows::Forms::TextBox());
			this->fcgLBTC2MP4Path = (gcnew System::Windows::Forms::Label());
			this->fcgLBMP4MuxerPath = (gcnew System::Windows::Forms::Label());
			this->fcgCXMP4CmdEx = (gcnew System::Windows::Forms::ComboBox());
			this->fcgLBMP4CmdEx = (gcnew System::Windows::Forms::Label());
			this->fcgCBMP4MuxerExt = (gcnew System::Windows::Forms::CheckBox());
			this->fcgtabPageMKV = (gcnew System::Windows::Forms::TabPage());
			this->fcgBTMKVMuxerPath = (gcnew System::Windows::Forms::Button());
			this->fcgTXMKVMuxerPath = (gcnew System::Windows::Forms::TextBox());
			this->fcgLBMKVMuxerPath = (gcnew System::Windows::Forms::Label());
			this->fcgCXMKVCmdEx = (gcnew System::Windows::Forms::ComboBox());
			this->fcgLBMKVMuxerCmdEx = (gcnew System::Windows::Forms::Label());
			this->fcgCBMKVMuxerExt = (gcnew System::Windows::Forms::CheckBox());
			this->fcgtabPageMPG = (gcnew System::Windows::Forms::TabPage());
			this->fcgBTMPGMuxerPath = (gcnew System::Windows::Forms::Button());
			this->fcgTXMPGMuxerPath = (gcnew System::Windows::Forms::TextBox());
			this->fcgLBMPGMuxerPath = (gcnew System::Windows::Forms::Label());
			this->fcgCXMPGCmdEx = (gcnew System::Windows::Forms::ComboBox());
			this->fcgLBMPGMuxerCmdEx = (gcnew System::Windows::Forms::Label());
			this->fcgCBMPGMuxerExt = (gcnew System::Windows::Forms::CheckBox());
			this->fcgtabPageMux = (gcnew System::Windows::Forms::TabPage());
			this->fcgCXMuxPriority = (gcnew System::Windows::Forms::ComboBox());
			this->fcgLBMuxPriority = (gcnew System::Windows::Forms::Label());
			this->fcgCBMuxMinimize = (gcnew System::Windows::Forms::CheckBox());
			this->fcgtabPageBat = (gcnew System::Windows::Forms::TabPage());
			this->fcgLBBatAfterString = (gcnew System::Windows::Forms::Label());
			this->fcgLBBatBeforeString = (gcnew System::Windows::Forms::Label());
			this->fcgPNSeparator = (gcnew System::Windows::Forms::Panel());
			this->fcgBTBatBeforePath = (gcnew System::Windows::Forms::Button());
			this->fcgTXBatBeforePath = (gcnew System::Windows::Forms::TextBox());
			this->fcgLBBatBeforePath = (gcnew System::Windows::Forms::Label());
			this->fcgCBWaitForBatBefore = (gcnew System::Windows::Forms::CheckBox());
			this->fcgCBRunBatBefore = (gcnew System::Windows::Forms::CheckBox());
			this->fcgBTBatAfterPath = (gcnew System::Windows::Forms::Button());
			this->fcgTXBatAfterPath = (gcnew System::Windows::Forms::TextBox());
			this->fcgLBBatAfterPath = (gcnew System::Windows::Forms::Label());
			this->fcgCBWaitForBatAfter = (gcnew System::Windows::Forms::CheckBox());
			this->fcgCBRunBatAfter = (gcnew System::Windows::Forms::CheckBox());
			this->fcgTXCmd = (gcnew System::Windows::Forms::TextBox());
			this->fcgBTCancel = (gcnew System::Windows::Forms::Button());
			this->fcgBTOK = (gcnew System::Windows::Forms::Button());
			this->fcgBTDefault = (gcnew System::Windows::Forms::Button());
			this->fcgLBVersionDate = (gcnew System::Windows::Forms::Label());
			this->fcgLBVersion = (gcnew System::Windows::Forms::Label());
			this->fcgfolderBrowserTemp = (gcnew System::Windows::Forms::FolderBrowserDialog());
			this->fcgOpenFileDialog = (gcnew System::Windows::Forms::OpenFileDialog());
			this->fcgTTEx = (gcnew System::Windows::Forms::ToolTip(this->components));
			this->fcgtabControlNVEnc = (gcnew System::Windows::Forms::TabControl());
			this->tabPageVideoEnc = (gcnew System::Windows::Forms::TabPage());
			this->fcgCXAdaptiveTransform = (gcnew System::Windows::Forms::ComboBox());
			this->fcgLBAdaptiveTransform = (gcnew System::Windows::Forms::Label());
			this->fcgLBFullrange = (gcnew System::Windows::Forms::Label());
			this->fcgCBFullrange = (gcnew System::Windows::Forms::CheckBox());
			this->fcgCXVideoFormat = (gcnew System::Windows::Forms::ComboBox());
			this->fcgLBVideoFormat = (gcnew System::Windows::Forms::Label());
			this->fcggroupBoxColor = (gcnew System::Windows::Forms::GroupBox());
			this->fcgCXTransfer = (gcnew System::Windows::Forms::ComboBox());
			this->fcgCXColorPrim = (gcnew System::Windows::Forms::ComboBox());
			this->fcgCXColorMatrix = (gcnew System::Windows::Forms::ComboBox());
			this->fcgLBTransfer = (gcnew System::Windows::Forms::Label());
			this->fcgLBColorPrim = (gcnew System::Windows::Forms::Label());
			this->fcgLBColorMatrix = (gcnew System::Windows::Forms::Label());
			this->fcgNURefFrames = (gcnew System::Windows::Forms::NumericUpDown());
			this->fcgLBRefFrames = (gcnew System::Windows::Forms::Label());
			this->fcgCXBDirectMode = (gcnew System::Windows::Forms::ComboBox());
			this->fcgLBBDirectMode = (gcnew System::Windows::Forms::Label());
			this->fcgLBMVPRecision = (gcnew System::Windows::Forms::Label());
			this->fcgCXMVPrecision = (gcnew System::Windows::Forms::ComboBox());
			this->fcgNUBframes = (gcnew System::Windows::Forms::NumericUpDown());
			this->fcgLBBframes = (gcnew System::Windows::Forms::Label());
			this->fcgGroupBoxQulaityStg = (gcnew System::Windows::Forms::GroupBox());
			this->fcgBTQualityStg = (gcnew System::Windows::Forms::Button());
			this->fcgCXQualityPreset = (gcnew System::Windows::Forms::ComboBox());
			this->fcgLBCABAC = (gcnew System::Windows::Forms::Label());
			this->fcgCBDeblock = (gcnew System::Windows::Forms::CheckBox());
			this->fcgPNBitrate = (gcnew System::Windows::Forms::Panel());
			this->fcgLBBitrate = (gcnew System::Windows::Forms::Label());
			this->fcgNUBitrate = (gcnew System::Windows::Forms::NumericUpDown());
			this->fcgLBBitrate2 = (gcnew System::Windows::Forms::Label());
			this->fcgNUMaxkbps = (gcnew System::Windows::Forms::NumericUpDown());
			this->fcgLBMaxkbps = (gcnew System::Windows::Forms::Label());
			this->fcgLBMaxBitrate2 = (gcnew System::Windows::Forms::Label());
			this->fcgPNQP = (gcnew System::Windows::Forms::Panel());
			this->fcgLBQPI = (gcnew System::Windows::Forms::Label());
			this->fcgNUQPI = (gcnew System::Windows::Forms::NumericUpDown());
			this->fcgNUQPP = (gcnew System::Windows::Forms::NumericUpDown());
			this->fcgNUQPB = (gcnew System::Windows::Forms::NumericUpDown());
			this->fcgLBQPP = (gcnew System::Windows::Forms::Label());
			this->fcgLBQPB = (gcnew System::Windows::Forms::Label());
			this->fcgNUSlices = (gcnew System::Windows::Forms::NumericUpDown());
			this->fcgCBCABAC = (gcnew System::Windows::Forms::CheckBox());
			this->fcgLBSlices = (gcnew System::Windows::Forms::Label());
			this->fcgLBGOPLengthAuto = (gcnew System::Windows::Forms::Label());
			this->fcgGroupBoxAspectRatio = (gcnew System::Windows::Forms::GroupBox());
			this->fcgLBAspectRatio = (gcnew System::Windows::Forms::Label());
			this->fcgNUAspectRatioY = (gcnew System::Windows::Forms::NumericUpDown());
			this->fcgNUAspectRatioX = (gcnew System::Windows::Forms::NumericUpDown());
			this->fcgCXAspectRatio = (gcnew System::Windows::Forms::ComboBox());
			this->fcgCXInterlaced = (gcnew System::Windows::Forms::ComboBox());
			this->fcgLBInterlaced = (gcnew System::Windows::Forms::Label());
			this->fcgNUGopLength = (gcnew System::Windows::Forms::NumericUpDown());
			this->fcgLBGOPLength = (gcnew System::Windows::Forms::Label());
			this->fcgCXCodecLevel = (gcnew System::Windows::Forms::ComboBox());
			this->fcgCXCodecProfile = (gcnew System::Windows::Forms::ComboBox());
			this->fcgLBCodecLevel = (gcnew System::Windows::Forms::Label());
			this->fcgLBCodecProfile = (gcnew System::Windows::Forms::Label());
			this->fcgLBEncMode = (gcnew System::Windows::Forms::Label());
			this->fcgCXEncMode = (gcnew System::Windows::Forms::ComboBox());
			this->tabPageExOpt = (gcnew System::Windows::Forms::TabPage());
			this->fcgCBAuoTcfileout = (gcnew System::Windows::Forms::CheckBox());
			this->fcgCBAFS = (gcnew System::Windows::Forms::CheckBox());
			this->fcgLBTempDir = (gcnew System::Windows::Forms::Label());
			this->fcgBTCustomTempDir = (gcnew System::Windows::Forms::Button());
			this->fcgTXCustomTempDir = (gcnew System::Windows::Forms::TextBox());
			this->fcgCXTempDir = (gcnew System::Windows::Forms::ComboBox());
			this->tabPageNVEncFeatures = (gcnew System::Windows::Forms::TabPage());
			this->fcgLBOSInfo = (gcnew System::Windows::Forms::Label());
			this->fcgLBOSInfoLabel = (gcnew System::Windows::Forms::Label());
			this->fcgLBCPUInfoOnFeatureTab = (gcnew System::Windows::Forms::Label());
			this->fcgLBCPUInfoLabelOnFeatureTab = (gcnew System::Windows::Forms::Label());
			this->label2 = (gcnew System::Windows::Forms::Label());
			this->fcgLBGPUInfoOnFeatureTab = (gcnew System::Windows::Forms::Label());
			this->fcgLBGPUInfoLabelOnFeatureTab = (gcnew System::Windows::Forms::Label());
			this->fcgDGVFeatures = (gcnew System::Windows::Forms::DataGridView());
			this->fcgCSExeFiles = (gcnew System::Windows::Forms::ContextMenuStrip(this->components));
			this->fcgTSExeFileshelp = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->fcgLBguiExBlog = (gcnew System::Windows::Forms::LinkLabel());
			this->fcgPBNVEncLogoDisabled = (gcnew System::Windows::Forms::PictureBox());
			this->fcgPBNVEncLogoEnabled = (gcnew System::Windows::Forms::PictureBox());
			this->fcgtoolStripSettings->SuspendLayout();
			this->fcggroupBoxAudio->SuspendLayout();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUAudioBitrate))->BeginInit();
			this->fcgtabControlMux->SuspendLayout();
			this->fcgtabPageMP4->SuspendLayout();
			this->fcgtabPageMKV->SuspendLayout();
			this->fcgtabPageMPG->SuspendLayout();
			this->fcgtabPageMux->SuspendLayout();
			this->fcgtabPageBat->SuspendLayout();
			this->fcgtabControlNVEnc->SuspendLayout();
			this->tabPageVideoEnc->SuspendLayout();
			this->fcggroupBoxColor->SuspendLayout();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNURefFrames))->BeginInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUBframes))->BeginInit();
			this->fcgGroupBoxQulaityStg->SuspendLayout();
			this->fcgPNBitrate->SuspendLayout();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUBitrate))->BeginInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUMaxkbps))->BeginInit();
			this->fcgPNQP->SuspendLayout();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUQPI))->BeginInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUQPP))->BeginInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUQPB))->BeginInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUSlices))->BeginInit();
			this->fcgGroupBoxAspectRatio->SuspendLayout();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUAspectRatioY))->BeginInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUAspectRatioX))->BeginInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUGopLength))->BeginInit();
			this->tabPageExOpt->SuspendLayout();
			this->tabPageNVEncFeatures->SuspendLayout();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgDGVFeatures))->BeginInit();
			this->fcgCSExeFiles->SuspendLayout();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgPBNVEncLogoDisabled))->BeginInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgPBNVEncLogoEnabled))->BeginInit();
			this->SuspendLayout();
			// 
			// fcgtoolStripSettings
			// 
			this->fcgtoolStripSettings->Items->AddRange(gcnew cli::array< System::Windows::Forms::ToolStripItem^  >(10) {
				this->fcgTSBSave,
					this->fcgTSBSaveNew, this->fcgTSBDelete, this->fcgtoolStripSeparator1, this->fcgTSSettings, this->fcgTSBBitrateCalc, this->toolStripSeparator2,
					this->fcgTSBOtherSettings, this->fcgTSLSettingsNotes, this->fcgTSTSettingsNotes
			});
			this->fcgtoolStripSettings->Location = System::Drawing::Point(0, 0);
			this->fcgtoolStripSettings->Name = L"fcgtoolStripSettings";
			this->fcgtoolStripSettings->Size = System::Drawing::Size(1008, 25);
			this->fcgtoolStripSettings->TabIndex = 1;
			this->fcgtoolStripSettings->Text = L"toolStrip1";
			// 
			// fcgTSBSave
			// 
			this->fcgTSBSave->Image = (cli::safe_cast<System::Drawing::Image^>(resources->GetObject(L"fcgTSBSave.Image")));
			this->fcgTSBSave->ImageTransparentColor = System::Drawing::Color::Magenta;
			this->fcgTSBSave->Name = L"fcgTSBSave";
			this->fcgTSBSave->Size = System::Drawing::Size(88, 22);
			this->fcgTSBSave->Text = L"上書き保存";
			this->fcgTSBSave->Click += gcnew System::EventHandler(this, &frmConfig::fcgTSBSave_Click);
			// 
			// fcgTSBSaveNew
			// 
			this->fcgTSBSaveNew->Image = (cli::safe_cast<System::Drawing::Image^>(resources->GetObject(L"fcgTSBSaveNew.Image")));
			this->fcgTSBSaveNew->ImageTransparentColor = System::Drawing::Color::Black;
			this->fcgTSBSaveNew->Name = L"fcgTSBSaveNew";
			this->fcgTSBSaveNew->Size = System::Drawing::Size(76, 22);
			this->fcgTSBSaveNew->Text = L"新規保存";
			this->fcgTSBSaveNew->Click += gcnew System::EventHandler(this, &frmConfig::fcgTSBSaveNew_Click);
			// 
			// fcgTSBDelete
			// 
			this->fcgTSBDelete->Image = (cli::safe_cast<System::Drawing::Image^>(resources->GetObject(L"fcgTSBDelete.Image")));
			this->fcgTSBDelete->ImageTransparentColor = System::Drawing::Color::Magenta;
			this->fcgTSBDelete->Name = L"fcgTSBDelete";
			this->fcgTSBDelete->Size = System::Drawing::Size(52, 22);
			this->fcgTSBDelete->Text = L"削除";
			this->fcgTSBDelete->Click += gcnew System::EventHandler(this, &frmConfig::fcgTSBDelete_Click);
			// 
			// fcgtoolStripSeparator1
			// 
			this->fcgtoolStripSeparator1->Name = L"fcgtoolStripSeparator1";
			this->fcgtoolStripSeparator1->Size = System::Drawing::Size(6, 25);
			// 
			// fcgTSSettings
			// 
			this->fcgTSSettings->Image = (cli::safe_cast<System::Drawing::Image^>(resources->GetObject(L"fcgTSSettings.Image")));
			this->fcgTSSettings->ImageTransparentColor = System::Drawing::Color::Magenta;
			this->fcgTSSettings->Name = L"fcgTSSettings";
			this->fcgTSSettings->Size = System::Drawing::Size(97, 22);
			this->fcgTSSettings->Text = L"プリセット";
			this->fcgTSSettings->DropDownItemClicked += gcnew System::Windows::Forms::ToolStripItemClickedEventHandler(this, &frmConfig::fcgTSSettings_DropDownItemClicked);
			this->fcgTSSettings->Click += gcnew System::EventHandler(this, &frmConfig::fcgTSSettings_Click);
			// 
			// fcgTSBBitrateCalc
			// 
			this->fcgTSBBitrateCalc->Alignment = System::Windows::Forms::ToolStripItemAlignment::Right;
			this->fcgTSBBitrateCalc->CheckOnClick = true;
			this->fcgTSBBitrateCalc->DisplayStyle = System::Windows::Forms::ToolStripItemDisplayStyle::Text;
			this->fcgTSBBitrateCalc->Image = (cli::safe_cast<System::Drawing::Image^>(resources->GetObject(L"fcgTSBBitrateCalc.Image")));
			this->fcgTSBBitrateCalc->ImageTransparentColor = System::Drawing::Color::Magenta;
			this->fcgTSBBitrateCalc->Name = L"fcgTSBBitrateCalc";
			this->fcgTSBBitrateCalc->Size = System::Drawing::Size(120, 22);
			this->fcgTSBBitrateCalc->Text = L"ビットレート計算機";
			this->fcgTSBBitrateCalc->CheckedChanged += gcnew System::EventHandler(this, &frmConfig::fcgTSBBitrateCalc_CheckedChanged);
			// 
			// toolStripSeparator2
			// 
			this->toolStripSeparator2->Alignment = System::Windows::Forms::ToolStripItemAlignment::Right;
			this->toolStripSeparator2->Name = L"toolStripSeparator2";
			this->toolStripSeparator2->Size = System::Drawing::Size(6, 25);
			// 
			// fcgTSBOtherSettings
			// 
			this->fcgTSBOtherSettings->Alignment = System::Windows::Forms::ToolStripItemAlignment::Right;
			this->fcgTSBOtherSettings->DisplayStyle = System::Windows::Forms::ToolStripItemDisplayStyle::Text;
			this->fcgTSBOtherSettings->Image = (cli::safe_cast<System::Drawing::Image^>(resources->GetObject(L"fcgTSBOtherSettings.Image")));
			this->fcgTSBOtherSettings->ImageTransparentColor = System::Drawing::Color::Magenta;
			this->fcgTSBOtherSettings->Name = L"fcgTSBOtherSettings";
			this->fcgTSBOtherSettings->Size = System::Drawing::Size(84, 22);
			this->fcgTSBOtherSettings->Text = L"その他の設定";
			this->fcgTSBOtherSettings->Click += gcnew System::EventHandler(this, &frmConfig::fcgTSBOtherSettings_Click);
			// 
			// fcgTSLSettingsNotes
			// 
			this->fcgTSLSettingsNotes->DoubleClickEnabled = true;
			this->fcgTSLSettingsNotes->Font = (gcnew System::Drawing::Font(L"Meiryo UI", 8.25F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(128)));
			this->fcgTSLSettingsNotes->Margin = System::Windows::Forms::Padding(3, 1, 0, 2);
			this->fcgTSLSettingsNotes->Name = L"fcgTSLSettingsNotes";
			this->fcgTSLSettingsNotes->Overflow = System::Windows::Forms::ToolStripItemOverflow::Never;
			this->fcgTSLSettingsNotes->Size = System::Drawing::Size(45, 22);
			this->fcgTSLSettingsNotes->Text = L"メモ表示";
			this->fcgTSLSettingsNotes->DoubleClick += gcnew System::EventHandler(this, &frmConfig::fcgTSLSettingsNotes_DoubleClick);
			// 
			// fcgTSTSettingsNotes
			// 
			this->fcgTSTSettingsNotes->BackColor = System::Drawing::SystemColors::Window;
			this->fcgTSTSettingsNotes->Font = (gcnew System::Drawing::Font(L"Meiryo UI", 8.25F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(128)));
			this->fcgTSTSettingsNotes->Margin = System::Windows::Forms::Padding(3, 0, 1, 0);
			this->fcgTSTSettingsNotes->Name = L"fcgTSTSettingsNotes";
			this->fcgTSTSettingsNotes->Size = System::Drawing::Size(200, 25);
			this->fcgTSTSettingsNotes->Text = L"メモ...";
			this->fcgTSTSettingsNotes->Visible = false;
			this->fcgTSTSettingsNotes->Leave += gcnew System::EventHandler(this, &frmConfig::fcgTSTSettingsNotes_Leave);
			this->fcgTSTSettingsNotes->KeyDown += gcnew System::Windows::Forms::KeyEventHandler(this, &frmConfig::fcgTSTSettingsNotes_KeyDown);
			this->fcgTSTSettingsNotes->TextChanged += gcnew System::EventHandler(this, &frmConfig::fcgTSTSettingsNotes_TextChanged);
			// 
			// fcggroupBoxAudio
			// 
			this->fcggroupBoxAudio->Controls->Add(this->fcgCXAudioDelayCut);
			this->fcggroupBoxAudio->Controls->Add(this->fcgLBAudioDelayCut);
			this->fcggroupBoxAudio->Controls->Add(this->fcgLBAudioEncTiming);
			this->fcggroupBoxAudio->Controls->Add(this->fcgCXAudioEncTiming);
			this->fcggroupBoxAudio->Controls->Add(this->fcgLBAudioTemp);
			this->fcggroupBoxAudio->Controls->Add(this->fcgCXAudioTempDir);
			this->fcggroupBoxAudio->Controls->Add(this->fcgTXCustomAudioTempDir);
			this->fcggroupBoxAudio->Controls->Add(this->fcgBTCustomAudioTempDir);
			this->fcggroupBoxAudio->Controls->Add(this->fcgCBAudioUsePipe);
			this->fcggroupBoxAudio->Controls->Add(this->fcgLBAudioBitrate);
			this->fcggroupBoxAudio->Controls->Add(this->fcgNUAudioBitrate);
			this->fcggroupBoxAudio->Controls->Add(this->fcgCBAudio2pass);
			this->fcggroupBoxAudio->Controls->Add(this->fcgCXAudioEncMode);
			this->fcggroupBoxAudio->Controls->Add(this->fcgLBAudioEncMode);
			this->fcggroupBoxAudio->Controls->Add(this->fcgBTAudioEncoderPath);
			this->fcggroupBoxAudio->Controls->Add(this->fcgTXAudioEncoderPath);
			this->fcggroupBoxAudio->Controls->Add(this->fcgLBAudioEncoderPath);
			this->fcggroupBoxAudio->Controls->Add(this->fcgCBAudioOnly);
			this->fcggroupBoxAudio->Controls->Add(this->fcgCBFAWCheck);
			this->fcggroupBoxAudio->Controls->Add(this->fcgCXAudioPriority);
			this->fcggroupBoxAudio->Controls->Add(this->fcgLBAudioPriority);
			this->fcggroupBoxAudio->Controls->Add(this->fcgCXAudioEncoder);
			this->fcggroupBoxAudio->Controls->Add(this->fcgLBAudioEncoder);
			this->fcggroupBoxAudio->Font = (gcnew System::Drawing::Font(L"Meiryo UI", 8.25F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(128)));
			this->fcggroupBoxAudio->Location = System::Drawing::Point(622, 25);
			this->fcggroupBoxAudio->Name = L"fcggroupBoxAudio";
			this->fcggroupBoxAudio->Size = System::Drawing::Size(379, 308);
			this->fcggroupBoxAudio->TabIndex = 2;
			this->fcggroupBoxAudio->TabStop = false;
			this->fcggroupBoxAudio->Text = L"音声";
			// 
			// fcgCXAudioDelayCut
			// 
			this->fcgCXAudioDelayCut->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
			this->fcgCXAudioDelayCut->FormattingEnabled = true;
			this->fcgCXAudioDelayCut->Location = System::Drawing::Point(296, 141);
			this->fcgCXAudioDelayCut->Name = L"fcgCXAudioDelayCut";
			this->fcgCXAudioDelayCut->Size = System::Drawing::Size(70, 22);
			this->fcgCXAudioDelayCut->TabIndex = 30;
			this->fcgCXAudioDelayCut->Tag = L"chValue";
			// 
			// fcgLBAudioDelayCut
			// 
			this->fcgLBAudioDelayCut->AutoSize = true;
			this->fcgLBAudioDelayCut->Location = System::Drawing::Point(229, 144);
			this->fcgLBAudioDelayCut->Name = L"fcgLBAudioDelayCut";
			this->fcgLBAudioDelayCut->Size = System::Drawing::Size(60, 14);
			this->fcgLBAudioDelayCut->TabIndex = 31;
			this->fcgLBAudioDelayCut->Text = L"ディレイカット";
			// 
			// fcgLBAudioEncTiming
			// 
			this->fcgLBAudioEncTiming->AutoSize = true;
			this->fcgLBAudioEncTiming->Location = System::Drawing::Point(256, 62);
			this->fcgLBAudioEncTiming->Name = L"fcgLBAudioEncTiming";
			this->fcgLBAudioEncTiming->Size = System::Drawing::Size(40, 14);
			this->fcgLBAudioEncTiming->TabIndex = 28;
			this->fcgLBAudioEncTiming->Text = L"処理順";
			// 
			// fcgCXAudioEncTiming
			// 
			this->fcgCXAudioEncTiming->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
			this->fcgCXAudioEncTiming->FormattingEnabled = true;
			this->fcgCXAudioEncTiming->Location = System::Drawing::Point(301, 59);
			this->fcgCXAudioEncTiming->Name = L"fcgCXAudioEncTiming";
			this->fcgCXAudioEncTiming->Size = System::Drawing::Size(68, 22);
			this->fcgCXAudioEncTiming->TabIndex = 27;
			this->fcgCXAudioEncTiming->Tag = L"chValue";
			// 
			// fcgLBAudioTemp
			// 
			this->fcgLBAudioTemp->AutoSize = true;
			this->fcgLBAudioTemp->Location = System::Drawing::Point(18, 252);
			this->fcgLBAudioTemp->Name = L"fcgLBAudioTemp";
			this->fcgLBAudioTemp->Size = System::Drawing::Size(114, 14);
			this->fcgLBAudioTemp->TabIndex = 26;
			this->fcgLBAudioTemp->Text = L"音声一時ファイル出力先";
			// 
			// fcgCXAudioTempDir
			// 
			this->fcgCXAudioTempDir->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
			this->fcgCXAudioTempDir->FormattingEnabled = true;
			this->fcgCXAudioTempDir->Location = System::Drawing::Point(146, 249);
			this->fcgCXAudioTempDir->Name = L"fcgCXAudioTempDir";
			this->fcgCXAudioTempDir->Size = System::Drawing::Size(150, 22);
			this->fcgCXAudioTempDir->TabIndex = 12;
			this->fcgCXAudioTempDir->Tag = L"chValue";
			// 
			// fcgTXCustomAudioTempDir
			// 
			this->fcgTXCustomAudioTempDir->Location = System::Drawing::Point(75, 276);
			this->fcgTXCustomAudioTempDir->Name = L"fcgTXCustomAudioTempDir";
			this->fcgTXCustomAudioTempDir->Size = System::Drawing::Size(245, 21);
			this->fcgTXCustomAudioTempDir->TabIndex = 13;
			this->fcgTXCustomAudioTempDir->TextChanged += gcnew System::EventHandler(this, &frmConfig::fcgTXCustomAudioTempDir_TextChanged);
			// 
			// fcgBTCustomAudioTempDir
			// 
			this->fcgBTCustomAudioTempDir->Location = System::Drawing::Point(326, 274);
			this->fcgBTCustomAudioTempDir->Name = L"fcgBTCustomAudioTempDir";
			this->fcgBTCustomAudioTempDir->Size = System::Drawing::Size(29, 23);
			this->fcgBTCustomAudioTempDir->TabIndex = 14;
			this->fcgBTCustomAudioTempDir->Text = L"...";
			this->fcgBTCustomAudioTempDir->UseVisualStyleBackColor = true;
			this->fcgBTCustomAudioTempDir->Click += gcnew System::EventHandler(this, &frmConfig::fcgBTCustomAudioTempDir_Click);
			// 
			// fcgCBAudioUsePipe
			// 
			this->fcgCBAudioUsePipe->AutoSize = true;
			this->fcgCBAudioUsePipe->Location = System::Drawing::Point(135, 142);
			this->fcgCBAudioUsePipe->Name = L"fcgCBAudioUsePipe";
			this->fcgCBAudioUsePipe->Size = System::Drawing::Size(73, 18);
			this->fcgCBAudioUsePipe->TabIndex = 10;
			this->fcgCBAudioUsePipe->Tag = L"chValue";
			this->fcgCBAudioUsePipe->Text = L"パイプ処理";
			this->fcgCBAudioUsePipe->UseVisualStyleBackColor = true;
			// 
			// fcgLBAudioBitrate
			// 
			this->fcgLBAudioBitrate->AutoSize = true;
			this->fcgLBAudioBitrate->Location = System::Drawing::Point(289, 169);
			this->fcgLBAudioBitrate->Name = L"fcgLBAudioBitrate";
			this->fcgLBAudioBitrate->Size = System::Drawing::Size(32, 14);
			this->fcgLBAudioBitrate->TabIndex = 20;
			this->fcgLBAudioBitrate->Text = L"kbps";
			// 
			// fcgNUAudioBitrate
			// 
			this->fcgNUAudioBitrate->Location = System::Drawing::Point(218, 165);
			this->fcgNUAudioBitrate->Name = L"fcgNUAudioBitrate";
			this->fcgNUAudioBitrate->Size = System::Drawing::Size(65, 21);
			this->fcgNUAudioBitrate->TabIndex = 8;
			this->fcgNUAudioBitrate->Tag = L"chValue";
			this->fcgNUAudioBitrate->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
			// 
			// fcgCBAudio2pass
			// 
			this->fcgCBAudio2pass->AutoSize = true;
			this->fcgCBAudio2pass->Location = System::Drawing::Point(64, 142);
			this->fcgCBAudio2pass->Name = L"fcgCBAudio2pass";
			this->fcgCBAudio2pass->Size = System::Drawing::Size(56, 18);
			this->fcgCBAudio2pass->TabIndex = 9;
			this->fcgCBAudio2pass->Tag = L"chValue";
			this->fcgCBAudio2pass->Text = L"2pass";
			this->fcgCBAudio2pass->UseVisualStyleBackColor = true;
			this->fcgCBAudio2pass->CheckedChanged += gcnew System::EventHandler(this, &frmConfig::fcgCBAudio2pass_CheckedChanged);
			// 
			// fcgCXAudioEncMode
			// 
			this->fcgCXAudioEncMode->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
			this->fcgCXAudioEncMode->FormattingEnabled = true;
			this->fcgCXAudioEncMode->Location = System::Drawing::Point(21, 164);
			this->fcgCXAudioEncMode->Name = L"fcgCXAudioEncMode";
			this->fcgCXAudioEncMode->Size = System::Drawing::Size(189, 22);
			this->fcgCXAudioEncMode->TabIndex = 7;
			this->fcgCXAudioEncMode->Tag = L"chValue";
			this->fcgCXAudioEncMode->SelectedIndexChanged += gcnew System::EventHandler(this, &frmConfig::fcgCXAudioEncMode_SelectedIndexChanged);
			// 
			// fcgLBAudioEncMode
			// 
			this->fcgLBAudioEncMode->AutoSize = true;
			this->fcgLBAudioEncMode->Location = System::Drawing::Point(10, 144);
			this->fcgLBAudioEncMode->Name = L"fcgLBAudioEncMode";
			this->fcgLBAudioEncMode->Size = System::Drawing::Size(32, 14);
			this->fcgLBAudioEncMode->TabIndex = 15;
			this->fcgLBAudioEncMode->Text = L"モード";
			// 
			// fcgBTAudioEncoderPath
			// 
			this->fcgBTAudioEncoderPath->Location = System::Drawing::Point(329, 98);
			this->fcgBTAudioEncoderPath->Name = L"fcgBTAudioEncoderPath";
			this->fcgBTAudioEncoderPath->Size = System::Drawing::Size(30, 23);
			this->fcgBTAudioEncoderPath->TabIndex = 6;
			this->fcgBTAudioEncoderPath->Text = L"...";
			this->fcgBTAudioEncoderPath->UseVisualStyleBackColor = true;
			this->fcgBTAudioEncoderPath->Click += gcnew System::EventHandler(this, &frmConfig::fcgBTAudioEncoderPath_Click);
			// 
			// fcgTXAudioEncoderPath
			// 
			this->fcgTXAudioEncoderPath->AllowDrop = true;
			this->fcgTXAudioEncoderPath->Location = System::Drawing::Point(20, 100);
			this->fcgTXAudioEncoderPath->Name = L"fcgTXAudioEncoderPath";
			this->fcgTXAudioEncoderPath->Size = System::Drawing::Size(303, 21);
			this->fcgTXAudioEncoderPath->TabIndex = 5;
			this->fcgTXAudioEncoderPath->TextChanged += gcnew System::EventHandler(this, &frmConfig::fcgTXAudioEncoderPath_TextChanged);
			this->fcgTXAudioEncoderPath->DragDrop += gcnew System::Windows::Forms::DragEventHandler(this, &frmConfig::fcgSetDragDropFilename_DragDrop);
			this->fcgTXAudioEncoderPath->DragEnter += gcnew System::Windows::Forms::DragEventHandler(this, &frmConfig::fcgSetDragDropFilename_Enter);
			// 
			// fcgLBAudioEncoderPath
			// 
			this->fcgLBAudioEncoderPath->AutoSize = true;
			this->fcgLBAudioEncoderPath->Location = System::Drawing::Point(17, 83);
			this->fcgLBAudioEncoderPath->Name = L"fcgLBAudioEncoderPath";
			this->fcgLBAudioEncoderPath->Size = System::Drawing::Size(49, 14);
			this->fcgLBAudioEncoderPath->TabIndex = 12;
			this->fcgLBAudioEncoderPath->Text = L"～の指定";
			// 
			// fcgCBAudioOnly
			// 
			this->fcgCBAudioOnly->AutoSize = true;
			this->fcgCBAudioOnly->ForeColor = System::Drawing::SystemColors::ControlDarkDark;
			this->fcgCBAudioOnly->Location = System::Drawing::Point(259, 14);
			this->fcgCBAudioOnly->Name = L"fcgCBAudioOnly";
			this->fcgCBAudioOnly->Size = System::Drawing::Size(89, 18);
			this->fcgCBAudioOnly->TabIndex = 1;
			this->fcgCBAudioOnly->Tag = L"chValue";
			this->fcgCBAudioOnly->Text = L"音声のみ出力";
			this->fcgCBAudioOnly->UseVisualStyleBackColor = true;
			// 
			// fcgCBFAWCheck
			// 
			this->fcgCBFAWCheck->AutoSize = true;
			this->fcgCBFAWCheck->Location = System::Drawing::Point(259, 36);
			this->fcgCBFAWCheck->Name = L"fcgCBFAWCheck";
			this->fcgCBFAWCheck->Size = System::Drawing::Size(81, 18);
			this->fcgCBFAWCheck->TabIndex = 2;
			this->fcgCBFAWCheck->Tag = L"chValue";
			this->fcgCBFAWCheck->Text = L"FAWCheck";
			this->fcgCBFAWCheck->UseVisualStyleBackColor = true;
			// 
			// fcgCXAudioPriority
			// 
			this->fcgCXAudioPriority->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
			this->fcgCXAudioPriority->FormattingEnabled = true;
			this->fcgCXAudioPriority->Location = System::Drawing::Point(146, 217);
			this->fcgCXAudioPriority->Name = L"fcgCXAudioPriority";
			this->fcgCXAudioPriority->Size = System::Drawing::Size(136, 22);
			this->fcgCXAudioPriority->TabIndex = 11;
			this->fcgCXAudioPriority->Tag = L"chValue";
			// 
			// fcgLBAudioPriority
			// 
			this->fcgLBAudioPriority->AutoSize = true;
			this->fcgLBAudioPriority->Location = System::Drawing::Point(19, 220);
			this->fcgLBAudioPriority->Name = L"fcgLBAudioPriority";
			this->fcgLBAudioPriority->Size = System::Drawing::Size(62, 14);
			this->fcgLBAudioPriority->TabIndex = 2;
			this->fcgLBAudioPriority->Text = L"音声優先度";
			// 
			// fcgCXAudioEncoder
			// 
			this->fcgCXAudioEncoder->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
			this->fcgCXAudioEncoder->FormattingEnabled = true;
			this->fcgCXAudioEncoder->Location = System::Drawing::Point(22, 42);
			this->fcgCXAudioEncoder->Name = L"fcgCXAudioEncoder";
			this->fcgCXAudioEncoder->Size = System::Drawing::Size(172, 22);
			this->fcgCXAudioEncoder->TabIndex = 0;
			this->fcgCXAudioEncoder->Tag = L"chValue";
			this->fcgCXAudioEncoder->SelectedIndexChanged += gcnew System::EventHandler(this, &frmConfig::fcgCXAudioEncoder_SelectedIndexChanged);
			// 
			// fcgLBAudioEncoder
			// 
			this->fcgLBAudioEncoder->AutoSize = true;
			this->fcgLBAudioEncoder->Location = System::Drawing::Point(11, 22);
			this->fcgLBAudioEncoder->Name = L"fcgLBAudioEncoder";
			this->fcgLBAudioEncoder->Size = System::Drawing::Size(48, 14);
			this->fcgLBAudioEncoder->TabIndex = 0;
			this->fcgLBAudioEncoder->Text = L"エンコーダ";
			// 
			// fcgtabControlMux
			// 
			this->fcgtabControlMux->Controls->Add(this->fcgtabPageMP4);
			this->fcgtabControlMux->Controls->Add(this->fcgtabPageMKV);
			this->fcgtabControlMux->Controls->Add(this->fcgtabPageMPG);
			this->fcgtabControlMux->Controls->Add(this->fcgtabPageMux);
			this->fcgtabControlMux->Controls->Add(this->fcgtabPageBat);
			this->fcgtabControlMux->Font = (gcnew System::Drawing::Font(L"Meiryo UI", 8.25F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(128)));
			this->fcgtabControlMux->Location = System::Drawing::Point(622, 339);
			this->fcgtabControlMux->Name = L"fcgtabControlMux";
			this->fcgtabControlMux->SelectedIndex = 0;
			this->fcgtabControlMux->Size = System::Drawing::Size(384, 201);
			this->fcgtabControlMux->TabIndex = 3;
			// 
			// fcgtabPageMP4
			// 
			this->fcgtabPageMP4->Controls->Add(this->fcgBTMP4RawPath);
			this->fcgtabPageMP4->Controls->Add(this->fcgTXMP4RawPath);
			this->fcgtabPageMP4->Controls->Add(this->fcgLBMP4RawPath);
			this->fcgtabPageMP4->Controls->Add(this->fcgCBMP4MuxApple);
			this->fcgtabPageMP4->Controls->Add(this->fcgBTMP4BoxTempDir);
			this->fcgtabPageMP4->Controls->Add(this->fcgTXMP4BoxTempDir);
			this->fcgtabPageMP4->Controls->Add(this->fcgCXMP4BoxTempDir);
			this->fcgtabPageMP4->Controls->Add(this->fcgLBMP4BoxTempDir);
			this->fcgtabPageMP4->Controls->Add(this->fcgBTTC2MP4Path);
			this->fcgtabPageMP4->Controls->Add(this->fcgTXTC2MP4Path);
			this->fcgtabPageMP4->Controls->Add(this->fcgBTMP4MuxerPath);
			this->fcgtabPageMP4->Controls->Add(this->fcgTXMP4MuxerPath);
			this->fcgtabPageMP4->Controls->Add(this->fcgLBTC2MP4Path);
			this->fcgtabPageMP4->Controls->Add(this->fcgLBMP4MuxerPath);
			this->fcgtabPageMP4->Controls->Add(this->fcgCXMP4CmdEx);
			this->fcgtabPageMP4->Controls->Add(this->fcgLBMP4CmdEx);
			this->fcgtabPageMP4->Controls->Add(this->fcgCBMP4MuxerExt);
			this->fcgtabPageMP4->Location = System::Drawing::Point(4, 23);
			this->fcgtabPageMP4->Name = L"fcgtabPageMP4";
			this->fcgtabPageMP4->Padding = System::Windows::Forms::Padding(3);
			this->fcgtabPageMP4->Size = System::Drawing::Size(376, 174);
			this->fcgtabPageMP4->TabIndex = 0;
			this->fcgtabPageMP4->Text = L"mp4";
			this->fcgtabPageMP4->UseVisualStyleBackColor = true;
			// 
			// fcgBTMP4RawPath
			// 
			this->fcgBTMP4RawPath->Location = System::Drawing::Point(340, 102);
			this->fcgBTMP4RawPath->Name = L"fcgBTMP4RawPath";
			this->fcgBTMP4RawPath->Size = System::Drawing::Size(30, 23);
			this->fcgBTMP4RawPath->TabIndex = 23;
			this->fcgBTMP4RawPath->Text = L"...";
			this->fcgBTMP4RawPath->UseVisualStyleBackColor = true;
			this->fcgBTMP4RawPath->Visible = false;
			this->fcgBTMP4RawPath->Click += gcnew System::EventHandler(this, &frmConfig::fcgBTMP4RawMuxerPath_Click);
			// 
			// fcgTXMP4RawPath
			// 
			this->fcgTXMP4RawPath->AllowDrop = true;
			this->fcgTXMP4RawPath->Location = System::Drawing::Point(136, 103);
			this->fcgTXMP4RawPath->Name = L"fcgTXMP4RawPath";
			this->fcgTXMP4RawPath->Size = System::Drawing::Size(202, 21);
			this->fcgTXMP4RawPath->TabIndex = 22;
			this->fcgTXMP4RawPath->Visible = false;
			this->fcgTXMP4RawPath->TextChanged += gcnew System::EventHandler(this, &frmConfig::fcgTXMP4RawMuxerPath_TextChanged);
			this->fcgTXMP4RawPath->DragDrop += gcnew System::Windows::Forms::DragEventHandler(this, &frmConfig::fcgSetDragDropFilename_DragDrop);
			this->fcgTXMP4RawPath->DragEnter += gcnew System::Windows::Forms::DragEventHandler(this, &frmConfig::fcgSetDragDropFilename_Enter);
			// 
			// fcgLBMP4RawPath
			// 
			this->fcgLBMP4RawPath->AutoSize = true;
			this->fcgLBMP4RawPath->Location = System::Drawing::Point(4, 106);
			this->fcgLBMP4RawPath->Name = L"fcgLBMP4RawPath";
			this->fcgLBMP4RawPath->Size = System::Drawing::Size(49, 14);
			this->fcgLBMP4RawPath->TabIndex = 21;
			this->fcgLBMP4RawPath->Text = L"～の指定";
			this->fcgLBMP4RawPath->Visible = false;
			// 
			// fcgCBMP4MuxApple
			// 
			this->fcgCBMP4MuxApple->AutoSize = true;
			this->fcgCBMP4MuxApple->Location = System::Drawing::Point(254, 34);
			this->fcgCBMP4MuxApple->Name = L"fcgCBMP4MuxApple";
			this->fcgCBMP4MuxApple->Size = System::Drawing::Size(109, 18);
			this->fcgCBMP4MuxApple->TabIndex = 20;
			this->fcgCBMP4MuxApple->Tag = L"chValue";
			this->fcgCBMP4MuxApple->Text = L"Apple形式に対応";
			this->fcgCBMP4MuxApple->UseVisualStyleBackColor = true;
			// 
			// fcgBTMP4BoxTempDir
			// 
			this->fcgBTMP4BoxTempDir->Location = System::Drawing::Point(340, 146);
			this->fcgBTMP4BoxTempDir->Name = L"fcgBTMP4BoxTempDir";
			this->fcgBTMP4BoxTempDir->Size = System::Drawing::Size(30, 23);
			this->fcgBTMP4BoxTempDir->TabIndex = 8;
			this->fcgBTMP4BoxTempDir->Text = L"...";
			this->fcgBTMP4BoxTempDir->UseVisualStyleBackColor = true;
			this->fcgBTMP4BoxTempDir->Click += gcnew System::EventHandler(this, &frmConfig::fcgBTMP4BoxTempDir_Click);
			// 
			// fcgTXMP4BoxTempDir
			// 
			this->fcgTXMP4BoxTempDir->Location = System::Drawing::Point(107, 147);
			this->fcgTXMP4BoxTempDir->Name = L"fcgTXMP4BoxTempDir";
			this->fcgTXMP4BoxTempDir->Size = System::Drawing::Size(227, 21);
			this->fcgTXMP4BoxTempDir->TabIndex = 7;
			this->fcgTXMP4BoxTempDir->TextChanged += gcnew System::EventHandler(this, &frmConfig::fcgTXMP4BoxTempDir_TextChanged);
			// 
			// fcgCXMP4BoxTempDir
			// 
			this->fcgCXMP4BoxTempDir->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
			this->fcgCXMP4BoxTempDir->FormattingEnabled = true;
			this->fcgCXMP4BoxTempDir->Location = System::Drawing::Point(145, 119);
			this->fcgCXMP4BoxTempDir->Name = L"fcgCXMP4BoxTempDir";
			this->fcgCXMP4BoxTempDir->Size = System::Drawing::Size(206, 22);
			this->fcgCXMP4BoxTempDir->TabIndex = 6;
			this->fcgCXMP4BoxTempDir->Tag = L"chValue";
			// 
			// fcgLBMP4BoxTempDir
			// 
			this->fcgLBMP4BoxTempDir->AutoSize = true;
			this->fcgLBMP4BoxTempDir->Location = System::Drawing::Point(25, 122);
			this->fcgLBMP4BoxTempDir->Name = L"fcgLBMP4BoxTempDir";
			this->fcgLBMP4BoxTempDir->Size = System::Drawing::Size(105, 14);
			this->fcgLBMP4BoxTempDir->TabIndex = 18;
			this->fcgLBMP4BoxTempDir->Text = L"mp4box一時フォルダ";
			// 
			// fcgBTTC2MP4Path
			// 
			this->fcgBTTC2MP4Path->Location = System::Drawing::Point(340, 80);
			this->fcgBTTC2MP4Path->Name = L"fcgBTTC2MP4Path";
			this->fcgBTTC2MP4Path->Size = System::Drawing::Size(30, 23);
			this->fcgBTTC2MP4Path->TabIndex = 5;
			this->fcgBTTC2MP4Path->Text = L"...";
			this->fcgBTTC2MP4Path->UseVisualStyleBackColor = true;
			this->fcgBTTC2MP4Path->Visible = false;
			this->fcgBTTC2MP4Path->Click += gcnew System::EventHandler(this, &frmConfig::fcgBTTC2MP4Path_Click);
			// 
			// fcgTXTC2MP4Path
			// 
			this->fcgTXTC2MP4Path->AllowDrop = true;
			this->fcgTXTC2MP4Path->Location = System::Drawing::Point(136, 81);
			this->fcgTXTC2MP4Path->Name = L"fcgTXTC2MP4Path";
			this->fcgTXTC2MP4Path->Size = System::Drawing::Size(202, 21);
			this->fcgTXTC2MP4Path->TabIndex = 4;
			this->fcgTXTC2MP4Path->Visible = false;
			this->fcgTXTC2MP4Path->TextChanged += gcnew System::EventHandler(this, &frmConfig::fcgTXTC2MP4Path_TextChanged);
			this->fcgTXTC2MP4Path->DragDrop += gcnew System::Windows::Forms::DragEventHandler(this, &frmConfig::fcgSetDragDropFilename_DragDrop);
			this->fcgTXTC2MP4Path->DragEnter += gcnew System::Windows::Forms::DragEventHandler(this, &frmConfig::fcgSetDragDropFilename_Enter);
			// 
			// fcgBTMP4MuxerPath
			// 
			this->fcgBTMP4MuxerPath->Location = System::Drawing::Point(340, 58);
			this->fcgBTMP4MuxerPath->Name = L"fcgBTMP4MuxerPath";
			this->fcgBTMP4MuxerPath->Size = System::Drawing::Size(30, 23);
			this->fcgBTMP4MuxerPath->TabIndex = 3;
			this->fcgBTMP4MuxerPath->Text = L"...";
			this->fcgBTMP4MuxerPath->UseVisualStyleBackColor = true;
			this->fcgBTMP4MuxerPath->Click += gcnew System::EventHandler(this, &frmConfig::fcgBTMP4MuxerPath_Click);
			// 
			// fcgTXMP4MuxerPath
			// 
			this->fcgTXMP4MuxerPath->AllowDrop = true;
			this->fcgTXMP4MuxerPath->Location = System::Drawing::Point(136, 59);
			this->fcgTXMP4MuxerPath->Name = L"fcgTXMP4MuxerPath";
			this->fcgTXMP4MuxerPath->Size = System::Drawing::Size(202, 21);
			this->fcgTXMP4MuxerPath->TabIndex = 2;
			this->fcgTXMP4MuxerPath->Tag = L"";
			this->fcgTXMP4MuxerPath->TextChanged += gcnew System::EventHandler(this, &frmConfig::fcgTXMP4MuxerPath_TextChanged);
			this->fcgTXMP4MuxerPath->DragDrop += gcnew System::Windows::Forms::DragEventHandler(this, &frmConfig::fcgSetDragDropFilename_DragDrop);
			this->fcgTXMP4MuxerPath->DragEnter += gcnew System::Windows::Forms::DragEventHandler(this, &frmConfig::fcgSetDragDropFilename_Enter);
			// 
			// fcgLBTC2MP4Path
			// 
			this->fcgLBTC2MP4Path->AutoSize = true;
			this->fcgLBTC2MP4Path->Location = System::Drawing::Point(4, 84);
			this->fcgLBTC2MP4Path->Name = L"fcgLBTC2MP4Path";
			this->fcgLBTC2MP4Path->Size = System::Drawing::Size(49, 14);
			this->fcgLBTC2MP4Path->TabIndex = 4;
			this->fcgLBTC2MP4Path->Text = L"～の指定";
			this->fcgLBTC2MP4Path->Visible = false;
			// 
			// fcgLBMP4MuxerPath
			// 
			this->fcgLBMP4MuxerPath->AutoSize = true;
			this->fcgLBMP4MuxerPath->Location = System::Drawing::Point(4, 62);
			this->fcgLBMP4MuxerPath->Name = L"fcgLBMP4MuxerPath";
			this->fcgLBMP4MuxerPath->Size = System::Drawing::Size(49, 14);
			this->fcgLBMP4MuxerPath->TabIndex = 3;
			this->fcgLBMP4MuxerPath->Text = L"～の指定";
			// 
			// fcgCXMP4CmdEx
			// 
			this->fcgCXMP4CmdEx->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
			this->fcgCXMP4CmdEx->FormattingEnabled = true;
			this->fcgCXMP4CmdEx->Location = System::Drawing::Point(213, 7);
			this->fcgCXMP4CmdEx->Name = L"fcgCXMP4CmdEx";
			this->fcgCXMP4CmdEx->Size = System::Drawing::Size(157, 22);
			this->fcgCXMP4CmdEx->TabIndex = 1;
			this->fcgCXMP4CmdEx->Tag = L"chValue";
			// 
			// fcgLBMP4CmdEx
			// 
			this->fcgLBMP4CmdEx->AutoSize = true;
			this->fcgLBMP4CmdEx->Location = System::Drawing::Point(139, 10);
			this->fcgLBMP4CmdEx->Name = L"fcgLBMP4CmdEx";
			this->fcgLBMP4CmdEx->Size = System::Drawing::Size(68, 14);
			this->fcgLBMP4CmdEx->TabIndex = 1;
			this->fcgLBMP4CmdEx->Text = L"拡張オプション";
			// 
			// fcgCBMP4MuxerExt
			// 
			this->fcgCBMP4MuxerExt->AutoSize = true;
			this->fcgCBMP4MuxerExt->Location = System::Drawing::Point(10, 9);
			this->fcgCBMP4MuxerExt->Name = L"fcgCBMP4MuxerExt";
			this->fcgCBMP4MuxerExt->Size = System::Drawing::Size(113, 18);
			this->fcgCBMP4MuxerExt->TabIndex = 0;
			this->fcgCBMP4MuxerExt->Tag = L"chValue";
			this->fcgCBMP4MuxerExt->Text = L"外部muxerを使用";
			this->fcgCBMP4MuxerExt->UseVisualStyleBackColor = true;
			// 
			// fcgtabPageMKV
			// 
			this->fcgtabPageMKV->Controls->Add(this->fcgBTMKVMuxerPath);
			this->fcgtabPageMKV->Controls->Add(this->fcgTXMKVMuxerPath);
			this->fcgtabPageMKV->Controls->Add(this->fcgLBMKVMuxerPath);
			this->fcgtabPageMKV->Controls->Add(this->fcgCXMKVCmdEx);
			this->fcgtabPageMKV->Controls->Add(this->fcgLBMKVMuxerCmdEx);
			this->fcgtabPageMKV->Controls->Add(this->fcgCBMKVMuxerExt);
			this->fcgtabPageMKV->Location = System::Drawing::Point(4, 23);
			this->fcgtabPageMKV->Name = L"fcgtabPageMKV";
			this->fcgtabPageMKV->Padding = System::Windows::Forms::Padding(3);
			this->fcgtabPageMKV->Size = System::Drawing::Size(376, 174);
			this->fcgtabPageMKV->TabIndex = 1;
			this->fcgtabPageMKV->Text = L"mkv";
			this->fcgtabPageMKV->UseVisualStyleBackColor = true;
			// 
			// fcgBTMKVMuxerPath
			// 
			this->fcgBTMKVMuxerPath->Location = System::Drawing::Point(340, 76);
			this->fcgBTMKVMuxerPath->Name = L"fcgBTMKVMuxerPath";
			this->fcgBTMKVMuxerPath->Size = System::Drawing::Size(30, 23);
			this->fcgBTMKVMuxerPath->TabIndex = 3;
			this->fcgBTMKVMuxerPath->Text = L"...";
			this->fcgBTMKVMuxerPath->UseVisualStyleBackColor = true;
			this->fcgBTMKVMuxerPath->Click += gcnew System::EventHandler(this, &frmConfig::fcgBTMKVMuxerPath_Click);
			// 
			// fcgTXMKVMuxerPath
			// 
			this->fcgTXMKVMuxerPath->Location = System::Drawing::Point(131, 77);
			this->fcgTXMKVMuxerPath->Name = L"fcgTXMKVMuxerPath";
			this->fcgTXMKVMuxerPath->Size = System::Drawing::Size(207, 21);
			this->fcgTXMKVMuxerPath->TabIndex = 2;
			this->fcgTXMKVMuxerPath->TextChanged += gcnew System::EventHandler(this, &frmConfig::fcgTXMKVMuxerPath_TextChanged);
			this->fcgTXMKVMuxerPath->DragDrop += gcnew System::Windows::Forms::DragEventHandler(this, &frmConfig::fcgSetDragDropFilename_DragDrop);
			this->fcgTXMKVMuxerPath->DragEnter += gcnew System::Windows::Forms::DragEventHandler(this, &frmConfig::fcgSetDragDropFilename_Enter);
			// 
			// fcgLBMKVMuxerPath
			// 
			this->fcgLBMKVMuxerPath->AutoSize = true;
			this->fcgLBMKVMuxerPath->Location = System::Drawing::Point(4, 80);
			this->fcgLBMKVMuxerPath->Name = L"fcgLBMKVMuxerPath";
			this->fcgLBMKVMuxerPath->Size = System::Drawing::Size(49, 14);
			this->fcgLBMKVMuxerPath->TabIndex = 19;
			this->fcgLBMKVMuxerPath->Text = L"～の指定";
			// 
			// fcgCXMKVCmdEx
			// 
			this->fcgCXMKVCmdEx->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
			this->fcgCXMKVCmdEx->FormattingEnabled = true;
			this->fcgCXMKVCmdEx->Location = System::Drawing::Point(213, 43);
			this->fcgCXMKVCmdEx->Name = L"fcgCXMKVCmdEx";
			this->fcgCXMKVCmdEx->Size = System::Drawing::Size(157, 22);
			this->fcgCXMKVCmdEx->TabIndex = 1;
			this->fcgCXMKVCmdEx->Tag = L"chValue";
			// 
			// fcgLBMKVMuxerCmdEx
			// 
			this->fcgLBMKVMuxerCmdEx->AutoSize = true;
			this->fcgLBMKVMuxerCmdEx->Location = System::Drawing::Point(139, 46);
			this->fcgLBMKVMuxerCmdEx->Name = L"fcgLBMKVMuxerCmdEx";
			this->fcgLBMKVMuxerCmdEx->Size = System::Drawing::Size(68, 14);
			this->fcgLBMKVMuxerCmdEx->TabIndex = 17;
			this->fcgLBMKVMuxerCmdEx->Text = L"拡張オプション";
			// 
			// fcgCBMKVMuxerExt
			// 
			this->fcgCBMKVMuxerExt->AutoSize = true;
			this->fcgCBMKVMuxerExt->Location = System::Drawing::Point(10, 45);
			this->fcgCBMKVMuxerExt->Name = L"fcgCBMKVMuxerExt";
			this->fcgCBMKVMuxerExt->Size = System::Drawing::Size(113, 18);
			this->fcgCBMKVMuxerExt->TabIndex = 0;
			this->fcgCBMKVMuxerExt->Tag = L"chValue";
			this->fcgCBMKVMuxerExt->Text = L"外部muxerを使用";
			this->fcgCBMKVMuxerExt->UseVisualStyleBackColor = true;
			// 
			// fcgtabPageMPG
			// 
			this->fcgtabPageMPG->Controls->Add(this->fcgBTMPGMuxerPath);
			this->fcgtabPageMPG->Controls->Add(this->fcgTXMPGMuxerPath);
			this->fcgtabPageMPG->Controls->Add(this->fcgLBMPGMuxerPath);
			this->fcgtabPageMPG->Controls->Add(this->fcgCXMPGCmdEx);
			this->fcgtabPageMPG->Controls->Add(this->fcgLBMPGMuxerCmdEx);
			this->fcgtabPageMPG->Controls->Add(this->fcgCBMPGMuxerExt);
			this->fcgtabPageMPG->Location = System::Drawing::Point(4, 23);
			this->fcgtabPageMPG->Name = L"fcgtabPageMPG";
			this->fcgtabPageMPG->Size = System::Drawing::Size(376, 174);
			this->fcgtabPageMPG->TabIndex = 4;
			this->fcgtabPageMPG->Text = L"mpg";
			this->fcgtabPageMPG->UseVisualStyleBackColor = true;
			// 
			// fcgBTMPGMuxerPath
			// 
			this->fcgBTMPGMuxerPath->Location = System::Drawing::Point(341, 92);
			this->fcgBTMPGMuxerPath->Name = L"fcgBTMPGMuxerPath";
			this->fcgBTMPGMuxerPath->Size = System::Drawing::Size(30, 23);
			this->fcgBTMPGMuxerPath->TabIndex = 23;
			this->fcgBTMPGMuxerPath->Text = L"...";
			this->fcgBTMPGMuxerPath->UseVisualStyleBackColor = true;
			this->fcgBTMPGMuxerPath->Click += gcnew System::EventHandler(this, &frmConfig::fcgBTMPGMuxerPath_Click);
			// 
			// fcgTXMPGMuxerPath
			// 
			this->fcgTXMPGMuxerPath->Location = System::Drawing::Point(132, 93);
			this->fcgTXMPGMuxerPath->Name = L"fcgTXMPGMuxerPath";
			this->fcgTXMPGMuxerPath->Size = System::Drawing::Size(207, 21);
			this->fcgTXMPGMuxerPath->TabIndex = 22;
			this->fcgTXMPGMuxerPath->TextChanged += gcnew System::EventHandler(this, &frmConfig::fcgTXMPGMuxerPath_TextChanged);
			this->fcgTXMPGMuxerPath->DragDrop += gcnew System::Windows::Forms::DragEventHandler(this, &frmConfig::fcgSetDragDropFilename_DragDrop);
			this->fcgTXMPGMuxerPath->DragEnter += gcnew System::Windows::Forms::DragEventHandler(this, &frmConfig::fcgSetDragDropFilename_Enter);
			// 
			// fcgLBMPGMuxerPath
			// 
			this->fcgLBMPGMuxerPath->AutoSize = true;
			this->fcgLBMPGMuxerPath->Location = System::Drawing::Point(5, 96);
			this->fcgLBMPGMuxerPath->Name = L"fcgLBMPGMuxerPath";
			this->fcgLBMPGMuxerPath->Size = System::Drawing::Size(49, 14);
			this->fcgLBMPGMuxerPath->TabIndex = 25;
			this->fcgLBMPGMuxerPath->Text = L"～の指定";
			// 
			// fcgCXMPGCmdEx
			// 
			this->fcgCXMPGCmdEx->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
			this->fcgCXMPGCmdEx->FormattingEnabled = true;
			this->fcgCXMPGCmdEx->Location = System::Drawing::Point(214, 59);
			this->fcgCXMPGCmdEx->Name = L"fcgCXMPGCmdEx";
			this->fcgCXMPGCmdEx->Size = System::Drawing::Size(157, 22);
			this->fcgCXMPGCmdEx->TabIndex = 21;
			this->fcgCXMPGCmdEx->Tag = L"chValue";
			// 
			// fcgLBMPGMuxerCmdEx
			// 
			this->fcgLBMPGMuxerCmdEx->AutoSize = true;
			this->fcgLBMPGMuxerCmdEx->Location = System::Drawing::Point(140, 62);
			this->fcgLBMPGMuxerCmdEx->Name = L"fcgLBMPGMuxerCmdEx";
			this->fcgLBMPGMuxerCmdEx->Size = System::Drawing::Size(68, 14);
			this->fcgLBMPGMuxerCmdEx->TabIndex = 24;
			this->fcgLBMPGMuxerCmdEx->Text = L"拡張オプション";
			// 
			// fcgCBMPGMuxerExt
			// 
			this->fcgCBMPGMuxerExt->AutoSize = true;
			this->fcgCBMPGMuxerExt->Location = System::Drawing::Point(11, 61);
			this->fcgCBMPGMuxerExt->Name = L"fcgCBMPGMuxerExt";
			this->fcgCBMPGMuxerExt->Size = System::Drawing::Size(113, 18);
			this->fcgCBMPGMuxerExt->TabIndex = 20;
			this->fcgCBMPGMuxerExt->Tag = L"chValue";
			this->fcgCBMPGMuxerExt->Text = L"外部muxerを使用";
			this->fcgCBMPGMuxerExt->UseVisualStyleBackColor = true;
			// 
			// fcgtabPageMux
			// 
			this->fcgtabPageMux->Controls->Add(this->fcgCXMuxPriority);
			this->fcgtabPageMux->Controls->Add(this->fcgLBMuxPriority);
			this->fcgtabPageMux->Controls->Add(this->fcgCBMuxMinimize);
			this->fcgtabPageMux->Location = System::Drawing::Point(4, 23);
			this->fcgtabPageMux->Name = L"fcgtabPageMux";
			this->fcgtabPageMux->Size = System::Drawing::Size(376, 174);
			this->fcgtabPageMux->TabIndex = 2;
			this->fcgtabPageMux->Text = L"Mux共通設定";
			this->fcgtabPageMux->UseVisualStyleBackColor = true;
			// 
			// fcgCXMuxPriority
			// 
			this->fcgCXMuxPriority->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
			this->fcgCXMuxPriority->FormattingEnabled = true;
			this->fcgCXMuxPriority->Location = System::Drawing::Point(102, 64);
			this->fcgCXMuxPriority->Name = L"fcgCXMuxPriority";
			this->fcgCXMuxPriority->Size = System::Drawing::Size(198, 22);
			this->fcgCXMuxPriority->TabIndex = 1;
			this->fcgCXMuxPriority->Tag = L"chValue";
			// 
			// fcgLBMuxPriority
			// 
			this->fcgLBMuxPriority->AutoSize = true;
			this->fcgLBMuxPriority->Location = System::Drawing::Point(15, 67);
			this->fcgLBMuxPriority->Name = L"fcgLBMuxPriority";
			this->fcgLBMuxPriority->Size = System::Drawing::Size(62, 14);
			this->fcgLBMuxPriority->TabIndex = 1;
			this->fcgLBMuxPriority->Text = L"Mux優先度";
			// 
			// fcgCBMuxMinimize
			// 
			this->fcgCBMuxMinimize->AutoSize = true;
			this->fcgCBMuxMinimize->Location = System::Drawing::Point(18, 26);
			this->fcgCBMuxMinimize->Name = L"fcgCBMuxMinimize";
			this->fcgCBMuxMinimize->Size = System::Drawing::Size(59, 18);
			this->fcgCBMuxMinimize->TabIndex = 0;
			this->fcgCBMuxMinimize->Tag = L"chValue";
			this->fcgCBMuxMinimize->Text = L"最小化";
			this->fcgCBMuxMinimize->UseVisualStyleBackColor = true;
			// 
			// fcgtabPageBat
			// 
			this->fcgtabPageBat->Controls->Add(this->fcgLBBatAfterString);
			this->fcgtabPageBat->Controls->Add(this->fcgLBBatBeforeString);
			this->fcgtabPageBat->Controls->Add(this->fcgPNSeparator);
			this->fcgtabPageBat->Controls->Add(this->fcgBTBatBeforePath);
			this->fcgtabPageBat->Controls->Add(this->fcgTXBatBeforePath);
			this->fcgtabPageBat->Controls->Add(this->fcgLBBatBeforePath);
			this->fcgtabPageBat->Controls->Add(this->fcgCBWaitForBatBefore);
			this->fcgtabPageBat->Controls->Add(this->fcgCBRunBatBefore);
			this->fcgtabPageBat->Controls->Add(this->fcgBTBatAfterPath);
			this->fcgtabPageBat->Controls->Add(this->fcgTXBatAfterPath);
			this->fcgtabPageBat->Controls->Add(this->fcgLBBatAfterPath);
			this->fcgtabPageBat->Controls->Add(this->fcgCBWaitForBatAfter);
			this->fcgtabPageBat->Controls->Add(this->fcgCBRunBatAfter);
			this->fcgtabPageBat->Location = System::Drawing::Point(4, 23);
			this->fcgtabPageBat->Name = L"fcgtabPageBat";
			this->fcgtabPageBat->Size = System::Drawing::Size(376, 174);
			this->fcgtabPageBat->TabIndex = 3;
			this->fcgtabPageBat->Text = L"エンコ前後バッチ処理";
			this->fcgtabPageBat->UseVisualStyleBackColor = true;
			// 
			// fcgLBBatAfterString
			// 
			this->fcgLBBatAfterString->AutoSize = true;
			this->fcgLBBatAfterString->Font = (gcnew System::Drawing::Font(L"Meiryo UI", 9, static_cast<System::Drawing::FontStyle>((System::Drawing::FontStyle::Italic | System::Drawing::FontStyle::Underline)),
				System::Drawing::GraphicsUnit::Point, static_cast<System::Byte>(128)));
			this->fcgLBBatAfterString->ForeColor = System::Drawing::SystemColors::ControlDarkDark;
			this->fcgLBBatAfterString->Location = System::Drawing::Point(304, 112);
			this->fcgLBBatAfterString->Name = L"fcgLBBatAfterString";
			this->fcgLBBatAfterString->Size = System::Drawing::Size(27, 15);
			this->fcgLBBatAfterString->TabIndex = 20;
			this->fcgLBBatAfterString->Text = L" 後& ";
			this->fcgLBBatAfterString->TextAlign = System::Drawing::ContentAlignment::TopCenter;
			// 
			// fcgLBBatBeforeString
			// 
			this->fcgLBBatBeforeString->AutoSize = true;
			this->fcgLBBatBeforeString->Font = (gcnew System::Drawing::Font(L"Meiryo UI", 9, static_cast<System::Drawing::FontStyle>((System::Drawing::FontStyle::Italic | System::Drawing::FontStyle::Underline)),
				System::Drawing::GraphicsUnit::Point, static_cast<System::Byte>(128)));
			this->fcgLBBatBeforeString->ForeColor = System::Drawing::SystemColors::ControlDarkDark;
			this->fcgLBBatBeforeString->Location = System::Drawing::Point(304, 14);
			this->fcgLBBatBeforeString->Name = L"fcgLBBatBeforeString";
			this->fcgLBBatBeforeString->Size = System::Drawing::Size(27, 15);
			this->fcgLBBatBeforeString->TabIndex = 19;
			this->fcgLBBatBeforeString->Text = L" 前& ";
			this->fcgLBBatBeforeString->TextAlign = System::Drawing::ContentAlignment::TopCenter;
			// 
			// fcgPNSeparator
			// 
			this->fcgPNSeparator->BorderStyle = System::Windows::Forms::BorderStyle::FixedSingle;
			this->fcgPNSeparator->Location = System::Drawing::Point(18, 88);
			this->fcgPNSeparator->Name = L"fcgPNSeparator";
			this->fcgPNSeparator->Size = System::Drawing::Size(342, 1);
			this->fcgPNSeparator->TabIndex = 18;
			// 
			// fcgBTBatBeforePath
			// 
			this->fcgBTBatBeforePath->Location = System::Drawing::Point(330, 55);
			this->fcgBTBatBeforePath->Name = L"fcgBTBatBeforePath";
			this->fcgBTBatBeforePath->Size = System::Drawing::Size(30, 23);
			this->fcgBTBatBeforePath->TabIndex = 17;
			this->fcgBTBatBeforePath->Tag = L"chValue";
			this->fcgBTBatBeforePath->Text = L"...";
			this->fcgBTBatBeforePath->UseVisualStyleBackColor = true;
			this->fcgBTBatBeforePath->Click += gcnew System::EventHandler(this, &frmConfig::fcgBTBatBeforePath_Click);
			// 
			// fcgTXBatBeforePath
			// 
			this->fcgTXBatBeforePath->AllowDrop = true;
			this->fcgTXBatBeforePath->Location = System::Drawing::Point(126, 56);
			this->fcgTXBatBeforePath->Name = L"fcgTXBatBeforePath";
			this->fcgTXBatBeforePath->Size = System::Drawing::Size(202, 21);
			this->fcgTXBatBeforePath->TabIndex = 16;
			this->fcgTXBatBeforePath->Tag = L"chValue";
			this->fcgTXBatBeforePath->DragDrop += gcnew System::Windows::Forms::DragEventHandler(this, &frmConfig::fcgSetDragDropFilename_DragDrop);
			this->fcgTXBatBeforePath->DragEnter += gcnew System::Windows::Forms::DragEventHandler(this, &frmConfig::fcgSetDragDropFilename_Enter);
			// 
			// fcgLBBatBeforePath
			// 
			this->fcgLBBatBeforePath->AutoSize = true;
			this->fcgLBBatBeforePath->Location = System::Drawing::Point(40, 59);
			this->fcgLBBatBeforePath->Name = L"fcgLBBatBeforePath";
			this->fcgLBBatBeforePath->Size = System::Drawing::Size(61, 14);
			this->fcgLBBatBeforePath->TabIndex = 15;
			this->fcgLBBatBeforePath->Text = L"バッチファイル";
			// 
			// fcgCBWaitForBatBefore
			// 
			this->fcgCBWaitForBatBefore->AutoSize = true;
			this->fcgCBWaitForBatBefore->Location = System::Drawing::Point(40, 30);
			this->fcgCBWaitForBatBefore->Name = L"fcgCBWaitForBatBefore";
			this->fcgCBWaitForBatBefore->Size = System::Drawing::Size(150, 18);
			this->fcgCBWaitForBatBefore->TabIndex = 14;
			this->fcgCBWaitForBatBefore->Tag = L"chValue";
			this->fcgCBWaitForBatBefore->Text = L"バッチ処理の終了を待機する";
			this->fcgCBWaitForBatBefore->UseVisualStyleBackColor = true;
			// 
			// fcgCBRunBatBefore
			// 
			this->fcgCBRunBatBefore->AutoSize = true;
			this->fcgCBRunBatBefore->Location = System::Drawing::Point(18, 6);
			this->fcgCBRunBatBefore->Name = L"fcgCBRunBatBefore";
			this->fcgCBRunBatBefore->Size = System::Drawing::Size(179, 18);
			this->fcgCBRunBatBefore->TabIndex = 13;
			this->fcgCBRunBatBefore->Tag = L"chValue";
			this->fcgCBRunBatBefore->Text = L"エンコード開始前、バッチ処理を行う";
			this->fcgCBRunBatBefore->UseVisualStyleBackColor = true;
			// 
			// fcgBTBatAfterPath
			// 
			this->fcgBTBatAfterPath->Location = System::Drawing::Point(330, 146);
			this->fcgBTBatAfterPath->Name = L"fcgBTBatAfterPath";
			this->fcgBTBatAfterPath->Size = System::Drawing::Size(30, 23);
			this->fcgBTBatAfterPath->TabIndex = 10;
			this->fcgBTBatAfterPath->Tag = L"chValue";
			this->fcgBTBatAfterPath->Text = L"...";
			this->fcgBTBatAfterPath->UseVisualStyleBackColor = true;
			this->fcgBTBatAfterPath->Click += gcnew System::EventHandler(this, &frmConfig::fcgBTBatAfterPath_Click);
			// 
			// fcgTXBatAfterPath
			// 
			this->fcgTXBatAfterPath->AllowDrop = true;
			this->fcgTXBatAfterPath->Location = System::Drawing::Point(126, 147);
			this->fcgTXBatAfterPath->Name = L"fcgTXBatAfterPath";
			this->fcgTXBatAfterPath->Size = System::Drawing::Size(202, 21);
			this->fcgTXBatAfterPath->TabIndex = 9;
			this->fcgTXBatAfterPath->Tag = L"chValue";
			this->fcgTXBatAfterPath->DragDrop += gcnew System::Windows::Forms::DragEventHandler(this, &frmConfig::fcgSetDragDropFilename_DragDrop);
			this->fcgTXBatAfterPath->DragEnter += gcnew System::Windows::Forms::DragEventHandler(this, &frmConfig::fcgSetDragDropFilename_Enter);
			// 
			// fcgLBBatAfterPath
			// 
			this->fcgLBBatAfterPath->AutoSize = true;
			this->fcgLBBatAfterPath->Location = System::Drawing::Point(40, 150);
			this->fcgLBBatAfterPath->Name = L"fcgLBBatAfterPath";
			this->fcgLBBatAfterPath->Size = System::Drawing::Size(61, 14);
			this->fcgLBBatAfterPath->TabIndex = 8;
			this->fcgLBBatAfterPath->Text = L"バッチファイル";
			// 
			// fcgCBWaitForBatAfter
			// 
			this->fcgCBWaitForBatAfter->AutoSize = true;
			this->fcgCBWaitForBatAfter->Location = System::Drawing::Point(40, 122);
			this->fcgCBWaitForBatAfter->Name = L"fcgCBWaitForBatAfter";
			this->fcgCBWaitForBatAfter->Size = System::Drawing::Size(150, 18);
			this->fcgCBWaitForBatAfter->TabIndex = 7;
			this->fcgCBWaitForBatAfter->Tag = L"chValue";
			this->fcgCBWaitForBatAfter->Text = L"バッチ処理の終了を待機する";
			this->fcgCBWaitForBatAfter->UseVisualStyleBackColor = true;
			// 
			// fcgCBRunBatAfter
			// 
			this->fcgCBRunBatAfter->AutoSize = true;
			this->fcgCBRunBatAfter->Location = System::Drawing::Point(18, 98);
			this->fcgCBRunBatAfter->Name = L"fcgCBRunBatAfter";
			this->fcgCBRunBatAfter->Size = System::Drawing::Size(179, 18);
			this->fcgCBRunBatAfter->TabIndex = 6;
			this->fcgCBRunBatAfter->Tag = L"chValue";
			this->fcgCBRunBatAfter->Text = L"エンコード終了後、バッチ処理を行う";
			this->fcgCBRunBatAfter->UseVisualStyleBackColor = true;
			// 
			// fcgTXCmd
			// 
			this->fcgTXCmd->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom)
				| System::Windows::Forms::AnchorStyles::Left)
				| System::Windows::Forms::AnchorStyles::Right));
			this->fcgTXCmd->Font = (gcnew System::Drawing::Font(L"Meiryo UI", 8.25F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(128)));
			this->fcgTXCmd->Location = System::Drawing::Point(9, 512);
			this->fcgTXCmd->Name = L"fcgTXCmd";
			this->fcgTXCmd->ReadOnly = true;
			this->fcgTXCmd->Size = System::Drawing::Size(992, 21);
			this->fcgTXCmd->TabIndex = 4;
			this->fcgTXCmd->Visible = false;
			this->fcgTXCmd->DoubleClick += gcnew System::EventHandler(this, &frmConfig::fcgTXCmd_DoubleClick);
			// 
			// fcgBTCancel
			// 
			this->fcgBTCancel->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((System::Windows::Forms::AnchorStyles::Bottom | System::Windows::Forms::AnchorStyles::Right));
			this->fcgBTCancel->Location = System::Drawing::Point(771, 546);
			this->fcgBTCancel->Name = L"fcgBTCancel";
			this->fcgBTCancel->Size = System::Drawing::Size(84, 28);
			this->fcgBTCancel->TabIndex = 5;
			this->fcgBTCancel->Text = L"キャンセル";
			this->fcgBTCancel->UseVisualStyleBackColor = true;
			this->fcgBTCancel->Click += gcnew System::EventHandler(this, &frmConfig::fcgBTCancel_Click);
			// 
			// fcgBTOK
			// 
			this->fcgBTOK->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((System::Windows::Forms::AnchorStyles::Bottom | System::Windows::Forms::AnchorStyles::Right));
			this->fcgBTOK->Location = System::Drawing::Point(893, 546);
			this->fcgBTOK->Name = L"fcgBTOK";
			this->fcgBTOK->Size = System::Drawing::Size(84, 28);
			this->fcgBTOK->TabIndex = 6;
			this->fcgBTOK->Text = L"OK";
			this->fcgBTOK->UseVisualStyleBackColor = true;
			this->fcgBTOK->Click += gcnew System::EventHandler(this, &frmConfig::fcgBTOK_Click);
			// 
			// fcgBTDefault
			// 
			this->fcgBTDefault->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((System::Windows::Forms::AnchorStyles::Bottom | System::Windows::Forms::AnchorStyles::Left));
			this->fcgBTDefault->Location = System::Drawing::Point(9, 548);
			this->fcgBTDefault->Name = L"fcgBTDefault";
			this->fcgBTDefault->Size = System::Drawing::Size(112, 28);
			this->fcgBTDefault->TabIndex = 7;
			this->fcgBTDefault->Text = L"デフォルト";
			this->fcgBTDefault->UseVisualStyleBackColor = true;
			this->fcgBTDefault->Click += gcnew System::EventHandler(this, &frmConfig::fcgBTDefault_Click);
			// 
			// fcgLBVersionDate
			// 
			this->fcgLBVersionDate->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((System::Windows::Forms::AnchorStyles::Bottom | System::Windows::Forms::AnchorStyles::Left));
			this->fcgLBVersionDate->AutoSize = true;
			this->fcgLBVersionDate->Font = (gcnew System::Drawing::Font(L"Meiryo UI", 8.25F, System::Drawing::FontStyle::Italic, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(128)));
			this->fcgLBVersionDate->ForeColor = System::Drawing::SystemColors::ControlDarkDark;
			this->fcgLBVersionDate->Location = System::Drawing::Point(445, 555);
			this->fcgLBVersionDate->Name = L"fcgLBVersionDate";
			this->fcgLBVersionDate->Size = System::Drawing::Size(47, 14);
			this->fcgLBVersionDate->TabIndex = 8;
			this->fcgLBVersionDate->Text = L"Version";
			// 
			// fcgLBVersion
			// 
			this->fcgLBVersion->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((System::Windows::Forms::AnchorStyles::Bottom | System::Windows::Forms::AnchorStyles::Left));
			this->fcgLBVersion->AutoSize = true;
			this->fcgLBVersion->Font = (gcnew System::Drawing::Font(L"Meiryo UI", 8.25F, System::Drawing::FontStyle::Italic, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(128)));
			this->fcgLBVersion->ForeColor = System::Drawing::SystemColors::ControlDarkDark;
			this->fcgLBVersion->Location = System::Drawing::Point(136, 555);
			this->fcgLBVersion->Name = L"fcgLBVersion";
			this->fcgLBVersion->Size = System::Drawing::Size(47, 14);
			this->fcgLBVersion->TabIndex = 9;
			this->fcgLBVersion->Text = L"Version";
			// 
			// fcgOpenFileDialog
			// 
			this->fcgOpenFileDialog->FileName = L"openFileDialog1";
			// 
			// fcgTTEx
			// 
			this->fcgTTEx->AutomaticDelay = 200;
			this->fcgTTEx->AutoPopDelay = 9999;
			this->fcgTTEx->InitialDelay = 200;
			this->fcgTTEx->IsBalloon = true;
			this->fcgTTEx->ReshowDelay = 50;
			this->fcgTTEx->ShowAlways = true;
			this->fcgTTEx->UseAnimation = false;
			this->fcgTTEx->UseFading = false;
			// 
			// fcgtabControlNVEnc
			// 
			this->fcgtabControlNVEnc->Controls->Add(this->tabPageVideoEnc);
			this->fcgtabControlNVEnc->Controls->Add(this->tabPageExOpt);
			this->fcgtabControlNVEnc->Controls->Add(this->tabPageNVEncFeatures);
			this->fcgtabControlNVEnc->Location = System::Drawing::Point(4, 31);
			this->fcgtabControlNVEnc->Name = L"fcgtabControlNVEnc";
			this->fcgtabControlNVEnc->SelectedIndex = 0;
			this->fcgtabControlNVEnc->Size = System::Drawing::Size(616, 509);
			this->fcgtabControlNVEnc->TabIndex = 49;
			// 
			// tabPageVideoEnc
			// 
			this->tabPageVideoEnc->Controls->Add(this->fcgCXAdaptiveTransform);
			this->tabPageVideoEnc->Controls->Add(this->fcgLBAdaptiveTransform);
			this->tabPageVideoEnc->Controls->Add(this->fcgLBFullrange);
			this->tabPageVideoEnc->Controls->Add(this->fcgCBFullrange);
			this->tabPageVideoEnc->Controls->Add(this->fcgCXVideoFormat);
			this->tabPageVideoEnc->Controls->Add(this->fcgLBVideoFormat);
			this->tabPageVideoEnc->Controls->Add(this->fcggroupBoxColor);
			this->tabPageVideoEnc->Controls->Add(this->fcgNURefFrames);
			this->tabPageVideoEnc->Controls->Add(this->fcgLBRefFrames);
			this->tabPageVideoEnc->Controls->Add(this->fcgCXBDirectMode);
			this->tabPageVideoEnc->Controls->Add(this->fcgLBBDirectMode);
			this->tabPageVideoEnc->Controls->Add(this->fcgLBMVPRecision);
			this->tabPageVideoEnc->Controls->Add(this->fcgCXMVPrecision);
			this->tabPageVideoEnc->Controls->Add(this->fcgNUBframes);
			this->tabPageVideoEnc->Controls->Add(this->fcgLBBframes);
			this->tabPageVideoEnc->Controls->Add(this->fcgGroupBoxQulaityStg);
			this->tabPageVideoEnc->Controls->Add(this->fcgLBCABAC);
			this->tabPageVideoEnc->Controls->Add(this->fcgCBDeblock);
			this->tabPageVideoEnc->Controls->Add(this->fcgPNBitrate);
			this->tabPageVideoEnc->Controls->Add(this->fcgPNQP);
			this->tabPageVideoEnc->Controls->Add(this->fcgNUSlices);
			this->tabPageVideoEnc->Controls->Add(this->fcgCBCABAC);
			this->tabPageVideoEnc->Controls->Add(this->fcgLBSlices);
			this->tabPageVideoEnc->Controls->Add(this->fcgLBGOPLengthAuto);
			this->tabPageVideoEnc->Controls->Add(this->fcgGroupBoxAspectRatio);
			this->tabPageVideoEnc->Controls->Add(this->fcgCXInterlaced);
			this->tabPageVideoEnc->Controls->Add(this->fcgLBInterlaced);
			this->tabPageVideoEnc->Controls->Add(this->fcgNUGopLength);
			this->tabPageVideoEnc->Controls->Add(this->fcgLBGOPLength);
			this->tabPageVideoEnc->Controls->Add(this->fcgCXCodecLevel);
			this->tabPageVideoEnc->Controls->Add(this->fcgCXCodecProfile);
			this->tabPageVideoEnc->Controls->Add(this->fcgLBCodecLevel);
			this->tabPageVideoEnc->Controls->Add(this->fcgLBCodecProfile);
			this->tabPageVideoEnc->Controls->Add(this->fcgLBEncMode);
			this->tabPageVideoEnc->Controls->Add(this->fcgCXEncMode);
			this->tabPageVideoEnc->Controls->Add(this->fcgPBNVEncLogoDisabled);
			this->tabPageVideoEnc->Controls->Add(this->fcgPBNVEncLogoEnabled);
			this->tabPageVideoEnc->Font = (gcnew System::Drawing::Font(L"Meiryo UI", 8.25F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(128)));
			this->tabPageVideoEnc->Location = System::Drawing::Point(4, 24);
			this->tabPageVideoEnc->Name = L"tabPageVideoEnc";
			this->tabPageVideoEnc->Padding = System::Windows::Forms::Padding(3);
			this->tabPageVideoEnc->Size = System::Drawing::Size(608, 481);
			this->tabPageVideoEnc->TabIndex = 0;
			this->tabPageVideoEnc->Text = L"動画エンコード";
			this->tabPageVideoEnc->UseVisualStyleBackColor = true;
			// 
			// fcgCXAdaptiveTransform
			// 
			this->fcgCXAdaptiveTransform->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
			this->fcgCXAdaptiveTransform->FormattingEnabled = true;
			this->fcgCXAdaptiveTransform->Location = System::Drawing::Point(131, 351);
			this->fcgCXAdaptiveTransform->Name = L"fcgCXAdaptiveTransform";
			this->fcgCXAdaptiveTransform->Size = System::Drawing::Size(144, 22);
			this->fcgCXAdaptiveTransform->TabIndex = 147;
			this->fcgCXAdaptiveTransform->Tag = L"chValue";
			// 
			// fcgLBAdaptiveTransform
			// 
			this->fcgLBAdaptiveTransform->AutoSize = true;
			this->fcgLBAdaptiveTransform->Location = System::Drawing::Point(13, 354);
			this->fcgLBAdaptiveTransform->Name = L"fcgLBAdaptiveTransform";
			this->fcgLBAdaptiveTransform->Size = System::Drawing::Size(108, 14);
			this->fcgLBAdaptiveTransform->TabIndex = 146;
			this->fcgLBAdaptiveTransform->Text = L"AdaptiveTransform";
			// 
			// fcgLBFullrange
			// 
			this->fcgLBFullrange->AutoSize = true;
			this->fcgLBFullrange->Location = System::Drawing::Point(363, 341);
			this->fcgLBFullrange->Name = L"fcgLBFullrange";
			this->fcgLBFullrange->Size = System::Drawing::Size(55, 14);
			this->fcgLBFullrange->TabIndex = 145;
			this->fcgLBFullrange->Text = L"fullrange";
			// 
			// fcgCBFullrange
			// 
			this->fcgCBFullrange->AutoSize = true;
			this->fcgCBFullrange->Location = System::Drawing::Point(462, 344);
			this->fcgCBFullrange->Name = L"fcgCBFullrange";
			this->fcgCBFullrange->Size = System::Drawing::Size(15, 14);
			this->fcgCBFullrange->TabIndex = 142;
			this->fcgCBFullrange->Tag = L"chValue";
			this->fcgCBFullrange->UseVisualStyleBackColor = true;
			// 
			// fcgCXVideoFormat
			// 
			this->fcgCXVideoFormat->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
			this->fcgCXVideoFormat->FormattingEnabled = true;
			this->fcgCXVideoFormat->Location = System::Drawing::Point(462, 312);
			this->fcgCXVideoFormat->Name = L"fcgCXVideoFormat";
			this->fcgCXVideoFormat->Size = System::Drawing::Size(121, 22);
			this->fcgCXVideoFormat->TabIndex = 141;
			this->fcgCXVideoFormat->Tag = L"chValue";
			// 
			// fcgLBVideoFormat
			// 
			this->fcgLBVideoFormat->AutoSize = true;
			this->fcgLBVideoFormat->Location = System::Drawing::Point(363, 313);
			this->fcgLBVideoFormat->Name = L"fcgLBVideoFormat";
			this->fcgLBVideoFormat->Size = System::Drawing::Size(73, 14);
			this->fcgLBVideoFormat->TabIndex = 144;
			this->fcgLBVideoFormat->Text = L"videoformat";
			// 
			// fcggroupBoxColor
			// 
			this->fcggroupBoxColor->Controls->Add(this->fcgCXTransfer);
			this->fcggroupBoxColor->Controls->Add(this->fcgCXColorPrim);
			this->fcggroupBoxColor->Controls->Add(this->fcgCXColorMatrix);
			this->fcggroupBoxColor->Controls->Add(this->fcgLBTransfer);
			this->fcggroupBoxColor->Controls->Add(this->fcgLBColorPrim);
			this->fcggroupBoxColor->Controls->Add(this->fcgLBColorMatrix);
			this->fcggroupBoxColor->Location = System::Drawing::Point(357, 365);
			this->fcggroupBoxColor->Name = L"fcggroupBoxColor";
			this->fcggroupBoxColor->Size = System::Drawing::Size(241, 103);
			this->fcggroupBoxColor->TabIndex = 143;
			this->fcggroupBoxColor->TabStop = false;
			this->fcggroupBoxColor->Text = L"色設定";
			// 
			// fcgCXTransfer
			// 
			this->fcgCXTransfer->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
			this->fcgCXTransfer->FormattingEnabled = true;
			this->fcgCXTransfer->Location = System::Drawing::Point(105, 72);
			this->fcgCXTransfer->Name = L"fcgCXTransfer";
			this->fcgCXTransfer->Size = System::Drawing::Size(121, 22);
			this->fcgCXTransfer->TabIndex = 2;
			this->fcgCXTransfer->Tag = L"chValue";
			// 
			// fcgCXColorPrim
			// 
			this->fcgCXColorPrim->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
			this->fcgCXColorPrim->FormattingEnabled = true;
			this->fcgCXColorPrim->Location = System::Drawing::Point(105, 44);
			this->fcgCXColorPrim->Name = L"fcgCXColorPrim";
			this->fcgCXColorPrim->Size = System::Drawing::Size(121, 22);
			this->fcgCXColorPrim->TabIndex = 1;
			this->fcgCXColorPrim->Tag = L"chValue";
			// 
			// fcgCXColorMatrix
			// 
			this->fcgCXColorMatrix->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
			this->fcgCXColorMatrix->FormattingEnabled = true;
			this->fcgCXColorMatrix->Location = System::Drawing::Point(105, 16);
			this->fcgCXColorMatrix->Name = L"fcgCXColorMatrix";
			this->fcgCXColorMatrix->Size = System::Drawing::Size(121, 22);
			this->fcgCXColorMatrix->TabIndex = 0;
			this->fcgCXColorMatrix->Tag = L"chValue";
			// 
			// fcgLBTransfer
			// 
			this->fcgLBTransfer->AutoSize = true;
			this->fcgLBTransfer->Location = System::Drawing::Point(18, 75);
			this->fcgLBTransfer->Name = L"fcgLBTransfer";
			this->fcgLBTransfer->Size = System::Drawing::Size(49, 14);
			this->fcgLBTransfer->TabIndex = 2;
			this->fcgLBTransfer->Text = L"transfer";
			// 
			// fcgLBColorPrim
			// 
			this->fcgLBColorPrim->AutoSize = true;
			this->fcgLBColorPrim->Location = System::Drawing::Point(18, 47);
			this->fcgLBColorPrim->Name = L"fcgLBColorPrim";
			this->fcgLBColorPrim->Size = System::Drawing::Size(61, 14);
			this->fcgLBColorPrim->TabIndex = 1;
			this->fcgLBColorPrim->Text = L"colorprim";
			// 
			// fcgLBColorMatrix
			// 
			this->fcgLBColorMatrix->AutoSize = true;
			this->fcgLBColorMatrix->Location = System::Drawing::Point(18, 19);
			this->fcgLBColorMatrix->Name = L"fcgLBColorMatrix";
			this->fcgLBColorMatrix->Size = System::Drawing::Size(70, 14);
			this->fcgLBColorMatrix->TabIndex = 0;
			this->fcgLBColorMatrix->Text = L"colormatrix";
			// 
			// fcgNURefFrames
			// 
			this->fcgNURefFrames->Location = System::Drawing::Point(132, 320);
			this->fcgNURefFrames->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 16, 0, 0, 0 });
			this->fcgNURefFrames->Name = L"fcgNURefFrames";
			this->fcgNURefFrames->Size = System::Drawing::Size(77, 21);
			this->fcgNURefFrames->TabIndex = 139;
			this->fcgNURefFrames->Tag = L"chValue";
			this->fcgNURefFrames->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
			// 
			// fcgLBRefFrames
			// 
			this->fcgLBRefFrames->AutoSize = true;
			this->fcgLBRefFrames->Location = System::Drawing::Point(14, 323);
			this->fcgLBRefFrames->Name = L"fcgLBRefFrames";
			this->fcgLBRefFrames->Size = System::Drawing::Size(51, 14);
			this->fcgLBRefFrames->TabIndex = 140;
			this->fcgLBRefFrames->Text = L"参照距離";
			// 
			// fcgCXBDirectMode
			// 
			this->fcgCXBDirectMode->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
			this->fcgCXBDirectMode->FormattingEnabled = true;
			this->fcgCXBDirectMode->Location = System::Drawing::Point(131, 288);
			this->fcgCXBDirectMode->Name = L"fcgCXBDirectMode";
			this->fcgCXBDirectMode->Size = System::Drawing::Size(144, 22);
			this->fcgCXBDirectMode->TabIndex = 138;
			this->fcgCXBDirectMode->Tag = L"chValue";
			// 
			// fcgLBBDirectMode
			// 
			this->fcgLBBDirectMode->AutoSize = true;
			this->fcgLBBDirectMode->Location = System::Drawing::Point(13, 291);
			this->fcgLBBDirectMode->Name = L"fcgLBBDirectMode";
			this->fcgLBBDirectMode->Size = System::Drawing::Size(70, 14);
			this->fcgLBBDirectMode->TabIndex = 137;
			this->fcgLBBDirectMode->Text = L"動き予測方式";
			// 
			// fcgLBMVPRecision
			// 
			this->fcgLBMVPRecision->AutoSize = true;
			this->fcgLBMVPRecision->Location = System::Drawing::Point(14, 259);
			this->fcgLBMVPRecision->Name = L"fcgLBMVPRecision";
			this->fcgLBMVPRecision->Size = System::Drawing::Size(70, 14);
			this->fcgLBMVPRecision->TabIndex = 135;
			this->fcgLBMVPRecision->Text = L"動き探索精度";
			// 
			// fcgCXMVPrecision
			// 
			this->fcgCXMVPrecision->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
			this->fcgCXMVPrecision->FormattingEnabled = true;
			this->fcgCXMVPrecision->Location = System::Drawing::Point(131, 256);
			this->fcgCXMVPrecision->Name = L"fcgCXMVPrecision";
			this->fcgCXMVPrecision->Size = System::Drawing::Size(144, 22);
			this->fcgCXMVPrecision->TabIndex = 134;
			this->fcgCXMVPrecision->Tag = L"chValue";
			// 
			// fcgNUBframes
			// 
			this->fcgNUBframes->Location = System::Drawing::Point(132, 226);
			this->fcgNUBframes->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 16, 0, 0, 0 });
			this->fcgNUBframes->Name = L"fcgNUBframes";
			this->fcgNUBframes->Size = System::Drawing::Size(77, 21);
			this->fcgNUBframes->TabIndex = 132;
			this->fcgNUBframes->Tag = L"chValue";
			this->fcgNUBframes->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
			// 
			// fcgLBBframes
			// 
			this->fcgLBBframes->AutoSize = true;
			this->fcgLBBframes->Location = System::Drawing::Point(14, 229);
			this->fcgLBBframes->Name = L"fcgLBBframes";
			this->fcgLBBframes->Size = System::Drawing::Size(58, 14);
			this->fcgLBBframes->TabIndex = 133;
			this->fcgLBBframes->Text = L"Bフレーム数";
			// 
			// fcgGroupBoxQulaityStg
			// 
			this->fcgGroupBoxQulaityStg->Controls->Add(this->fcgBTQualityStg);
			this->fcgGroupBoxQulaityStg->Controls->Add(this->fcgCXQualityPreset);
			this->fcgGroupBoxQulaityStg->Location = System::Drawing::Point(6, 383);
			this->fcgGroupBoxQulaityStg->Name = L"fcgGroupBoxQulaityStg";
			this->fcgGroupBoxQulaityStg->Size = System::Drawing::Size(219, 92);
			this->fcgGroupBoxQulaityStg->TabIndex = 28;
			this->fcgGroupBoxQulaityStg->TabStop = false;
			this->fcgGroupBoxQulaityStg->Text = L"品質設定";
			this->fcgGroupBoxQulaityStg->Visible = false;
			// 
			// fcgBTQualityStg
			// 
			this->fcgBTQualityStg->Location = System::Drawing::Point(106, 52);
			this->fcgBTQualityStg->Name = L"fcgBTQualityStg";
			this->fcgBTQualityStg->Size = System::Drawing::Size(97, 28);
			this->fcgBTQualityStg->TabIndex = 7;
			this->fcgBTQualityStg->Text = L"ロードして反映";
			this->fcgBTQualityStg->UseVisualStyleBackColor = true;
			this->fcgBTQualityStg->Click += gcnew System::EventHandler(this, &frmConfig::fcgBTQualityStg_Click);
			// 
			// fcgCXQualityPreset
			// 
			this->fcgCXQualityPreset->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
			this->fcgCXQualityPreset->FormattingEnabled = true;
			this->fcgCXQualityPreset->Location = System::Drawing::Point(26, 24);
			this->fcgCXQualityPreset->Name = L"fcgCXQualityPreset";
			this->fcgCXQualityPreset->Size = System::Drawing::Size(174, 22);
			this->fcgCXQualityPreset->TabIndex = 0;
			this->fcgCXQualityPreset->Tag = L"chValue";
			// 
			// fcgLBCABAC
			// 
			this->fcgLBCABAC->AutoSize = true;
			this->fcgLBCABAC->Location = System::Drawing::Point(360, 242);
			this->fcgLBCABAC->Name = L"fcgLBCABAC";
			this->fcgLBCABAC->Size = System::Drawing::Size(42, 14);
			this->fcgLBCABAC->TabIndex = 131;
			this->fcgLBCABAC->Text = L"CABAC";
			// 
			// fcgCBDeblock
			// 
			this->fcgCBDeblock->AutoSize = true;
			this->fcgCBDeblock->Location = System::Drawing::Point(374, 279);
			this->fcgCBDeblock->Name = L"fcgCBDeblock";
			this->fcgCBDeblock->Size = System::Drawing::Size(141, 18);
			this->fcgCBDeblock->TabIndex = 25;
			this->fcgCBDeblock->Tag = L"chValue";
			this->fcgCBDeblock->Text = L"インループ デブロックフィルタ";
			this->fcgCBDeblock->UseVisualStyleBackColor = true;
			this->fcgCBDeblock->Visible = false;
			// 
			// fcgPNBitrate
			// 
			this->fcgPNBitrate->Controls->Add(this->fcgLBBitrate);
			this->fcgPNBitrate->Controls->Add(this->fcgNUBitrate);
			this->fcgPNBitrate->Controls->Add(this->fcgLBBitrate2);
			this->fcgPNBitrate->Controls->Add(this->fcgNUMaxkbps);
			this->fcgPNBitrate->Controls->Add(this->fcgLBMaxkbps);
			this->fcgPNBitrate->Controls->Add(this->fcgLBMaxBitrate2);
			this->fcgPNBitrate->Location = System::Drawing::Point(8, 114);
			this->fcgPNBitrate->Name = L"fcgPNBitrate";
			this->fcgPNBitrate->Size = System::Drawing::Size(289, 54);
			this->fcgPNBitrate->TabIndex = 114;
			// 
			// fcgLBBitrate
			// 
			this->fcgLBBitrate->AutoSize = true;
			this->fcgLBBitrate->Location = System::Drawing::Point(5, 4);
			this->fcgLBBitrate->Name = L"fcgLBBitrate";
			this->fcgLBBitrate->Size = System::Drawing::Size(54, 14);
			this->fcgLBBitrate->TabIndex = 66;
			this->fcgLBBitrate->Text = L"ビットレート";
			// 
			// fcgNUBitrate
			// 
			this->fcgNUBitrate->Location = System::Drawing::Point(124, 2);
			this->fcgNUBitrate->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 65535, 0, 0, 0 });
			this->fcgNUBitrate->Name = L"fcgNUBitrate";
			this->fcgNUBitrate->Size = System::Drawing::Size(77, 21);
			this->fcgNUBitrate->TabIndex = 5;
			this->fcgNUBitrate->Tag = L"chValue";
			this->fcgNUBitrate->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
			// 
			// fcgLBBitrate2
			// 
			this->fcgLBBitrate2->AutoSize = true;
			this->fcgLBBitrate2->Location = System::Drawing::Point(207, 4);
			this->fcgLBBitrate2->Name = L"fcgLBBitrate2";
			this->fcgLBBitrate2->Size = System::Drawing::Size(32, 14);
			this->fcgLBBitrate2->TabIndex = 69;
			this->fcgLBBitrate2->Text = L"kbps";
			// 
			// fcgNUMaxkbps
			// 
			this->fcgNUMaxkbps->Location = System::Drawing::Point(124, 29);
			this->fcgNUMaxkbps->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 65535, 0, 0, 0 });
			this->fcgNUMaxkbps->Name = L"fcgNUMaxkbps";
			this->fcgNUMaxkbps->Size = System::Drawing::Size(77, 21);
			this->fcgNUMaxkbps->TabIndex = 6;
			this->fcgNUMaxkbps->Tag = L"chValue";
			this->fcgNUMaxkbps->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
			// 
			// fcgLBMaxkbps
			// 
			this->fcgLBMaxkbps->AutoSize = true;
			this->fcgLBMaxkbps->Location = System::Drawing::Point(5, 33);
			this->fcgLBMaxkbps->Name = L"fcgLBMaxkbps";
			this->fcgLBMaxkbps->Size = System::Drawing::Size(76, 14);
			this->fcgLBMaxkbps->TabIndex = 78;
			this->fcgLBMaxkbps->Text = L"最大ビットレート";
			// 
			// fcgLBMaxBitrate2
			// 
			this->fcgLBMaxBitrate2->AutoSize = true;
			this->fcgLBMaxBitrate2->Location = System::Drawing::Point(207, 31);
			this->fcgLBMaxBitrate2->Name = L"fcgLBMaxBitrate2";
			this->fcgLBMaxBitrate2->Size = System::Drawing::Size(32, 14);
			this->fcgLBMaxBitrate2->TabIndex = 80;
			this->fcgLBMaxBitrate2->Text = L"kbps";
			// 
			// fcgPNQP
			// 
			this->fcgPNQP->Controls->Add(this->fcgLBQPI);
			this->fcgPNQP->Controls->Add(this->fcgNUQPI);
			this->fcgPNQP->Controls->Add(this->fcgNUQPP);
			this->fcgPNQP->Controls->Add(this->fcgNUQPB);
			this->fcgPNQP->Controls->Add(this->fcgLBQPP);
			this->fcgPNQP->Controls->Add(this->fcgLBQPB);
			this->fcgPNQP->Location = System::Drawing::Point(8, 114);
			this->fcgPNQP->Name = L"fcgPNQP";
			this->fcgPNQP->Size = System::Drawing::Size(289, 79);
			this->fcgPNQP->TabIndex = 113;
			// 
			// fcgLBQPI
			// 
			this->fcgLBQPI->AutoSize = true;
			this->fcgLBQPI->Location = System::Drawing::Point(10, 4);
			this->fcgLBQPI->Name = L"fcgLBQPI";
			this->fcgLBQPI->Size = System::Drawing::Size(66, 14);
			this->fcgLBQPI->TabIndex = 75;
			this->fcgLBQPI->Text = L"QP I frame";
			// 
			// fcgNUQPI
			// 
			this->fcgNUQPI->Location = System::Drawing::Point(124, 2);
			this->fcgNUQPI->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 51, 0, 0, 0 });
			this->fcgNUQPI->Name = L"fcgNUQPI";
			this->fcgNUQPI->Size = System::Drawing::Size(77, 21);
			this->fcgNUQPI->TabIndex = 7;
			this->fcgNUQPI->Tag = L"chValue";
			this->fcgNUQPI->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
			// 
			// fcgNUQPP
			// 
			this->fcgNUQPP->Location = System::Drawing::Point(124, 29);
			this->fcgNUQPP->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 51, 0, 0, 0 });
			this->fcgNUQPP->Name = L"fcgNUQPP";
			this->fcgNUQPP->Size = System::Drawing::Size(77, 21);
			this->fcgNUQPP->TabIndex = 8;
			this->fcgNUQPP->Tag = L"chValue";
			this->fcgNUQPP->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
			// 
			// fcgNUQPB
			// 
			this->fcgNUQPB->Location = System::Drawing::Point(124, 55);
			this->fcgNUQPB->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 51, 0, 0, 0 });
			this->fcgNUQPB->Name = L"fcgNUQPB";
			this->fcgNUQPB->Size = System::Drawing::Size(77, 21);
			this->fcgNUQPB->TabIndex = 9;
			this->fcgNUQPB->Tag = L"chValue";
			this->fcgNUQPB->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
			// 
			// fcgLBQPP
			// 
			this->fcgLBQPP->AutoSize = true;
			this->fcgLBQPP->Location = System::Drawing::Point(10, 31);
			this->fcgLBQPP->Name = L"fcgLBQPP";
			this->fcgLBQPP->Size = System::Drawing::Size(69, 14);
			this->fcgLBQPP->TabIndex = 76;
			this->fcgLBQPP->Text = L"QP P frame";
			// 
			// fcgLBQPB
			// 
			this->fcgLBQPB->AutoSize = true;
			this->fcgLBQPB->Location = System::Drawing::Point(6, 57);
			this->fcgLBQPB->Name = L"fcgLBQPB";
			this->fcgLBQPB->Size = System::Drawing::Size(74, 14);
			this->fcgLBQPB->TabIndex = 77;
			this->fcgLBQPB->Text = L"QP B frames";
			// 
			// fcgNUSlices
			// 
			this->fcgNUSlices->Location = System::Drawing::Point(460, 206);
			this->fcgNUSlices->Minimum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 1, 0, 0, 0 });
			this->fcgNUSlices->Name = L"fcgNUSlices";
			this->fcgNUSlices->Size = System::Drawing::Size(70, 21);
			this->fcgNUSlices->TabIndex = 23;
			this->fcgNUSlices->Tag = L"chValue";
			this->fcgNUSlices->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
			this->fcgNUSlices->Value = System::Decimal(gcnew cli::array< System::Int32 >(4) { 1, 0, 0, 0 });
			// 
			// fcgCBCABAC
			// 
			this->fcgCBCABAC->AutoSize = true;
			this->fcgCBCABAC->Location = System::Drawing::Point(460, 243);
			this->fcgCBCABAC->Name = L"fcgCBCABAC";
			this->fcgCBCABAC->Size = System::Drawing::Size(15, 14);
			this->fcgCBCABAC->TabIndex = 24;
			this->fcgCBCABAC->Tag = L"chValue";
			this->fcgCBCABAC->UseVisualStyleBackColor = true;
			// 
			// fcgLBSlices
			// 
			this->fcgLBSlices->AutoSize = true;
			this->fcgLBSlices->Location = System::Drawing::Point(360, 208);
			this->fcgLBSlices->Name = L"fcgLBSlices";
			this->fcgLBSlices->Size = System::Drawing::Size(50, 14);
			this->fcgLBSlices->TabIndex = 103;
			this->fcgLBSlices->Text = L"スライス数";
			// 
			// fcgLBGOPLengthAuto
			// 
			this->fcgLBGOPLengthAuto->AutoSize = true;
			this->fcgLBGOPLengthAuto->Location = System::Drawing::Point(214, 203);
			this->fcgLBGOPLengthAuto->Name = L"fcgLBGOPLengthAuto";
			this->fcgLBGOPLengthAuto->Size = System::Drawing::Size(66, 14);
			this->fcgLBGOPLengthAuto->TabIndex = 101;
			this->fcgLBGOPLengthAuto->Text = L"※\"0\"で自動";
			// 
			// fcgGroupBoxAspectRatio
			// 
			this->fcgGroupBoxAspectRatio->Controls->Add(this->fcgLBAspectRatio);
			this->fcgGroupBoxAspectRatio->Controls->Add(this->fcgNUAspectRatioY);
			this->fcgGroupBoxAspectRatio->Controls->Add(this->fcgNUAspectRatioX);
			this->fcgGroupBoxAspectRatio->Controls->Add(this->fcgCXAspectRatio);
			this->fcgGroupBoxAspectRatio->Location = System::Drawing::Point(357, 9);
			this->fcgGroupBoxAspectRatio->Name = L"fcgGroupBoxAspectRatio";
			this->fcgGroupBoxAspectRatio->Size = System::Drawing::Size(241, 92);
			this->fcgGroupBoxAspectRatio->TabIndex = 27;
			this->fcgGroupBoxAspectRatio->TabStop = false;
			this->fcgGroupBoxAspectRatio->Text = L"アスペクト比";
			this->fcgGroupBoxAspectRatio->Visible = false;
			// 
			// fcgLBAspectRatio
			// 
			this->fcgLBAspectRatio->AutoSize = true;
			this->fcgLBAspectRatio->Location = System::Drawing::Point(131, 58);
			this->fcgLBAspectRatio->Name = L"fcgLBAspectRatio";
			this->fcgLBAspectRatio->Size = System::Drawing::Size(12, 14);
			this->fcgLBAspectRatio->TabIndex = 3;
			this->fcgLBAspectRatio->Text = L":";
			// 
			// fcgNUAspectRatioY
			// 
			this->fcgNUAspectRatioY->Location = System::Drawing::Point(149, 56);
			this->fcgNUAspectRatioY->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 65535, 0, 0, 0 });
			this->fcgNUAspectRatioY->Name = L"fcgNUAspectRatioY";
			this->fcgNUAspectRatioY->Size = System::Drawing::Size(60, 21);
			this->fcgNUAspectRatioY->TabIndex = 2;
			this->fcgNUAspectRatioY->Tag = L"chValue";
			this->fcgNUAspectRatioY->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
			// 
			// fcgNUAspectRatioX
			// 
			this->fcgNUAspectRatioX->Location = System::Drawing::Point(65, 56);
			this->fcgNUAspectRatioX->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 65535, 0, 0, 0 });
			this->fcgNUAspectRatioX->Name = L"fcgNUAspectRatioX";
			this->fcgNUAspectRatioX->Size = System::Drawing::Size(60, 21);
			this->fcgNUAspectRatioX->TabIndex = 1;
			this->fcgNUAspectRatioX->Tag = L"chValue";
			this->fcgNUAspectRatioX->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
			// 
			// fcgCXAspectRatio
			// 
			this->fcgCXAspectRatio->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
			this->fcgCXAspectRatio->FormattingEnabled = true;
			this->fcgCXAspectRatio->Location = System::Drawing::Point(26, 24);
			this->fcgCXAspectRatio->Name = L"fcgCXAspectRatio";
			this->fcgCXAspectRatio->Size = System::Drawing::Size(197, 22);
			this->fcgCXAspectRatio->TabIndex = 0;
			this->fcgCXAspectRatio->Tag = L"chValue";
			// 
			// fcgCXInterlaced
			// 
			this->fcgCXInterlaced->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
			this->fcgCXInterlaced->FormattingEnabled = true;
			this->fcgCXInterlaced->Location = System::Drawing::Point(459, 108);
			this->fcgCXInterlaced->Name = L"fcgCXInterlaced";
			this->fcgCXInterlaced->Size = System::Drawing::Size(121, 22);
			this->fcgCXInterlaced->TabIndex = 20;
			this->fcgCXInterlaced->Tag = L"chValue";
			// 
			// fcgLBInterlaced
			// 
			this->fcgLBInterlaced->AutoSize = true;
			this->fcgLBInterlaced->Location = System::Drawing::Point(360, 111);
			this->fcgLBInterlaced->Name = L"fcgLBInterlaced";
			this->fcgLBInterlaced->Size = System::Drawing::Size(64, 14);
			this->fcgLBInterlaced->TabIndex = 86;
			this->fcgLBInterlaced->Text = L"フレームタイプ";
			// 
			// fcgNUGopLength
			// 
			this->fcgNUGopLength->Location = System::Drawing::Point(132, 200);
			this->fcgNUGopLength->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 120000, 0, 0, 0 });
			this->fcgNUGopLength->Name = L"fcgNUGopLength";
			this->fcgNUGopLength->Size = System::Drawing::Size(77, 21);
			this->fcgNUGopLength->TabIndex = 10;
			this->fcgNUGopLength->Tag = L"chValue";
			this->fcgNUGopLength->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
			// 
			// fcgLBGOPLength
			// 
			this->fcgLBGOPLength->AutoSize = true;
			this->fcgLBGOPLength->Location = System::Drawing::Point(14, 203);
			this->fcgLBGOPLength->Name = L"fcgLBGOPLength";
			this->fcgLBGOPLength->Size = System::Drawing::Size(41, 14);
			this->fcgLBGOPLength->TabIndex = 85;
			this->fcgLBGOPLength->Text = L"GOP長";
			// 
			// fcgCXCodecLevel
			// 
			this->fcgCXCodecLevel->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
			this->fcgCXCodecLevel->FormattingEnabled = true;
			this->fcgCXCodecLevel->Location = System::Drawing::Point(459, 172);
			this->fcgCXCodecLevel->Name = L"fcgCXCodecLevel";
			this->fcgCXCodecLevel->Size = System::Drawing::Size(121, 22);
			this->fcgCXCodecLevel->TabIndex = 22;
			this->fcgCXCodecLevel->Tag = L"chValue";
			// 
			// fcgCXCodecProfile
			// 
			this->fcgCXCodecProfile->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
			this->fcgCXCodecProfile->FormattingEnabled = true;
			this->fcgCXCodecProfile->Location = System::Drawing::Point(459, 140);
			this->fcgCXCodecProfile->Name = L"fcgCXCodecProfile";
			this->fcgCXCodecProfile->Size = System::Drawing::Size(121, 22);
			this->fcgCXCodecProfile->TabIndex = 21;
			this->fcgCXCodecProfile->Tag = L"chValue";
			// 
			// fcgLBCodecLevel
			// 
			this->fcgLBCodecLevel->AutoSize = true;
			this->fcgLBCodecLevel->Location = System::Drawing::Point(360, 175);
			this->fcgLBCodecLevel->Name = L"fcgLBCodecLevel";
			this->fcgLBCodecLevel->Size = System::Drawing::Size(33, 14);
			this->fcgLBCodecLevel->TabIndex = 84;
			this->fcgLBCodecLevel->Text = L"レベル";
			// 
			// fcgLBCodecProfile
			// 
			this->fcgLBCodecProfile->AutoSize = true;
			this->fcgLBCodecProfile->Location = System::Drawing::Point(360, 143);
			this->fcgLBCodecProfile->Name = L"fcgLBCodecProfile";
			this->fcgLBCodecProfile->Size = System::Drawing::Size(53, 14);
			this->fcgLBCodecProfile->TabIndex = 83;
			this->fcgLBCodecProfile->Text = L"プロファイル";
			// 
			// fcgLBEncMode
			// 
			this->fcgLBEncMode->AutoSize = true;
			this->fcgLBEncMode->Location = System::Drawing::Point(13, 87);
			this->fcgLBEncMode->Name = L"fcgLBEncMode";
			this->fcgLBEncMode->Size = System::Drawing::Size(32, 14);
			this->fcgLBEncMode->TabIndex = 79;
			this->fcgLBEncMode->Text = L"モード";
			// 
			// fcgCXEncMode
			// 
			this->fcgCXEncMode->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
			this->fcgCXEncMode->FormattingEnabled = true;
			this->fcgCXEncMode->Items->AddRange(gcnew cli::array< System::Object^  >(3) { L"高品質", L"標準", L"高速" });
			this->fcgCXEncMode->Location = System::Drawing::Point(81, 84);
			this->fcgCXEncMode->Name = L"fcgCXEncMode";
			this->fcgCXEncMode->Size = System::Drawing::Size(160, 22);
			this->fcgCXEncMode->TabIndex = 4;
			this->fcgCXEncMode->Tag = L"chValue";
			this->fcgCXEncMode->SelectedIndexChanged += gcnew System::EventHandler(this, &frmConfig::fcgChangeEnabled);
			// 
			// tabPageExOpt
			// 
			this->tabPageExOpt->Controls->Add(this->fcgCBAuoTcfileout);
			this->tabPageExOpt->Controls->Add(this->fcgCBAFS);
			this->tabPageExOpt->Controls->Add(this->fcgLBTempDir);
			this->tabPageExOpt->Controls->Add(this->fcgBTCustomTempDir);
			this->tabPageExOpt->Controls->Add(this->fcgTXCustomTempDir);
			this->tabPageExOpt->Controls->Add(this->fcgCXTempDir);
			this->tabPageExOpt->Font = (gcnew System::Drawing::Font(L"Meiryo UI", 8.25F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(128)));
			this->tabPageExOpt->Location = System::Drawing::Point(4, 24);
			this->tabPageExOpt->Name = L"tabPageExOpt";
			this->tabPageExOpt->Size = System::Drawing::Size(608, 481);
			this->tabPageExOpt->TabIndex = 1;
			this->tabPageExOpt->Text = L"その他";
			this->tabPageExOpt->UseVisualStyleBackColor = true;
			// 
			// fcgCBAuoTcfileout
			// 
			this->fcgCBAuoTcfileout->AutoSize = true;
			this->fcgCBAuoTcfileout->Location = System::Drawing::Point(301, 72);
			this->fcgCBAuoTcfileout->Name = L"fcgCBAuoTcfileout";
			this->fcgCBAuoTcfileout->Size = System::Drawing::Size(98, 18);
			this->fcgCBAuoTcfileout->TabIndex = 73;
			this->fcgCBAuoTcfileout->Tag = L"chValue";
			this->fcgCBAuoTcfileout->Text = L"タイムコード出力";
			this->fcgCBAuoTcfileout->UseVisualStyleBackColor = true;
			// 
			// fcgCBAFS
			// 
			this->fcgCBAFS->AutoSize = true;
			this->fcgCBAFS->Location = System::Drawing::Point(301, 39);
			this->fcgCBAFS->Name = L"fcgCBAFS";
			this->fcgCBAFS->Size = System::Drawing::Size(183, 18);
			this->fcgCBAFS->TabIndex = 72;
			this->fcgCBAFS->Tag = L"chValue";
			this->fcgCBAFS->Text = L"自動フィールドシフト(afs)を使用する";
			this->fcgCBAFS->UseVisualStyleBackColor = true;
			// 
			// fcgLBTempDir
			// 
			this->fcgLBTempDir->AutoSize = true;
			this->fcgLBTempDir->Location = System::Drawing::Point(15, 9);
			this->fcgLBTempDir->Name = L"fcgLBTempDir";
			this->fcgLBTempDir->Size = System::Drawing::Size(60, 14);
			this->fcgLBTempDir->TabIndex = 67;
			this->fcgLBTempDir->Text = L"一時フォルダ";
			// 
			// fcgBTCustomTempDir
			// 
			this->fcgBTCustomTempDir->Location = System::Drawing::Point(215, 62);
			this->fcgBTCustomTempDir->Name = L"fcgBTCustomTempDir";
			this->fcgBTCustomTempDir->Size = System::Drawing::Size(29, 23);
			this->fcgBTCustomTempDir->TabIndex = 64;
			this->fcgBTCustomTempDir->Text = L"...";
			this->fcgBTCustomTempDir->UseVisualStyleBackColor = true;
			this->fcgBTCustomTempDir->Click += gcnew System::EventHandler(this, &frmConfig::fcgBTCustomTempDir_Click);
			// 
			// fcgTXCustomTempDir
			// 
			this->fcgTXCustomTempDir->Location = System::Drawing::Point(30, 63);
			this->fcgTXCustomTempDir->Name = L"fcgTXCustomTempDir";
			this->fcgTXCustomTempDir->Size = System::Drawing::Size(182, 21);
			this->fcgTXCustomTempDir->TabIndex = 63;
			this->fcgTXCustomTempDir->Tag = L"";
			this->fcgTXCustomTempDir->TextChanged += gcnew System::EventHandler(this, &frmConfig::fcgTXCustomTempDir_TextChanged);
			// 
			// fcgCXTempDir
			// 
			this->fcgCXTempDir->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
			this->fcgCXTempDir->FormattingEnabled = true;
			this->fcgCXTempDir->Location = System::Drawing::Point(18, 35);
			this->fcgCXTempDir->Name = L"fcgCXTempDir";
			this->fcgCXTempDir->Size = System::Drawing::Size(209, 22);
			this->fcgCXTempDir->TabIndex = 62;
			this->fcgCXTempDir->Tag = L"chValue";
			// 
			// tabPageNVEncFeatures
			// 
			this->tabPageNVEncFeatures->Controls->Add(this->fcgLBOSInfo);
			this->tabPageNVEncFeatures->Controls->Add(this->fcgLBOSInfoLabel);
			this->tabPageNVEncFeatures->Controls->Add(this->fcgLBCPUInfoOnFeatureTab);
			this->tabPageNVEncFeatures->Controls->Add(this->fcgLBCPUInfoLabelOnFeatureTab);
			this->tabPageNVEncFeatures->Controls->Add(this->label2);
			this->tabPageNVEncFeatures->Controls->Add(this->fcgLBGPUInfoOnFeatureTab);
			this->tabPageNVEncFeatures->Controls->Add(this->fcgLBGPUInfoLabelOnFeatureTab);
			this->tabPageNVEncFeatures->Controls->Add(this->fcgDGVFeatures);
			this->tabPageNVEncFeatures->Location = System::Drawing::Point(4, 24);
			this->tabPageNVEncFeatures->Name = L"tabPageNVEncFeatures";
			this->tabPageNVEncFeatures->Size = System::Drawing::Size(608, 481);
			this->tabPageNVEncFeatures->TabIndex = 2;
			this->tabPageNVEncFeatures->Text = L"情報";
			this->tabPageNVEncFeatures->UseVisualStyleBackColor = true;
			// 
			// fcgLBOSInfo
			// 
			this->fcgLBOSInfo->AutoSize = true;
			this->fcgLBOSInfo->Font = (gcnew System::Drawing::Font(L"Meiryo UI", 9.75F, System::Drawing::FontStyle::Italic, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(128)));
			this->fcgLBOSInfo->ForeColor = System::Drawing::Color::DarkViolet;
			this->fcgLBOSInfo->Location = System::Drawing::Point(96, 51);
			this->fcgLBOSInfo->Name = L"fcgLBOSInfo";
			this->fcgLBOSInfo->Size = System::Drawing::Size(26, 17);
			this->fcgLBOSInfo->TabIndex = 121;
			this->fcgLBOSInfo->Text = L"OS";
			// 
			// fcgLBOSInfoLabel
			// 
			this->fcgLBOSInfoLabel->AutoSize = true;
			this->fcgLBOSInfoLabel->Font = (gcnew System::Drawing::Font(L"Meiryo UI", 11.25F, System::Drawing::FontStyle::Italic, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(128)));
			this->fcgLBOSInfoLabel->ForeColor = System::Drawing::Color::Blue;
			this->fcgLBOSInfoLabel->Location = System::Drawing::Point(33, 49);
			this->fcgLBOSInfoLabel->Name = L"fcgLBOSInfoLabel";
			this->fcgLBOSInfoLabel->Size = System::Drawing::Size(30, 19);
			this->fcgLBOSInfoLabel->TabIndex = 120;
			this->fcgLBOSInfoLabel->Text = L"OS";
			// 
			// fcgLBCPUInfoOnFeatureTab
			// 
			this->fcgLBCPUInfoOnFeatureTab->AutoSize = true;
			this->fcgLBCPUInfoOnFeatureTab->Font = (gcnew System::Drawing::Font(L"Meiryo UI", 9.75F, System::Drawing::FontStyle::Italic, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(128)));
			this->fcgLBCPUInfoOnFeatureTab->ForeColor = System::Drawing::Color::DarkViolet;
			this->fcgLBCPUInfoOnFeatureTab->Location = System::Drawing::Point(96, 78);
			this->fcgLBCPUInfoOnFeatureTab->Name = L"fcgLBCPUInfoOnFeatureTab";
			this->fcgLBCPUInfoOnFeatureTab->Size = System::Drawing::Size(34, 17);
			this->fcgLBCPUInfoOnFeatureTab->TabIndex = 119;
			this->fcgLBCPUInfoOnFeatureTab->Text = L"CPU";
			// 
			// fcgLBCPUInfoLabelOnFeatureTab
			// 
			this->fcgLBCPUInfoLabelOnFeatureTab->AutoSize = true;
			this->fcgLBCPUInfoLabelOnFeatureTab->Font = (gcnew System::Drawing::Font(L"Meiryo UI", 11.25F, System::Drawing::FontStyle::Italic,
				System::Drawing::GraphicsUnit::Point, static_cast<System::Byte>(128)));
			this->fcgLBCPUInfoLabelOnFeatureTab->ForeColor = System::Drawing::Color::Blue;
			this->fcgLBCPUInfoLabelOnFeatureTab->Location = System::Drawing::Point(32, 76);
			this->fcgLBCPUInfoLabelOnFeatureTab->Name = L"fcgLBCPUInfoLabelOnFeatureTab";
			this->fcgLBCPUInfoLabelOnFeatureTab->Size = System::Drawing::Size(39, 19);
			this->fcgLBCPUInfoLabelOnFeatureTab->TabIndex = 118;
			this->fcgLBCPUInfoLabelOnFeatureTab->Text = L"CPU";
			// 
			// label2
			// 
			this->label2->AutoSize = true;
			this->label2->Font = (gcnew System::Drawing::Font(L"Meiryo UI", 11.25F, System::Drawing::FontStyle::Italic, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(128)));
			this->label2->ForeColor = System::Drawing::SystemColors::ControlDarkDark;
			this->label2->Location = System::Drawing::Point(14, 15);
			this->label2->Name = L"label2";
			this->label2->Size = System::Drawing::Size(314, 19);
			this->label2->TabIndex = 117;
			this->label2->Text = L"現在の環境でサポートされる機能を表示しています。";
			// 
			// fcgLBGPUInfoOnFeatureTab
			// 
			this->fcgLBGPUInfoOnFeatureTab->AutoSize = true;
			this->fcgLBGPUInfoOnFeatureTab->Font = (gcnew System::Drawing::Font(L"Meiryo UI", 9.75F, System::Drawing::FontStyle::Italic, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(128)));
			this->fcgLBGPUInfoOnFeatureTab->ForeColor = System::Drawing::Color::DarkViolet;
			this->fcgLBGPUInfoOnFeatureTab->Location = System::Drawing::Point(96, 105);
			this->fcgLBGPUInfoOnFeatureTab->Name = L"fcgLBGPUInfoOnFeatureTab";
			this->fcgLBGPUInfoOnFeatureTab->Size = System::Drawing::Size(35, 17);
			this->fcgLBGPUInfoOnFeatureTab->TabIndex = 116;
			this->fcgLBGPUInfoOnFeatureTab->Text = L"GPU";
			// 
			// fcgLBGPUInfoLabelOnFeatureTab
			// 
			this->fcgLBGPUInfoLabelOnFeatureTab->AutoSize = true;
			this->fcgLBGPUInfoLabelOnFeatureTab->Font = (gcnew System::Drawing::Font(L"Meiryo UI", 11.25F, System::Drawing::FontStyle::Italic,
				System::Drawing::GraphicsUnit::Point, static_cast<System::Byte>(128)));
			this->fcgLBGPUInfoLabelOnFeatureTab->ForeColor = System::Drawing::Color::Blue;
			this->fcgLBGPUInfoLabelOnFeatureTab->Location = System::Drawing::Point(32, 103);
			this->fcgLBGPUInfoLabelOnFeatureTab->Name = L"fcgLBGPUInfoLabelOnFeatureTab";
			this->fcgLBGPUInfoLabelOnFeatureTab->Size = System::Drawing::Size(40, 19);
			this->fcgLBGPUInfoLabelOnFeatureTab->TabIndex = 115;
			this->fcgLBGPUInfoLabelOnFeatureTab->Text = L"GPU";
			// 
			// fcgDGVFeatures
			// 
			this->fcgDGVFeatures->BackgroundColor = System::Drawing::SystemColors::Window;
			this->fcgDGVFeatures->ColumnHeadersHeightSizeMode = System::Windows::Forms::DataGridViewColumnHeadersHeightSizeMode::AutoSize;
			this->fcgDGVFeatures->Location = System::Drawing::Point(4, 152);
			this->fcgDGVFeatures->Name = L"fcgDGVFeatures";
			this->fcgDGVFeatures->RowTemplate->Height = 21;
			this->fcgDGVFeatures->Size = System::Drawing::Size(601, 323);
			this->fcgDGVFeatures->TabIndex = 0;
			// 
			// fcgCSExeFiles
			// 
			this->fcgCSExeFiles->Items->AddRange(gcnew cli::array< System::Windows::Forms::ToolStripItem^  >(1) { this->fcgTSExeFileshelp });
			this->fcgCSExeFiles->Name = L"fcgCSx264";
			this->fcgCSExeFiles->Size = System::Drawing::Size(137, 26);
			// 
			// fcgTSExeFileshelp
			// 
			this->fcgTSExeFileshelp->Name = L"fcgTSExeFileshelp";
			this->fcgTSExeFileshelp->Size = System::Drawing::Size(136, 22);
			this->fcgTSExeFileshelp->Text = L"helpを表示";
			this->fcgTSExeFileshelp->Click += gcnew System::EventHandler(this, &frmConfig::fcgTSExeFileshelp_Click);
			// 
			// fcgLBguiExBlog
			// 
			this->fcgLBguiExBlog->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((System::Windows::Forms::AnchorStyles::Bottom | System::Windows::Forms::AnchorStyles::Left));
			this->fcgLBguiExBlog->AutoSize = true;
			this->fcgLBguiExBlog->Font = (gcnew System::Drawing::Font(L"Meiryo UI", 8.25F, System::Drawing::FontStyle::Italic, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(128)));
			this->fcgLBguiExBlog->LinkColor = System::Drawing::Color::Gray;
			this->fcgLBguiExBlog->Location = System::Drawing::Point(639, 555);
			this->fcgLBguiExBlog->Name = L"fcgLBguiExBlog";
			this->fcgLBguiExBlog->Size = System::Drawing::Size(79, 14);
			this->fcgLBguiExBlog->TabIndex = 50;
			this->fcgLBguiExBlog->TabStop = true;
			this->fcgLBguiExBlog->Text = L"NVEncについて";
			this->fcgLBguiExBlog->VisitedLinkColor = System::Drawing::Color::Gray;
			this->fcgLBguiExBlog->LinkClicked += gcnew System::Windows::Forms::LinkLabelLinkClickedEventHandler(this, &frmConfig::fcgLBguiExBlog_LinkClicked);
			// 
			// fcgPBNVEncLogoDisabled
			// 
			this->fcgPBNVEncLogoDisabled->Image = (cli::safe_cast<System::Drawing::Image^>(resources->GetObject(L"fcgPBNVEncLogoDisabled.Image")));
			this->fcgPBNVEncLogoDisabled->Location = System::Drawing::Point(6, 3);
			this->fcgPBNVEncLogoDisabled->Name = L"fcgPBNVEncLogoDisabled";
			this->fcgPBNVEncLogoDisabled->Size = System::Drawing::Size(219, 75);
			this->fcgPBNVEncLogoDisabled->SizeMode = System::Windows::Forms::PictureBoxSizeMode::StretchImage;
			this->fcgPBNVEncLogoDisabled->TabIndex = 148;
			this->fcgPBNVEncLogoDisabled->TabStop = false;
			// 
			// fcgPBNVEncLogoEnabled
			// 
			this->fcgPBNVEncLogoEnabled->Image = (cli::safe_cast<System::Drawing::Image^>(resources->GetObject(L"fcgPBNVEncLogoEnabled.Image")));
			this->fcgPBNVEncLogoEnabled->Location = System::Drawing::Point(6, 3);
			this->fcgPBNVEncLogoEnabled->Name = L"fcgPBNVEncLogoEnabled";
			this->fcgPBNVEncLogoEnabled->Size = System::Drawing::Size(219, 75);
			this->fcgPBNVEncLogoEnabled->SizeMode = System::Windows::Forms::PictureBoxSizeMode::StretchImage;
			this->fcgPBNVEncLogoEnabled->TabIndex = 149;
			this->fcgPBNVEncLogoEnabled->TabStop = false;
			// 
			// frmConfig
			// 
			this->AutoScaleDimensions = System::Drawing::SizeF(96, 96);
			this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Dpi;
			this->ClientSize = System::Drawing::Size(1008, 577);
			this->Controls->Add(this->fcgLBguiExBlog);
			this->Controls->Add(this->fcgtabControlMux);
			this->Controls->Add(this->fcgtabControlNVEnc);
			this->Controls->Add(this->fcgLBVersion);
			this->Controls->Add(this->fcgLBVersionDate);
			this->Controls->Add(this->fcgBTDefault);
			this->Controls->Add(this->fcgBTOK);
			this->Controls->Add(this->fcgBTCancel);
			this->Controls->Add(this->fcgTXCmd);
			this->Controls->Add(this->fcggroupBoxAudio);
			this->Controls->Add(this->fcgtoolStripSettings);
			this->Font = (gcnew System::Drawing::Font(L"Meiryo UI", 9, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(128)));
			this->FormBorderStyle = System::Windows::Forms::FormBorderStyle::FixedDialog;
			this->MaximizeBox = false;
			this->Name = L"frmConfig";
			this->ShowIcon = false;
			this->Text = L"Aviutl 出力 プラグイン";
			this->Load += gcnew System::EventHandler(this, &frmConfig::frmConfig_Load);
			this->fcgtoolStripSettings->ResumeLayout(false);
			this->fcgtoolStripSettings->PerformLayout();
			this->fcggroupBoxAudio->ResumeLayout(false);
			this->fcggroupBoxAudio->PerformLayout();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUAudioBitrate))->EndInit();
			this->fcgtabControlMux->ResumeLayout(false);
			this->fcgtabPageMP4->ResumeLayout(false);
			this->fcgtabPageMP4->PerformLayout();
			this->fcgtabPageMKV->ResumeLayout(false);
			this->fcgtabPageMKV->PerformLayout();
			this->fcgtabPageMPG->ResumeLayout(false);
			this->fcgtabPageMPG->PerformLayout();
			this->fcgtabPageMux->ResumeLayout(false);
			this->fcgtabPageMux->PerformLayout();
			this->fcgtabPageBat->ResumeLayout(false);
			this->fcgtabPageBat->PerformLayout();
			this->fcgtabControlNVEnc->ResumeLayout(false);
			this->tabPageVideoEnc->ResumeLayout(false);
			this->tabPageVideoEnc->PerformLayout();
			this->fcggroupBoxColor->ResumeLayout(false);
			this->fcggroupBoxColor->PerformLayout();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNURefFrames))->EndInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUBframes))->EndInit();
			this->fcgGroupBoxQulaityStg->ResumeLayout(false);
			this->fcgPNBitrate->ResumeLayout(false);
			this->fcgPNBitrate->PerformLayout();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUBitrate))->EndInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUMaxkbps))->EndInit();
			this->fcgPNQP->ResumeLayout(false);
			this->fcgPNQP->PerformLayout();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUQPI))->EndInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUQPP))->EndInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUQPB))->EndInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUSlices))->EndInit();
			this->fcgGroupBoxAspectRatio->ResumeLayout(false);
			this->fcgGroupBoxAspectRatio->PerformLayout();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUAspectRatioY))->EndInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUAspectRatioX))->EndInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUGopLength))->EndInit();
			this->tabPageExOpt->ResumeLayout(false);
			this->tabPageExOpt->PerformLayout();
			this->tabPageNVEncFeatures->ResumeLayout(false);
			this->tabPageNVEncFeatures->PerformLayout();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgDGVFeatures))->EndInit();
			this->fcgCSExeFiles->ResumeLayout(false);
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgPBNVEncLogoDisabled))->EndInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgPBNVEncLogoEnabled))->EndInit();
			this->ResumeLayout(false);
			this->PerformLayout();

		}
#pragma endregion
	private:
		delegate void SetEnvironmentInfoDelegate();
		String^ StrCPUInfo;
		String^ StrGPUInfo;
		SetEnvironmentInfoDelegate ^getEnvironmentInfoDelegate;

		NVEncParam *paramCache;
		DataTable^ dataTableNVEncFeatures;
		
		
		const SYSTEM_DATA *sys_dat;
		CONF_GUIEX *conf;
		LocalSettings LocalStg;
		bool CurrentPipeEnabled;
		bool stgChanged;
		String^ CurrentStgDir;
		ToolStripMenuItem^ CheckedStgMenuItem;
		CONF_GUIEX *cnf_stgSelected;
		String^ lastQualityStr;
#ifdef HIDE_MPEG2
		TabPage^ tabPageMpgMux;
#endif
	private:
		System::Int32 GetCurrentAudioDefaultBitrate();
		delegate System::Void qualityTimerChangeDelegate();
		System::Void InitComboBox();
		System::Void setAudioDisplay();
		System::Void AudioEncodeModeChanged();
		System::Void InitStgFileList();
		System::Void RebuildStgFileDropDown(String^ stgDir);
		System::Void RebuildStgFileDropDown(ToolStripDropDownItem^ TS, String^ dir);
		System::Void SetLocalStg();
		System::Void LoadLocalStg();
		System::Void SaveLocalStg();
		System::Boolean CheckLocalStg();
		System::Void SetTXMaxLen(TextBox^ TX, int max_len);
		System::Void SetTXMaxLenAll();
		System::Void InitForm();
		System::Void ConfToFrm(CONF_GUIEX *cnf);
		System::Void FrmToConf(CONF_GUIEX *cnf);
		System::Void SetChangedEvent(Control^ control, System::EventHandler^ _event);
		System::Void SetAllCheckChangedEvents(Control ^top);
		System::Void SaveToStgFile(String^ stgName);
		System::Void DeleteStgFile(ToolStripMenuItem^ mItem);
		System::Boolean EnableSettingsNoteChange(bool Enable);
		System::Void fcgTSLSettingsNotes_DoubleClick(System::Object^  sender, System::EventArgs^  e);
		System::Void fcgTSTSettingsNotes_Leave(System::Object^  sender, System::EventArgs^  e);
		System::Void fcgTSTSettingsNotes_KeyDown(System::Object^  sender, System::Windows::Forms::KeyEventArgs^  e);
		System::Void fcgTSTSettingsNotes_TextChanged(System::Object^  sender, System::EventArgs^  e);
		System::Void GetfcgTSLSettingsNotes(char *notes, int nSize);
		System::Void SetfcgTSLSettingsNotes(const char *notes);
		System::Void SetfcgTSLSettingsNotes(String^ notes);
		System::Void fcgTSBSave_Click(System::Object^  sender, System::EventArgs^  e);
		System::Void fcgTSBSaveNew_Click(System::Object^  sender, System::EventArgs^  e);
		System::Void fcgTSBDelete_Click(System::Object^  sender, System::EventArgs^  e);
		System::Void fcgTSSettings_DropDownItemClicked(System::Object^  sender, System::Windows::Forms::ToolStripItemClickedEventArgs^  e);
		System::Void UncheckAllDropDownItem(ToolStripItem^ mItem);
		ToolStripMenuItem^ fcgTSSettingsSearchItem(String^ stgPath, ToolStripItem^ mItem);
		ToolStripMenuItem^ fcgTSSettingsSearchItem(String^ stgPath);
		System::Void CheckTSSettingsDropDownItem(ToolStripMenuItem^ mItem);
		System::Void CheckTSItemsEnabled(CONF_GUIEX *current_conf);
		System::Void fcgChangeMuxerVisible(System::Object^  sender, System::EventArgs^  e);
		System::Void SetHelpToolTips();
		System::Void ShowExehelp(String^ ExePath, String^ args);
		System::Void fcgTSBOtherSettings_Click(System::Object^  sender, System::EventArgs^  e);
		System::Void fcgChangeEnabled(System::Object^  sender, System::EventArgs^  e);
		System::Void fcgTSBBitrateCalc_CheckedChanged(System::Object^  sender, System::EventArgs^  e);
		System::Void CloseBitrateCalc();
		System::Void SetfbcBTABEnable(bool enable, int max);
		System::Void SetfbcBTVBEnable(bool enable);
		System::Void fcgCBAudio2pass_CheckedChanged(System::Object^  sender, System::EventArgs^  e);
		System::Void fcgCXAudioEncoder_SelectedIndexChanged(System::Object^  sender, System::EventArgs^  e);
		System::Void fcgCXAudioEncMode_SelectedIndexChanged(System::Object^  sender, System::EventArgs^  e);
		System::Void AdjustLocation();
		System::Void ActivateToolTip(bool Enable);
		System::Void SetStgEscKey(bool Enable);
		System::Void SetToolStripEvents(ToolStrip^ TS, System::Windows::Forms::MouseEventHandler^ _event);
		System::Void SetEnvironmentInfo();
		System::Void SetupFeatureTable();
	public:
		System::Void InitData(CONF_GUIEX *set_config, const SYSTEM_DATA *system_data);
		System::Void SetVideoBitrate(int bitrate);
		System::Void SetAudioBitrate(int bitrate);
		System::Void InformfbcClosed();
	private:
		System::Void fcgTSItem_MouseDown(System::Object^  sender, System::Windows::Forms::MouseEventArgs^  e) {
			EnableSettingsNoteChange(false);
		}
	private:
		System::Void frmConfig_KeyDown(System::Object^  sender, System::Windows::Forms::KeyEventArgs^  e) {
			if (e->KeyCode == Keys::Escape)
				this->Close();
		}
	private:
		System::Void NUSelectAll(System::Object^  sender, System::EventArgs^  e) {
			 NumericUpDown^ NU = dynamic_cast<NumericUpDown^>(sender);
			 NU->Select(0, NU->Text->Length);
		 }
	private:
		System::Void setComboBox(ComboBox^ CX, const X264_OPTION_STR * list) {
			CX->BeginUpdate();
			CX->Items->Clear();
			for (int i = 0; list[i].desc; i++)
				CX->Items->Add(String(list[i].desc).ToString());
			CX->EndUpdate();
		}
	private:
		System::Void setComboBox(ComboBox^ CX, const CX_DESC * list) {
			CX->BeginUpdate();
			CX->Items->Clear();
			for (int i = 0; list[i].desc; i++)
				CX->Items->Add(String(list[i].desc).ToString());
			CX->EndUpdate();
		}
	private:
		System::Void setComboBox(ComboBox^ CX, const WCHAR * const * list) {
			CX->BeginUpdate();
			CX->Items->Clear();
			for (int i = 0; list[i]; i++)
				CX->Items->Add(String(list[i]).ToString());
			CX->EndUpdate();
		}
	private:
		System::Void setPriorityList(ComboBox^ CX) {
			CX->BeginUpdate();
			CX->Items->Clear();
			for (int i = 0; priority_table[i].text; i++)
				CX->Items->Add(String(priority_table[i].text).ToString());
			CX->EndUpdate();
		}
	private:
		System::Void setMuxerCmdExNames(ComboBox^ CX, int muxer_index) {
			CX->BeginUpdate();
			CX->Items->Clear();
			MUXER_SETTINGS *mstg = &sys_dat->exstg->s_mux[muxer_index];
			for (int i = 0; i < mstg->ex_count; i++)
				CX->Items->Add(String(mstg->ex_cmd[i].name).ToString());
			CX->EndUpdate();
		}
	private:
		System::Void setAudioEncoderNames() {
			fcgCXAudioEncoder->BeginUpdate();
			fcgCXAudioEncoder->Items->Clear();
			//fcgCXAudioEncoder->Items->AddRange(reinterpret_cast<array<String^>^>(LocalStg.audEncName->ToArray(String::typeid)));
			fcgCXAudioEncoder->Items->AddRange(LocalStg.audEncName->ToArray());
			fcgCXAudioEncoder->EndUpdate();
		}
	private:
		System::Void TX_LimitbyBytes(System::Object^  sender, System::ComponentModel::CancelEventArgs^ e) {
			int maxLength = 0;
			int stringBytes = 0;
			TextBox^ TX = nullptr;
			if ((TX = dynamic_cast<TextBox^>(sender)) == nullptr)
				return;
			stringBytes = CountStringBytes(TX->Text);
			maxLength = TX->MaxLength;
			if (stringBytes > maxLength - 1) {
				e->Cancel = true;
				MessageBox::Show(this, L"入力された文字数が多すぎます。減らしてください。", L"エラー", MessageBoxButtons::OK, MessageBoxIcon::Error);
			}
		}
	private:
		System::Boolean openExeFile(TextBox^ TX, String^ ExeName) {
			//WinXPにおいて、OpenFileDialogはCurrentDirctoryを勝手に変更しやがるので、
			//一度保存し、あとから再適用する
			String^ CurrentDir = Directory::GetCurrentDirectory();
			OpenFileDialog ofd;
			ofd.Multiselect = false;
			ofd.FileName = ExeName;
			ofd.Filter = MakeExeFilter(ExeName);
			if (Directory::Exists(LocalStg.LastAppDir))
				ofd.InitialDirectory = Path::GetFullPath(LocalStg.LastAppDir);
			else if (File::Exists(TX->Text))
				ofd.InitialDirectory = Path::GetFullPath(Path::GetDirectoryName(TX->Text));
			else
				ofd.InitialDirectory = String(sys_dat->aviutl_dir).ToString();
			bool ret = (ofd.ShowDialog() == Windows::Forms::DialogResult::OK);
			if (ret) {
				if (sys_dat->exstg->s_local.get_relative_path)
					ofd.FileName = GetRelativePath(ofd.FileName, CurrentDir);
				LocalStg.LastAppDir = Path::GetDirectoryName(Path::GetFullPath(ofd.FileName));
				TX->Text = ofd.FileName;
				TX->SelectionStart = TX->Text->Length;
			}
			Directory::SetCurrentDirectory(CurrentDir);
			return ret;
		}
	private: 
		System::Void fcgBTMP4MuxerPath_Click(System::Object^  sender, System::EventArgs^  e) {
			openExeFile(fcgTXMP4MuxerPath, LocalStg.MP4MuxerExeName);
		}
	private: 
		System::Void fcgBTTC2MP4Path_Click(System::Object^  sender, System::EventArgs^  e) {
			openExeFile(fcgTXTC2MP4Path, LocalStg.TC2MP4ExeName);
		}
	private:
		System::Void fcgBTMP4RawMuxerPath_Click(System::Object^  sender, System::EventArgs^  e) {
			openExeFile(fcgTXMP4RawPath, LocalStg.MP4RawExeName);
		}
	private: 
		System::Void fcgBTAudioEncoderPath_Click(System::Object^  sender, System::EventArgs^  e) {
			int index = fcgCXAudioEncoder->SelectedIndex;
			openExeFile(fcgTXAudioEncoderPath, LocalStg.audEncExeName[index]);
		}
	private: 
		System::Void fcgBTMKVMuxerPath_Click(System::Object^  sender, System::EventArgs^  e) {
			openExeFile(fcgTXMKVMuxerPath, LocalStg.MKVMuxerExeName);
		}
	private:
		System::Void fcgBTMPGMuxerPath_Click(System::Object^  sender, System::EventArgs^  e) {
			openExeFile(fcgTXMPGMuxerPath, LocalStg.MPGMuxerExeName);
		}
	private:
		System::Void openTempFolder(TextBox^ TX) {
			FolderBrowserDialog^ fbd = fcgfolderBrowserTemp;
			if (Directory::Exists(TX->Text))
				fbd->SelectedPath = TX->Text;
			if (fbd->ShowDialog() == Windows::Forms::DialogResult::OK) {
				if (sys_dat->exstg->s_local.get_relative_path)
					fbd->SelectedPath = GetRelativePath(fbd->SelectedPath);
				TX->Text = fbd->SelectedPath;
				TX->SelectionStart = TX->Text->Length;
			}
		}
	private: 
		System::Void fcgBTCustomAudioTempDir_Click(System::Object^  sender, System::EventArgs^  e) {
			openTempFolder(fcgTXCustomAudioTempDir);
		}
	private: 
		System::Void fcgBTMP4BoxTempDir_Click(System::Object^  sender, System::EventArgs^  e) {
			openTempFolder(fcgTXMP4BoxTempDir);
		}
	private: 
		System::Void fcgBTCustomTempDir_Click(System::Object^  sender, System::EventArgs^  e) {
			openTempFolder(fcgTXCustomTempDir);
		}
	private:
		System::Boolean openAndSetFilePath(TextBox^ TX, String^ fileTypeName) {
			return openAndSetFilePath(TX, fileTypeName, nullptr, nullptr);
		}
	private:
		System::Boolean openAndSetFilePath(TextBox^ TX, String^ fileTypeName, String^ ext) {
			return openAndSetFilePath(TX, fileTypeName, ext, nullptr);
		}
	private:
		System::Boolean openAndSetFilePath(TextBox^ TX, String^ fileTypeName, String^ ext, String^ dir) {
			//WinXPにおいて、OpenFileDialogはCurrentDirctoryを勝手に変更しやがるので、
			//一度保存し、あとから再適用する
			String^ CurrentDir = Directory::GetCurrentDirectory();
			//設定
			if (ext == nullptr)
				ext = L".*";
			OpenFileDialog^ ofd = fcgOpenFileDialog;
			ofd->FileName = L"";
			if (dir != nullptr && Directory::Exists(dir))
				ofd->InitialDirectory = dir;
			if (TX->Text->Length) {
				String^ fileName = nullptr;
				try {
					fileName = Path::GetFileName(TX->Text);
				} catch (...) {
					//invalid charによる例外は破棄
				}
				if (fileName != nullptr)
					ofd->FileName = fileName;
			}
			ofd->Multiselect = false;
			ofd->Filter = fileTypeName + L"(*" + ext + L")|*" + ext;
			bool ret = (ofd->ShowDialog() == Windows::Forms::DialogResult::OK);
			if (ret) {
				if (sys_dat->exstg->s_local.get_relative_path)
					ofd->FileName = GetRelativePath(ofd->FileName, CurrentDir);
				TX->Text = ofd->FileName;
				TX->SelectionStart = TX->Text->Length;
			}
			Directory::SetCurrentDirectory(CurrentDir);
			return ret;
		}
	private:
		System::Void fcgBTBatAfterPath_Click(System::Object^  sender, System::EventArgs^  e) {
			if (openAndSetFilePath(fcgTXBatAfterPath, L"バッチファイル", ".bat", LocalStg.LastBatDir))
				LocalStg.LastBatDir = Path::GetDirectoryName(fcgTXBatAfterPath->Text);
		}
	private:
		System::Void fcgBTBatBeforePath_Click(System::Object^  sender, System::EventArgs^  e) {
			if (openAndSetFilePath(fcgTXBatBeforePath, L"バッチファイル", ".bat", LocalStg.LastBatDir))
				LocalStg.LastBatDir = Path::GetDirectoryName(fcgTXBatBeforePath->Text);
		}
	private:
		System::Void SetCXIndex(ComboBox^ CX, int index) {
			CX->SelectedIndex = clamp(index, 0, CX->Items->Count - 1);
		}
	private:
		System::Void SetNUValue(NumericUpDown^ NU, Decimal d) {
			NU->Value = clamp(d, NU->Minimum, NU->Maximum);
		}
	private:
		System::Void SetNUValue(NumericUpDown^ NU, int i) {
			NU->Value = clamp(Convert::ToDecimal(i), NU->Minimum, NU->Maximum);
		}
	private:
		System::Void SetNUValue(NumericUpDown^ NU, unsigned int i) {
			NU->Value = clamp(Convert::ToDecimal(i), NU->Minimum, NU->Maximum);
		}
	private:
		System::Void SetNUValue(NumericUpDown^ NU, float f) {
			NU->Value = clamp(Convert::ToDecimal(f), NU->Minimum, NU->Maximum);
		}
	private: 
		System::Void frmConfig_Load(System::Object^  sender, System::EventArgs^  e) {
			InitForm();
		}
	private: 
		System::Void fcgBTOK_Click(System::Object^  sender, System::EventArgs^  e) {
			if (CheckLocalStg())
				return;
			init_CONF_GUIEX(conf, false);
			FrmToConf(conf);
			SaveLocalStg();
			ZeroMemory(conf->oth.notes, sizeof(conf->oth.notes));
			this->Close();
		}
	private: 
		System::Void fcgBTCancel_Click(System::Object^  sender, System::EventArgs^  e) {
			this->Close();
		}
	private: 
		System::Void fcgBTDefault_Click(System::Object^  sender, System::EventArgs^  e) {
			init_CONF_GUIEX(conf, FALSE);
			ConfToFrm(conf);
		}
	private:
		System::Void ChangePresetNameDisplay(bool changed) {
			if (CheckedStgMenuItem != nullptr) {
				fcgTSSettings->Text = (changed) ? L"[" + CheckedStgMenuItem->Text + L"]*" : CheckedStgMenuItem->Text;
				fcgTSBSave->Enabled = changed;
			}
		}
	private:
		System::Void CheckOtherChanges(System::Object^  sender, System::EventArgs^  e) {
			if (CheckedStgMenuItem == nullptr)
				return;
			CONF_GUIEX check_change;
			init_CONF_GUIEX(&check_change, FALSE);
			FrmToConf(&check_change);
			ChangePresetNameDisplay(memcmp(&check_change, cnf_stgSelected, sizeof(CONF_GUIEX)) != 0);
		}
	private: 
		System::Void fcgTXCmd_DoubleClick(System::Object^  sender, System::EventArgs^  e) {
			int offset = (fcgTXCmd->Multiline) ? -fcgTXCmdfulloffset : fcgTXCmdfulloffset;
			fcgTXCmd->Height += offset;
			this->Height += offset;
			fcgTXCmd->Multiline = !fcgTXCmd->Multiline;
		}
	private: 
		System::Void fcgTSSettings_Click(System::Object^  sender, System::EventArgs^  e) {
			if (EnableSettingsNoteChange(false))
				fcgTSSettings->ShowDropDown();
		}
	private: 
		System::Void fcgTXAudioEncoderPath_TextChanged(System::Object^  sender, System::EventArgs^  e) {
			LocalStg.audEncPath[fcgCXAudioEncoder->SelectedIndex] = fcgTXAudioEncoderPath->Text;
			fcgBTAudioEncoderPath->ContextMenuStrip = (File::Exists(fcgTXAudioEncoderPath->Text)) ? fcgCSExeFiles : nullptr;
		}
	private: 
		System::Void fcgTXMP4MuxerPath_TextChanged(System::Object^  sender, System::EventArgs^  e) {
			LocalStg.MP4MuxerPath = fcgTXMP4MuxerPath->Text;
			fcgBTMP4MuxerPath->ContextMenuStrip = (File::Exists(fcgTXMP4MuxerPath->Text)) ? fcgCSExeFiles : nullptr;
		}
	private: 
		System::Void fcgTXTC2MP4Path_TextChanged(System::Object^  sender, System::EventArgs^  e) {
			LocalStg.TC2MP4Path = fcgTXTC2MP4Path->Text;
			fcgBTTC2MP4Path->ContextMenuStrip = (File::Exists(fcgTXTC2MP4Path->Text)) ? fcgCSExeFiles : nullptr;
		}
	private:
		System::Void fcgTXMP4RawMuxerPath_TextChanged(System::Object^  sender, System::EventArgs^  e) {
			LocalStg.MP4RawPath = fcgTXMP4RawPath->Text;
			fcgBTMP4RawPath->ContextMenuStrip = (File::Exists(fcgTXMP4RawPath->Text)) ? fcgCSExeFiles : nullptr;
		}
	private: 
		System::Void fcgTXMKVMuxerPath_TextChanged(System::Object^  sender, System::EventArgs^  e) {
			LocalStg.MKVMuxerPath = fcgTXMKVMuxerPath->Text;
			fcgBTMKVMuxerPath->ContextMenuStrip = (File::Exists(fcgTXMKVMuxerPath->Text)) ? fcgCSExeFiles : nullptr;
		}
	private:
		System::Void fcgTXMPGMuxerPath_TextChanged(System::Object^  sender, System::EventArgs^  e) {
			LocalStg.MPGMuxerPath = fcgTXMPGMuxerPath->Text;
			fcgBTMPGMuxerPath->ContextMenuStrip = (File::Exists(fcgTXMPGMuxerPath->Text)) ? fcgCSExeFiles : nullptr;
		}
	private: 
		System::Void fcgTXMP4BoxTempDir_TextChanged(System::Object^  sender, System::EventArgs^  e) {
			LocalStg.CustomMP4TmpDir = fcgTXMP4BoxTempDir->Text;
		}
	private: 
		System::Void fcgTXCustomAudioTempDir_TextChanged(System::Object^  sender, System::EventArgs^  e) {
			LocalStg.CustomAudTmpDir = fcgTXCustomAudioTempDir->Text;
		}
	private: 
		System::Void fcgTXCustomTempDir_TextChanged(System::Object^  sender, System::EventArgs^  e) {
			LocalStg.CustomTmpDir = fcgTXCustomTempDir->Text;
		}
	private:
		System::Void fcgSetDragDropFilename_Enter(System::Object^  sender, DragEventArgs^  e) {
			e->Effect = (e->Data->GetDataPresent(DataFormats::FileDrop)) ? DragDropEffects::Copy : DragDropEffects::None;
		}
	private:
		System::Void fcgSetDragDropFilename_DragDrop(System::Object^  sender, DragEventArgs^  e) {
			TextBox^ TX = dynamic_cast<TextBox^>(sender);
			array<System::String ^>^ filelist = dynamic_cast<array<System::String ^>^>(e->Data->GetData(DataFormats::FileDrop, false));
			if (filelist == nullptr || TX == nullptr)
				return;
			String^ filePath = filelist[0]; //複数だった場合は先頭のものを使用
			if (sys_dat->exstg->s_local.get_relative_path)
				filePath = GetRelativePath(filePath);
			TX->Text = filePath;
		}
	private:
		System::Void fcgTSExeFileshelp_Click(System::Object^  sender, System::EventArgs^  e) {
			System::Windows::Forms::ToolStripMenuItem^ TS = dynamic_cast<System::Windows::Forms::ToolStripMenuItem^>(sender);
			if (TS == nullptr) return;
			System::Windows::Forms::ContextMenuStrip^ CS = dynamic_cast<System::Windows::Forms::ContextMenuStrip^>(TS->Owner);
			if (CS == nullptr) return;

			//Name, args, Path の順番
			array<ExeControls>^ ControlList = {
				{ fcgBTAudioEncoderPath->Name,   fcgTXAudioEncoderPath->Text,   sys_dat->exstg->s_aud[fcgCXAudioEncoder->SelectedIndex].cmd_help },
				{ fcgBTMP4MuxerPath->Name,       fcgTXMP4MuxerPath->Text,       sys_dat->exstg->s_mux[MUXER_MP4].help_cmd },
				{ fcgBTTC2MP4Path->Name,         fcgTXTC2MP4Path->Text,         sys_dat->exstg->s_mux[MUXER_TC2MP4].help_cmd },
				{ fcgBTMP4RawPath->Name,         fcgTXMP4RawPath->Text,         sys_dat->exstg->s_mux[MUXER_MP4_RAW].help_cmd },
				{ fcgBTMKVMuxerPath->Name,       fcgTXMKVMuxerPath->Text,       sys_dat->exstg->s_mux[MUXER_MKV].help_cmd },
				{ fcgBTMPGMuxerPath->Name,       fcgTXMPGMuxerPath->Text,       sys_dat->exstg->s_mux[MUXER_MPG].help_cmd }
			};
			for (int i = 0; i < ControlList->Length; i++) {
				if (NULL == String::Compare(CS->SourceControl->Name, ControlList[i].Name)) {
					ShowExehelp(ControlList[i].Path, String(ControlList[i].args).ToString());
					return;
				}
			}
			MessageBox::Show(L"ヘルプ表示用のコマンドが設定されていません。", L"エラー", MessageBoxButtons::OK, MessageBoxIcon::Error);
		}
	private:
		System::Void fcgLBguiExBlog_LinkClicked(System::Object^  sender, System::Windows::Forms::LinkLabelLinkClickedEventArgs^  e) {
			fcgLBguiExBlog->LinkVisited = true;
			try {
				System::Diagnostics::Process::Start(String(sys_dat->exstg->blog_url).ToString());
			} catch (...) {
				//いちいちメッセージとか出さない
			};
		}
	private:
		System::Void fcgBTQualityStg_Click(System::Object^  sender, System::EventArgs^  e) {
			CONF_GUIEX cnf;
			FrmToConf(&cnf);
			auto presetList = paramCache->GetCachedNVEncH264Preset();
			memcpy(&cnf.nvenc, &presetList[fcgCXQualityPreset->SelectedIndex], sizeof(cnf.nvenc));
			ConfToFrm(&cnf);
		}
};
}
