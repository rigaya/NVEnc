// -----------------------------------------------------------------------------------------
// x264guiEx/x265guiEx/svtAV1guiEx/ffmpegOut/QSVEnc/NVEnc/VCEEnc by rigaya
// -----------------------------------------------------------------------------------------
// The MIT License
//
// Copyright (c) 2010-2022 rigaya
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
// --------------------------------------------------------------------------------------------

#pragma once

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
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
#include "NVEncCmd.h"

using namespace System;
using namespace System::ComponentModel;
using namespace System::Collections;
using namespace System::Windows::Forms;
using namespace System::Data;
using namespace System::Drawing;
using namespace System::IO;
using namespace System::Threading::Tasks;


namespace AUO_NAME_R {

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
            InitData(_conf, _sys_dat);
            dwStgReader = nullptr;
            themeMode = AuoTheme::DefaultLight;
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
            if (dwStgReader != nullptr)
                delete dwStgReader;
            if (cnf_stgSelected) free(cnf_stgSelected); cnf_stgSelected = NULL;
        }




    private: System::Windows::Forms::ToolStrip^  fcgtoolStripSettings;

    private: System::Windows::Forms::TabControl^  fcgtabControlMux;
    private: System::Windows::Forms::TabPage^  fcgtabPageMP4;
    private: System::Windows::Forms::TabPage^  fcgtabPageMKV;


    private: System::Windows::Forms::Button^  fcgBTCancel;

    private: System::Windows::Forms::Button^  fcgBTOK;
    private: System::Windows::Forms::Button^  fcgBTDefault;




    private: System::Windows::Forms::ToolStripButton^  fcgTSBSave;

    private: System::Windows::Forms::ToolStripButton^  fcgTSBSaveNew;

    private: System::Windows::Forms::ToolStripButton^  fcgTSBDelete;

    private: System::Windows::Forms::ToolStripSeparator^  fcgtoolStripSeparator1;
    private: System::Windows::Forms::ToolStripDropDownButton^  fcgTSSettings;





























































































































































































































































    private: System::Windows::Forms::TabPage^  fcgtabPageMux;












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










private: System::Windows::Forms::ToolTip^  fcgTTEx;






private: System::Windows::Forms::ToolStripSeparator^  toolStripSeparator2;
private: System::Windows::Forms::ToolStripButton^  fcgTSBOtherSettings;















































































































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








private: System::Windows::Forms::Label^  fcgLBGOPLengthAuto;


































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




private: System::Windows::Forms::Label^  fcgLBBatAfterString;

private: System::Windows::Forms::Label^  fcgLBBatBeforeString;
private: System::Windows::Forms::Panel^  fcgPNSeparator;
private: System::Windows::Forms::Button^  fcgBTBatBeforePath;
private: System::Windows::Forms::TextBox^  fcgTXBatBeforePath;
private: System::Windows::Forms::Label^  fcgLBBatBeforePath;
private: System::Windows::Forms::CheckBox^  fcgCBWaitForBatBefore;
private: System::Windows::Forms::CheckBox^  fcgCBRunBatBefore;
private: System::Windows::Forms::LinkLabel^  fcgLBguiExBlog;







































private: System::Windows::Forms::NumericUpDown^  fcgNUBframes;
private: System::Windows::Forms::Label^  fcgLBBframes;













private: System::Windows::Forms::NumericUpDown^  fcgNURefFrames;
private: System::Windows::Forms::Label^  fcgLBRefFrames;














private: System::Windows::Forms::PictureBox^  fcgPBNVEncLogoEnabled;

private: System::Windows::Forms::PictureBox^  fcgPBNVEncLogoDisabled;
private: System::Windows::Forms::Label^  label1;
private: System::Windows::Forms::ComboBox^  fcgCXEncCodec;
private: System::Windows::Forms::Panel^  fcgPNH264;


private: System::Windows::Forms::Panel^  fcgPNHEVC;


private: System::Windows::Forms::Label^  fcgLBHEVCProfile;
private: System::Windows::Forms::Label^  fxgLBHEVCTier;
private: System::Windows::Forms::ComboBox^  fcgCXHEVCTier;





private: System::Windows::Forms::ComboBox^  fxgCXHEVCLevel;



























private: System::Windows::Forms::CheckBox^  fcgCBAFS;
private: System::Windows::Forms::NumericUpDown^  fcgNUVBVBufsize;
private: System::Windows::Forms::Label^  fcgLBVBVBufsize;
private: System::Windows::Forms::Label^  fcgLBBluray;
private: System::Windows::Forms::CheckBox^  fcgCBBluray;
private: System::Windows::Forms::TabControl^  fcgtabControlAudio;
private: System::Windows::Forms::TabPage^  fcgtabPageAudioMain;





















private: System::Windows::Forms::TabPage^  fcgtabPageAudioOther;
private: System::Windows::Forms::Panel^  panel2;
private: System::Windows::Forms::Label^  fcgLBBatAfterAudioString;
private: System::Windows::Forms::Label^  fcgLBBatBeforeAudioString;
private: System::Windows::Forms::Button^  fcgBTBatAfterAudioPath;
private: System::Windows::Forms::TextBox^  fcgTXBatAfterAudioPath;
private: System::Windows::Forms::Label^  fcgLBBatAfterAudioPath;
private: System::Windows::Forms::CheckBox^  fcgCBRunBatAfterAudio;
private: System::Windows::Forms::Panel^  panel1;
private: System::Windows::Forms::Button^  fcgBTBatBeforeAudioPath;
private: System::Windows::Forms::TextBox^  fcgTXBatBeforeAudioPath;
private: System::Windows::Forms::Label^  fcgLBBatBeforeAudioPath;
private: System::Windows::Forms::CheckBox^  fcgCBRunBatBeforeAudio;
private: System::Windows::Forms::ComboBox^  fcgCXAudioPriority;
private: System::Windows::Forms::Label^  fcgLBAudioPriority;

private: System::Windows::Forms::Label^  fcgLBAQ;














private: System::Windows::Forms::Label^  fcgLBLookaheadDisable;
private: System::Windows::Forms::NumericUpDown^  fcgNULookaheadDepth;

private: System::Windows::Forms::Label^  fcgLBAQStrengthAuto;
private: System::Windows::Forms::NumericUpDown^  fcgNUAQStrength;
private: System::Windows::Forms::Label^  fcgLBAQStrength;
private: System::Windows::Forms::ComboBox^  fcgCXAQ;

private: System::Windows::Forms::Label^  fcgLBLookaheadDepth;
private: System::Windows::Forms::Label^  fcgLBHEVCOutBitDepth;
private: System::Windows::Forms::ComboBox^  fcgCXHEVCOutBitDepth;





























private: System::Windows::Forms::TabPage^  tabPageVideoDetail;
private: System::Windows::Forms::Label^  fcgLBSlices;

private: System::Windows::Forms::NumericUpDown^  fcgNUSlices;

private: System::Windows::Forms::Panel^  fcgPNH264Detail;
private: System::Windows::Forms::Label^  fcgLBDeblock;


private: System::Windows::Forms::ComboBox^  fcgCXAdaptiveTransform;
private: System::Windows::Forms::Label^  fcgLBAdaptiveTransform;
private: System::Windows::Forms::ComboBox^  fcgCXBDirectMode;
private: System::Windows::Forms::Label^  fcgLBBDirectMode;
private: System::Windows::Forms::Label^  fcgLBMVPrecision;





private: System::Windows::Forms::ComboBox^  fcgCXMVPrecision;
private: System::Windows::Forms::CheckBox^  fcgCBDeblock;


private: System::Windows::Forms::CheckBox^  fcgCBCABAC;
private: System::Windows::Forms::Label^  fcgLBCABAC;


private: System::Windows::Forms::Panel^  fcgPNHEVCDetail;
private: System::Windows::Forms::Label^  fcgLBHEVCMinCUSize;


private: System::Windows::Forms::ComboBox^  fcgCXHEVCMinCUSize;
private: System::Windows::Forms::Label^  fcgLBHEVCMaxCUSize;


private: System::Windows::Forms::ComboBox^  fcgCXHEVCMaxCUSize;

private: System::Windows::Forms::Label^  label3;
private: System::Windows::Forms::Label^  fcgLBDevice;

private: System::Windows::Forms::ComboBox^  fcgCXDevice;
private: System::Windows::Forms::GroupBox^  groupBoxQPDetail;
private: System::Windows::Forms::Label^  fcgLBQPDetailB;
private: System::Windows::Forms::Label^  fcgLBQPDetailP;
private: System::Windows::Forms::Label^  fcgLBQPDetailI;
private: System::Windows::Forms::CheckBox^  fcgCBQPInit;

private: System::Windows::Forms::Label^  label16;
private: System::Windows::Forms::NumericUpDown^  fcgNUQPInitB;

private: System::Windows::Forms::Label^  label18;
private: System::Windows::Forms::NumericUpDown^  fcgNUQPInitP;

private: System::Windows::Forms::NumericUpDown^  fcgNUQPInitI;

private: System::Windows::Forms::CheckBox^  fcgCBQPMin;

private: System::Windows::Forms::Label^  label11;
private: System::Windows::Forms::NumericUpDown^  fcgNUQPMinB;

private: System::Windows::Forms::Label^  label13;
private: System::Windows::Forms::NumericUpDown^  fcgNUQPMinP;

private: System::Windows::Forms::NumericUpDown^  fcgNUQPMinI;

private: System::Windows::Forms::CheckBox^  fcgCBQPMax;
private: System::Windows::Forms::Label^  label8;
private: System::Windows::Forms::NumericUpDown^  fcgNUQPMaxB;

private: System::Windows::Forms::Label^  label7;
private: System::Windows::Forms::NumericUpDown^  fcgNUQPMaxP;

private: System::Windows::Forms::NumericUpDown^  fcgNUQPMaxI;
private: System::Windows::Forms::Label^  fcgLBWeightP;
private: System::Windows::Forms::CheckBox^  fcgCBWeightP;



private: System::Windows::Forms::NumericUpDown^  fcgNUVBRTragetQuality;
private: System::Windows::Forms::Label^  fcgLBVBRTragetQuality;
private: System::Windows::Forms::Label^  fcgLBVBRTragetQuality2;








































private: System::Windows::Forms::Label^  fcgLBCudaSchdule;
private: System::Windows::Forms::ComboBox^  fcgCXCudaSchdule;
































































private: System::Windows::Forms::GroupBox^  fcggroupBoxVppDetailEnahance;

private: System::Windows::Forms::Panel^  fcgPNVppUnsharp;
private: System::Windows::Forms::Label^  fcgLBVppUnsharpThreshold;
private: System::Windows::Forms::Label^  fcgLBVppUnsharpWeight;
private: System::Windows::Forms::Label^  fcgLBVppUnsharpRadius;
private: System::Windows::Forms::NumericUpDown^  fcgNUVppUnsharpThreshold;
private: System::Windows::Forms::NumericUpDown^  fcgNUVppUnsharpWeight;
private: System::Windows::Forms::NumericUpDown^  fcgNUVppUnsharpRadius;
private: System::Windows::Forms::Panel^  fcgPNVppEdgelevel;

private: System::Windows::Forms::Label^  label10;


private: System::Windows::Forms::NumericUpDown^  numericUpDown4;
private: System::Windows::Forms::ComboBox^  fcgCXVppDetailEnhance;
private: System::Windows::Forms::GroupBox^  fcggroupBoxVppDeinterlace;







































private: System::Windows::Forms::CheckBox^  fcgCBVppDebandEnable;
private: System::Windows::Forms::GroupBox^  fcggroupBoxVppDeband;
private: System::Windows::Forms::CheckBox^  fcgCBVppDebandRandEachFrame;
private: System::Windows::Forms::CheckBox^  fcgCBVppDebandBlurFirst;
private: System::Windows::Forms::Label^  fcgLBVppDebandSample;
private: System::Windows::Forms::Label^  fcgLBVppDebandDitherC;
private: System::Windows::Forms::Label^  fcgLBVppDebandDitherY;
private: System::Windows::Forms::Label^  fcgLBVppDebandDither;
private: System::Windows::Forms::Label^  fcgLBVppDebandThreCr;
private: System::Windows::Forms::Label^  fcgLBVppDebandThreCb;
private: System::Windows::Forms::Label^  fcgLBVppDebandThreY;
private: System::Windows::Forms::NumericUpDown^  fcgNUVppDebandDitherC;
private: System::Windows::Forms::NumericUpDown^  fcgNUVppDebandDitherY;
private: System::Windows::Forms::NumericUpDown^  fcgNUVppDebandThreCr;
private: System::Windows::Forms::NumericUpDown^  fcgNUVppDebandThreCb;
private: System::Windows::Forms::Label^  fcgLBVppDebandThreshold;
private: System::Windows::Forms::Label^  fcgLBVppDebandRange;
private: System::Windows::Forms::ComboBox^  fcgCXVppDebandSample;
private: System::Windows::Forms::NumericUpDown^  fcgNUVppDebandThreY;
private: System::Windows::Forms::NumericUpDown^  fcgNUVppDebandRange;
private: System::Windows::Forms::GroupBox^  fcggroupBoxVppDenoise;
private: System::Windows::Forms::Panel^  fcgPNVppDenoiseKnn;
private: System::Windows::Forms::Label^  fcgLBVppDenoiseKnnThreshold;
private: System::Windows::Forms::Label^  fcgLBVppDenoiseKnnStrength;
private: System::Windows::Forms::Label^  fcgLBVppDenoiseKnnRadius;
private: System::Windows::Forms::NumericUpDown^  fcgNUVppDenoiseKnnThreshold;
private: System::Windows::Forms::NumericUpDown^  fcgNUVppDenoiseKnnStrength;
private: System::Windows::Forms::NumericUpDown^  fcgNUVppDenoiseKnnRadius;
private: System::Windows::Forms::Panel^  fcgPNVppDenoisePmd;
private: System::Windows::Forms::Label^  fcgLBVppDenoisePmdThreshold;
private: System::Windows::Forms::Label^  fcgLBVppDenoisePmdStrength;
private: System::Windows::Forms::Label^  fcgLBVppDenoisePmdApplyCount;
private: System::Windows::Forms::NumericUpDown^  fcgNUVppDenoisePmdThreshold;
private: System::Windows::Forms::NumericUpDown^  fcgNUVppDenoisePmdStrength;
private: System::Windows::Forms::NumericUpDown^  fcgNUVppDenoisePmdApplyCount;
private: System::Windows::Forms::ComboBox^  fcgCXVppDenoiseMethod;
private: System::Windows::Forms::CheckBox^  fcgCBVppResize;
private: System::Windows::Forms::GroupBox^  fcggroupBoxResize;
private: System::Windows::Forms::ComboBox^  fcgCXVppResizeAlg;
private: System::Windows::Forms::Label^  fcgLBVppResize;
private: System::Windows::Forms::NumericUpDown^  fcgNUVppResizeHeight;
private: System::Windows::Forms::NumericUpDown^  fcgNUVppResizeWidth;
private: System::Windows::Forms::Label^  fcgLBVppEdgelevelThreshold;
private: System::Windows::Forms::Label^  fcgLBVppEdgelevelBlack;


private: System::Windows::Forms::Label^  fcgLBVppEdgelevelStrength;
private: System::Windows::Forms::NumericUpDown^  fcgNUVppEdgelevelThreshold;
private: System::Windows::Forms::NumericUpDown^  fcgNUVppEdgelevelBlack;




private: System::Windows::Forms::NumericUpDown^  fcgNUVppEdgelevelStrength;
private: System::Windows::Forms::Label^  fcgLBVppEdgelevelWhite;
private: System::Windows::Forms::NumericUpDown^  fcgNUVppEdgelevelWhite;
private: System::Windows::Forms::CheckBox^  fcgCBVppPerfMonitor;
private: System::Windows::Forms::CheckBox^  fcgCBVppTweakEnable;
private: System::Windows::Forms::GroupBox^  fcggroupBoxVppTweak;


private: System::Windows::Forms::TrackBar^  fcgTBVppTweakHue;
private: System::Windows::Forms::Label^  fcgLBVppTweakHue;
private: System::Windows::Forms::TrackBar^  fcgTBVppTweakSaturation;
private: System::Windows::Forms::Label^  fcgLBVppTweakSaturation;
private: System::Windows::Forms::TrackBar^  fcgTBVppTweakGamma;
private: System::Windows::Forms::Label^  fcgLBVppTweakGamma;
private: System::Windows::Forms::TrackBar^  fcgTBVppTweakContrast;
private: System::Windows::Forms::Label^  fcgLBVppTweakContrast;
private: System::Windows::Forms::TrackBar^  fcgTBVppTweakBrightness;
private: System::Windows::Forms::NumericUpDown^  fcgNUVppTweakGamma;
private: System::Windows::Forms::NumericUpDown^  fcgNUVppTweakSaturation;
private: System::Windows::Forms::NumericUpDown^  fcgNUVppTweakHue;
private: System::Windows::Forms::Label^  fcgLBVppTweakBrightness;
private: System::Windows::Forms::NumericUpDown^  fcgNUVppTweakContrast;
private: System::Windows::Forms::NumericUpDown^  fcgNUVppTweakBrightness;

private: System::Windows::Forms::TextBox^  fcgTXCmd;

private: System::Windows::Forms::Label^  fcgLBVideoEncoderPath;
private: System::Windows::Forms::Button^  fcgBTVideoEncoderPath;


private: System::Windows::Forms::TextBox^  fcgTXVideoEncoderPath;
private: System::Windows::Forms::ComboBox^  fcgCXBrefMode;

private: System::Windows::Forms::Label^  fcgLBBrefMode;
private: System::Windows::Forms::Label^  fcgLBQuality;
private: System::Windows::Forms::ComboBox^  fcgCXQuality;
private: System::Windows::Forms::Panel^  fcgPNVppAfs;
private: System::Windows::Forms::TrackBar^  fcgTBVppAfsThreCMotion;
private: System::Windows::Forms::Label^  fcgLBVppAfsThreCMotion;
private: System::Windows::Forms::TrackBar^  fcgTBVppAfsThreYMotion;
private: System::Windows::Forms::Label^  fcgLBVppAfsThreYmotion;
private: System::Windows::Forms::TrackBar^  fcgTBVppAfsThreDeint;
private: System::Windows::Forms::Label^  fcgLBVppAfsThreDeint;
private: System::Windows::Forms::TrackBar^  fcgTBVppAfsThreShift;
private: System::Windows::Forms::Label^  fcgLBVppAfsThreShift;
private: System::Windows::Forms::TrackBar^  fcgTBVppAfsCoeffShift;
private: System::Windows::Forms::Label^  fcgLBVppAfsCoeffShift;
private: System::Windows::Forms::Label^  label20;
private: System::Windows::Forms::Label^  label19;
private: System::Windows::Forms::Label^  label4;
private: System::Windows::Forms::Label^  fcgLBVppAfsUp;
private: System::Windows::Forms::NumericUpDown^  fcgNUVppAfsRight;
private: System::Windows::Forms::NumericUpDown^  fcgNUVppAfsLeft;
private: System::Windows::Forms::NumericUpDown^  fcgNUVppAfsBottom;
private: System::Windows::Forms::NumericUpDown^  fcgNUVppAfsUp;
private: System::Windows::Forms::TrackBar^  fcgTBVppAfsMethodSwitch;
private: System::Windows::Forms::CheckBox^  fcgCBVppAfs24fps;
private: System::Windows::Forms::CheckBox^  fcgCBVppAfsTune;
private: System::Windows::Forms::CheckBox^  fcgCBVppAfsSmooth;
private: System::Windows::Forms::CheckBox^  fcgCBVppAfsDrop;
private: System::Windows::Forms::CheckBox^  fcgCBVppAfsShift;
private: System::Windows::Forms::Label^  fcgLBVppAfsAnalyze;
private: System::Windows::Forms::NumericUpDown^  fcgNUVppAfsThreCMotion;
private: System::Windows::Forms::NumericUpDown^  fcgNUVppAfsThreShift;
private: System::Windows::Forms::NumericUpDown^  fcgNUVppAfsThreDeint;
private: System::Windows::Forms::NumericUpDown^  fcgNUVppAfsThreYMotion;
private: System::Windows::Forms::Label^  fcgLBVppAfsMethodSwitch;
private: System::Windows::Forms::ComboBox^  fcgCXVppAfsAnalyze;
private: System::Windows::Forms::NumericUpDown^  fcgNUVppAfsCoeffShift;
private: System::Windows::Forms::NumericUpDown^  fcgNUVppAfsMethodSwitch;
private: System::Windows::Forms::Label^  fcgLBVppDeinterlace;

private: System::Windows::Forms::ComboBox^  fcgCXVppDeinterlace;
private: System::Windows::Forms::Panel^  fcgPNVppNnedi;
private: System::Windows::Forms::Label^  fcgLBVppNnediErrorType;
private: System::Windows::Forms::ComboBox^  fcgCXVppNnediErrorType;
private: System::Windows::Forms::Label^  fcgLBVppNnediPrescreen;
private: System::Windows::Forms::ComboBox^  fcgCXVppNnediPrescreen;
private: System::Windows::Forms::Label^  fcgLBVppNnediPrec;
private: System::Windows::Forms::ComboBox^  fcgCXVppNnediPrec;
private: System::Windows::Forms::Label^  fcgLBVppNnediQual;
private: System::Windows::Forms::ComboBox^  fcgCXVppNnediQual;
private: System::Windows::Forms::Label^  fcgLBVppNnediNsize;
private: System::Windows::Forms::ComboBox^  fcgCXVppNnediNsize;
private: System::Windows::Forms::Label^  fcgLBVppNnediNns;
private: System::Windows::Forms::ComboBox^  fcgCXVppNnediNns;
private: System::Windows::Forms::CheckBox ^fcgCBLossless;
private: System::Windows::Forms::Label ^fcgLBLossless;
private: System::Windows::Forms::CheckBox ^fcgCBPSNR;

private: System::Windows::Forms::Label ^fcgLBPSNR;
private: System::Windows::Forms::CheckBox ^fcgCBSSIM;


private: System::Windows::Forms::Label ^fcgLBSSIM;
private: System::Windows::Forms::Panel ^fcgPNVppYadif;
private: System::Windows::Forms::Label ^fcgLBVppYadifMode;
private: System::Windows::Forms::ComboBox ^fcgCXVppYadifMode;
private: System::Windows::Forms::Label ^fcgLBFullrange;
private: System::Windows::Forms::CheckBox ^fcgCBFullrange;
private: System::Windows::Forms::ComboBox ^fcgCXVideoFormat;
private: System::Windows::Forms::Label ^fcgLBVideoFormat;
private: System::Windows::Forms::GroupBox ^fcggroupBoxColor;
private: System::Windows::Forms::ComboBox ^fcgCXTransfer;
private: System::Windows::Forms::ComboBox ^fcgCXColorPrim;
private: System::Windows::Forms::ComboBox ^fcgCXColorMatrix;
private: System::Windows::Forms::Label ^fcgLBTransfer;
private: System::Windows::Forms::Label ^fcgLBColorPrim;
private: System::Windows::Forms::Label ^fcgLBColorMatrix;
private: System::Windows::Forms::Panel^ fcgPNVppDenoiseSmooth;

private: System::Windows::Forms::Label^ fcgLBVppDenoiseSmoothQP;


private: System::Windows::Forms::Label^ fcgLBVppDenoiseSmoothQuality;
private: System::Windows::Forms::NumericUpDown^ fcgNUVppDenoiseSmoothQP;




private: System::Windows::Forms::NumericUpDown^ fcgNUVppDenoiseSmoothQuality;
private: System::Windows::Forms::CheckBox ^fcgCBFAWCheck;

private: System::Windows::Forms::Panel ^fcgPNAudioInternal;
private: System::Windows::Forms::Label ^fcgLBAudioBitrateInternal;

private: System::Windows::Forms::NumericUpDown ^fcgNUAudioBitrateInternal;

private: System::Windows::Forms::ComboBox ^fcgCXAudioEncModeInternal;

private: System::Windows::Forms::Label ^label2;
private: System::Windows::Forms::ComboBox ^fcgCXAudioEncoderInternal;


private: System::Windows::Forms::Panel ^fcgPNAudioExt;
private: System::Windows::Forms::ComboBox ^fcgCXAudioDelayCut;
private: System::Windows::Forms::Label ^fcgLBAudioDelayCut;
private: System::Windows::Forms::Label ^fcgCBAudioEncTiming;
private: System::Windows::Forms::ComboBox ^fcgCXAudioEncTiming;
private: System::Windows::Forms::ComboBox ^fcgCXAudioTempDir;
private: System::Windows::Forms::TextBox ^fcgTXCustomAudioTempDir;
private: System::Windows::Forms::Button ^fcgBTCustomAudioTempDir;
private: System::Windows::Forms::CheckBox ^fcgCBAudioUsePipe;
private: System::Windows::Forms::NumericUpDown ^fcgNUAudioBitrate;
private: System::Windows::Forms::CheckBox ^fcgCBAudio2pass;
private: System::Windows::Forms::ComboBox ^fcgCXAudioEncMode;
private: System::Windows::Forms::Label ^fcgLBAudioEncMode;
private: System::Windows::Forms::Button ^fcgBTAudioEncoderPath;
private: System::Windows::Forms::TextBox ^fcgTXAudioEncoderPath;
private: System::Windows::Forms::Label ^fcgLBAudioEncoderPath;
private: System::Windows::Forms::CheckBox ^fcgCBAudioOnly;
private: System::Windows::Forms::ComboBox ^fcgCXAudioEncoder;
private: System::Windows::Forms::Label ^fcgLBAudioTemp;
private: System::Windows::Forms::Label ^fcgLBAudioBitrate;
private: System::Windows::Forms::CheckBox ^fcgCBAudioUseExt;
private: System::Windows::Forms::TabPage ^fcgtabPageInternal;
private: System::Windows::Forms::ComboBox ^fcgCXInternalCmdEx;
private: System::Windows::Forms::Label ^fcgLBMuxInternalExt;
private: System::Windows::Forms::TabPage ^tabPageOther;
private: System::Windows::Forms::CheckBox ^fcgCBLogDebug;
private: System::Windows::Forms::CheckBox ^fcgCBAuoTcfileout;
private: System::Windows::Forms::Label ^fcgLBTempDir;
private: System::Windows::Forms::Button ^fcgBTCustomTempDir;
private: System::Windows::Forms::TextBox ^fcgTXCustomTempDir;
private: System::Windows::Forms::ComboBox ^fcgCXTempDir;
private: System::Windows::Forms::CheckBox ^fcgCBPerfMonitor;
private: System::Windows::Forms::GroupBox ^fcggroupBoxCmdEx;
private: System::Windows::Forms::TextBox ^fcgTXCmdEx;
private: System::Windows::Forms::Label ^fcgLBMultiPass;
private: System::Windows::Forms::ComboBox ^fcgCXMultiPass;
private: System::Windows::Forms::Panel^  fcgPNVppWarpsharp;
private: System::Windows::Forms::Label^  fcgLBVppWarpsharpBlurType;
private: System::Windows::Forms::NumericUpDown^  fcgNUVppWarpsharpBlurType;



private: System::Windows::Forms::Label^  fcgLBVppWarpsharpDepth;
private: System::Windows::Forms::Label^  fcgLBVppWarpsharpBlur;


private: System::Windows::Forms::Label^  fcgLBVppWarpsharpThreshold;
private: System::Windows::Forms::NumericUpDown^  fcgNUVppWarpsharpDepth;
private: System::Windows::Forms::NumericUpDown^  fcgNUVppWarpsharpBlur;



private: System::Windows::Forms::NumericUpDown^  fcgNUVppWarpsharpThreshold;

private: System::Windows::Forms::Label^  label14;
private: System::Windows::Forms::NumericUpDown^  numericUpDown6;
private: System::Windows::Forms::Label^  fcgLBChromaQPOffset;
private: System::Windows::Forms::NumericUpDown^  fcgNUChromaQPOffset;
private: System::Windows::Forms::Panel^  fcgPNVppDenoiseConv3D;

private: System::Windows::Forms::Label^  fcgLBVppDenoiseConv3DThreshTemporal;
private: System::Windows::Forms::Label^  fcgLBVppDenoiseConv3DThreshSpatial;




private: System::Windows::Forms::Label^  fcgLBVppDenoiseConv3DThreshCTemporal;

private: System::Windows::Forms::Label^  fcgLBVppDenoiseConv3DThreshCSpatial;
private: System::Windows::Forms::NumericUpDown^  fcgNUVppDenoiseConv3DThreshCTemporal;




private: System::Windows::Forms::NumericUpDown^  fcgNUVppDenoiseConv3DThreshCSpatial;
private: System::Windows::Forms::Label^  fcgLBVppDenoiseConv3DThreshYTemporal;





private: System::Windows::Forms::Label^  fcgLBVppDenoiseConv3DThreshYSpatial;
private: System::Windows::Forms::NumericUpDown^  fcgNUVppDenoiseConv3DThreshYTemporal;



private: System::Windows::Forms::NumericUpDown^  fcgNUVppDenoiseConv3DThreshYSpatial;
private: System::Windows::Forms::ComboBox^  fcgCXVppDenoiseConv3DMatrix;
private: System::Windows::Forms::Label^  fcgLBVppDenoiseConv3DMatrix;
private: System::Windows::Forms::Panel^  fcgPNHideToolStripBorder;









































































































































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
            this->fcgtabPageInternal = (gcnew System::Windows::Forms::TabPage());
            this->fcgCXInternalCmdEx = (gcnew System::Windows::Forms::ComboBox());
            this->fcgLBMuxInternalExt = (gcnew System::Windows::Forms::Label());
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
            this->fcgLBMultiPass = (gcnew System::Windows::Forms::Label());
            this->fcgCXMultiPass = (gcnew System::Windows::Forms::ComboBox());
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
            this->fcgLBQuality = (gcnew System::Windows::Forms::Label());
            this->fcgCXQuality = (gcnew System::Windows::Forms::ComboBox());
            this->fcgCXBrefMode = (gcnew System::Windows::Forms::ComboBox());
            this->fcgLBBrefMode = (gcnew System::Windows::Forms::Label());
            this->fcgBTVideoEncoderPath = (gcnew System::Windows::Forms::Button());
            this->fcgTXVideoEncoderPath = (gcnew System::Windows::Forms::TextBox());
            this->fcgLBVideoEncoderPath = (gcnew System::Windows::Forms::Label());
            this->fcgLBWeightP = (gcnew System::Windows::Forms::Label());
            this->fcgLBInterlaced = (gcnew System::Windows::Forms::Label());
            this->fcgCBWeightP = (gcnew System::Windows::Forms::CheckBox());
            this->label3 = (gcnew System::Windows::Forms::Label());
            this->fcgLBLookaheadDisable = (gcnew System::Windows::Forms::Label());
            this->fcgNULookaheadDepth = (gcnew System::Windows::Forms::NumericUpDown());
            this->fcgLBAQStrengthAuto = (gcnew System::Windows::Forms::Label());
            this->fcgCXInterlaced = (gcnew System::Windows::Forms::ComboBox());
            this->fcgNUAQStrength = (gcnew System::Windows::Forms::NumericUpDown());
            this->fcgLBAQStrength = (gcnew System::Windows::Forms::Label());
            this->fcgCXAQ = (gcnew System::Windows::Forms::ComboBox());
            this->fcgLBLookaheadDepth = (gcnew System::Windows::Forms::Label());
            this->fcgLBAQ = (gcnew System::Windows::Forms::Label());
            this->fcgNUVBVBufsize = (gcnew System::Windows::Forms::NumericUpDown());
            this->fcgGroupBoxAspectRatio = (gcnew System::Windows::Forms::GroupBox());
            this->fcgLBAspectRatio = (gcnew System::Windows::Forms::Label());
            this->fcgNUAspectRatioY = (gcnew System::Windows::Forms::NumericUpDown());
            this->fcgNUAspectRatioX = (gcnew System::Windows::Forms::NumericUpDown());
            this->fcgCXAspectRatio = (gcnew System::Windows::Forms::ComboBox());
            this->fcgLBVBVBufsize = (gcnew System::Windows::Forms::Label());
            this->fcgCBAFS = (gcnew System::Windows::Forms::CheckBox());
            this->label1 = (gcnew System::Windows::Forms::Label());
            this->fcgCXEncCodec = (gcnew System::Windows::Forms::ComboBox());
            this->fcgNURefFrames = (gcnew System::Windows::Forms::NumericUpDown());
            this->fcgLBRefFrames = (gcnew System::Windows::Forms::Label());
            this->fcgNUBframes = (gcnew System::Windows::Forms::NumericUpDown());
            this->fcgLBBframes = (gcnew System::Windows::Forms::Label());
            this->fcgLBGOPLengthAuto = (gcnew System::Windows::Forms::Label());
            this->fcgNUGopLength = (gcnew System::Windows::Forms::NumericUpDown());
            this->fcgLBGOPLength = (gcnew System::Windows::Forms::Label());
            this->fcgLBEncMode = (gcnew System::Windows::Forms::Label());
            this->fcgCXEncMode = (gcnew System::Windows::Forms::ComboBox());
            this->fcgPBNVEncLogoEnabled = (gcnew System::Windows::Forms::PictureBox());
            this->fcgPBNVEncLogoDisabled = (gcnew System::Windows::Forms::PictureBox());
            this->fcgPNH264 = (gcnew System::Windows::Forms::Panel());
            this->fcgLBBluray = (gcnew System::Windows::Forms::Label());
            this->fcgCBBluray = (gcnew System::Windows::Forms::CheckBox());
            this->fcgLBCodecProfile = (gcnew System::Windows::Forms::Label());
            this->fcgLBCodecLevel = (gcnew System::Windows::Forms::Label());
            this->fcgCXCodecProfile = (gcnew System::Windows::Forms::ComboBox());
            this->fcgCXCodecLevel = (gcnew System::Windows::Forms::ComboBox());
            this->fcgPNQP = (gcnew System::Windows::Forms::Panel());
            this->fcgLBQPI = (gcnew System::Windows::Forms::Label());
            this->fcgNUQPI = (gcnew System::Windows::Forms::NumericUpDown());
            this->fcgNUQPP = (gcnew System::Windows::Forms::NumericUpDown());
            this->fcgNUQPB = (gcnew System::Windows::Forms::NumericUpDown());
            this->fcgLBQPP = (gcnew System::Windows::Forms::Label());
            this->fcgLBQPB = (gcnew System::Windows::Forms::Label());
            this->fcgPNBitrate = (gcnew System::Windows::Forms::Panel());
            this->fcgLBVBRTragetQuality2 = (gcnew System::Windows::Forms::Label());
            this->fcgNUVBRTragetQuality = (gcnew System::Windows::Forms::NumericUpDown());
            this->fcgLBVBRTragetQuality = (gcnew System::Windows::Forms::Label());
            this->fcgLBBitrate = (gcnew System::Windows::Forms::Label());
            this->fcgNUBitrate = (gcnew System::Windows::Forms::NumericUpDown());
            this->fcgLBBitrate2 = (gcnew System::Windows::Forms::Label());
            this->fcgNUMaxkbps = (gcnew System::Windows::Forms::NumericUpDown());
            this->fcgLBMaxkbps = (gcnew System::Windows::Forms::Label());
            this->fcgLBMaxBitrate2 = (gcnew System::Windows::Forms::Label());
            this->fcgPNHEVC = (gcnew System::Windows::Forms::Panel());
            this->fxgLBHEVCTier = (gcnew System::Windows::Forms::Label());
            this->fcgLBHEVCOutBitDepth = (gcnew System::Windows::Forms::Label());
            this->fcgCXHEVCOutBitDepth = (gcnew System::Windows::Forms::ComboBox());
            this->fcgLBHEVCProfile = (gcnew System::Windows::Forms::Label());
            this->fcgCXHEVCTier = (gcnew System::Windows::Forms::ComboBox());
            this->fxgCXHEVCLevel = (gcnew System::Windows::Forms::ComboBox());
            this->tabPageVideoDetail = (gcnew System::Windows::Forms::TabPage());
            this->fcgCBPSNR = (gcnew System::Windows::Forms::CheckBox());
            this->fcgLBPSNR = (gcnew System::Windows::Forms::Label());
            this->fcgCBSSIM = (gcnew System::Windows::Forms::CheckBox());
            this->fcgLBSSIM = (gcnew System::Windows::Forms::Label());
            this->fcgCBLossless = (gcnew System::Windows::Forms::CheckBox());
            this->fcgLBLossless = (gcnew System::Windows::Forms::Label());
            this->fcgLBCudaSchdule = (gcnew System::Windows::Forms::Label());
            this->fcgCXCudaSchdule = (gcnew System::Windows::Forms::ComboBox());
            this->groupBoxQPDetail = (gcnew System::Windows::Forms::GroupBox());
            this->fcgLBChromaQPOffset = (gcnew System::Windows::Forms::Label());
            this->fcgNUChromaQPOffset = (gcnew System::Windows::Forms::NumericUpDown());
            this->fcgLBQPDetailB = (gcnew System::Windows::Forms::Label());
            this->fcgLBQPDetailP = (gcnew System::Windows::Forms::Label());
            this->fcgLBQPDetailI = (gcnew System::Windows::Forms::Label());
            this->fcgCBQPInit = (gcnew System::Windows::Forms::CheckBox());
            this->label16 = (gcnew System::Windows::Forms::Label());
            this->fcgNUQPInitB = (gcnew System::Windows::Forms::NumericUpDown());
            this->label18 = (gcnew System::Windows::Forms::Label());
            this->fcgNUQPInitP = (gcnew System::Windows::Forms::NumericUpDown());
            this->fcgNUQPInitI = (gcnew System::Windows::Forms::NumericUpDown());
            this->fcgCBQPMin = (gcnew System::Windows::Forms::CheckBox());
            this->label11 = (gcnew System::Windows::Forms::Label());
            this->fcgNUQPMinB = (gcnew System::Windows::Forms::NumericUpDown());
            this->label13 = (gcnew System::Windows::Forms::Label());
            this->fcgNUQPMinP = (gcnew System::Windows::Forms::NumericUpDown());
            this->fcgNUQPMinI = (gcnew System::Windows::Forms::NumericUpDown());
            this->fcgCBQPMax = (gcnew System::Windows::Forms::CheckBox());
            this->label8 = (gcnew System::Windows::Forms::Label());
            this->fcgNUQPMaxB = (gcnew System::Windows::Forms::NumericUpDown());
            this->label7 = (gcnew System::Windows::Forms::Label());
            this->fcgNUQPMaxP = (gcnew System::Windows::Forms::NumericUpDown());
            this->fcgNUQPMaxI = (gcnew System::Windows::Forms::NumericUpDown());
            this->fcgLBDevice = (gcnew System::Windows::Forms::Label());
            this->fcgCXDevice = (gcnew System::Windows::Forms::ComboBox());
            this->fcgLBSlices = (gcnew System::Windows::Forms::Label());
            this->fcgNUSlices = (gcnew System::Windows::Forms::NumericUpDown());
            this->fcgPNH264Detail = (gcnew System::Windows::Forms::Panel());
            this->fcgLBDeblock = (gcnew System::Windows::Forms::Label());
            this->fcgCXAdaptiveTransform = (gcnew System::Windows::Forms::ComboBox());
            this->fcgLBAdaptiveTransform = (gcnew System::Windows::Forms::Label());
            this->fcgCXBDirectMode = (gcnew System::Windows::Forms::ComboBox());
            this->fcgLBBDirectMode = (gcnew System::Windows::Forms::Label());
            this->fcgLBMVPrecision = (gcnew System::Windows::Forms::Label());
            this->fcgCXMVPrecision = (gcnew System::Windows::Forms::ComboBox());
            this->fcgCBDeblock = (gcnew System::Windows::Forms::CheckBox());
            this->fcgCBCABAC = (gcnew System::Windows::Forms::CheckBox());
            this->fcgLBCABAC = (gcnew System::Windows::Forms::Label());
            this->fcgPNHEVCDetail = (gcnew System::Windows::Forms::Panel());
            this->fcgLBHEVCMinCUSize = (gcnew System::Windows::Forms::Label());
            this->fcgCXHEVCMinCUSize = (gcnew System::Windows::Forms::ComboBox());
            this->fcgLBHEVCMaxCUSize = (gcnew System::Windows::Forms::Label());
            this->fcgCXHEVCMaxCUSize = (gcnew System::Windows::Forms::ComboBox());
            this->tabPageExOpt = (gcnew System::Windows::Forms::TabPage());
            this->fcgCBVppTweakEnable = (gcnew System::Windows::Forms::CheckBox());
            this->fcggroupBoxVppTweak = (gcnew System::Windows::Forms::GroupBox());
            this->fcgTBVppTweakHue = (gcnew System::Windows::Forms::TrackBar());
            this->fcgLBVppTweakHue = (gcnew System::Windows::Forms::Label());
            this->fcgTBVppTweakSaturation = (gcnew System::Windows::Forms::TrackBar());
            this->fcgLBVppTweakSaturation = (gcnew System::Windows::Forms::Label());
            this->fcgTBVppTweakGamma = (gcnew System::Windows::Forms::TrackBar());
            this->fcgLBVppTweakGamma = (gcnew System::Windows::Forms::Label());
            this->fcgTBVppTweakContrast = (gcnew System::Windows::Forms::TrackBar());
            this->fcgLBVppTweakContrast = (gcnew System::Windows::Forms::Label());
            this->fcgTBVppTweakBrightness = (gcnew System::Windows::Forms::TrackBar());
            this->fcgNUVppTweakGamma = (gcnew System::Windows::Forms::NumericUpDown());
            this->fcgNUVppTweakSaturation = (gcnew System::Windows::Forms::NumericUpDown());
            this->fcgNUVppTweakHue = (gcnew System::Windows::Forms::NumericUpDown());
            this->fcgLBVppTweakBrightness = (gcnew System::Windows::Forms::Label());
            this->fcgNUVppTweakContrast = (gcnew System::Windows::Forms::NumericUpDown());
            this->fcgNUVppTweakBrightness = (gcnew System::Windows::Forms::NumericUpDown());
            this->fcgCBVppPerfMonitor = (gcnew System::Windows::Forms::CheckBox());
            this->fcggroupBoxVppDetailEnahance = (gcnew System::Windows::Forms::GroupBox());
            this->fcgPNVppWarpsharp = (gcnew System::Windows::Forms::Panel());
            this->fcgLBVppWarpsharpBlurType = (gcnew System::Windows::Forms::Label());
            this->fcgNUVppWarpsharpBlurType = (gcnew System::Windows::Forms::NumericUpDown());
            this->fcgLBVppWarpsharpDepth = (gcnew System::Windows::Forms::Label());
            this->fcgLBVppWarpsharpBlur = (gcnew System::Windows::Forms::Label());
            this->fcgLBVppWarpsharpThreshold = (gcnew System::Windows::Forms::Label());
            this->fcgNUVppWarpsharpDepth = (gcnew System::Windows::Forms::NumericUpDown());
            this->fcgNUVppWarpsharpBlur = (gcnew System::Windows::Forms::NumericUpDown());
            this->fcgNUVppWarpsharpThreshold = (gcnew System::Windows::Forms::NumericUpDown());
            this->label14 = (gcnew System::Windows::Forms::Label());
            this->numericUpDown6 = (gcnew System::Windows::Forms::NumericUpDown());
            this->fcgCXVppDetailEnhance = (gcnew System::Windows::Forms::ComboBox());
            this->fcgPNVppEdgelevel = (gcnew System::Windows::Forms::Panel());
            this->fcgLBVppEdgelevelWhite = (gcnew System::Windows::Forms::Label());
            this->fcgNUVppEdgelevelWhite = (gcnew System::Windows::Forms::NumericUpDown());
            this->fcgLBVppEdgelevelThreshold = (gcnew System::Windows::Forms::Label());
            this->fcgLBVppEdgelevelBlack = (gcnew System::Windows::Forms::Label());
            this->fcgLBVppEdgelevelStrength = (gcnew System::Windows::Forms::Label());
            this->fcgNUVppEdgelevelThreshold = (gcnew System::Windows::Forms::NumericUpDown());
            this->fcgNUVppEdgelevelBlack = (gcnew System::Windows::Forms::NumericUpDown());
            this->fcgNUVppEdgelevelStrength = (gcnew System::Windows::Forms::NumericUpDown());
            this->label10 = (gcnew System::Windows::Forms::Label());
            this->numericUpDown4 = (gcnew System::Windows::Forms::NumericUpDown());
            this->fcgPNVppUnsharp = (gcnew System::Windows::Forms::Panel());
            this->fcgLBVppUnsharpThreshold = (gcnew System::Windows::Forms::Label());
            this->fcgLBVppUnsharpWeight = (gcnew System::Windows::Forms::Label());
            this->fcgLBVppUnsharpRadius = (gcnew System::Windows::Forms::Label());
            this->fcgNUVppUnsharpThreshold = (gcnew System::Windows::Forms::NumericUpDown());
            this->fcgNUVppUnsharpWeight = (gcnew System::Windows::Forms::NumericUpDown());
            this->fcgNUVppUnsharpRadius = (gcnew System::Windows::Forms::NumericUpDown());
            this->fcggroupBoxVppDeinterlace = (gcnew System::Windows::Forms::GroupBox());
            this->fcgLBVppDeinterlace = (gcnew System::Windows::Forms::Label());
            this->fcgCXVppDeinterlace = (gcnew System::Windows::Forms::ComboBox());
            this->fcgPNVppYadif = (gcnew System::Windows::Forms::Panel());
            this->fcgLBVppYadifMode = (gcnew System::Windows::Forms::Label());
            this->fcgCXVppYadifMode = (gcnew System::Windows::Forms::ComboBox());
            this->fcgPNVppNnedi = (gcnew System::Windows::Forms::Panel());
            this->fcgLBVppNnediErrorType = (gcnew System::Windows::Forms::Label());
            this->fcgCXVppNnediErrorType = (gcnew System::Windows::Forms::ComboBox());
            this->fcgLBVppNnediPrescreen = (gcnew System::Windows::Forms::Label());
            this->fcgCXVppNnediPrescreen = (gcnew System::Windows::Forms::ComboBox());
            this->fcgLBVppNnediPrec = (gcnew System::Windows::Forms::Label());
            this->fcgCXVppNnediPrec = (gcnew System::Windows::Forms::ComboBox());
            this->fcgLBVppNnediQual = (gcnew System::Windows::Forms::Label());
            this->fcgCXVppNnediQual = (gcnew System::Windows::Forms::ComboBox());
            this->fcgLBVppNnediNsize = (gcnew System::Windows::Forms::Label());
            this->fcgCXVppNnediNsize = (gcnew System::Windows::Forms::ComboBox());
            this->fcgLBVppNnediNns = (gcnew System::Windows::Forms::Label());
            this->fcgCXVppNnediNns = (gcnew System::Windows::Forms::ComboBox());
            this->fcgPNVppAfs = (gcnew System::Windows::Forms::Panel());
            this->fcgTBVppAfsThreCMotion = (gcnew System::Windows::Forms::TrackBar());
            this->fcgLBVppAfsThreCMotion = (gcnew System::Windows::Forms::Label());
            this->fcgTBVppAfsThreYMotion = (gcnew System::Windows::Forms::TrackBar());
            this->fcgLBVppAfsThreYmotion = (gcnew System::Windows::Forms::Label());
            this->fcgTBVppAfsThreDeint = (gcnew System::Windows::Forms::TrackBar());
            this->fcgLBVppAfsThreDeint = (gcnew System::Windows::Forms::Label());
            this->fcgTBVppAfsThreShift = (gcnew System::Windows::Forms::TrackBar());
            this->fcgLBVppAfsThreShift = (gcnew System::Windows::Forms::Label());
            this->fcgTBVppAfsCoeffShift = (gcnew System::Windows::Forms::TrackBar());
            this->fcgLBVppAfsCoeffShift = (gcnew System::Windows::Forms::Label());
            this->label20 = (gcnew System::Windows::Forms::Label());
            this->label19 = (gcnew System::Windows::Forms::Label());
            this->label4 = (gcnew System::Windows::Forms::Label());
            this->fcgLBVppAfsUp = (gcnew System::Windows::Forms::Label());
            this->fcgNUVppAfsRight = (gcnew System::Windows::Forms::NumericUpDown());
            this->fcgNUVppAfsLeft = (gcnew System::Windows::Forms::NumericUpDown());
            this->fcgNUVppAfsBottom = (gcnew System::Windows::Forms::NumericUpDown());
            this->fcgNUVppAfsUp = (gcnew System::Windows::Forms::NumericUpDown());
            this->fcgTBVppAfsMethodSwitch = (gcnew System::Windows::Forms::TrackBar());
            this->fcgCBVppAfs24fps = (gcnew System::Windows::Forms::CheckBox());
            this->fcgCBVppAfsTune = (gcnew System::Windows::Forms::CheckBox());
            this->fcgCBVppAfsSmooth = (gcnew System::Windows::Forms::CheckBox());
            this->fcgCBVppAfsDrop = (gcnew System::Windows::Forms::CheckBox());
            this->fcgCBVppAfsShift = (gcnew System::Windows::Forms::CheckBox());
            this->fcgLBVppAfsAnalyze = (gcnew System::Windows::Forms::Label());
            this->fcgNUVppAfsThreCMotion = (gcnew System::Windows::Forms::NumericUpDown());
            this->fcgNUVppAfsThreShift = (gcnew System::Windows::Forms::NumericUpDown());
            this->fcgNUVppAfsThreDeint = (gcnew System::Windows::Forms::NumericUpDown());
            this->fcgNUVppAfsThreYMotion = (gcnew System::Windows::Forms::NumericUpDown());
            this->fcgLBVppAfsMethodSwitch = (gcnew System::Windows::Forms::Label());
            this->fcgCXVppAfsAnalyze = (gcnew System::Windows::Forms::ComboBox());
            this->fcgNUVppAfsCoeffShift = (gcnew System::Windows::Forms::NumericUpDown());
            this->fcgNUVppAfsMethodSwitch = (gcnew System::Windows::Forms::NumericUpDown());
            this->fcgCBVppDebandEnable = (gcnew System::Windows::Forms::CheckBox());
            this->fcggroupBoxVppDeband = (gcnew System::Windows::Forms::GroupBox());
            this->fcgCBVppDebandRandEachFrame = (gcnew System::Windows::Forms::CheckBox());
            this->fcgCBVppDebandBlurFirst = (gcnew System::Windows::Forms::CheckBox());
            this->fcgLBVppDebandSample = (gcnew System::Windows::Forms::Label());
            this->fcgLBVppDebandDitherC = (gcnew System::Windows::Forms::Label());
            this->fcgLBVppDebandDitherY = (gcnew System::Windows::Forms::Label());
            this->fcgLBVppDebandDither = (gcnew System::Windows::Forms::Label());
            this->fcgLBVppDebandThreCr = (gcnew System::Windows::Forms::Label());
            this->fcgLBVppDebandThreCb = (gcnew System::Windows::Forms::Label());
            this->fcgLBVppDebandThreY = (gcnew System::Windows::Forms::Label());
            this->fcgNUVppDebandDitherC = (gcnew System::Windows::Forms::NumericUpDown());
            this->fcgNUVppDebandDitherY = (gcnew System::Windows::Forms::NumericUpDown());
            this->fcgNUVppDebandThreCr = (gcnew System::Windows::Forms::NumericUpDown());
            this->fcgNUVppDebandThreCb = (gcnew System::Windows::Forms::NumericUpDown());
            this->fcgLBVppDebandThreshold = (gcnew System::Windows::Forms::Label());
            this->fcgLBVppDebandRange = (gcnew System::Windows::Forms::Label());
            this->fcgCXVppDebandSample = (gcnew System::Windows::Forms::ComboBox());
            this->fcgNUVppDebandThreY = (gcnew System::Windows::Forms::NumericUpDown());
            this->fcgNUVppDebandRange = (gcnew System::Windows::Forms::NumericUpDown());
            this->fcggroupBoxVppDenoise = (gcnew System::Windows::Forms::GroupBox());
            this->fcgPNVppDenoiseConv3D = (gcnew System::Windows::Forms::Panel());
            this->fcgCXVppDenoiseConv3DMatrix = (gcnew System::Windows::Forms::ComboBox());
            this->fcgLBVppDenoiseConv3DMatrix = (gcnew System::Windows::Forms::Label());
            this->fcgLBVppDenoiseConv3DThreshTemporal = (gcnew System::Windows::Forms::Label());
            this->fcgLBVppDenoiseConv3DThreshSpatial = (gcnew System::Windows::Forms::Label());
            this->fcgLBVppDenoiseConv3DThreshCTemporal = (gcnew System::Windows::Forms::Label());
            this->fcgLBVppDenoiseConv3DThreshCSpatial = (gcnew System::Windows::Forms::Label());
            this->fcgNUVppDenoiseConv3DThreshCTemporal = (gcnew System::Windows::Forms::NumericUpDown());
            this->fcgNUVppDenoiseConv3DThreshCSpatial = (gcnew System::Windows::Forms::NumericUpDown());
            this->fcgLBVppDenoiseConv3DThreshYTemporal = (gcnew System::Windows::Forms::Label());
            this->fcgLBVppDenoiseConv3DThreshYSpatial = (gcnew System::Windows::Forms::Label());
            this->fcgNUVppDenoiseConv3DThreshYTemporal = (gcnew System::Windows::Forms::NumericUpDown());
            this->fcgNUVppDenoiseConv3DThreshYSpatial = (gcnew System::Windows::Forms::NumericUpDown());
            this->fcgPNVppDenoiseSmooth = (gcnew System::Windows::Forms::Panel());
            this->fcgLBVppDenoiseSmoothQP = (gcnew System::Windows::Forms::Label());
            this->fcgLBVppDenoiseSmoothQuality = (gcnew System::Windows::Forms::Label());
            this->fcgNUVppDenoiseSmoothQP = (gcnew System::Windows::Forms::NumericUpDown());
            this->fcgNUVppDenoiseSmoothQuality = (gcnew System::Windows::Forms::NumericUpDown());
            this->fcgPNVppDenoiseKnn = (gcnew System::Windows::Forms::Panel());
            this->fcgLBVppDenoiseKnnThreshold = (gcnew System::Windows::Forms::Label());
            this->fcgLBVppDenoiseKnnStrength = (gcnew System::Windows::Forms::Label());
            this->fcgLBVppDenoiseKnnRadius = (gcnew System::Windows::Forms::Label());
            this->fcgNUVppDenoiseKnnThreshold = (gcnew System::Windows::Forms::NumericUpDown());
            this->fcgNUVppDenoiseKnnStrength = (gcnew System::Windows::Forms::NumericUpDown());
            this->fcgNUVppDenoiseKnnRadius = (gcnew System::Windows::Forms::NumericUpDown());
            this->fcgCXVppDenoiseMethod = (gcnew System::Windows::Forms::ComboBox());
            this->fcgPNVppDenoisePmd = (gcnew System::Windows::Forms::Panel());
            this->fcgLBVppDenoisePmdThreshold = (gcnew System::Windows::Forms::Label());
            this->fcgLBVppDenoisePmdStrength = (gcnew System::Windows::Forms::Label());
            this->fcgLBVppDenoisePmdApplyCount = (gcnew System::Windows::Forms::Label());
            this->fcgNUVppDenoisePmdThreshold = (gcnew System::Windows::Forms::NumericUpDown());
            this->fcgNUVppDenoisePmdStrength = (gcnew System::Windows::Forms::NumericUpDown());
            this->fcgNUVppDenoisePmdApplyCount = (gcnew System::Windows::Forms::NumericUpDown());
            this->fcgCBVppResize = (gcnew System::Windows::Forms::CheckBox());
            this->fcggroupBoxResize = (gcnew System::Windows::Forms::GroupBox());
            this->fcgCXVppResizeAlg = (gcnew System::Windows::Forms::ComboBox());
            this->fcgLBVppResize = (gcnew System::Windows::Forms::Label());
            this->fcgNUVppResizeHeight = (gcnew System::Windows::Forms::NumericUpDown());
            this->fcgNUVppResizeWidth = (gcnew System::Windows::Forms::NumericUpDown());
            this->tabPageOther = (gcnew System::Windows::Forms::TabPage());
            this->fcggroupBoxCmdEx = (gcnew System::Windows::Forms::GroupBox());
            this->fcgTXCmdEx = (gcnew System::Windows::Forms::TextBox());
            this->fcgCBLogDebug = (gcnew System::Windows::Forms::CheckBox());
            this->fcgCBAuoTcfileout = (gcnew System::Windows::Forms::CheckBox());
            this->fcgLBTempDir = (gcnew System::Windows::Forms::Label());
            this->fcgBTCustomTempDir = (gcnew System::Windows::Forms::Button());
            this->fcgTXCustomTempDir = (gcnew System::Windows::Forms::TextBox());
            this->fcgCXTempDir = (gcnew System::Windows::Forms::ComboBox());
            this->fcgCBPerfMonitor = (gcnew System::Windows::Forms::CheckBox());
            this->fcgCSExeFiles = (gcnew System::Windows::Forms::ContextMenuStrip(this->components));
            this->fcgTSExeFileshelp = (gcnew System::Windows::Forms::ToolStripMenuItem());
            this->fcgLBguiExBlog = (gcnew System::Windows::Forms::LinkLabel());
            this->fcgtabControlAudio = (gcnew System::Windows::Forms::TabControl());
            this->fcgtabPageAudioMain = (gcnew System::Windows::Forms::TabPage());
            this->fcgCBAudioUseExt = (gcnew System::Windows::Forms::CheckBox());
            this->fcgCBFAWCheck = (gcnew System::Windows::Forms::CheckBox());
            this->fcgPNAudioExt = (gcnew System::Windows::Forms::Panel());
            this->fcgCXAudioDelayCut = (gcnew System::Windows::Forms::ComboBox());
            this->fcgLBAudioDelayCut = (gcnew System::Windows::Forms::Label());
            this->fcgCBAudioEncTiming = (gcnew System::Windows::Forms::Label());
            this->fcgCXAudioEncTiming = (gcnew System::Windows::Forms::ComboBox());
            this->fcgCXAudioTempDir = (gcnew System::Windows::Forms::ComboBox());
            this->fcgTXCustomAudioTempDir = (gcnew System::Windows::Forms::TextBox());
            this->fcgBTCustomAudioTempDir = (gcnew System::Windows::Forms::Button());
            this->fcgCBAudioUsePipe = (gcnew System::Windows::Forms::CheckBox());
            this->fcgNUAudioBitrate = (gcnew System::Windows::Forms::NumericUpDown());
            this->fcgCBAudio2pass = (gcnew System::Windows::Forms::CheckBox());
            this->fcgCXAudioEncMode = (gcnew System::Windows::Forms::ComboBox());
            this->fcgLBAudioEncMode = (gcnew System::Windows::Forms::Label());
            this->fcgBTAudioEncoderPath = (gcnew System::Windows::Forms::Button());
            this->fcgTXAudioEncoderPath = (gcnew System::Windows::Forms::TextBox());
            this->fcgLBAudioEncoderPath = (gcnew System::Windows::Forms::Label());
            this->fcgCBAudioOnly = (gcnew System::Windows::Forms::CheckBox());
            this->fcgCXAudioEncoder = (gcnew System::Windows::Forms::ComboBox());
            this->fcgLBAudioTemp = (gcnew System::Windows::Forms::Label());
            this->fcgLBAudioBitrate = (gcnew System::Windows::Forms::Label());
            this->fcgPNAudioInternal = (gcnew System::Windows::Forms::Panel());
            this->fcgLBAudioBitrateInternal = (gcnew System::Windows::Forms::Label());
            this->fcgNUAudioBitrateInternal = (gcnew System::Windows::Forms::NumericUpDown());
            this->fcgCXAudioEncModeInternal = (gcnew System::Windows::Forms::ComboBox());
            this->label2 = (gcnew System::Windows::Forms::Label());
            this->fcgCXAudioEncoderInternal = (gcnew System::Windows::Forms::ComboBox());
            this->fcgtabPageAudioOther = (gcnew System::Windows::Forms::TabPage());
            this->panel2 = (gcnew System::Windows::Forms::Panel());
            this->fcgLBBatAfterAudioString = (gcnew System::Windows::Forms::Label());
            this->fcgLBBatBeforeAudioString = (gcnew System::Windows::Forms::Label());
            this->fcgBTBatAfterAudioPath = (gcnew System::Windows::Forms::Button());
            this->fcgTXBatAfterAudioPath = (gcnew System::Windows::Forms::TextBox());
            this->fcgLBBatAfterAudioPath = (gcnew System::Windows::Forms::Label());
            this->fcgCBRunBatAfterAudio = (gcnew System::Windows::Forms::CheckBox());
            this->panel1 = (gcnew System::Windows::Forms::Panel());
            this->fcgBTBatBeforeAudioPath = (gcnew System::Windows::Forms::Button());
            this->fcgTXBatBeforeAudioPath = (gcnew System::Windows::Forms::TextBox());
            this->fcgLBBatBeforeAudioPath = (gcnew System::Windows::Forms::Label());
            this->fcgCBRunBatBeforeAudio = (gcnew System::Windows::Forms::CheckBox());
            this->fcgCXAudioPriority = (gcnew System::Windows::Forms::ComboBox());
            this->fcgLBAudioPriority = (gcnew System::Windows::Forms::Label());
            this->fcgTXCmd = (gcnew System::Windows::Forms::TextBox());
            this->fcgPNHideToolStripBorder = (gcnew System::Windows::Forms::Panel());
            this->fcgtoolStripSettings->SuspendLayout();
            this->fcgtabControlMux->SuspendLayout();
            this->fcgtabPageMP4->SuspendLayout();
            this->fcgtabPageMKV->SuspendLayout();
            this->fcgtabPageMPG->SuspendLayout();
            this->fcgtabPageMux->SuspendLayout();
            this->fcgtabPageBat->SuspendLayout();
            this->fcgtabPageInternal->SuspendLayout();
            this->fcgtabControlNVEnc->SuspendLayout();
            this->tabPageVideoEnc->SuspendLayout();
            this->fcggroupBoxColor->SuspendLayout();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNULookaheadDepth))->BeginInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUAQStrength))->BeginInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUVBVBufsize))->BeginInit();
            this->fcgGroupBoxAspectRatio->SuspendLayout();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUAspectRatioY))->BeginInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUAspectRatioX))->BeginInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNURefFrames))->BeginInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUBframes))->BeginInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUGopLength))->BeginInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgPBNVEncLogoEnabled))->BeginInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgPBNVEncLogoDisabled))->BeginInit();
            this->fcgPNH264->SuspendLayout();
            this->fcgPNQP->SuspendLayout();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUQPI))->BeginInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUQPP))->BeginInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUQPB))->BeginInit();
            this->fcgPNBitrate->SuspendLayout();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUVBRTragetQuality))->BeginInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUBitrate))->BeginInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUMaxkbps))->BeginInit();
            this->fcgPNHEVC->SuspendLayout();
            this->tabPageVideoDetail->SuspendLayout();
            this->groupBoxQPDetail->SuspendLayout();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUChromaQPOffset))->BeginInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUQPInitB))->BeginInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUQPInitP))->BeginInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUQPInitI))->BeginInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUQPMinB))->BeginInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUQPMinP))->BeginInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUQPMinI))->BeginInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUQPMaxB))->BeginInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUQPMaxP))->BeginInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUQPMaxI))->BeginInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUSlices))->BeginInit();
            this->fcgPNH264Detail->SuspendLayout();
            this->fcgPNHEVCDetail->SuspendLayout();
            this->tabPageExOpt->SuspendLayout();
            this->fcggroupBoxVppTweak->SuspendLayout();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgTBVppTweakHue))->BeginInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgTBVppTweakSaturation))->BeginInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgTBVppTweakGamma))->BeginInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgTBVppTweakContrast))->BeginInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgTBVppTweakBrightness))->BeginInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUVppTweakGamma))->BeginInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUVppTweakSaturation))->BeginInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUVppTweakHue))->BeginInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUVppTweakContrast))->BeginInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUVppTweakBrightness))->BeginInit();
            this->fcggroupBoxVppDetailEnahance->SuspendLayout();
            this->fcgPNVppWarpsharp->SuspendLayout();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUVppWarpsharpBlurType))->BeginInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUVppWarpsharpDepth))->BeginInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUVppWarpsharpBlur))->BeginInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUVppWarpsharpThreshold))->BeginInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->numericUpDown6))->BeginInit();
            this->fcgPNVppEdgelevel->SuspendLayout();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUVppEdgelevelWhite))->BeginInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUVppEdgelevelThreshold))->BeginInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUVppEdgelevelBlack))->BeginInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUVppEdgelevelStrength))->BeginInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->numericUpDown4))->BeginInit();
            this->fcgPNVppUnsharp->SuspendLayout();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUVppUnsharpThreshold))->BeginInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUVppUnsharpWeight))->BeginInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUVppUnsharpRadius))->BeginInit();
            this->fcggroupBoxVppDeinterlace->SuspendLayout();
            this->fcgPNVppYadif->SuspendLayout();
            this->fcgPNVppNnedi->SuspendLayout();
            this->fcgPNVppAfs->SuspendLayout();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgTBVppAfsThreCMotion))->BeginInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgTBVppAfsThreYMotion))->BeginInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgTBVppAfsThreDeint))->BeginInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgTBVppAfsThreShift))->BeginInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgTBVppAfsCoeffShift))->BeginInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUVppAfsRight))->BeginInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUVppAfsLeft))->BeginInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUVppAfsBottom))->BeginInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUVppAfsUp))->BeginInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgTBVppAfsMethodSwitch))->BeginInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUVppAfsThreCMotion))->BeginInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUVppAfsThreShift))->BeginInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUVppAfsThreDeint))->BeginInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUVppAfsThreYMotion))->BeginInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUVppAfsCoeffShift))->BeginInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUVppAfsMethodSwitch))->BeginInit();
            this->fcggroupBoxVppDeband->SuspendLayout();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUVppDebandDitherC))->BeginInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUVppDebandDitherY))->BeginInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUVppDebandThreCr))->BeginInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUVppDebandThreCb))->BeginInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUVppDebandThreY))->BeginInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUVppDebandRange))->BeginInit();
            this->fcggroupBoxVppDenoise->SuspendLayout();
            this->fcgPNVppDenoiseConv3D->SuspendLayout();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUVppDenoiseConv3DThreshCTemporal))->BeginInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUVppDenoiseConv3DThreshCSpatial))->BeginInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUVppDenoiseConv3DThreshYTemporal))->BeginInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUVppDenoiseConv3DThreshYSpatial))->BeginInit();
            this->fcgPNVppDenoiseSmooth->SuspendLayout();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUVppDenoiseSmoothQP))->BeginInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUVppDenoiseSmoothQuality))->BeginInit();
            this->fcgPNVppDenoiseKnn->SuspendLayout();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUVppDenoiseKnnThreshold))->BeginInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUVppDenoiseKnnStrength))->BeginInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUVppDenoiseKnnRadius))->BeginInit();
            this->fcgPNVppDenoisePmd->SuspendLayout();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUVppDenoisePmdThreshold))->BeginInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUVppDenoisePmdStrength))->BeginInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUVppDenoisePmdApplyCount))->BeginInit();
            this->fcggroupBoxResize->SuspendLayout();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUVppResizeHeight))->BeginInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUVppResizeWidth))->BeginInit();
            this->tabPageOther->SuspendLayout();
            this->fcggroupBoxCmdEx->SuspendLayout();
            this->fcgCSExeFiles->SuspendLayout();
            this->fcgtabControlAudio->SuspendLayout();
            this->fcgtabPageAudioMain->SuspendLayout();
            this->fcgPNAudioExt->SuspendLayout();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUAudioBitrate))->BeginInit();
            this->fcgPNAudioInternal->SuspendLayout();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUAudioBitrateInternal))->BeginInit();
            this->fcgtabPageAudioOther->SuspendLayout();
            this->SuspendLayout();
            // 
            // fcgtoolStripSettings
            // 
            this->fcgtoolStripSettings->ImageScalingSize = System::Drawing::Size(18, 18);
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
            this->fcgTSBSave->Size = System::Drawing::Size(86, 22);
            this->fcgTSBSave->Text = L"上書き保存";
            this->fcgTSBSave->Click += gcnew System::EventHandler(this, &frmConfig::fcgTSBSave_Click);
            // 
            // fcgTSBSaveNew
            // 
            this->fcgTSBSaveNew->Image = (cli::safe_cast<System::Drawing::Image^>(resources->GetObject(L"fcgTSBSaveNew.Image")));
            this->fcgTSBSaveNew->ImageTransparentColor = System::Drawing::Color::Black;
            this->fcgTSBSaveNew->Name = L"fcgTSBSaveNew";
            this->fcgTSBSaveNew->Size = System::Drawing::Size(77, 22);
            this->fcgTSBSaveNew->Text = L"新規保存";
            this->fcgTSBSaveNew->Click += gcnew System::EventHandler(this, &frmConfig::fcgTSBSaveNew_Click);
            // 
            // fcgTSBDelete
            // 
            this->fcgTSBDelete->Image = (cli::safe_cast<System::Drawing::Image^>(resources->GetObject(L"fcgTSBDelete.Image")));
            this->fcgTSBDelete->ImageTransparentColor = System::Drawing::Color::Magenta;
            this->fcgTSBDelete->Name = L"fcgTSBDelete";
            this->fcgTSBDelete->Size = System::Drawing::Size(53, 22);
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
            this->fcgTSSettings->Size = System::Drawing::Size(81, 22);
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
            this->fcgTSBBitrateCalc->Size = System::Drawing::Size(96, 22);
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
            this->fcgTSBOtherSettings->Size = System::Drawing::Size(76, 22);
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
            // fcgtabControlMux
            // 
            this->fcgtabControlMux->Controls->Add(this->fcgtabPageMP4);
            this->fcgtabControlMux->Controls->Add(this->fcgtabPageMKV);
            this->fcgtabControlMux->Controls->Add(this->fcgtabPageMPG);
            this->fcgtabControlMux->Controls->Add(this->fcgtabPageMux);
            this->fcgtabControlMux->Controls->Add(this->fcgtabPageBat);
            this->fcgtabControlMux->Controls->Add(this->fcgtabPageInternal);
            this->fcgtabControlMux->Font = (gcnew System::Drawing::Font(L"Meiryo UI", 8.25F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
                static_cast<System::Byte>(128)));
            this->fcgtabControlMux->Location = System::Drawing::Point(622, 339);
            this->fcgtabControlMux->Name = L"fcgtabControlMux";
            this->fcgtabControlMux->SelectedIndex = 0;
            this->fcgtabControlMux->Size = System::Drawing::Size(384, 214);
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
            this->fcgtabPageMP4->Size = System::Drawing::Size(376, 187);
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
            this->fcgTXMP4RawPath->Enter += gcnew System::EventHandler(this, &frmConfig::fcgTXMP4RawPath_Enter);
            this->fcgTXMP4RawPath->Leave += gcnew System::EventHandler(this, &frmConfig::fcgTXMP4RawPath_Leave);
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
            this->fcgTXTC2MP4Path->Enter += gcnew System::EventHandler(this, &frmConfig::fcgTXTC2MP4Path_Enter);
            this->fcgTXTC2MP4Path->Leave += gcnew System::EventHandler(this, &frmConfig::fcgTXTC2MP4Path_Leave);
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
            this->fcgTXMP4MuxerPath->Enter += gcnew System::EventHandler(this, &frmConfig::fcgTXMP4MuxerPath_Enter);
            this->fcgTXMP4MuxerPath->Leave += gcnew System::EventHandler(this, &frmConfig::fcgTXMP4MuxerPath_Leave);
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
            this->fcgtabPageMKV->Size = System::Drawing::Size(376, 187);
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
            this->fcgTXMKVMuxerPath->Enter += gcnew System::EventHandler(this, &frmConfig::fcgTXMKVMuxerPath_Enter);
            this->fcgTXMKVMuxerPath->Leave += gcnew System::EventHandler(this, &frmConfig::fcgTXMKVMuxerPath_Leave);
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
            this->fcgtabPageMPG->Size = System::Drawing::Size(376, 187);
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
            this->fcgTXMPGMuxerPath->Enter += gcnew System::EventHandler(this, &frmConfig::fcgTXMPGMuxerPath_Enter);
            this->fcgTXMPGMuxerPath->Leave += gcnew System::EventHandler(this, &frmConfig::fcgTXMPGMuxerPath_Leave);
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
            this->fcgtabPageMux->Size = System::Drawing::Size(376, 187);
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
            this->fcgtabPageBat->Size = System::Drawing::Size(376, 187);
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
            // fcgtabPageInternal
            // 
            this->fcgtabPageInternal->Controls->Add(this->fcgCXInternalCmdEx);
            this->fcgtabPageInternal->Controls->Add(this->fcgLBMuxInternalExt);
            this->fcgtabPageInternal->Location = System::Drawing::Point(4, 23);
            this->fcgtabPageInternal->Name = L"fcgtabPageInternal";
            this->fcgtabPageInternal->Size = System::Drawing::Size(376, 187);
            this->fcgtabPageInternal->TabIndex = 5;
            this->fcgtabPageInternal->Text = L"mux";
            this->fcgtabPageInternal->UseVisualStyleBackColor = true;
            // 
            // fcgCXInternalCmdEx
            // 
            this->fcgCXInternalCmdEx->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
            this->fcgCXInternalCmdEx->FormattingEnabled = true;
            this->fcgCXInternalCmdEx->Location = System::Drawing::Point(86, 17);
            this->fcgCXInternalCmdEx->Name = L"fcgCXInternalCmdEx";
            this->fcgCXInternalCmdEx->Size = System::Drawing::Size(157, 22);
            this->fcgCXInternalCmdEx->TabIndex = 2;
            this->fcgCXInternalCmdEx->Tag = L"chValue";
            // 
            // fcgLBMuxInternalExt
            // 
            this->fcgLBMuxInternalExt->AutoSize = true;
            this->fcgLBMuxInternalExt->Location = System::Drawing::Point(12, 20);
            this->fcgLBMuxInternalExt->Name = L"fcgLBMuxInternalExt";
            this->fcgLBMuxInternalExt->Size = System::Drawing::Size(68, 14);
            this->fcgLBMuxInternalExt->TabIndex = 3;
            this->fcgLBMuxInternalExt->Text = L"拡張オプション";
            // 
            // fcgBTCancel
            // 
            this->fcgBTCancel->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((System::Windows::Forms::AnchorStyles::Bottom | System::Windows::Forms::AnchorStyles::Right));
            this->fcgBTCancel->Location = System::Drawing::Point(771, 586);
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
            this->fcgBTOK->Location = System::Drawing::Point(893, 586);
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
            this->fcgBTDefault->Location = System::Drawing::Point(9, 589);
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
            this->fcgLBVersionDate->Location = System::Drawing::Point(445, 596);
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
            this->fcgLBVersion->Location = System::Drawing::Point(136, 596);
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
            this->fcgtabControlNVEnc->Controls->Add(this->tabPageVideoDetail);
            this->fcgtabControlNVEnc->Controls->Add(this->tabPageExOpt);
            this->fcgtabControlNVEnc->Controls->Add(this->tabPageOther);
            this->fcgtabControlNVEnc->Location = System::Drawing::Point(4, 31);
            this->fcgtabControlNVEnc->Name = L"fcgtabControlNVEnc";
            this->fcgtabControlNVEnc->SelectedIndex = 0;
            this->fcgtabControlNVEnc->Size = System::Drawing::Size(616, 522);
            this->fcgtabControlNVEnc->TabIndex = 49;
            // 
            // tabPageVideoEnc
            // 
            this->tabPageVideoEnc->Controls->Add(this->fcgLBMultiPass);
            this->tabPageVideoEnc->Controls->Add(this->fcgCXMultiPass);
            this->tabPageVideoEnc->Controls->Add(this->fcgLBFullrange);
            this->tabPageVideoEnc->Controls->Add(this->fcgCBFullrange);
            this->tabPageVideoEnc->Controls->Add(this->fcgCXVideoFormat);
            this->tabPageVideoEnc->Controls->Add(this->fcgLBVideoFormat);
            this->tabPageVideoEnc->Controls->Add(this->fcggroupBoxColor);
            this->tabPageVideoEnc->Controls->Add(this->fcgLBQuality);
            this->tabPageVideoEnc->Controls->Add(this->fcgCXQuality);
            this->tabPageVideoEnc->Controls->Add(this->fcgCXBrefMode);
            this->tabPageVideoEnc->Controls->Add(this->fcgLBBrefMode);
            this->tabPageVideoEnc->Controls->Add(this->fcgBTVideoEncoderPath);
            this->tabPageVideoEnc->Controls->Add(this->fcgTXVideoEncoderPath);
            this->tabPageVideoEnc->Controls->Add(this->fcgLBVideoEncoderPath);
            this->tabPageVideoEnc->Controls->Add(this->fcgLBWeightP);
            this->tabPageVideoEnc->Controls->Add(this->fcgLBInterlaced);
            this->tabPageVideoEnc->Controls->Add(this->fcgCBWeightP);
            this->tabPageVideoEnc->Controls->Add(this->label3);
            this->tabPageVideoEnc->Controls->Add(this->fcgLBLookaheadDisable);
            this->tabPageVideoEnc->Controls->Add(this->fcgNULookaheadDepth);
            this->tabPageVideoEnc->Controls->Add(this->fcgLBAQStrengthAuto);
            this->tabPageVideoEnc->Controls->Add(this->fcgCXInterlaced);
            this->tabPageVideoEnc->Controls->Add(this->fcgNUAQStrength);
            this->tabPageVideoEnc->Controls->Add(this->fcgLBAQStrength);
            this->tabPageVideoEnc->Controls->Add(this->fcgCXAQ);
            this->tabPageVideoEnc->Controls->Add(this->fcgLBLookaheadDepth);
            this->tabPageVideoEnc->Controls->Add(this->fcgLBAQ);
            this->tabPageVideoEnc->Controls->Add(this->fcgNUVBVBufsize);
            this->tabPageVideoEnc->Controls->Add(this->fcgGroupBoxAspectRatio);
            this->tabPageVideoEnc->Controls->Add(this->fcgLBVBVBufsize);
            this->tabPageVideoEnc->Controls->Add(this->fcgCBAFS);
            this->tabPageVideoEnc->Controls->Add(this->label1);
            this->tabPageVideoEnc->Controls->Add(this->fcgCXEncCodec);
            this->tabPageVideoEnc->Controls->Add(this->fcgNURefFrames);
            this->tabPageVideoEnc->Controls->Add(this->fcgLBRefFrames);
            this->tabPageVideoEnc->Controls->Add(this->fcgNUBframes);
            this->tabPageVideoEnc->Controls->Add(this->fcgLBBframes);
            this->tabPageVideoEnc->Controls->Add(this->fcgLBGOPLengthAuto);
            this->tabPageVideoEnc->Controls->Add(this->fcgNUGopLength);
            this->tabPageVideoEnc->Controls->Add(this->fcgLBGOPLength);
            this->tabPageVideoEnc->Controls->Add(this->fcgLBEncMode);
            this->tabPageVideoEnc->Controls->Add(this->fcgCXEncMode);
            this->tabPageVideoEnc->Controls->Add(this->fcgPBNVEncLogoEnabled);
            this->tabPageVideoEnc->Controls->Add(this->fcgPBNVEncLogoDisabled);
            this->tabPageVideoEnc->Controls->Add(this->fcgPNH264);
            this->tabPageVideoEnc->Controls->Add(this->fcgPNQP);
            this->tabPageVideoEnc->Controls->Add(this->fcgPNBitrate);
            this->tabPageVideoEnc->Controls->Add(this->fcgPNHEVC);
            this->tabPageVideoEnc->Font = (gcnew System::Drawing::Font(L"Meiryo UI", 8.25F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
                static_cast<System::Byte>(128)));
            this->tabPageVideoEnc->Location = System::Drawing::Point(4, 24);
            this->tabPageVideoEnc->Name = L"tabPageVideoEnc";
            this->tabPageVideoEnc->Padding = System::Windows::Forms::Padding(3);
            this->tabPageVideoEnc->Size = System::Drawing::Size(608, 494);
            this->tabPageVideoEnc->TabIndex = 0;
            this->tabPageVideoEnc->Text = L"動画エンコード";
            this->tabPageVideoEnc->UseVisualStyleBackColor = true;
            // 
            // fcgLBMultiPass
            // 
            this->fcgLBMultiPass->AutoSize = true;
            this->fcgLBMultiPass->Location = System::Drawing::Point(13, 198);
            this->fcgLBMultiPass->Name = L"fcgLBMultiPass";
            this->fcgLBMultiPass->Size = System::Drawing::Size(50, 14);
            this->fcgLBMultiPass->TabIndex = 150;
            this->fcgLBMultiPass->Text = L"マルチパス";
            // 
            // fcgCXMultiPass
            // 
            this->fcgCXMultiPass->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
            this->fcgCXMultiPass->FormattingEnabled = true;
            this->fcgCXMultiPass->Items->AddRange(gcnew cli::array< System::Object^  >(3) { L"高品質", L"標準", L"高速" });
            this->fcgCXMultiPass->Location = System::Drawing::Point(81, 195);
            this->fcgCXMultiPass->Name = L"fcgCXMultiPass";
            this->fcgCXMultiPass->Size = System::Drawing::Size(185, 22);
            this->fcgCXMultiPass->TabIndex = 151;
            this->fcgCXMultiPass->Tag = L"reCmd";
            // 
            // fcgLBFullrange
            // 
            this->fcgLBFullrange->AutoSize = true;
            this->fcgLBFullrange->Location = System::Drawing::Point(361, 297);
            this->fcgLBFullrange->Name = L"fcgLBFullrange";
            this->fcgLBFullrange->Size = System::Drawing::Size(55, 14);
            this->fcgLBFullrange->TabIndex = 102;
            this->fcgLBFullrange->Text = L"fullrange";
            // 
            // fcgCBFullrange
            // 
            this->fcgCBFullrange->AutoSize = true;
            this->fcgCBFullrange->Location = System::Drawing::Point(468, 301);
            this->fcgCBFullrange->Name = L"fcgCBFullrange";
            this->fcgCBFullrange->Size = System::Drawing::Size(15, 14);
            this->fcgCBFullrange->TabIndex = 103;
            this->fcgCBFullrange->Tag = L"reCmd";
            this->fcgCBFullrange->UseVisualStyleBackColor = true;
            // 
            // fcgCXVideoFormat
            // 
            this->fcgCXVideoFormat->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
            this->fcgCXVideoFormat->FormattingEnabled = true;
            this->fcgCXVideoFormat->Location = System::Drawing::Point(468, 268);
            this->fcgCXVideoFormat->Name = L"fcgCXVideoFormat";
            this->fcgCXVideoFormat->Size = System::Drawing::Size(121, 22);
            this->fcgCXVideoFormat->TabIndex = 101;
            this->fcgCXVideoFormat->Tag = L"reCmd";
            // 
            // fcgLBVideoFormat
            // 
            this->fcgLBVideoFormat->AutoSize = true;
            this->fcgLBVideoFormat->Location = System::Drawing::Point(361, 270);
            this->fcgLBVideoFormat->Name = L"fcgLBVideoFormat";
            this->fcgLBVideoFormat->Size = System::Drawing::Size(73, 14);
            this->fcgLBVideoFormat->TabIndex = 100;
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
            this->fcggroupBoxColor->Location = System::Drawing::Point(357, 317);
            this->fcggroupBoxColor->Name = L"fcggroupBoxColor";
            this->fcggroupBoxColor->Size = System::Drawing::Size(241, 103);
            this->fcggroupBoxColor->TabIndex = 110;
            this->fcggroupBoxColor->TabStop = false;
            this->fcggroupBoxColor->Text = L"色設定";
            // 
            // fcgCXTransfer
            // 
            this->fcgCXTransfer->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
            this->fcgCXTransfer->FormattingEnabled = true;
            this->fcgCXTransfer->Location = System::Drawing::Point(111, 72);
            this->fcgCXTransfer->Name = L"fcgCXTransfer";
            this->fcgCXTransfer->Size = System::Drawing::Size(121, 22);
            this->fcgCXTransfer->TabIndex = 2;
            this->fcgCXTransfer->Tag = L"reCmd";
            // 
            // fcgCXColorPrim
            // 
            this->fcgCXColorPrim->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
            this->fcgCXColorPrim->FormattingEnabled = true;
            this->fcgCXColorPrim->Location = System::Drawing::Point(111, 44);
            this->fcgCXColorPrim->Name = L"fcgCXColorPrim";
            this->fcgCXColorPrim->Size = System::Drawing::Size(121, 22);
            this->fcgCXColorPrim->TabIndex = 3;
            this->fcgCXColorPrim->Tag = L"reCmd";
            // 
            // fcgCXColorMatrix
            // 
            this->fcgCXColorMatrix->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
            this->fcgCXColorMatrix->FormattingEnabled = true;
            this->fcgCXColorMatrix->Location = System::Drawing::Point(111, 16);
            this->fcgCXColorMatrix->Name = L"fcgCXColorMatrix";
            this->fcgCXColorMatrix->Size = System::Drawing::Size(121, 22);
            this->fcgCXColorMatrix->TabIndex = 1;
            this->fcgCXColorMatrix->Tag = L"reCmd";
            // 
            // fcgLBTransfer
            // 
            this->fcgLBTransfer->AutoSize = true;
            this->fcgLBTransfer->Location = System::Drawing::Point(18, 75);
            this->fcgLBTransfer->Name = L"fcgLBTransfer";
            this->fcgLBTransfer->Size = System::Drawing::Size(49, 14);
            this->fcgLBTransfer->TabIndex = 4;
            this->fcgLBTransfer->Text = L"transfer";
            // 
            // fcgLBColorPrim
            // 
            this->fcgLBColorPrim->AutoSize = true;
            this->fcgLBColorPrim->Location = System::Drawing::Point(18, 47);
            this->fcgLBColorPrim->Name = L"fcgLBColorPrim";
            this->fcgLBColorPrim->Size = System::Drawing::Size(61, 14);
            this->fcgLBColorPrim->TabIndex = 2;
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
            // fcgLBQuality
            // 
            this->fcgLBQuality->AutoSize = true;
            this->fcgLBQuality->Location = System::Drawing::Point(13, 170);
            this->fcgLBQuality->Name = L"fcgLBQuality";
            this->fcgLBQuality->Size = System::Drawing::Size(29, 14);
            this->fcgLBQuality->TabIndex = 8;
            this->fcgLBQuality->Text = L"品質";
            // 
            // fcgCXQuality
            // 
            this->fcgCXQuality->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
            this->fcgCXQuality->FormattingEnabled = true;
            this->fcgCXQuality->Items->AddRange(gcnew cli::array< System::Object^  >(3) { L"高品質", L"標準", L"高速" });
            this->fcgCXQuality->Location = System::Drawing::Point(81, 167);
            this->fcgCXQuality->Name = L"fcgCXQuality";
            this->fcgCXQuality->Size = System::Drawing::Size(185, 22);
            this->fcgCXQuality->TabIndex = 9;
            this->fcgCXQuality->Tag = L"reCmd";
            // 
            // fcgCXBrefMode
            // 
            this->fcgCXBrefMode->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
            this->fcgCXBrefMode->FormattingEnabled = true;
            this->fcgCXBrefMode->Location = System::Drawing::Point(132, 387);
            this->fcgCXBrefMode->Name = L"fcgCXBrefMode";
            this->fcgCXBrefMode->Size = System::Drawing::Size(122, 22);
            this->fcgCXBrefMode->TabIndex = 39;
            this->fcgCXBrefMode->Tag = L"reCmd";
            // 
            // fcgLBBrefMode
            // 
            this->fcgLBBrefMode->AutoSize = true;
            this->fcgLBBrefMode->Location = System::Drawing::Point(14, 389);
            this->fcgLBBrefMode->Name = L"fcgLBBrefMode";
            this->fcgLBBrefMode->Size = System::Drawing::Size(94, 14);
            this->fcgLBBrefMode->TabIndex = 38;
            this->fcgLBBrefMode->Text = L"Bフレーム参照モード";
            // 
            // fcgBTVideoEncoderPath
            // 
            this->fcgBTVideoEncoderPath->Location = System::Drawing::Point(303, 81);
            this->fcgBTVideoEncoderPath->Name = L"fcgBTVideoEncoderPath";
            this->fcgBTVideoEncoderPath->Size = System::Drawing::Size(30, 23);
            this->fcgBTVideoEncoderPath->TabIndex = 3;
            this->fcgBTVideoEncoderPath->Text = L"...";
            this->fcgBTVideoEncoderPath->UseVisualStyleBackColor = true;
            this->fcgBTVideoEncoderPath->Click += gcnew System::EventHandler(this, &frmConfig::fcgBTVideoEncoderPath_Click);
            // 
            // fcgTXVideoEncoderPath
            // 
            this->fcgTXVideoEncoderPath->AllowDrop = true;
            this->fcgTXVideoEncoderPath->Location = System::Drawing::Point(81, 82);
            this->fcgTXVideoEncoderPath->Name = L"fcgTXVideoEncoderPath";
            this->fcgTXVideoEncoderPath->Size = System::Drawing::Size(216, 21);
            this->fcgTXVideoEncoderPath->TabIndex = 2;
            this->fcgTXVideoEncoderPath->TextChanged += gcnew System::EventHandler(this, &frmConfig::fcgTXVideoEncoderPath_TextChanged);
            this->fcgTXVideoEncoderPath->DragDrop += gcnew System::Windows::Forms::DragEventHandler(this, &frmConfig::fcgSetDragDropFilename_DragDrop);
            this->fcgTXVideoEncoderPath->DragEnter += gcnew System::Windows::Forms::DragEventHandler(this, &frmConfig::fcgSetDragDropFilename_Enter);
            this->fcgTXVideoEncoderPath->Enter += gcnew System::EventHandler(this, &frmConfig::fcgTXVideoEncoderPath_Enter);
            this->fcgTXVideoEncoderPath->Leave += gcnew System::EventHandler(this, &frmConfig::fcgTXVideoEncoderPath_Leave);
            // 
            // fcgLBVideoEncoderPath
            // 
            this->fcgLBVideoEncoderPath->AutoSize = true;
            this->fcgLBVideoEncoderPath->Location = System::Drawing::Point(6, 82);
            this->fcgLBVideoEncoderPath->Name = L"fcgLBVideoEncoderPath";
            this->fcgLBVideoEncoderPath->Size = System::Drawing::Size(49, 14);
            this->fcgLBVideoEncoderPath->TabIndex = 1;
            this->fcgLBVideoEncoderPath->Text = L"～の指定";
            // 
            // fcgLBWeightP
            // 
            this->fcgLBWeightP->AutoSize = true;
            this->fcgLBWeightP->Location = System::Drawing::Point(15, 470);
            this->fcgLBWeightP->Name = L"fcgLBWeightP";
            this->fcgLBWeightP->Size = System::Drawing::Size(87, 14);
            this->fcgLBWeightP->TabIndex = 45;
            this->fcgLBWeightP->Text = L"重み付きPフレーム";
            // 
            // fcgLBInterlaced
            // 
            this->fcgLBInterlaced->AutoSize = true;
            this->fcgLBInterlaced->Location = System::Drawing::Point(357, 142);
            this->fcgLBInterlaced->Name = L"fcgLBInterlaced";
            this->fcgLBInterlaced->Size = System::Drawing::Size(86, 14);
            this->fcgLBInterlaced->TabIndex = 70;
            this->fcgLBInterlaced->Text = L"入力フレームタイプ";
            // 
            // fcgCBWeightP
            // 
            this->fcgCBWeightP->AutoSize = true;
            this->fcgCBWeightP->Location = System::Drawing::Point(132, 473);
            this->fcgCBWeightP->Name = L"fcgCBWeightP";
            this->fcgCBWeightP->Size = System::Drawing::Size(15, 14);
            this->fcgCBWeightP->TabIndex = 46;
            this->fcgCBWeightP->Tag = L"reCmd";
            this->fcgCBWeightP->UseVisualStyleBackColor = true;
            // 
            // label3
            // 
            this->label3->AutoSize = true;
            this->label3->Location = System::Drawing::Point(215, 310);
            this->label3->Name = L"label3";
            this->label3->Size = System::Drawing::Size(32, 14);
            this->label3->TabIndex = 32;
            this->label3->Text = L"kbps";
            // 
            // fcgLBLookaheadDisable
            // 
            this->fcgLBLookaheadDisable->AutoSize = true;
            this->fcgLBLookaheadDisable->Location = System::Drawing::Point(215, 445);
            this->fcgLBLookaheadDisable->Name = L"fcgLBLookaheadDisable";
            this->fcgLBLookaheadDisable->Size = System::Drawing::Size(66, 14);
            this->fcgLBLookaheadDisable->TabIndex = 44;
            this->fcgLBLookaheadDisable->Text = L"※\"0\"で無効";
            // 
            // fcgNULookaheadDepth
            // 
            this->fcgNULookaheadDepth->Location = System::Drawing::Point(132, 443);
            this->fcgNULookaheadDepth->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 32, 0, 0, 0 });
            this->fcgNULookaheadDepth->Name = L"fcgNULookaheadDepth";
            this->fcgNULookaheadDepth->Size = System::Drawing::Size(77, 21);
            this->fcgNULookaheadDepth->TabIndex = 43;
            this->fcgNULookaheadDepth->Tag = L"reCmd";
            this->fcgNULookaheadDepth->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
            // 
            // fcgLBAQStrengthAuto
            // 
            this->fcgLBAQStrengthAuto->AutoSize = true;
            this->fcgLBAQStrengthAuto->Location = System::Drawing::Point(536, 465);
            this->fcgLBAQStrengthAuto->Name = L"fcgLBAQStrengthAuto";
            this->fcgLBAQStrengthAuto->Size = System::Drawing::Size(66, 14);
            this->fcgLBAQStrengthAuto->TabIndex = 124;
            this->fcgLBAQStrengthAuto->Text = L"※\"0\"で自動";
            // 
            // fcgCXInterlaced
            // 
            this->fcgCXInterlaced->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
            this->fcgCXInterlaced->FormattingEnabled = true;
            this->fcgCXInterlaced->Location = System::Drawing::Point(468, 138);
            this->fcgCXInterlaced->Name = L"fcgCXInterlaced";
            this->fcgCXInterlaced->Size = System::Drawing::Size(121, 22);
            this->fcgCXInterlaced->TabIndex = 71;
            this->fcgCXInterlaced->Tag = L"reCmd";
            // 
            // fcgNUAQStrength
            // 
            this->fcgNUAQStrength->Location = System::Drawing::Point(468, 463);
            this->fcgNUAQStrength->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 15, 0, 0, 0 });
            this->fcgNUAQStrength->Name = L"fcgNUAQStrength";
            this->fcgNUAQStrength->Size = System::Drawing::Size(57, 21);
            this->fcgNUAQStrength->TabIndex = 123;
            this->fcgNUAQStrength->Tag = L"reCmd";
            this->fcgNUAQStrength->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
            // 
            // fcgLBAQStrength
            // 
            this->fcgLBAQStrength->AutoSize = true;
            this->fcgLBAQStrength->Location = System::Drawing::Point(350, 464);
            this->fcgLBAQStrength->Name = L"fcgLBAQStrength";
            this->fcgLBAQStrength->Size = System::Drawing::Size(107, 14);
            this->fcgLBAQStrength->TabIndex = 122;
            this->fcgLBAQStrength->Text = L"AQstrength (1-15)";
            // 
            // fcgCXAQ
            // 
            this->fcgCXAQ->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
            this->fcgCXAQ->FormattingEnabled = true;
            this->fcgCXAQ->Location = System::Drawing::Point(468, 432);
            this->fcgCXAQ->Name = L"fcgCXAQ";
            this->fcgCXAQ->Size = System::Drawing::Size(122, 22);
            this->fcgCXAQ->TabIndex = 121;
            this->fcgCXAQ->Tag = L"reCmd";
            // 
            // fcgLBLookaheadDepth
            // 
            this->fcgLBLookaheadDepth->AutoSize = true;
            this->fcgLBLookaheadDepth->Location = System::Drawing::Point(14, 445);
            this->fcgLBLookaheadDepth->Name = L"fcgLBLookaheadDepth";
            this->fcgLBLookaheadDepth->Size = System::Drawing::Size(100, 14);
            this->fcgLBLookaheadDepth->TabIndex = 42;
            this->fcgLBLookaheadDepth->Text = L"Lookahead depth";
            // 
            // fcgLBAQ
            // 
            this->fcgLBAQ->AutoSize = true;
            this->fcgLBAQ->Location = System::Drawing::Point(350, 434);
            this->fcgLBAQ->Name = L"fcgLBAQ";
            this->fcgLBAQ->Size = System::Drawing::Size(102, 14);
            this->fcgLBAQ->TabIndex = 120;
            this->fcgLBAQ->Text = L"適応的量子化 (AQ)";
            // 
            // fcgNUVBVBufsize
            // 
            this->fcgNUVBVBufsize->Increment = System::Decimal(gcnew cli::array< System::Int32 >(4) { 500, 0, 0, 0 });
            this->fcgNUVBVBufsize->Location = System::Drawing::Point(132, 306);
            this->fcgNUVBVBufsize->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 128000, 0, 0, 0 });
            this->fcgNUVBVBufsize->Name = L"fcgNUVBVBufsize";
            this->fcgNUVBVBufsize->Size = System::Drawing::Size(77, 21);
            this->fcgNUVBVBufsize->TabIndex = 31;
            this->fcgNUVBVBufsize->Tag = L"reCmd";
            this->fcgNUVBVBufsize->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
            // 
            // fcgGroupBoxAspectRatio
            // 
            this->fcgGroupBoxAspectRatio->Controls->Add(this->fcgLBAspectRatio);
            this->fcgGroupBoxAspectRatio->Controls->Add(this->fcgNUAspectRatioY);
            this->fcgGroupBoxAspectRatio->Controls->Add(this->fcgNUAspectRatioX);
            this->fcgGroupBoxAspectRatio->Controls->Add(this->fcgCXAspectRatio);
            this->fcgGroupBoxAspectRatio->Location = System::Drawing::Point(353, 43);
            this->fcgGroupBoxAspectRatio->Name = L"fcgGroupBoxAspectRatio";
            this->fcgGroupBoxAspectRatio->Size = System::Drawing::Size(241, 76);
            this->fcgGroupBoxAspectRatio->TabIndex = 61;
            this->fcgGroupBoxAspectRatio->TabStop = false;
            this->fcgGroupBoxAspectRatio->Text = L"アスペクト比";
            // 
            // fcgLBAspectRatio
            // 
            this->fcgLBAspectRatio->AutoSize = true;
            this->fcgLBAspectRatio->Location = System::Drawing::Point(131, 47);
            this->fcgLBAspectRatio->Name = L"fcgLBAspectRatio";
            this->fcgLBAspectRatio->Size = System::Drawing::Size(12, 14);
            this->fcgLBAspectRatio->TabIndex = 3;
            this->fcgLBAspectRatio->Text = L":";
            // 
            // fcgNUAspectRatioY
            // 
            this->fcgNUAspectRatioY->Location = System::Drawing::Point(149, 45);
            this->fcgNUAspectRatioY->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 65535, 0, 0, 0 });
            this->fcgNUAspectRatioY->Name = L"fcgNUAspectRatioY";
            this->fcgNUAspectRatioY->Size = System::Drawing::Size(60, 21);
            this->fcgNUAspectRatioY->TabIndex = 2;
            this->fcgNUAspectRatioY->Tag = L"reCmd";
            this->fcgNUAspectRatioY->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
            // 
            // fcgNUAspectRatioX
            // 
            this->fcgNUAspectRatioX->Location = System::Drawing::Point(65, 45);
            this->fcgNUAspectRatioX->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 65535, 0, 0, 0 });
            this->fcgNUAspectRatioX->Name = L"fcgNUAspectRatioX";
            this->fcgNUAspectRatioX->Size = System::Drawing::Size(60, 21);
            this->fcgNUAspectRatioX->TabIndex = 1;
            this->fcgNUAspectRatioX->Tag = L"reCmd";
            this->fcgNUAspectRatioX->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
            // 
            // fcgCXAspectRatio
            // 
            this->fcgCXAspectRatio->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
            this->fcgCXAspectRatio->FormattingEnabled = true;
            this->fcgCXAspectRatio->Location = System::Drawing::Point(26, 18);
            this->fcgCXAspectRatio->Name = L"fcgCXAspectRatio";
            this->fcgCXAspectRatio->Size = System::Drawing::Size(197, 22);
            this->fcgCXAspectRatio->TabIndex = 0;
            this->fcgCXAspectRatio->Tag = L"reCmd";
            // 
            // fcgLBVBVBufsize
            // 
            this->fcgLBVBVBufsize->AutoSize = true;
            this->fcgLBVBVBufsize->Location = System::Drawing::Point(14, 308);
            this->fcgLBVBVBufsize->Name = L"fcgLBVBVBufsize";
            this->fcgLBVBVBufsize->Size = System::Drawing::Size(84, 14);
            this->fcgLBVBVBufsize->TabIndex = 30;
            this->fcgLBVBVBufsize->Text = L"VBVバッファサイズ";
            // 
            // fcgCBAFS
            // 
            this->fcgCBAFS->AutoSize = true;
            this->fcgCBAFS->Location = System::Drawing::Point(362, 16);
            this->fcgCBAFS->Name = L"fcgCBAFS";
            this->fcgCBAFS->Size = System::Drawing::Size(183, 18);
            this->fcgCBAFS->TabIndex = 60;
            this->fcgCBAFS->Tag = L"chValue";
            this->fcgCBAFS->Text = L"自動フィールドシフト(afs)を使用する";
            this->fcgCBAFS->UseVisualStyleBackColor = true;
            // 
            // label1
            // 
            this->label1->AutoSize = true;
            this->label1->Location = System::Drawing::Point(13, 114);
            this->label1->Name = L"label1";
            this->label1->Size = System::Drawing::Size(51, 14);
            this->label1->TabIndex = 4;
            this->label1->Text = L"出力形式";
            // 
            // fcgCXEncCodec
            // 
            this->fcgCXEncCodec->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
            this->fcgCXEncCodec->FormattingEnabled = true;
            this->fcgCXEncCodec->Items->AddRange(gcnew cli::array< System::Object^  >(3) { L"高品質", L"標準", L"高速" });
            this->fcgCXEncCodec->Location = System::Drawing::Point(81, 111);
            this->fcgCXEncCodec->Name = L"fcgCXEncCodec";
            this->fcgCXEncCodec->Size = System::Drawing::Size(185, 22);
            this->fcgCXEncCodec->TabIndex = 5;
            this->fcgCXEncCodec->Tag = L"reCmd";
            this->fcgCXEncCodec->SelectedIndexChanged += gcnew System::EventHandler(this, &frmConfig::fcgChangeEnabled);
            // 
            // fcgNURefFrames
            // 
            this->fcgNURefFrames->Location = System::Drawing::Point(132, 415);
            this->fcgNURefFrames->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 16, 0, 0, 0 });
            this->fcgNURefFrames->Name = L"fcgNURefFrames";
            this->fcgNURefFrames->Size = System::Drawing::Size(77, 21);
            this->fcgNURefFrames->TabIndex = 41;
            this->fcgNURefFrames->Tag = L"reCmd";
            this->fcgNURefFrames->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
            // 
            // fcgLBRefFrames
            // 
            this->fcgLBRefFrames->AutoSize = true;
            this->fcgLBRefFrames->Location = System::Drawing::Point(13, 418);
            this->fcgLBRefFrames->Name = L"fcgLBRefFrames";
            this->fcgLBRefFrames->Size = System::Drawing::Size(51, 14);
            this->fcgLBRefFrames->TabIndex = 40;
            this->fcgLBRefFrames->Text = L"参照距離";
            // 
            // fcgNUBframes
            // 
            this->fcgNUBframes->Location = System::Drawing::Point(132, 361);
            this->fcgNUBframes->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 16, 0, 0, 0 });
            this->fcgNUBframes->Name = L"fcgNUBframes";
            this->fcgNUBframes->Size = System::Drawing::Size(77, 21);
            this->fcgNUBframes->TabIndex = 37;
            this->fcgNUBframes->Tag = L"reCmd";
            this->fcgNUBframes->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
            // 
            // fcgLBBframes
            // 
            this->fcgLBBframes->AutoSize = true;
            this->fcgLBBframes->Location = System::Drawing::Point(14, 364);
            this->fcgLBBframes->Name = L"fcgLBBframes";
            this->fcgLBBframes->Size = System::Drawing::Size(58, 14);
            this->fcgLBBframes->TabIndex = 36;
            this->fcgLBBframes->Text = L"Bフレーム数";
            // 
            // fcgLBGOPLengthAuto
            // 
            this->fcgLBGOPLengthAuto->AutoSize = true;
            this->fcgLBGOPLengthAuto->Location = System::Drawing::Point(214, 336);
            this->fcgLBGOPLengthAuto->Name = L"fcgLBGOPLengthAuto";
            this->fcgLBGOPLengthAuto->Size = System::Drawing::Size(66, 14);
            this->fcgLBGOPLengthAuto->TabIndex = 35;
            this->fcgLBGOPLengthAuto->Text = L"※\"0\"で自動";
            // 
            // fcgNUGopLength
            // 
            this->fcgNUGopLength->Location = System::Drawing::Point(132, 333);
            this->fcgNUGopLength->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 120000, 0, 0, 0 });
            this->fcgNUGopLength->Name = L"fcgNUGopLength";
            this->fcgNUGopLength->Size = System::Drawing::Size(77, 21);
            this->fcgNUGopLength->TabIndex = 34;
            this->fcgNUGopLength->Tag = L"reCmd";
            this->fcgNUGopLength->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
            // 
            // fcgLBGOPLength
            // 
            this->fcgLBGOPLength->AutoSize = true;
            this->fcgLBGOPLength->Location = System::Drawing::Point(14, 336);
            this->fcgLBGOPLength->Name = L"fcgLBGOPLength";
            this->fcgLBGOPLength->Size = System::Drawing::Size(41, 14);
            this->fcgLBGOPLength->TabIndex = 33;
            this->fcgLBGOPLength->Text = L"GOP長";
            // 
            // fcgLBEncMode
            // 
            this->fcgLBEncMode->AutoSize = true;
            this->fcgLBEncMode->Location = System::Drawing::Point(13, 142);
            this->fcgLBEncMode->Name = L"fcgLBEncMode";
            this->fcgLBEncMode->Size = System::Drawing::Size(32, 14);
            this->fcgLBEncMode->TabIndex = 6;
            this->fcgLBEncMode->Text = L"モード";
            // 
            // fcgCXEncMode
            // 
            this->fcgCXEncMode->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
            this->fcgCXEncMode->FormattingEnabled = true;
            this->fcgCXEncMode->Items->AddRange(gcnew cli::array< System::Object^  >(3) { L"高品質", L"標準", L"高速" });
            this->fcgCXEncMode->Location = System::Drawing::Point(81, 139);
            this->fcgCXEncMode->Name = L"fcgCXEncMode";
            this->fcgCXEncMode->Size = System::Drawing::Size(185, 22);
            this->fcgCXEncMode->TabIndex = 7;
            this->fcgCXEncMode->Tag = L"reCmd";
            this->fcgCXEncMode->SelectedIndexChanged += gcnew System::EventHandler(this, &frmConfig::fcgChangeEnabled);
            // 
            // fcgPBNVEncLogoEnabled
            // 
            this->fcgPBNVEncLogoEnabled->Image = (cli::safe_cast<System::Drawing::Image^>(resources->GetObject(L"fcgPBNVEncLogoEnabled.Image")));
            this->fcgPBNVEncLogoEnabled->Location = System::Drawing::Point(6, 1);
            this->fcgPBNVEncLogoEnabled->Name = L"fcgPBNVEncLogoEnabled";
            this->fcgPBNVEncLogoEnabled->Size = System::Drawing::Size(219, 75);
            this->fcgPBNVEncLogoEnabled->SizeMode = System::Windows::Forms::PictureBoxSizeMode::StretchImage;
            this->fcgPBNVEncLogoEnabled->TabIndex = 148;
            this->fcgPBNVEncLogoEnabled->TabStop = false;
            // 
            // fcgPBNVEncLogoDisabled
            // 
            this->fcgPBNVEncLogoDisabled->Image = (cli::safe_cast<System::Drawing::Image^>(resources->GetObject(L"fcgPBNVEncLogoDisabled.Image")));
            this->fcgPBNVEncLogoDisabled->Location = System::Drawing::Point(6, 1);
            this->fcgPBNVEncLogoDisabled->Name = L"fcgPBNVEncLogoDisabled";
            this->fcgPBNVEncLogoDisabled->Size = System::Drawing::Size(219, 75);
            this->fcgPBNVEncLogoDisabled->SizeMode = System::Windows::Forms::PictureBoxSizeMode::StretchImage;
            this->fcgPBNVEncLogoDisabled->TabIndex = 149;
            this->fcgPBNVEncLogoDisabled->TabStop = false;
            // 
            // fcgPNH264
            // 
            this->fcgPNH264->Controls->Add(this->fcgLBBluray);
            this->fcgPNH264->Controls->Add(this->fcgCBBluray);
            this->fcgPNH264->Controls->Add(this->fcgLBCodecProfile);
            this->fcgPNH264->Controls->Add(this->fcgLBCodecLevel);
            this->fcgPNH264->Controls->Add(this->fcgCXCodecProfile);
            this->fcgPNH264->Controls->Add(this->fcgCXCodecLevel);
            this->fcgPNH264->Location = System::Drawing::Point(341, 165);
            this->fcgPNH264->Name = L"fcgPNH264";
            this->fcgPNH264->Size = System::Drawing::Size(264, 91);
            this->fcgPNH264->TabIndex = 90;
            // 
            // fcgLBBluray
            // 
            this->fcgLBBluray->AutoSize = true;
            this->fcgLBBluray->Location = System::Drawing::Point(16, 8);
            this->fcgLBBluray->Name = L"fcgLBBluray";
            this->fcgLBBluray->Size = System::Drawing::Size(74, 14);
            this->fcgLBBluray->TabIndex = 91;
            this->fcgLBBluray->Text = L"Bluray用出力";
            // 
            // fcgCBBluray
            // 
            this->fcgCBBluray->AutoSize = true;
            this->fcgCBBluray->Location = System::Drawing::Point(128, 9);
            this->fcgCBBluray->Name = L"fcgCBBluray";
            this->fcgCBBluray->Size = System::Drawing::Size(15, 14);
            this->fcgCBBluray->TabIndex = 92;
            this->fcgCBBluray->Tag = L"reCmd";
            this->fcgCBBluray->UseVisualStyleBackColor = true;
            // 
            // fcgLBCodecProfile
            // 
            this->fcgLBCodecProfile->AutoSize = true;
            this->fcgLBCodecProfile->Location = System::Drawing::Point(15, 38);
            this->fcgLBCodecProfile->Name = L"fcgLBCodecProfile";
            this->fcgLBCodecProfile->Size = System::Drawing::Size(53, 14);
            this->fcgLBCodecProfile->TabIndex = 93;
            this->fcgLBCodecProfile->Text = L"プロファイル";
            // 
            // fcgLBCodecLevel
            // 
            this->fcgLBCodecLevel->AutoSize = true;
            this->fcgLBCodecLevel->Location = System::Drawing::Point(15, 69);
            this->fcgLBCodecLevel->Name = L"fcgLBCodecLevel";
            this->fcgLBCodecLevel->Size = System::Drawing::Size(33, 14);
            this->fcgLBCodecLevel->TabIndex = 95;
            this->fcgLBCodecLevel->Text = L"レベル";
            // 
            // fcgCXCodecProfile
            // 
            this->fcgCXCodecProfile->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
            this->fcgCXCodecProfile->FormattingEnabled = true;
            this->fcgCXCodecProfile->Location = System::Drawing::Point(126, 35);
            this->fcgCXCodecProfile->Name = L"fcgCXCodecProfile";
            this->fcgCXCodecProfile->Size = System::Drawing::Size(121, 22);
            this->fcgCXCodecProfile->TabIndex = 94;
            this->fcgCXCodecProfile->Tag = L"reCmd";
            // 
            // fcgCXCodecLevel
            // 
            this->fcgCXCodecLevel->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
            this->fcgCXCodecLevel->FormattingEnabled = true;
            this->fcgCXCodecLevel->Location = System::Drawing::Point(126, 65);
            this->fcgCXCodecLevel->Name = L"fcgCXCodecLevel";
            this->fcgCXCodecLevel->Size = System::Drawing::Size(121, 22);
            this->fcgCXCodecLevel->TabIndex = 96;
            this->fcgCXCodecLevel->Tag = L"reCmd";
            // 
            // fcgPNQP
            // 
            this->fcgPNQP->Controls->Add(this->fcgLBQPI);
            this->fcgPNQP->Controls->Add(this->fcgNUQPI);
            this->fcgPNQP->Controls->Add(this->fcgNUQPP);
            this->fcgPNQP->Controls->Add(this->fcgNUQPB);
            this->fcgPNQP->Controls->Add(this->fcgLBQPP);
            this->fcgPNQP->Controls->Add(this->fcgLBQPB);
            this->fcgPNQP->Location = System::Drawing::Point(8, 223);
            this->fcgPNQP->Name = L"fcgPNQP";
            this->fcgPNQP->Size = System::Drawing::Size(289, 79);
            this->fcgPNQP->TabIndex = 10;
            // 
            // fcgLBQPI
            // 
            this->fcgLBQPI->AutoSize = true;
            this->fcgLBQPI->Location = System::Drawing::Point(10, 4);
            this->fcgLBQPI->Name = L"fcgLBQPI";
            this->fcgLBQPI->Size = System::Drawing::Size(66, 14);
            this->fcgLBQPI->TabIndex = 11;
            this->fcgLBQPI->Text = L"QP I frame";
            // 
            // fcgNUQPI
            // 
            this->fcgNUQPI->Location = System::Drawing::Point(124, 2);
            this->fcgNUQPI->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 51, 0, 0, 0 });
            this->fcgNUQPI->Name = L"fcgNUQPI";
            this->fcgNUQPI->Size = System::Drawing::Size(77, 21);
            this->fcgNUQPI->TabIndex = 12;
            this->fcgNUQPI->Tag = L"reCmd";
            this->fcgNUQPI->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
            // 
            // fcgNUQPP
            // 
            this->fcgNUQPP->Location = System::Drawing::Point(124, 29);
            this->fcgNUQPP->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 51, 0, 0, 0 });
            this->fcgNUQPP->Name = L"fcgNUQPP";
            this->fcgNUQPP->Size = System::Drawing::Size(77, 21);
            this->fcgNUQPP->TabIndex = 14;
            this->fcgNUQPP->Tag = L"reCmd";
            this->fcgNUQPP->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
            // 
            // fcgNUQPB
            // 
            this->fcgNUQPB->Location = System::Drawing::Point(124, 55);
            this->fcgNUQPB->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 51, 0, 0, 0 });
            this->fcgNUQPB->Name = L"fcgNUQPB";
            this->fcgNUQPB->Size = System::Drawing::Size(77, 21);
            this->fcgNUQPB->TabIndex = 16;
            this->fcgNUQPB->Tag = L"reCmd";
            this->fcgNUQPB->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
            // 
            // fcgLBQPP
            // 
            this->fcgLBQPP->AutoSize = true;
            this->fcgLBQPP->Location = System::Drawing::Point(10, 31);
            this->fcgLBQPP->Name = L"fcgLBQPP";
            this->fcgLBQPP->Size = System::Drawing::Size(69, 14);
            this->fcgLBQPP->TabIndex = 13;
            this->fcgLBQPP->Text = L"QP P frame";
            // 
            // fcgLBQPB
            // 
            this->fcgLBQPB->AutoSize = true;
            this->fcgLBQPB->Location = System::Drawing::Point(10, 57);
            this->fcgLBQPB->Name = L"fcgLBQPB";
            this->fcgLBQPB->Size = System::Drawing::Size(69, 14);
            this->fcgLBQPB->TabIndex = 15;
            this->fcgLBQPB->Text = L"QP B frame";
            // 
            // fcgPNBitrate
            // 
            this->fcgPNBitrate->Controls->Add(this->fcgLBVBRTragetQuality2);
            this->fcgPNBitrate->Controls->Add(this->fcgNUVBRTragetQuality);
            this->fcgPNBitrate->Controls->Add(this->fcgLBVBRTragetQuality);
            this->fcgPNBitrate->Controls->Add(this->fcgLBBitrate);
            this->fcgPNBitrate->Controls->Add(this->fcgNUBitrate);
            this->fcgPNBitrate->Controls->Add(this->fcgLBBitrate2);
            this->fcgPNBitrate->Controls->Add(this->fcgNUMaxkbps);
            this->fcgPNBitrate->Controls->Add(this->fcgLBMaxkbps);
            this->fcgPNBitrate->Controls->Add(this->fcgLBMaxBitrate2);
            this->fcgPNBitrate->Location = System::Drawing::Point(8, 223);
            this->fcgPNBitrate->Name = L"fcgPNBitrate";
            this->fcgPNBitrate->Size = System::Drawing::Size(289, 79);
            this->fcgPNBitrate->TabIndex = 20;
            // 
            // fcgLBVBRTragetQuality2
            // 
            this->fcgLBVBRTragetQuality2->AutoSize = true;
            this->fcgLBVBRTragetQuality2->Location = System::Drawing::Point(207, 57);
            this->fcgLBVBRTragetQuality2->Name = L"fcgLBVBRTragetQuality2";
            this->fcgLBVBRTragetQuality2->Size = System::Drawing::Size(66, 14);
            this->fcgLBVBRTragetQuality2->TabIndex = 29;
            this->fcgLBVBRTragetQuality2->Text = L"※\"0\"で自動";
            // 
            // fcgNUVBRTragetQuality
            // 
            this->fcgNUVBRTragetQuality->DecimalPlaces = 2;
            this->fcgNUVBRTragetQuality->Increment = System::Decimal(gcnew cli::array< System::Int32 >(4) { 5, 0, 0, 65536 });
            this->fcgNUVBRTragetQuality->Location = System::Drawing::Point(124, 55);
            this->fcgNUVBRTragetQuality->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 51, 0, 0, 0 });
            this->fcgNUVBRTragetQuality->Name = L"fcgNUVBRTragetQuality";
            this->fcgNUVBRTragetQuality->Size = System::Drawing::Size(77, 21);
            this->fcgNUVBRTragetQuality->TabIndex = 28;
            this->fcgNUVBRTragetQuality->Tag = L"reCmd";
            this->fcgNUVBRTragetQuality->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
            // 
            // fcgLBVBRTragetQuality
            // 
            this->fcgLBVBRTragetQuality->AutoSize = true;
            this->fcgLBVBRTragetQuality->Location = System::Drawing::Point(5, 59);
            this->fcgLBVBRTragetQuality->Name = L"fcgLBVBRTragetQuality";
            this->fcgLBVBRTragetQuality->Size = System::Drawing::Size(74, 14);
            this->fcgLBVBRTragetQuality->TabIndex = 27;
            this->fcgLBVBRTragetQuality->Text = L"VBR品質目標";
            // 
            // fcgLBBitrate
            // 
            this->fcgLBBitrate->AutoSize = true;
            this->fcgLBBitrate->Location = System::Drawing::Point(5, 4);
            this->fcgLBBitrate->Name = L"fcgLBBitrate";
            this->fcgLBBitrate->Size = System::Drawing::Size(54, 14);
            this->fcgLBBitrate->TabIndex = 21;
            this->fcgLBBitrate->Text = L"ビットレート";
            // 
            // fcgNUBitrate
            // 
            this->fcgNUBitrate->Location = System::Drawing::Point(124, 2);
            this->fcgNUBitrate->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 65536000, 0, 0, 0 });
            this->fcgNUBitrate->Name = L"fcgNUBitrate";
            this->fcgNUBitrate->Size = System::Drawing::Size(77, 21);
            this->fcgNUBitrate->TabIndex = 22;
            this->fcgNUBitrate->Tag = L"reCmd";
            this->fcgNUBitrate->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
            // 
            // fcgLBBitrate2
            // 
            this->fcgLBBitrate2->AutoSize = true;
            this->fcgLBBitrate2->Location = System::Drawing::Point(207, 4);
            this->fcgLBBitrate2->Name = L"fcgLBBitrate2";
            this->fcgLBBitrate2->Size = System::Drawing::Size(32, 14);
            this->fcgLBBitrate2->TabIndex = 23;
            this->fcgLBBitrate2->Text = L"kbps";
            // 
            // fcgNUMaxkbps
            // 
            this->fcgNUMaxkbps->Location = System::Drawing::Point(124, 29);
            this->fcgNUMaxkbps->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 65536000, 0, 0, 0 });
            this->fcgNUMaxkbps->Name = L"fcgNUMaxkbps";
            this->fcgNUMaxkbps->Size = System::Drawing::Size(77, 21);
            this->fcgNUMaxkbps->TabIndex = 25;
            this->fcgNUMaxkbps->Tag = L"reCmd";
            this->fcgNUMaxkbps->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
            // 
            // fcgLBMaxkbps
            // 
            this->fcgLBMaxkbps->AutoSize = true;
            this->fcgLBMaxkbps->Location = System::Drawing::Point(5, 33);
            this->fcgLBMaxkbps->Name = L"fcgLBMaxkbps";
            this->fcgLBMaxkbps->Size = System::Drawing::Size(76, 14);
            this->fcgLBMaxkbps->TabIndex = 24;
            this->fcgLBMaxkbps->Text = L"最大ビットレート";
            // 
            // fcgLBMaxBitrate2
            // 
            this->fcgLBMaxBitrate2->AutoSize = true;
            this->fcgLBMaxBitrate2->Location = System::Drawing::Point(207, 31);
            this->fcgLBMaxBitrate2->Name = L"fcgLBMaxBitrate2";
            this->fcgLBMaxBitrate2->Size = System::Drawing::Size(32, 14);
            this->fcgLBMaxBitrate2->TabIndex = 26;
            this->fcgLBMaxBitrate2->Text = L"kbps";
            // 
            // fcgPNHEVC
            // 
            this->fcgPNHEVC->Controls->Add(this->fxgLBHEVCTier);
            this->fcgPNHEVC->Controls->Add(this->fcgLBHEVCOutBitDepth);
            this->fcgPNHEVC->Controls->Add(this->fcgCXHEVCOutBitDepth);
            this->fcgPNHEVC->Controls->Add(this->fcgLBHEVCProfile);
            this->fcgPNHEVC->Controls->Add(this->fcgCXHEVCTier);
            this->fcgPNHEVC->Controls->Add(this->fxgCXHEVCLevel);
            this->fcgPNHEVC->Location = System::Drawing::Point(341, 165);
            this->fcgPNHEVC->Name = L"fcgPNHEVC";
            this->fcgPNHEVC->Size = System::Drawing::Size(264, 91);
            this->fcgPNHEVC->TabIndex = 80;
            // 
            // fxgLBHEVCTier
            // 
            this->fxgLBHEVCTier->AutoSize = true;
            this->fxgLBHEVCTier->Location = System::Drawing::Point(15, 69);
            this->fxgLBHEVCTier->Name = L"fxgLBHEVCTier";
            this->fxgLBHEVCTier->Size = System::Drawing::Size(33, 14);
            this->fxgLBHEVCTier->TabIndex = 85;
            this->fxgLBHEVCTier->Text = L"レベル";
            // 
            // fcgLBHEVCOutBitDepth
            // 
            this->fcgLBHEVCOutBitDepth->AutoSize = true;
            this->fcgLBHEVCOutBitDepth->Location = System::Drawing::Point(15, 7);
            this->fcgLBHEVCOutBitDepth->Name = L"fcgLBHEVCOutBitDepth";
            this->fcgLBHEVCOutBitDepth->Size = System::Drawing::Size(73, 14);
            this->fcgLBHEVCOutBitDepth->TabIndex = 81;
            this->fcgLBHEVCOutBitDepth->Text = L"出力ビット深度";
            // 
            // fcgCXHEVCOutBitDepth
            // 
            this->fcgCXHEVCOutBitDepth->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
            this->fcgCXHEVCOutBitDepth->FormattingEnabled = true;
            this->fcgCXHEVCOutBitDepth->Location = System::Drawing::Point(126, 5);
            this->fcgCXHEVCOutBitDepth->Name = L"fcgCXHEVCOutBitDepth";
            this->fcgCXHEVCOutBitDepth->Size = System::Drawing::Size(121, 22);
            this->fcgCXHEVCOutBitDepth->TabIndex = 82;
            this->fcgCXHEVCOutBitDepth->Tag = L"reCmd";
            // 
            // fcgLBHEVCProfile
            // 
            this->fcgLBHEVCProfile->AutoSize = true;
            this->fcgLBHEVCProfile->Location = System::Drawing::Point(15, 38);
            this->fcgLBHEVCProfile->Name = L"fcgLBHEVCProfile";
            this->fcgLBHEVCProfile->Size = System::Drawing::Size(53, 14);
            this->fcgLBHEVCProfile->TabIndex = 83;
            this->fcgLBHEVCProfile->Text = L"プロファイル";
            // 
            // fcgCXHEVCTier
            // 
            this->fcgCXHEVCTier->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
            this->fcgCXHEVCTier->FormattingEnabled = true;
            this->fcgCXHEVCTier->Location = System::Drawing::Point(126, 35);
            this->fcgCXHEVCTier->Name = L"fcgCXHEVCTier";
            this->fcgCXHEVCTier->Size = System::Drawing::Size(121, 22);
            this->fcgCXHEVCTier->TabIndex = 84;
            this->fcgCXHEVCTier->Tag = L"reCmd";
            // 
            // fxgCXHEVCLevel
            // 
            this->fxgCXHEVCLevel->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
            this->fxgCXHEVCLevel->FormattingEnabled = true;
            this->fxgCXHEVCLevel->Location = System::Drawing::Point(126, 65);
            this->fxgCXHEVCLevel->Name = L"fxgCXHEVCLevel";
            this->fxgCXHEVCLevel->Size = System::Drawing::Size(121, 22);
            this->fxgCXHEVCLevel->TabIndex = 86;
            this->fxgCXHEVCLevel->Tag = L"reCmd";
            // 
            // tabPageVideoDetail
            // 
            this->tabPageVideoDetail->Controls->Add(this->fcgCBPSNR);
            this->tabPageVideoDetail->Controls->Add(this->fcgLBPSNR);
            this->tabPageVideoDetail->Controls->Add(this->fcgCBSSIM);
            this->tabPageVideoDetail->Controls->Add(this->fcgLBSSIM);
            this->tabPageVideoDetail->Controls->Add(this->fcgCBLossless);
            this->tabPageVideoDetail->Controls->Add(this->fcgLBLossless);
            this->tabPageVideoDetail->Controls->Add(this->fcgLBCudaSchdule);
            this->tabPageVideoDetail->Controls->Add(this->fcgCXCudaSchdule);
            this->tabPageVideoDetail->Controls->Add(this->groupBoxQPDetail);
            this->tabPageVideoDetail->Controls->Add(this->fcgLBDevice);
            this->tabPageVideoDetail->Controls->Add(this->fcgCXDevice);
            this->tabPageVideoDetail->Controls->Add(this->fcgLBSlices);
            this->tabPageVideoDetail->Controls->Add(this->fcgNUSlices);
            this->tabPageVideoDetail->Controls->Add(this->fcgPNH264Detail);
            this->tabPageVideoDetail->Controls->Add(this->fcgPNHEVCDetail);
            this->tabPageVideoDetail->Location = System::Drawing::Point(4, 24);
            this->tabPageVideoDetail->Name = L"tabPageVideoDetail";
            this->tabPageVideoDetail->Size = System::Drawing::Size(608, 494);
            this->tabPageVideoDetail->TabIndex = 3;
            this->tabPageVideoDetail->Text = L"詳細設定";
            this->tabPageVideoDetail->UseVisualStyleBackColor = true;
            // 
            // fcgCBPSNR
            // 
            this->fcgCBPSNR->AutoSize = true;
            this->fcgCBPSNR->Location = System::Drawing::Point(133, 352);
            this->fcgCBPSNR->Name = L"fcgCBPSNR";
            this->fcgCBPSNR->Size = System::Drawing::Size(15, 14);
            this->fcgCBPSNR->TabIndex = 33;
            this->fcgCBPSNR->Tag = L"reCmd";
            this->fcgCBPSNR->UseVisualStyleBackColor = true;
            // 
            // fcgLBPSNR
            // 
            this->fcgLBPSNR->AutoSize = true;
            this->fcgLBPSNR->Location = System::Drawing::Point(21, 351);
            this->fcgLBPSNR->Name = L"fcgLBPSNR";
            this->fcgLBPSNR->Size = System::Drawing::Size(32, 15);
            this->fcgLBPSNR->TabIndex = 32;
            this->fcgLBPSNR->Text = L"psnr";
            // 
            // fcgCBSSIM
            // 
            this->fcgCBSSIM->AutoSize = true;
            this->fcgCBSSIM->Location = System::Drawing::Point(133, 328);
            this->fcgCBSSIM->Name = L"fcgCBSSIM";
            this->fcgCBSSIM->Size = System::Drawing::Size(15, 14);
            this->fcgCBSSIM->TabIndex = 31;
            this->fcgCBSSIM->Tag = L"reCmd";
            this->fcgCBSSIM->UseVisualStyleBackColor = true;
            // 
            // fcgLBSSIM
            // 
            this->fcgLBSSIM->AutoSize = true;
            this->fcgLBSSIM->Location = System::Drawing::Point(21, 327);
            this->fcgLBSSIM->Name = L"fcgLBSSIM";
            this->fcgLBSSIM->Size = System::Drawing::Size(34, 15);
            this->fcgLBSSIM->TabIndex = 30;
            this->fcgLBSSIM->Text = L"ssim";
            // 
            // fcgCBLossless
            // 
            this->fcgCBLossless->AutoSize = true;
            this->fcgCBLossless->Location = System::Drawing::Point(133, 87);
            this->fcgCBLossless->Name = L"fcgCBLossless";
            this->fcgCBLossless->Size = System::Drawing::Size(15, 14);
            this->fcgCBLossless->TabIndex = 5;
            this->fcgCBLossless->Tag = L"reCmd";
            this->fcgCBLossless->UseVisualStyleBackColor = true;
            // 
            // fcgLBLossless
            // 
            this->fcgLBLossless->AutoSize = true;
            this->fcgLBLossless->Location = System::Drawing::Point(21, 86);
            this->fcgLBLossless->Name = L"fcgLBLossless";
            this->fcgLBLossless->Size = System::Drawing::Size(51, 15);
            this->fcgLBLossless->TabIndex = 4;
            this->fcgLBLossless->Text = L"lossless";
            // 
            // fcgLBCudaSchdule
            // 
            this->fcgLBCudaSchdule->AutoSize = true;
            this->fcgLBCudaSchdule->Location = System::Drawing::Point(20, 54);
            this->fcgLBCudaSchdule->Name = L"fcgLBCudaSchdule";
            this->fcgLBCudaSchdule->Size = System::Drawing::Size(97, 15);
            this->fcgLBCudaSchdule->TabIndex = 2;
            this->fcgLBCudaSchdule->Text = L"CUDAスケジュール";
            // 
            // fcgCXCudaSchdule
            // 
            this->fcgCXCudaSchdule->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
            this->fcgCXCudaSchdule->FormattingEnabled = true;
            this->fcgCXCudaSchdule->Location = System::Drawing::Point(129, 51);
            this->fcgCXCudaSchdule->Name = L"fcgCXCudaSchdule";
            this->fcgCXCudaSchdule->Size = System::Drawing::Size(194, 23);
            this->fcgCXCudaSchdule->TabIndex = 3;
            this->fcgCXCudaSchdule->Tag = L"reCmd";
            // 
            // groupBoxQPDetail
            // 
            this->groupBoxQPDetail->Controls->Add(this->fcgLBChromaQPOffset);
            this->groupBoxQPDetail->Controls->Add(this->fcgNUChromaQPOffset);
            this->groupBoxQPDetail->Controls->Add(this->fcgLBQPDetailB);
            this->groupBoxQPDetail->Controls->Add(this->fcgLBQPDetailP);
            this->groupBoxQPDetail->Controls->Add(this->fcgLBQPDetailI);
            this->groupBoxQPDetail->Controls->Add(this->fcgCBQPInit);
            this->groupBoxQPDetail->Controls->Add(this->label16);
            this->groupBoxQPDetail->Controls->Add(this->fcgNUQPInitB);
            this->groupBoxQPDetail->Controls->Add(this->label18);
            this->groupBoxQPDetail->Controls->Add(this->fcgNUQPInitP);
            this->groupBoxQPDetail->Controls->Add(this->fcgNUQPInitI);
            this->groupBoxQPDetail->Controls->Add(this->fcgCBQPMin);
            this->groupBoxQPDetail->Controls->Add(this->label11);
            this->groupBoxQPDetail->Controls->Add(this->fcgNUQPMinB);
            this->groupBoxQPDetail->Controls->Add(this->label13);
            this->groupBoxQPDetail->Controls->Add(this->fcgNUQPMinP);
            this->groupBoxQPDetail->Controls->Add(this->fcgNUQPMinI);
            this->groupBoxQPDetail->Controls->Add(this->fcgCBQPMax);
            this->groupBoxQPDetail->Controls->Add(this->label8);
            this->groupBoxQPDetail->Controls->Add(this->fcgNUQPMaxB);
            this->groupBoxQPDetail->Controls->Add(this->label7);
            this->groupBoxQPDetail->Controls->Add(this->fcgNUQPMaxP);
            this->groupBoxQPDetail->Controls->Add(this->fcgNUQPMaxI);
            this->groupBoxQPDetail->Location = System::Drawing::Point(14, 115);
            this->groupBoxQPDetail->Name = L"groupBoxQPDetail";
            this->groupBoxQPDetail->Size = System::Drawing::Size(285, 172);
            this->groupBoxQPDetail->TabIndex = 10;
            this->groupBoxQPDetail->TabStop = false;
            this->groupBoxQPDetail->Text = L"QP詳細設定";
            // 
            // fcgLBChromaQPOffset
            // 
            this->fcgLBChromaQPOffset->AutoSize = true;
            this->fcgLBChromaQPOffset->Location = System::Drawing::Point(10, 136);
            this->fcgLBChromaQPOffset->Name = L"fcgLBChromaQPOffset";
            this->fcgLBChromaQPOffset->Size = System::Drawing::Size(89, 15);
            this->fcgLBChromaQPOffset->TabIndex = 12;
            this->fcgLBChromaQPOffset->Text = L"色差QPオフセット";
            // 
            // fcgNUChromaQPOffset
            // 
            this->fcgNUChromaQPOffset->Location = System::Drawing::Point(115, 134);
            this->fcgNUChromaQPOffset->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 51, 0, 0, 0 });
            this->fcgNUChromaQPOffset->Minimum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 51, 0, 0, System::Int32::MinValue });
            this->fcgNUChromaQPOffset->Name = L"fcgNUChromaQPOffset";
            this->fcgNUChromaQPOffset->Size = System::Drawing::Size(71, 23);
            this->fcgNUChromaQPOffset->TabIndex = 13;
            this->fcgNUChromaQPOffset->Tag = L"reCmd";
            this->fcgNUChromaQPOffset->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
            // 
            // fcgLBQPDetailB
            // 
            this->fcgLBQPDetailB->AutoSize = true;
            this->fcgLBQPDetailB->Location = System::Drawing::Point(214, 19);
            this->fcgLBQPDetailB->Name = L"fcgLBQPDetailB";
            this->fcgLBQPDetailB->Size = System::Drawing::Size(52, 15);
            this->fcgLBQPDetailB->TabIndex = 162;
            this->fcgLBQPDetailB->Text = L"Bフレーム";
            // 
            // fcgLBQPDetailP
            // 
            this->fcgLBQPDetailP->AutoSize = true;
            this->fcgLBQPDetailP->Location = System::Drawing::Point(148, 19);
            this->fcgLBQPDetailP->Name = L"fcgLBQPDetailP";
            this->fcgLBQPDetailP->Size = System::Drawing::Size(51, 15);
            this->fcgLBQPDetailP->TabIndex = 161;
            this->fcgLBQPDetailP->Text = L"Pフレーム";
            // 
            // fcgLBQPDetailI
            // 
            this->fcgLBQPDetailI->AutoSize = true;
            this->fcgLBQPDetailI->Location = System::Drawing::Point(80, 19);
            this->fcgLBQPDetailI->Name = L"fcgLBQPDetailI";
            this->fcgLBQPDetailI->Size = System::Drawing::Size(49, 15);
            this->fcgLBQPDetailI->TabIndex = 160;
            this->fcgLBQPDetailI->Text = L"Iフレーム";
            // 
            // fcgCBQPInit
            // 
            this->fcgCBQPInit->AutoSize = true;
            this->fcgCBQPInit->Location = System::Drawing::Point(10, 100);
            this->fcgCBQPInit->Name = L"fcgCBQPInit";
            this->fcgCBQPInit->Size = System::Drawing::Size(62, 19);
            this->fcgCBQPInit->TabIndex = 8;
            this->fcgCBQPInit->Tag = L"reCmd";
            this->fcgCBQPInit->Text = L"初期値";
            this->fcgCBQPInit->UseVisualStyleBackColor = true;
            // 
            // label16
            // 
            this->label16->AutoSize = true;
            this->label16->Location = System::Drawing::Point(199, 100);
            this->label16->Name = L"label16";
            this->label16->Size = System::Drawing::Size(12, 15);
            this->label16->TabIndex = 32;
            this->label16->Text = L":";
            // 
            // fcgNUQPInitB
            // 
            this->fcgNUQPInitB->Location = System::Drawing::Point(215, 98);
            this->fcgNUQPInitB->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 65535, 0, 0, 0 });
            this->fcgNUQPInitB->Name = L"fcgNUQPInitB";
            this->fcgNUQPInitB->Size = System::Drawing::Size(48, 23);
            this->fcgNUQPInitB->TabIndex = 11;
            this->fcgNUQPInitB->Tag = L"reCmd";
            this->fcgNUQPInitB->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
            // 
            // label18
            // 
            this->label18->AutoSize = true;
            this->label18->Location = System::Drawing::Point(133, 100);
            this->label18->Name = L"label18";
            this->label18->Size = System::Drawing::Size(12, 15);
            this->label18->TabIndex = 30;
            this->label18->Text = L":";
            // 
            // fcgNUQPInitP
            // 
            this->fcgNUQPInitP->Location = System::Drawing::Point(149, 98);
            this->fcgNUQPInitP->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 51, 0, 0, 0 });
            this->fcgNUQPInitP->Name = L"fcgNUQPInitP";
            this->fcgNUQPInitP->Size = System::Drawing::Size(48, 23);
            this->fcgNUQPInitP->TabIndex = 10;
            this->fcgNUQPInitP->Tag = L"reCmd";
            this->fcgNUQPInitP->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
            // 
            // fcgNUQPInitI
            // 
            this->fcgNUQPInitI->Location = System::Drawing::Point(83, 98);
            this->fcgNUQPInitI->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 51, 0, 0, 0 });
            this->fcgNUQPInitI->Name = L"fcgNUQPInitI";
            this->fcgNUQPInitI->Size = System::Drawing::Size(48, 23);
            this->fcgNUQPInitI->TabIndex = 9;
            this->fcgNUQPInitI->Tag = L"reCmd";
            this->fcgNUQPInitI->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
            // 
            // fcgCBQPMin
            // 
            this->fcgCBQPMin->AutoSize = true;
            this->fcgCBQPMin->Location = System::Drawing::Point(10, 43);
            this->fcgCBQPMin->Name = L"fcgCBQPMin";
            this->fcgCBQPMin->Size = System::Drawing::Size(50, 19);
            this->fcgCBQPMin->TabIndex = 0;
            this->fcgCBQPMin->Tag = L"reCmd";
            this->fcgCBQPMin->Text = L"下限";
            this->fcgCBQPMin->UseVisualStyleBackColor = true;
            // 
            // label11
            // 
            this->label11->AutoSize = true;
            this->label11->Location = System::Drawing::Point(199, 43);
            this->label11->Name = L"label11";
            this->label11->Size = System::Drawing::Size(12, 15);
            this->label11->TabIndex = 26;
            this->label11->Text = L":";
            // 
            // fcgNUQPMinB
            // 
            this->fcgNUQPMinB->Location = System::Drawing::Point(215, 41);
            this->fcgNUQPMinB->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 51, 0, 0, 0 });
            this->fcgNUQPMinB->Name = L"fcgNUQPMinB";
            this->fcgNUQPMinB->Size = System::Drawing::Size(48, 23);
            this->fcgNUQPMinB->TabIndex = 3;
            this->fcgNUQPMinB->Tag = L"reCmd";
            this->fcgNUQPMinB->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
            // 
            // label13
            // 
            this->label13->AutoSize = true;
            this->label13->Location = System::Drawing::Point(133, 43);
            this->label13->Name = L"label13";
            this->label13->Size = System::Drawing::Size(12, 15);
            this->label13->TabIndex = 24;
            this->label13->Text = L":";
            // 
            // fcgNUQPMinP
            // 
            this->fcgNUQPMinP->Location = System::Drawing::Point(149, 41);
            this->fcgNUQPMinP->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 51, 0, 0, 0 });
            this->fcgNUQPMinP->Name = L"fcgNUQPMinP";
            this->fcgNUQPMinP->Size = System::Drawing::Size(48, 23);
            this->fcgNUQPMinP->TabIndex = 2;
            this->fcgNUQPMinP->Tag = L"reCmd";
            this->fcgNUQPMinP->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
            // 
            // fcgNUQPMinI
            // 
            this->fcgNUQPMinI->Location = System::Drawing::Point(83, 41);
            this->fcgNUQPMinI->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 51, 0, 0, 0 });
            this->fcgNUQPMinI->Name = L"fcgNUQPMinI";
            this->fcgNUQPMinI->Size = System::Drawing::Size(48, 23);
            this->fcgNUQPMinI->TabIndex = 1;
            this->fcgNUQPMinI->Tag = L"reCmd";
            this->fcgNUQPMinI->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
            // 
            // fcgCBQPMax
            // 
            this->fcgCBQPMax->AutoSize = true;
            this->fcgCBQPMax->Location = System::Drawing::Point(10, 71);
            this->fcgCBQPMax->Name = L"fcgCBQPMax";
            this->fcgCBQPMax->Size = System::Drawing::Size(50, 19);
            this->fcgCBQPMax->TabIndex = 4;
            this->fcgCBQPMax->Tag = L"reCmd";
            this->fcgCBQPMax->Text = L"上限";
            this->fcgCBQPMax->UseVisualStyleBackColor = true;
            // 
            // label8
            // 
            this->label8->AutoSize = true;
            this->label8->Location = System::Drawing::Point(199, 71);
            this->label8->Name = L"label8";
            this->label8->Size = System::Drawing::Size(12, 15);
            this->label8->TabIndex = 5;
            this->label8->Text = L":";
            // 
            // fcgNUQPMaxB
            // 
            this->fcgNUQPMaxB->Location = System::Drawing::Point(215, 69);
            this->fcgNUQPMaxB->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 51, 0, 0, 0 });
            this->fcgNUQPMaxB->Name = L"fcgNUQPMaxB";
            this->fcgNUQPMaxB->Size = System::Drawing::Size(48, 23);
            this->fcgNUQPMaxB->TabIndex = 7;
            this->fcgNUQPMaxB->Tag = L"reCmd";
            this->fcgNUQPMaxB->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
            // 
            // label7
            // 
            this->label7->AutoSize = true;
            this->label7->Location = System::Drawing::Point(133, 71);
            this->label7->Name = L"label7";
            this->label7->Size = System::Drawing::Size(12, 15);
            this->label7->TabIndex = 3;
            this->label7->Text = L":";
            // 
            // fcgNUQPMaxP
            // 
            this->fcgNUQPMaxP->Location = System::Drawing::Point(149, 69);
            this->fcgNUQPMaxP->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 51, 0, 0, 0 });
            this->fcgNUQPMaxP->Name = L"fcgNUQPMaxP";
            this->fcgNUQPMaxP->Size = System::Drawing::Size(48, 23);
            this->fcgNUQPMaxP->TabIndex = 6;
            this->fcgNUQPMaxP->Tag = L"reCmd";
            this->fcgNUQPMaxP->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
            // 
            // fcgNUQPMaxI
            // 
            this->fcgNUQPMaxI->Location = System::Drawing::Point(83, 69);
            this->fcgNUQPMaxI->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 51, 0, 0, 0 });
            this->fcgNUQPMaxI->Name = L"fcgNUQPMaxI";
            this->fcgNUQPMaxI->Size = System::Drawing::Size(48, 23);
            this->fcgNUQPMaxI->TabIndex = 5;
            this->fcgNUQPMaxI->Tag = L"reCmd";
            this->fcgNUQPMaxI->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
            // 
            // fcgLBDevice
            // 
            this->fcgLBDevice->AutoSize = true;
            this->fcgLBDevice->Location = System::Drawing::Point(20, 23);
            this->fcgLBDevice->Name = L"fcgLBDevice";
            this->fcgLBDevice->Size = System::Drawing::Size(45, 15);
            this->fcgLBDevice->TabIndex = 0;
            this->fcgLBDevice->Text = L"デバイス";
            // 
            // fcgCXDevice
            // 
            this->fcgCXDevice->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
            this->fcgCXDevice->FormattingEnabled = true;
            this->fcgCXDevice->Location = System::Drawing::Point(129, 20);
            this->fcgCXDevice->Name = L"fcgCXDevice";
            this->fcgCXDevice->Size = System::Drawing::Size(194, 23);
            this->fcgCXDevice->TabIndex = 1;
            this->fcgCXDevice->Tag = L"reCmd";
            // 
            // fcgLBSlices
            // 
            this->fcgLBSlices->AutoSize = true;
            this->fcgLBSlices->Location = System::Drawing::Point(21, 295);
            this->fcgLBSlices->Name = L"fcgLBSlices";
            this->fcgLBSlices->Size = System::Drawing::Size(54, 15);
            this->fcgLBSlices->TabIndex = 20;
            this->fcgLBSlices->Text = L"スライス数";
            this->fcgLBSlices->Visible = false;
            // 
            // fcgNUSlices
            // 
            this->fcgNUSlices->Location = System::Drawing::Point(130, 293);
            this->fcgNUSlices->Minimum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 1, 0, 0, 0 });
            this->fcgNUSlices->Name = L"fcgNUSlices";
            this->fcgNUSlices->Size = System::Drawing::Size(70, 23);
            this->fcgNUSlices->TabIndex = 21;
            this->fcgNUSlices->Tag = L"reCmd";
            this->fcgNUSlices->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
            this->fcgNUSlices->Value = System::Decimal(gcnew cli::array< System::Int32 >(4) { 1, 0, 0, 0 });
            this->fcgNUSlices->Visible = false;
            // 
            // fcgPNH264Detail
            // 
            this->fcgPNH264Detail->Controls->Add(this->fcgLBDeblock);
            this->fcgPNH264Detail->Controls->Add(this->fcgCXAdaptiveTransform);
            this->fcgPNH264Detail->Controls->Add(this->fcgLBAdaptiveTransform);
            this->fcgPNH264Detail->Controls->Add(this->fcgCXBDirectMode);
            this->fcgPNH264Detail->Controls->Add(this->fcgLBBDirectMode);
            this->fcgPNH264Detail->Controls->Add(this->fcgLBMVPrecision);
            this->fcgPNH264Detail->Controls->Add(this->fcgCXMVPrecision);
            this->fcgPNH264Detail->Controls->Add(this->fcgCBDeblock);
            this->fcgPNH264Detail->Controls->Add(this->fcgCBCABAC);
            this->fcgPNH264Detail->Controls->Add(this->fcgLBCABAC);
            this->fcgPNH264Detail->Location = System::Drawing::Point(340, 13);
            this->fcgPNH264Detail->Name = L"fcgPNH264Detail";
            this->fcgPNH264Detail->Size = System::Drawing::Size(264, 170);
            this->fcgPNH264Detail->TabIndex = 60;
            // 
            // fcgLBDeblock
            // 
            this->fcgLBDeblock->AutoSize = true;
            this->fcgLBDeblock->Location = System::Drawing::Point(15, 39);
            this->fcgLBDeblock->Name = L"fcgLBDeblock";
            this->fcgLBDeblock->Size = System::Drawing::Size(83, 15);
            this->fcgLBDeblock->TabIndex = 63;
            this->fcgLBDeblock->Text = L"デブロックフィルタ";
            // 
            // fcgCXAdaptiveTransform
            // 
            this->fcgCXAdaptiveTransform->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
            this->fcgCXAdaptiveTransform->FormattingEnabled = true;
            this->fcgCXAdaptiveTransform->Location = System::Drawing::Point(125, 65);
            this->fcgCXAdaptiveTransform->Name = L"fcgCXAdaptiveTransform";
            this->fcgCXAdaptiveTransform->Size = System::Drawing::Size(122, 23);
            this->fcgCXAdaptiveTransform->TabIndex = 66;
            this->fcgCXAdaptiveTransform->Tag = L"reCmd";
            // 
            // fcgLBAdaptiveTransform
            // 
            this->fcgLBAdaptiveTransform->AutoSize = true;
            this->fcgLBAdaptiveTransform->Location = System::Drawing::Point(15, 68);
            this->fcgLBAdaptiveTransform->Name = L"fcgLBAdaptiveTransform";
            this->fcgLBAdaptiveTransform->Size = System::Drawing::Size(109, 15);
            this->fcgLBAdaptiveTransform->TabIndex = 65;
            this->fcgLBAdaptiveTransform->Text = L"Adapt. Transform";
            // 
            // fcgCXBDirectMode
            // 
            this->fcgCXBDirectMode->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
            this->fcgCXBDirectMode->FormattingEnabled = true;
            this->fcgCXBDirectMode->Location = System::Drawing::Point(125, 128);
            this->fcgCXBDirectMode->Name = L"fcgCXBDirectMode";
            this->fcgCXBDirectMode->Size = System::Drawing::Size(122, 23);
            this->fcgCXBDirectMode->TabIndex = 70;
            this->fcgCXBDirectMode->Tag = L"reCmd";
            // 
            // fcgLBBDirectMode
            // 
            this->fcgLBBDirectMode->AutoSize = true;
            this->fcgLBBDirectMode->Location = System::Drawing::Point(17, 131);
            this->fcgLBBDirectMode->Name = L"fcgLBBDirectMode";
            this->fcgLBBDirectMode->Size = System::Drawing::Size(76, 15);
            this->fcgLBBDirectMode->TabIndex = 69;
            this->fcgLBBDirectMode->Text = L"動き予測方式";
            // 
            // fcgLBMVPrecision
            // 
            this->fcgLBMVPrecision->AutoSize = true;
            this->fcgLBMVPrecision->Location = System::Drawing::Point(15, 100);
            this->fcgLBMVPrecision->Name = L"fcgLBMVPrecision";
            this->fcgLBMVPrecision->Size = System::Drawing::Size(76, 15);
            this->fcgLBMVPrecision->TabIndex = 67;
            this->fcgLBMVPrecision->Text = L"動き探索精度";
            // 
            // fcgCXMVPrecision
            // 
            this->fcgCXMVPrecision->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
            this->fcgCXMVPrecision->FormattingEnabled = true;
            this->fcgCXMVPrecision->Location = System::Drawing::Point(125, 96);
            this->fcgCXMVPrecision->Name = L"fcgCXMVPrecision";
            this->fcgCXMVPrecision->Size = System::Drawing::Size(122, 23);
            this->fcgCXMVPrecision->TabIndex = 68;
            this->fcgCXMVPrecision->Tag = L"reCmd";
            // 
            // fcgCBDeblock
            // 
            this->fcgCBDeblock->AutoSize = true;
            this->fcgCBDeblock->Location = System::Drawing::Point(127, 40);
            this->fcgCBDeblock->Name = L"fcgCBDeblock";
            this->fcgCBDeblock->Size = System::Drawing::Size(15, 14);
            this->fcgCBDeblock->TabIndex = 64;
            this->fcgCBDeblock->Tag = L"reCmd";
            this->fcgCBDeblock->UseVisualStyleBackColor = true;
            // 
            // fcgCBCABAC
            // 
            this->fcgCBCABAC->AutoSize = true;
            this->fcgCBCABAC->Location = System::Drawing::Point(127, 13);
            this->fcgCBCABAC->Name = L"fcgCBCABAC";
            this->fcgCBCABAC->Size = System::Drawing::Size(15, 14);
            this->fcgCBCABAC->TabIndex = 62;
            this->fcgCBCABAC->Tag = L"reCmd";
            this->fcgCBCABAC->UseVisualStyleBackColor = true;
            // 
            // fcgLBCABAC
            // 
            this->fcgLBCABAC->AutoSize = true;
            this->fcgLBCABAC->Location = System::Drawing::Point(15, 12);
            this->fcgLBCABAC->Name = L"fcgLBCABAC";
            this->fcgLBCABAC->Size = System::Drawing::Size(47, 15);
            this->fcgLBCABAC->TabIndex = 61;
            this->fcgLBCABAC->Text = L"CABAC";
            // 
            // fcgPNHEVCDetail
            // 
            this->fcgPNHEVCDetail->Controls->Add(this->fcgLBHEVCMinCUSize);
            this->fcgPNHEVCDetail->Controls->Add(this->fcgCXHEVCMinCUSize);
            this->fcgPNHEVCDetail->Controls->Add(this->fcgLBHEVCMaxCUSize);
            this->fcgPNHEVCDetail->Controls->Add(this->fcgCXHEVCMaxCUSize);
            this->fcgPNHEVCDetail->Location = System::Drawing::Point(340, 13);
            this->fcgPNHEVCDetail->Name = L"fcgPNHEVCDetail";
            this->fcgPNHEVCDetail->Size = System::Drawing::Size(264, 170);
            this->fcgPNHEVCDetail->TabIndex = 40;
            // 
            // fcgLBHEVCMinCUSize
            // 
            this->fcgLBHEVCMinCUSize->AutoSize = true;
            this->fcgLBHEVCMinCUSize->Location = System::Drawing::Point(20, 42);
            this->fcgLBHEVCMinCUSize->Name = L"fcgLBHEVCMinCUSize";
            this->fcgLBHEVCMinCUSize->Size = System::Drawing::Size(77, 15);
            this->fcgLBHEVCMinCUSize->TabIndex = 149;
            this->fcgLBHEVCMinCUSize->Text = L"最小CUサイズ";
            // 
            // fcgCXHEVCMinCUSize
            // 
            this->fcgCXHEVCMinCUSize->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
            this->fcgCXHEVCMinCUSize->FormattingEnabled = true;
            this->fcgCXHEVCMinCUSize->Location = System::Drawing::Point(129, 39);
            this->fcgCXHEVCMinCUSize->Name = L"fcgCXHEVCMinCUSize";
            this->fcgCXHEVCMinCUSize->Size = System::Drawing::Size(121, 23);
            this->fcgCXHEVCMinCUSize->TabIndex = 42;
            this->fcgCXHEVCMinCUSize->Tag = L"reCmd";
            // 
            // fcgLBHEVCMaxCUSize
            // 
            this->fcgLBHEVCMaxCUSize->AutoSize = true;
            this->fcgLBHEVCMaxCUSize->Location = System::Drawing::Point(20, 12);
            this->fcgLBHEVCMaxCUSize->Name = L"fcgLBHEVCMaxCUSize";
            this->fcgLBHEVCMaxCUSize->Size = System::Drawing::Size(77, 15);
            this->fcgLBHEVCMaxCUSize->TabIndex = 147;
            this->fcgLBHEVCMaxCUSize->Text = L"最大CUサイズ";
            // 
            // fcgCXHEVCMaxCUSize
            // 
            this->fcgCXHEVCMaxCUSize->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
            this->fcgCXHEVCMaxCUSize->FormattingEnabled = true;
            this->fcgCXHEVCMaxCUSize->Location = System::Drawing::Point(129, 9);
            this->fcgCXHEVCMaxCUSize->Name = L"fcgCXHEVCMaxCUSize";
            this->fcgCXHEVCMaxCUSize->Size = System::Drawing::Size(121, 23);
            this->fcgCXHEVCMaxCUSize->TabIndex = 41;
            this->fcgCXHEVCMaxCUSize->Tag = L"reCmd";
            // 
            // tabPageExOpt
            // 
            this->tabPageExOpt->Controls->Add(this->fcgCBVppTweakEnable);
            this->tabPageExOpt->Controls->Add(this->fcggroupBoxVppTweak);
            this->tabPageExOpt->Controls->Add(this->fcgCBVppPerfMonitor);
            this->tabPageExOpt->Controls->Add(this->fcggroupBoxVppDetailEnahance);
            this->tabPageExOpt->Controls->Add(this->fcggroupBoxVppDeinterlace);
            this->tabPageExOpt->Controls->Add(this->fcgCBVppDebandEnable);
            this->tabPageExOpt->Controls->Add(this->fcggroupBoxVppDeband);
            this->tabPageExOpt->Controls->Add(this->fcggroupBoxVppDenoise);
            this->tabPageExOpt->Controls->Add(this->fcgCBVppResize);
            this->tabPageExOpt->Controls->Add(this->fcggroupBoxResize);
            this->tabPageExOpt->Font = (gcnew System::Drawing::Font(L"Meiryo UI", 8.25F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
                static_cast<System::Byte>(128)));
            this->tabPageExOpt->Location = System::Drawing::Point(4, 24);
            this->tabPageExOpt->Name = L"tabPageExOpt";
            this->tabPageExOpt->Size = System::Drawing::Size(608, 494);
            this->tabPageExOpt->TabIndex = 1;
            this->tabPageExOpt->Text = L"フィルタ";
            this->tabPageExOpt->UseVisualStyleBackColor = true;
            // 
            // fcgCBVppTweakEnable
            // 
            this->fcgCBVppTweakEnable->AutoSize = true;
            this->fcgCBVppTweakEnable->Location = System::Drawing::Point(370, 346);
            this->fcgCBVppTweakEnable->Name = L"fcgCBVppTweakEnable";
            this->fcgCBVppTweakEnable->Size = System::Drawing::Size(70, 18);
            this->fcgCBVppTweakEnable->TabIndex = 50;
            this->fcgCBVppTweakEnable->Tag = L"reCmd";
            this->fcgCBVppTweakEnable->Text = L"色調補正";
            this->fcgCBVppTweakEnable->UseVisualStyleBackColor = true;
            this->fcgCBVppTweakEnable->CheckedChanged += gcnew System::EventHandler(this, &frmConfig::fcgChangeEnabled);
            // 
            // fcggroupBoxVppTweak
            // 
            this->fcggroupBoxVppTweak->Controls->Add(this->fcgTBVppTweakHue);
            this->fcggroupBoxVppTweak->Controls->Add(this->fcgLBVppTweakHue);
            this->fcggroupBoxVppTweak->Controls->Add(this->fcgTBVppTweakSaturation);
            this->fcggroupBoxVppTweak->Controls->Add(this->fcgLBVppTweakSaturation);
            this->fcggroupBoxVppTweak->Controls->Add(this->fcgTBVppTweakGamma);
            this->fcggroupBoxVppTweak->Controls->Add(this->fcgLBVppTweakGamma);
            this->fcggroupBoxVppTweak->Controls->Add(this->fcgTBVppTweakContrast);
            this->fcggroupBoxVppTweak->Controls->Add(this->fcgLBVppTweakContrast);
            this->fcggroupBoxVppTweak->Controls->Add(this->fcgTBVppTweakBrightness);
            this->fcggroupBoxVppTweak->Controls->Add(this->fcgNUVppTweakGamma);
            this->fcggroupBoxVppTweak->Controls->Add(this->fcgNUVppTweakSaturation);
            this->fcggroupBoxVppTweak->Controls->Add(this->fcgNUVppTweakHue);
            this->fcggroupBoxVppTweak->Controls->Add(this->fcgLBVppTweakBrightness);
            this->fcggroupBoxVppTweak->Controls->Add(this->fcgNUVppTweakContrast);
            this->fcggroupBoxVppTweak->Controls->Add(this->fcgNUVppTweakBrightness);
            this->fcggroupBoxVppTweak->Location = System::Drawing::Point(342, 346);
            this->fcggroupBoxVppTweak->Name = L"fcggroupBoxVppTweak";
            this->fcggroupBoxVppTweak->Size = System::Drawing::Size(263, 145);
            this->fcggroupBoxVppTweak->TabIndex = 51;
            this->fcggroupBoxVppTweak->TabStop = false;
            // 
            // fcgTBVppTweakHue
            // 
            this->fcgTBVppTweakHue->AutoSize = false;
            this->fcgTBVppTweakHue->LargeChange = 10;
            this->fcgTBVppTweakHue->Location = System::Drawing::Point(60, 119);
            this->fcgTBVppTweakHue->Maximum = 180;
            this->fcgTBVppTweakHue->Minimum = -180;
            this->fcgTBVppTweakHue->Name = L"fcgTBVppTweakHue";
            this->fcgTBVppTweakHue->Size = System::Drawing::Size(115, 18);
            this->fcgTBVppTweakHue->TabIndex = 13;
            this->fcgTBVppTweakHue->TickStyle = System::Windows::Forms::TickStyle::None;
            this->fcgTBVppTweakHue->Scroll += gcnew System::EventHandler(this, &frmConfig::fcgTBVppTweakScroll);
            // 
            // fcgLBVppTweakHue
            // 
            this->fcgLBVppTweakHue->AutoSize = true;
            this->fcgLBVppTweakHue->Location = System::Drawing::Point(9, 119);
            this->fcgLBVppTweakHue->Name = L"fcgLBVppTweakHue";
            this->fcgLBVppTweakHue->Size = System::Drawing::Size(29, 14);
            this->fcgLBVppTweakHue->TabIndex = 12;
            this->fcgLBVppTweakHue->Text = L"色相";
            // 
            // fcgTBVppTweakSaturation
            // 
            this->fcgTBVppTweakSaturation->AutoSize = false;
            this->fcgTBVppTweakSaturation->Location = System::Drawing::Point(60, 94);
            this->fcgTBVppTweakSaturation->Maximum = 200;
            this->fcgTBVppTweakSaturation->Name = L"fcgTBVppTweakSaturation";
            this->fcgTBVppTweakSaturation->Size = System::Drawing::Size(115, 18);
            this->fcgTBVppTweakSaturation->TabIndex = 10;
            this->fcgTBVppTweakSaturation->TickStyle = System::Windows::Forms::TickStyle::None;
            this->fcgTBVppTweakSaturation->Value = 100;
            this->fcgTBVppTweakSaturation->Scroll += gcnew System::EventHandler(this, &frmConfig::fcgTBVppTweakScroll);
            // 
            // fcgLBVppTweakSaturation
            // 
            this->fcgLBVppTweakSaturation->AutoSize = true;
            this->fcgLBVppTweakSaturation->Location = System::Drawing::Point(9, 94);
            this->fcgLBVppTweakSaturation->Name = L"fcgLBVppTweakSaturation";
            this->fcgLBVppTweakSaturation->Size = System::Drawing::Size(29, 14);
            this->fcgLBVppTweakSaturation->TabIndex = 9;
            this->fcgLBVppTweakSaturation->Text = L"彩度";
            // 
            // fcgTBVppTweakGamma
            // 
            this->fcgTBVppTweakGamma->AutoSize = false;
            this->fcgTBVppTweakGamma->LargeChange = 10;
            this->fcgTBVppTweakGamma->Location = System::Drawing::Point(60, 69);
            this->fcgTBVppTweakGamma->Maximum = 200;
            this->fcgTBVppTweakGamma->Minimum = 1;
            this->fcgTBVppTweakGamma->Name = L"fcgTBVppTweakGamma";
            this->fcgTBVppTweakGamma->Size = System::Drawing::Size(115, 18);
            this->fcgTBVppTweakGamma->TabIndex = 7;
            this->fcgTBVppTweakGamma->TickStyle = System::Windows::Forms::TickStyle::None;
            this->fcgTBVppTweakGamma->Value = 100;
            this->fcgTBVppTweakGamma->Scroll += gcnew System::EventHandler(this, &frmConfig::fcgTBVppTweakScroll);
            // 
            // fcgLBVppTweakGamma
            // 
            this->fcgLBVppTweakGamma->AutoSize = true;
            this->fcgLBVppTweakGamma->Location = System::Drawing::Point(9, 69);
            this->fcgLBVppTweakGamma->Name = L"fcgLBVppTweakGamma";
            this->fcgLBVppTweakGamma->Size = System::Drawing::Size(32, 14);
            this->fcgLBVppTweakGamma->TabIndex = 6;
            this->fcgLBVppTweakGamma->Text = L"ガンマ";
            // 
            // fcgTBVppTweakContrast
            // 
            this->fcgTBVppTweakContrast->AutoSize = false;
            this->fcgTBVppTweakContrast->LargeChange = 10;
            this->fcgTBVppTweakContrast->Location = System::Drawing::Point(60, 44);
            this->fcgTBVppTweakContrast->Maximum = 200;
            this->fcgTBVppTweakContrast->Minimum = -200;
            this->fcgTBVppTweakContrast->Name = L"fcgTBVppTweakContrast";
            this->fcgTBVppTweakContrast->Size = System::Drawing::Size(115, 18);
            this->fcgTBVppTweakContrast->TabIndex = 4;
            this->fcgTBVppTweakContrast->TickStyle = System::Windows::Forms::TickStyle::None;
            this->fcgTBVppTweakContrast->Value = 100;
            this->fcgTBVppTweakContrast->Scroll += gcnew System::EventHandler(this, &frmConfig::fcgTBVppTweakScroll);
            // 
            // fcgLBVppTweakContrast
            // 
            this->fcgLBVppTweakContrast->AutoSize = true;
            this->fcgLBVppTweakContrast->Location = System::Drawing::Point(9, 44);
            this->fcgLBVppTweakContrast->Name = L"fcgLBVppTweakContrast";
            this->fcgLBVppTweakContrast->Size = System::Drawing::Size(55, 14);
            this->fcgLBVppTweakContrast->TabIndex = 3;
            this->fcgLBVppTweakContrast->Text = L"コントラスト";
            // 
            // fcgTBVppTweakBrightness
            // 
            this->fcgTBVppTweakBrightness->AutoSize = false;
            this->fcgTBVppTweakBrightness->LargeChange = 10;
            this->fcgTBVppTweakBrightness->Location = System::Drawing::Point(60, 19);
            this->fcgTBVppTweakBrightness->Maximum = 100;
            this->fcgTBVppTweakBrightness->Minimum = -100;
            this->fcgTBVppTweakBrightness->Name = L"fcgTBVppTweakBrightness";
            this->fcgTBVppTweakBrightness->Size = System::Drawing::Size(115, 18);
            this->fcgTBVppTweakBrightness->TabIndex = 1;
            this->fcgTBVppTweakBrightness->TickStyle = System::Windows::Forms::TickStyle::None;
            this->fcgTBVppTweakBrightness->Scroll += gcnew System::EventHandler(this, &frmConfig::fcgTBVppTweakScroll);
            // 
            // fcgNUVppTweakGamma
            // 
            this->fcgNUVppTweakGamma->Increment = System::Decimal(gcnew cli::array< System::Int32 >(4) { 10, 0, 0, 0 });
            this->fcgNUVppTweakGamma->Location = System::Drawing::Point(181, 69);
            this->fcgNUVppTweakGamma->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 1000, 0, 0, 0 });
            this->fcgNUVppTweakGamma->Minimum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 1, 0, 0, 0 });
            this->fcgNUVppTweakGamma->Name = L"fcgNUVppTweakGamma";
            this->fcgNUVppTweakGamma->Size = System::Drawing::Size(60, 21);
            this->fcgNUVppTweakGamma->TabIndex = 8;
            this->fcgNUVppTweakGamma->Tag = L"reCmd";
            this->fcgNUVppTweakGamma->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
            this->fcgNUVppTweakGamma->Value = System::Decimal(gcnew cli::array< System::Int32 >(4) { 100, 0, 0, 0 });
            this->fcgNUVppTweakGamma->ValueChanged += gcnew System::EventHandler(this, &frmConfig::fcgNUVppTweakValueChanged);
            // 
            // fcgNUVppTweakSaturation
            // 
            this->fcgNUVppTweakSaturation->Increment = System::Decimal(gcnew cli::array< System::Int32 >(4) { 10, 0, 0, 0 });
            this->fcgNUVppTweakSaturation->Location = System::Drawing::Point(181, 94);
            this->fcgNUVppTweakSaturation->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 300, 0, 0, 0 });
            this->fcgNUVppTweakSaturation->Name = L"fcgNUVppTweakSaturation";
            this->fcgNUVppTweakSaturation->Size = System::Drawing::Size(60, 21);
            this->fcgNUVppTweakSaturation->TabIndex = 11;
            this->fcgNUVppTweakSaturation->Tag = L"reCmd";
            this->fcgNUVppTweakSaturation->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
            this->fcgNUVppTweakSaturation->Value = System::Decimal(gcnew cli::array< System::Int32 >(4) { 100, 0, 0, 0 });
            this->fcgNUVppTweakSaturation->ValueChanged += gcnew System::EventHandler(this, &frmConfig::fcgNUVppTweakValueChanged);
            // 
            // fcgNUVppTweakHue
            // 
            this->fcgNUVppTweakHue->Increment = System::Decimal(gcnew cli::array< System::Int32 >(4) { 10, 0, 0, 0 });
            this->fcgNUVppTweakHue->Location = System::Drawing::Point(181, 119);
            this->fcgNUVppTweakHue->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 180, 0, 0, 0 });
            this->fcgNUVppTweakHue->Minimum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 180, 0, 0, System::Int32::MinValue });
            this->fcgNUVppTweakHue->Name = L"fcgNUVppTweakHue";
            this->fcgNUVppTweakHue->Size = System::Drawing::Size(60, 21);
            this->fcgNUVppTweakHue->TabIndex = 14;
            this->fcgNUVppTweakHue->Tag = L"reCmd";
            this->fcgNUVppTweakHue->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
            this->fcgNUVppTweakHue->ValueChanged += gcnew System::EventHandler(this, &frmConfig::fcgNUVppTweakValueChanged);
            // 
            // fcgLBVppTweakBrightness
            // 
            this->fcgLBVppTweakBrightness->AutoSize = true;
            this->fcgLBVppTweakBrightness->Location = System::Drawing::Point(9, 19);
            this->fcgLBVppTweakBrightness->Name = L"fcgLBVppTweakBrightness";
            this->fcgLBVppTweakBrightness->Size = System::Drawing::Size(29, 14);
            this->fcgLBVppTweakBrightness->TabIndex = 0;
            this->fcgLBVppTweakBrightness->Text = L"輝度";
            // 
            // fcgNUVppTweakContrast
            // 
            this->fcgNUVppTweakContrast->Increment = System::Decimal(gcnew cli::array< System::Int32 >(4) { 10, 0, 0, 0 });
            this->fcgNUVppTweakContrast->Location = System::Drawing::Point(181, 44);
            this->fcgNUVppTweakContrast->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 200, 0, 0, 0 });
            this->fcgNUVppTweakContrast->Minimum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 200, 0, 0, System::Int32::MinValue });
            this->fcgNUVppTweakContrast->Name = L"fcgNUVppTweakContrast";
            this->fcgNUVppTweakContrast->Size = System::Drawing::Size(60, 21);
            this->fcgNUVppTweakContrast->TabIndex = 5;
            this->fcgNUVppTweakContrast->Tag = L"reCmd";
            this->fcgNUVppTweakContrast->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
            this->fcgNUVppTweakContrast->Value = System::Decimal(gcnew cli::array< System::Int32 >(4) { 100, 0, 0, 0 });
            this->fcgNUVppTweakContrast->ValueChanged += gcnew System::EventHandler(this, &frmConfig::fcgNUVppTweakValueChanged);
            // 
            // fcgNUVppTweakBrightness
            // 
            this->fcgNUVppTweakBrightness->Increment = System::Decimal(gcnew cli::array< System::Int32 >(4) { 10, 0, 0, 0 });
            this->fcgNUVppTweakBrightness->Location = System::Drawing::Point(181, 19);
            this->fcgNUVppTweakBrightness->Minimum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 100, 0, 0, System::Int32::MinValue });
            this->fcgNUVppTweakBrightness->Name = L"fcgNUVppTweakBrightness";
            this->fcgNUVppTweakBrightness->Size = System::Drawing::Size(60, 21);
            this->fcgNUVppTweakBrightness->TabIndex = 2;
            this->fcgNUVppTweakBrightness->Tag = L"reCmd";
            this->fcgNUVppTweakBrightness->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
            this->fcgNUVppTweakBrightness->ValueChanged += gcnew System::EventHandler(this, &frmConfig::fcgNUVppTweakValueChanged);
            // 
            // fcgCBVppPerfMonitor
            // 
            this->fcgCBVppPerfMonitor->AutoSize = true;
            this->fcgCBVppPerfMonitor->Location = System::Drawing::Point(13, 459);
            this->fcgCBVppPerfMonitor->Name = L"fcgCBVppPerfMonitor";
            this->fcgCBVppPerfMonitor->Size = System::Drawing::Size(112, 18);
            this->fcgCBVppPerfMonitor->TabIndex = 30;
            this->fcgCBVppPerfMonitor->Tag = L"reCmd";
            this->fcgCBVppPerfMonitor->Text = L"パフォーマンスチェック";
            this->fcgCBVppPerfMonitor->UseVisualStyleBackColor = true;
            // 
            // fcggroupBoxVppDetailEnahance
            // 
            this->fcggroupBoxVppDetailEnahance->Controls->Add(this->fcgPNVppWarpsharp);
            this->fcggroupBoxVppDetailEnahance->Controls->Add(this->fcgCXVppDetailEnhance);
            this->fcggroupBoxVppDetailEnahance->Controls->Add(this->fcgPNVppEdgelevel);
            this->fcggroupBoxVppDetailEnahance->Controls->Add(this->fcgPNVppUnsharp);
            this->fcggroupBoxVppDetailEnahance->Location = System::Drawing::Point(3, 199);
            this->fcggroupBoxVppDetailEnahance->Name = L"fcggroupBoxVppDetailEnahance";
            this->fcggroupBoxVppDetailEnahance->Size = System::Drawing::Size(320, 110);
            this->fcggroupBoxVppDetailEnahance->TabIndex = 11;
            this->fcggroupBoxVppDetailEnahance->TabStop = false;
            this->fcggroupBoxVppDetailEnahance->Text = L"ディテール・輪郭強調";
            // 
            // fcgPNVppWarpsharp
            // 
            this->fcgPNVppWarpsharp->Controls->Add(this->fcgLBVppWarpsharpBlurType);
            this->fcgPNVppWarpsharp->Controls->Add(this->fcgNUVppWarpsharpBlurType);
            this->fcgPNVppWarpsharp->Controls->Add(this->fcgLBVppWarpsharpDepth);
            this->fcgPNVppWarpsharp->Controls->Add(this->fcgLBVppWarpsharpBlur);
            this->fcgPNVppWarpsharp->Controls->Add(this->fcgLBVppWarpsharpThreshold);
            this->fcgPNVppWarpsharp->Controls->Add(this->fcgNUVppWarpsharpDepth);
            this->fcgPNVppWarpsharp->Controls->Add(this->fcgNUVppWarpsharpBlur);
            this->fcgPNVppWarpsharp->Controls->Add(this->fcgNUVppWarpsharpThreshold);
            this->fcgPNVppWarpsharp->Controls->Add(this->label14);
            this->fcgPNVppWarpsharp->Controls->Add(this->numericUpDown6);
            this->fcgPNVppWarpsharp->Location = System::Drawing::Point(3, 43);
            this->fcgPNVppWarpsharp->Name = L"fcgPNVppWarpsharp";
            this->fcgPNVppWarpsharp->Size = System::Drawing::Size(311, 61);
            this->fcgPNVppWarpsharp->TabIndex = 66;
            // 
            // fcgLBVppWarpsharpBlurType
            // 
            this->fcgLBVppWarpsharpBlurType->AutoSize = true;
            this->fcgLBVppWarpsharpBlurType->Location = System::Drawing::Point(182, 37);
            this->fcgLBVppWarpsharpBlurType->Name = L"fcgLBVppWarpsharpBlurType";
            this->fcgLBVppWarpsharpBlurType->Size = System::Drawing::Size(30, 14);
            this->fcgLBVppWarpsharpBlurType->TabIndex = 7;
            this->fcgLBVppWarpsharpBlurType->Text = L"type";
            // 
            // fcgNUVppWarpsharpBlurType
            // 
            this->fcgNUVppWarpsharpBlurType->Location = System::Drawing::Point(239, 34);
            this->fcgNUVppWarpsharpBlurType->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 1, 0, 0, 0 });
            this->fcgNUVppWarpsharpBlurType->Name = L"fcgNUVppWarpsharpBlurType";
            this->fcgNUVppWarpsharpBlurType->Size = System::Drawing::Size(60, 21);
            this->fcgNUVppWarpsharpBlurType->TabIndex = 8;
            this->fcgNUVppWarpsharpBlurType->Tag = L"reCmd";
            this->fcgNUVppWarpsharpBlurType->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
            this->fcgNUVppWarpsharpBlurType->Value = System::Decimal(gcnew cli::array< System::Int32 >(4) { 1, 0, 0, 0 });
            // 
            // fcgLBVppWarpsharpDepth
            // 
            this->fcgLBVppWarpsharpDepth->AutoSize = true;
            this->fcgLBVppWarpsharpDepth->Location = System::Drawing::Point(182, 9);
            this->fcgLBVppWarpsharpDepth->Name = L"fcgLBVppWarpsharpDepth";
            this->fcgLBVppWarpsharpDepth->Size = System::Drawing::Size(29, 14);
            this->fcgLBVppWarpsharpDepth->TabIndex = 3;
            this->fcgLBVppWarpsharpDepth->Text = L"深度";
            // 
            // fcgLBVppWarpsharpBlur
            // 
            this->fcgLBVppWarpsharpBlur->AutoSize = true;
            this->fcgLBVppWarpsharpBlur->Location = System::Drawing::Point(8, 37);
            this->fcgLBVppWarpsharpBlur->Name = L"fcgLBVppWarpsharpBlur";
            this->fcgLBVppWarpsharpBlur->Size = System::Drawing::Size(29, 14);
            this->fcgLBVppWarpsharpBlur->TabIndex = 5;
            this->fcgLBVppWarpsharpBlur->Text = L"blur";
            // 
            // fcgLBVppWarpsharpThreshold
            // 
            this->fcgLBVppWarpsharpThreshold->AutoSize = true;
            this->fcgLBVppWarpsharpThreshold->Location = System::Drawing::Point(8, 10);
            this->fcgLBVppWarpsharpThreshold->Name = L"fcgLBVppWarpsharpThreshold";
            this->fcgLBVppWarpsharpThreshold->Size = System::Drawing::Size(29, 14);
            this->fcgLBVppWarpsharpThreshold->TabIndex = 1;
            this->fcgLBVppWarpsharpThreshold->Text = L"閾値";
            // 
            // fcgNUVppWarpsharpDepth
            // 
            this->fcgNUVppWarpsharpDepth->Location = System::Drawing::Point(239, 7);
            this->fcgNUVppWarpsharpDepth->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 127, 0, 0, 0 });
            this->fcgNUVppWarpsharpDepth->Minimum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 128, 0, 0, System::Int32::MinValue });
            this->fcgNUVppWarpsharpDepth->Name = L"fcgNUVppWarpsharpDepth";
            this->fcgNUVppWarpsharpDepth->Size = System::Drawing::Size(60, 21);
            this->fcgNUVppWarpsharpDepth->TabIndex = 4;
            this->fcgNUVppWarpsharpDepth->Tag = L"reCmd";
            this->fcgNUVppWarpsharpDepth->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
            this->fcgNUVppWarpsharpDepth->Value = System::Decimal(gcnew cli::array< System::Int32 >(4) { 1, 0, 0, 0 });
            // 
            // fcgNUVppWarpsharpBlur
            // 
            this->fcgNUVppWarpsharpBlur->Location = System::Drawing::Point(68, 34);
            this->fcgNUVppWarpsharpBlur->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 127, 0, 0, 0 });
            this->fcgNUVppWarpsharpBlur->Name = L"fcgNUVppWarpsharpBlur";
            this->fcgNUVppWarpsharpBlur->Size = System::Drawing::Size(60, 21);
            this->fcgNUVppWarpsharpBlur->TabIndex = 6;
            this->fcgNUVppWarpsharpBlur->Tag = L"reCmd";
            this->fcgNUVppWarpsharpBlur->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
            this->fcgNUVppWarpsharpBlur->Value = System::Decimal(gcnew cli::array< System::Int32 >(4) { 1, 0, 0, 0 });
            // 
            // fcgNUVppWarpsharpThreshold
            // 
            this->fcgNUVppWarpsharpThreshold->Location = System::Drawing::Point(68, 7);
            this->fcgNUVppWarpsharpThreshold->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 255, 0, 0, 0 });
            this->fcgNUVppWarpsharpThreshold->Name = L"fcgNUVppWarpsharpThreshold";
            this->fcgNUVppWarpsharpThreshold->Size = System::Drawing::Size(60, 21);
            this->fcgNUVppWarpsharpThreshold->TabIndex = 2;
            this->fcgNUVppWarpsharpThreshold->Tag = L"reCmd";
            this->fcgNUVppWarpsharpThreshold->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
            this->fcgNUVppWarpsharpThreshold->Value = System::Decimal(gcnew cli::array< System::Int32 >(4) { 3, 0, 0, 0 });
            // 
            // label14
            // 
            this->label14->AutoSize = true;
            this->label14->Location = System::Drawing::Point(69, 66);
            this->label14->Name = L"label14";
            this->label14->Size = System::Drawing::Size(29, 14);
            this->label14->TabIndex = 11;
            this->label14->Text = L"閾値";
            // 
            // numericUpDown6
            // 
            this->numericUpDown6->Location = System::Drawing::Point(129, 64);
            this->numericUpDown6->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 255, 0, 0, 0 });
            this->numericUpDown6->Name = L"numericUpDown6";
            this->numericUpDown6->Size = System::Drawing::Size(60, 21);
            this->numericUpDown6->TabIndex = 8;
            this->numericUpDown6->Tag = L"chValue";
            this->numericUpDown6->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
            this->numericUpDown6->Value = System::Decimal(gcnew cli::array< System::Int32 >(4) { 100, 0, 0, 0 });
            // 
            // fcgCXVppDetailEnhance
            // 
            this->fcgCXVppDetailEnhance->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
            this->fcgCXVppDetailEnhance->FormattingEnabled = true;
            this->fcgCXVppDetailEnhance->Location = System::Drawing::Point(26, 20);
            this->fcgCXVppDetailEnhance->Name = L"fcgCXVppDetailEnhance";
            this->fcgCXVppDetailEnhance->Size = System::Drawing::Size(191, 22);
            this->fcgCXVppDetailEnhance->TabIndex = 0;
            this->fcgCXVppDetailEnhance->Tag = L"reCmd";
            this->fcgCXVppDetailEnhance->SelectedIndexChanged += gcnew System::EventHandler(this, &frmConfig::fcgChangeEnabled);
            // 
            // fcgPNVppEdgelevel
            // 
            this->fcgPNVppEdgelevel->Controls->Add(this->fcgLBVppEdgelevelWhite);
            this->fcgPNVppEdgelevel->Controls->Add(this->fcgNUVppEdgelevelWhite);
            this->fcgPNVppEdgelevel->Controls->Add(this->fcgLBVppEdgelevelThreshold);
            this->fcgPNVppEdgelevel->Controls->Add(this->fcgLBVppEdgelevelBlack);
            this->fcgPNVppEdgelevel->Controls->Add(this->fcgLBVppEdgelevelStrength);
            this->fcgPNVppEdgelevel->Controls->Add(this->fcgNUVppEdgelevelThreshold);
            this->fcgPNVppEdgelevel->Controls->Add(this->fcgNUVppEdgelevelBlack);
            this->fcgPNVppEdgelevel->Controls->Add(this->fcgNUVppEdgelevelStrength);
            this->fcgPNVppEdgelevel->Controls->Add(this->label10);
            this->fcgPNVppEdgelevel->Controls->Add(this->numericUpDown4);
            this->fcgPNVppEdgelevel->Location = System::Drawing::Point(3, 43);
            this->fcgPNVppEdgelevel->Name = L"fcgPNVppEdgelevel";
            this->fcgPNVppEdgelevel->Size = System::Drawing::Size(311, 61);
            this->fcgPNVppEdgelevel->TabIndex = 1;
            // 
            // fcgLBVppEdgelevelWhite
            // 
            this->fcgLBVppEdgelevelWhite->AutoSize = true;
            this->fcgLBVppEdgelevelWhite->Location = System::Drawing::Point(182, 37);
            this->fcgLBVppEdgelevelWhite->Name = L"fcgLBVppEdgelevelWhite";
            this->fcgLBVppEdgelevelWhite->Size = System::Drawing::Size(18, 14);
            this->fcgLBVppEdgelevelWhite->TabIndex = 7;
            this->fcgLBVppEdgelevelWhite->Text = L"白";
            // 
            // fcgNUVppEdgelevelWhite
            // 
            this->fcgNUVppEdgelevelWhite->DecimalPlaces = 1;
            this->fcgNUVppEdgelevelWhite->Location = System::Drawing::Point(239, 34);
            this->fcgNUVppEdgelevelWhite->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 31, 0, 0, 0 });
            this->fcgNUVppEdgelevelWhite->Name = L"fcgNUVppEdgelevelWhite";
            this->fcgNUVppEdgelevelWhite->Size = System::Drawing::Size(60, 21);
            this->fcgNUVppEdgelevelWhite->TabIndex = 8;
            this->fcgNUVppEdgelevelWhite->Tag = L"reCmd";
            this->fcgNUVppEdgelevelWhite->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
            this->fcgNUVppEdgelevelWhite->Value = System::Decimal(gcnew cli::array< System::Int32 >(4) { 1, 0, 0, 0 });
            // 
            // fcgLBVppEdgelevelThreshold
            // 
            this->fcgLBVppEdgelevelThreshold->AutoSize = true;
            this->fcgLBVppEdgelevelThreshold->Location = System::Drawing::Point(182, 9);
            this->fcgLBVppEdgelevelThreshold->Name = L"fcgLBVppEdgelevelThreshold";
            this->fcgLBVppEdgelevelThreshold->Size = System::Drawing::Size(29, 14);
            this->fcgLBVppEdgelevelThreshold->TabIndex = 3;
            this->fcgLBVppEdgelevelThreshold->Text = L"閾値";
            // 
            // fcgLBVppEdgelevelBlack
            // 
            this->fcgLBVppEdgelevelBlack->AutoSize = true;
            this->fcgLBVppEdgelevelBlack->Location = System::Drawing::Point(8, 37);
            this->fcgLBVppEdgelevelBlack->Name = L"fcgLBVppEdgelevelBlack";
            this->fcgLBVppEdgelevelBlack->Size = System::Drawing::Size(18, 14);
            this->fcgLBVppEdgelevelBlack->TabIndex = 5;
            this->fcgLBVppEdgelevelBlack->Text = L"黒";
            // 
            // fcgLBVppEdgelevelStrength
            // 
            this->fcgLBVppEdgelevelStrength->AutoSize = true;
            this->fcgLBVppEdgelevelStrength->Location = System::Drawing::Point(8, 10);
            this->fcgLBVppEdgelevelStrength->Name = L"fcgLBVppEdgelevelStrength";
            this->fcgLBVppEdgelevelStrength->Size = System::Drawing::Size(29, 14);
            this->fcgLBVppEdgelevelStrength->TabIndex = 1;
            this->fcgLBVppEdgelevelStrength->Text = L"特性";
            // 
            // fcgNUVppEdgelevelThreshold
            // 
            this->fcgNUVppEdgelevelThreshold->Location = System::Drawing::Point(239, 7);
            this->fcgNUVppEdgelevelThreshold->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 255, 0, 0, 0 });
            this->fcgNUVppEdgelevelThreshold->Name = L"fcgNUVppEdgelevelThreshold";
            this->fcgNUVppEdgelevelThreshold->Size = System::Drawing::Size(60, 21);
            this->fcgNUVppEdgelevelThreshold->TabIndex = 4;
            this->fcgNUVppEdgelevelThreshold->Tag = L"reCmd";
            this->fcgNUVppEdgelevelThreshold->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
            this->fcgNUVppEdgelevelThreshold->Value = System::Decimal(gcnew cli::array< System::Int32 >(4) { 1, 0, 0, 0 });
            // 
            // fcgNUVppEdgelevelBlack
            // 
            this->fcgNUVppEdgelevelBlack->DecimalPlaces = 1;
            this->fcgNUVppEdgelevelBlack->Location = System::Drawing::Point(68, 34);
            this->fcgNUVppEdgelevelBlack->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 31, 0, 0, 0 });
            this->fcgNUVppEdgelevelBlack->Name = L"fcgNUVppEdgelevelBlack";
            this->fcgNUVppEdgelevelBlack->Size = System::Drawing::Size(60, 21);
            this->fcgNUVppEdgelevelBlack->TabIndex = 6;
            this->fcgNUVppEdgelevelBlack->Tag = L"reCmd";
            this->fcgNUVppEdgelevelBlack->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
            this->fcgNUVppEdgelevelBlack->Value = System::Decimal(gcnew cli::array< System::Int32 >(4) { 1, 0, 0, 0 });
            // 
            // fcgNUVppEdgelevelStrength
            // 
            this->fcgNUVppEdgelevelStrength->Location = System::Drawing::Point(68, 7);
            this->fcgNUVppEdgelevelStrength->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 31, 0, 0, 0 });
            this->fcgNUVppEdgelevelStrength->Minimum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 31, 0, 0, System::Int32::MinValue });
            this->fcgNUVppEdgelevelStrength->Name = L"fcgNUVppEdgelevelStrength";
            this->fcgNUVppEdgelevelStrength->Size = System::Drawing::Size(60, 21);
            this->fcgNUVppEdgelevelStrength->TabIndex = 2;
            this->fcgNUVppEdgelevelStrength->Tag = L"reCmd";
            this->fcgNUVppEdgelevelStrength->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
            this->fcgNUVppEdgelevelStrength->Value = System::Decimal(gcnew cli::array< System::Int32 >(4) { 3, 0, 0, 0 });
            // 
            // label10
            // 
            this->label10->AutoSize = true;
            this->label10->Location = System::Drawing::Point(69, 66);
            this->label10->Name = L"label10";
            this->label10->Size = System::Drawing::Size(29, 14);
            this->label10->TabIndex = 11;
            this->label10->Text = L"閾値";
            // 
            // numericUpDown4
            // 
            this->numericUpDown4->Location = System::Drawing::Point(129, 64);
            this->numericUpDown4->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 255, 0, 0, 0 });
            this->numericUpDown4->Name = L"numericUpDown4";
            this->numericUpDown4->Size = System::Drawing::Size(60, 21);
            this->numericUpDown4->TabIndex = 8;
            this->numericUpDown4->Tag = L"chValue";
            this->numericUpDown4->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
            this->numericUpDown4->Value = System::Decimal(gcnew cli::array< System::Int32 >(4) { 100, 0, 0, 0 });
            // 
            // fcgPNVppUnsharp
            // 
            this->fcgPNVppUnsharp->Controls->Add(this->fcgLBVppUnsharpThreshold);
            this->fcgPNVppUnsharp->Controls->Add(this->fcgLBVppUnsharpWeight);
            this->fcgPNVppUnsharp->Controls->Add(this->fcgLBVppUnsharpRadius);
            this->fcgPNVppUnsharp->Controls->Add(this->fcgNUVppUnsharpThreshold);
            this->fcgPNVppUnsharp->Controls->Add(this->fcgNUVppUnsharpWeight);
            this->fcgPNVppUnsharp->Controls->Add(this->fcgNUVppUnsharpRadius);
            this->fcgPNVppUnsharp->Location = System::Drawing::Point(3, 43);
            this->fcgPNVppUnsharp->Name = L"fcgPNVppUnsharp";
            this->fcgPNVppUnsharp->Size = System::Drawing::Size(310, 61);
            this->fcgPNVppUnsharp->TabIndex = 65;
            // 
            // fcgLBVppUnsharpThreshold
            // 
            this->fcgLBVppUnsharpThreshold->AutoSize = true;
            this->fcgLBVppUnsharpThreshold->Location = System::Drawing::Point(184, 9);
            this->fcgLBVppUnsharpThreshold->Name = L"fcgLBVppUnsharpThreshold";
            this->fcgLBVppUnsharpThreshold->Size = System::Drawing::Size(29, 14);
            this->fcgLBVppUnsharpThreshold->TabIndex = 11;
            this->fcgLBVppUnsharpThreshold->Text = L"閾値";
            // 
            // fcgLBVppUnsharpWeight
            // 
            this->fcgLBVppUnsharpWeight->AutoSize = true;
            this->fcgLBVppUnsharpWeight->Location = System::Drawing::Point(10, 37);
            this->fcgLBVppUnsharpWeight->Name = L"fcgLBVppUnsharpWeight";
            this->fcgLBVppUnsharpWeight->Size = System::Drawing::Size(26, 14);
            this->fcgLBVppUnsharpWeight->TabIndex = 10;
            this->fcgLBVppUnsharpWeight->Text = L"強さ";
            // 
            // fcgLBVppUnsharpRadius
            // 
            this->fcgLBVppUnsharpRadius->AutoSize = true;
            this->fcgLBVppUnsharpRadius->Location = System::Drawing::Point(10, 10);
            this->fcgLBVppUnsharpRadius->Name = L"fcgLBVppUnsharpRadius";
            this->fcgLBVppUnsharpRadius->Size = System::Drawing::Size(29, 14);
            this->fcgLBVppUnsharpRadius->TabIndex = 9;
            this->fcgLBVppUnsharpRadius->Text = L"半径";
            // 
            // fcgNUVppUnsharpThreshold
            // 
            this->fcgNUVppUnsharpThreshold->Location = System::Drawing::Point(244, 7);
            this->fcgNUVppUnsharpThreshold->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 255, 0, 0, 0 });
            this->fcgNUVppUnsharpThreshold->Name = L"fcgNUVppUnsharpThreshold";
            this->fcgNUVppUnsharpThreshold->Size = System::Drawing::Size(60, 21);
            this->fcgNUVppUnsharpThreshold->TabIndex = 8;
            this->fcgNUVppUnsharpThreshold->Tag = L"reCmd";
            this->fcgNUVppUnsharpThreshold->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
            this->fcgNUVppUnsharpThreshold->Value = System::Decimal(gcnew cli::array< System::Int32 >(4) { 1, 0, 0, 0 });
            // 
            // fcgNUVppUnsharpWeight
            // 
            this->fcgNUVppUnsharpWeight->DecimalPlaces = 1;
            this->fcgNUVppUnsharpWeight->Increment = System::Decimal(gcnew cli::array< System::Int32 >(4) { 5, 0, 0, 65536 });
            this->fcgNUVppUnsharpWeight->Location = System::Drawing::Point(70, 34);
            this->fcgNUVppUnsharpWeight->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 10, 0, 0, 0 });
            this->fcgNUVppUnsharpWeight->Name = L"fcgNUVppUnsharpWeight";
            this->fcgNUVppUnsharpWeight->Size = System::Drawing::Size(60, 21);
            this->fcgNUVppUnsharpWeight->TabIndex = 7;
            this->fcgNUVppUnsharpWeight->Tag = L"reCmd";
            this->fcgNUVppUnsharpWeight->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
            this->fcgNUVppUnsharpWeight->Value = System::Decimal(gcnew cli::array< System::Int32 >(4) { 1, 0, 0, 0 });
            // 
            // fcgNUVppUnsharpRadius
            // 
            this->fcgNUVppUnsharpRadius->Location = System::Drawing::Point(70, 7);
            this->fcgNUVppUnsharpRadius->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 9, 0, 0, 0 });
            this->fcgNUVppUnsharpRadius->Minimum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 1, 0, 0, 0 });
            this->fcgNUVppUnsharpRadius->Name = L"fcgNUVppUnsharpRadius";
            this->fcgNUVppUnsharpRadius->Size = System::Drawing::Size(60, 21);
            this->fcgNUVppUnsharpRadius->TabIndex = 6;
            this->fcgNUVppUnsharpRadius->Tag = L"reCmd";
            this->fcgNUVppUnsharpRadius->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
            this->fcgNUVppUnsharpRadius->Value = System::Decimal(gcnew cli::array< System::Int32 >(4) { 3, 0, 0, 0 });
            // 
            // fcggroupBoxVppDeinterlace
            // 
            this->fcggroupBoxVppDeinterlace->Controls->Add(this->fcgLBVppDeinterlace);
            this->fcggroupBoxVppDeinterlace->Controls->Add(this->fcgCXVppDeinterlace);
            this->fcggroupBoxVppDeinterlace->Controls->Add(this->fcgPNVppYadif);
            this->fcggroupBoxVppDeinterlace->Controls->Add(this->fcgPNVppNnedi);
            this->fcggroupBoxVppDeinterlace->Controls->Add(this->fcgPNVppAfs);
            this->fcggroupBoxVppDeinterlace->Location = System::Drawing::Point(342, 3);
            this->fcggroupBoxVppDeinterlace->Name = L"fcggroupBoxVppDeinterlace";
            this->fcggroupBoxVppDeinterlace->Size = System::Drawing::Size(262, 338);
            this->fcggroupBoxVppDeinterlace->TabIndex = 40;
            this->fcggroupBoxVppDeinterlace->TabStop = false;
            this->fcggroupBoxVppDeinterlace->Text = L"インタレ解除";
            // 
            // fcgLBVppDeinterlace
            // 
            this->fcgLBVppDeinterlace->AutoSize = true;
            this->fcgLBVppDeinterlace->Location = System::Drawing::Point(15, 19);
            this->fcgLBVppDeinterlace->Name = L"fcgLBVppDeinterlace";
            this->fcgLBVppDeinterlace->Size = System::Drawing::Size(54, 14);
            this->fcgLBVppDeinterlace->TabIndex = 0;
            this->fcgLBVppDeinterlace->Text = L"解除モード";
            // 
            // fcgCXVppDeinterlace
            // 
            this->fcgCXVppDeinterlace->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
            this->fcgCXVppDeinterlace->FormattingEnabled = true;
            this->fcgCXVppDeinterlace->Location = System::Drawing::Point(83, 15);
            this->fcgCXVppDeinterlace->Name = L"fcgCXVppDeinterlace";
            this->fcgCXVppDeinterlace->Size = System::Drawing::Size(164, 22);
            this->fcgCXVppDeinterlace->TabIndex = 1;
            this->fcgCXVppDeinterlace->Tag = L"reCmd";
            this->fcgCXVppDeinterlace->SelectedIndexChanged += gcnew System::EventHandler(this, &frmConfig::fcgChangeEnabled);
            // 
            // fcgPNVppYadif
            // 
            this->fcgPNVppYadif->Controls->Add(this->fcgLBVppYadifMode);
            this->fcgPNVppYadif->Controls->Add(this->fcgCXVppYadifMode);
            this->fcgPNVppYadif->Location = System::Drawing::Point(6, 41);
            this->fcgPNVppYadif->Name = L"fcgPNVppYadif";
            this->fcgPNVppYadif->Size = System::Drawing::Size(251, 294);
            this->fcgPNVppYadif->TabIndex = 2;
            // 
            // fcgLBVppYadifMode
            // 
            this->fcgLBVppYadifMode->AutoSize = true;
            this->fcgLBVppYadifMode->Location = System::Drawing::Point(14, 12);
            this->fcgLBVppYadifMode->Name = L"fcgLBVppYadifMode";
            this->fcgLBVppYadifMode->Size = System::Drawing::Size(38, 14);
            this->fcgLBVppYadifMode->TabIndex = 78;
            this->fcgLBVppYadifMode->Text = L"mode";
            // 
            // fcgCXVppYadifMode
            // 
            this->fcgCXVppYadifMode->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
            this->fcgCXVppYadifMode->FormattingEnabled = true;
            this->fcgCXVppYadifMode->Location = System::Drawing::Point(81, 9);
            this->fcgCXVppYadifMode->Name = L"fcgCXVppYadifMode";
            this->fcgCXVppYadifMode->Size = System::Drawing::Size(160, 22);
            this->fcgCXVppYadifMode->TabIndex = 0;
            this->fcgCXVppYadifMode->Tag = L"reCmd";
            // 
            // fcgPNVppNnedi
            // 
            this->fcgPNVppNnedi->Controls->Add(this->fcgLBVppNnediErrorType);
            this->fcgPNVppNnedi->Controls->Add(this->fcgCXVppNnediErrorType);
            this->fcgPNVppNnedi->Controls->Add(this->fcgLBVppNnediPrescreen);
            this->fcgPNVppNnedi->Controls->Add(this->fcgCXVppNnediPrescreen);
            this->fcgPNVppNnedi->Controls->Add(this->fcgLBVppNnediPrec);
            this->fcgPNVppNnedi->Controls->Add(this->fcgCXVppNnediPrec);
            this->fcgPNVppNnedi->Controls->Add(this->fcgLBVppNnediQual);
            this->fcgPNVppNnedi->Controls->Add(this->fcgCXVppNnediQual);
            this->fcgPNVppNnedi->Controls->Add(this->fcgLBVppNnediNsize);
            this->fcgPNVppNnedi->Controls->Add(this->fcgCXVppNnediNsize);
            this->fcgPNVppNnedi->Controls->Add(this->fcgLBVppNnediNns);
            this->fcgPNVppNnedi->Controls->Add(this->fcgCXVppNnediNns);
            this->fcgPNVppNnedi->Location = System::Drawing::Point(6, 41);
            this->fcgPNVppNnedi->Name = L"fcgPNVppNnedi";
            this->fcgPNVppNnedi->Size = System::Drawing::Size(251, 294);
            this->fcgPNVppNnedi->TabIndex = 1;
            // 
            // fcgLBVppNnediErrorType
            // 
            this->fcgLBVppNnediErrorType->AutoSize = true;
            this->fcgLBVppNnediErrorType->Location = System::Drawing::Point(14, 152);
            this->fcgLBVppNnediErrorType->Name = L"fcgLBVppNnediErrorType";
            this->fcgLBVppNnediErrorType->Size = System::Drawing::Size(58, 14);
            this->fcgLBVppNnediErrorType->TabIndex = 10;
            this->fcgLBVppNnediErrorType->Text = L"errortype";
            // 
            // fcgCXVppNnediErrorType
            // 
            this->fcgCXVppNnediErrorType->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
            this->fcgCXVppNnediErrorType->FormattingEnabled = true;
            this->fcgCXVppNnediErrorType->Location = System::Drawing::Point(81, 149);
            this->fcgCXVppNnediErrorType->Name = L"fcgCXVppNnediErrorType";
            this->fcgCXVppNnediErrorType->Size = System::Drawing::Size(160, 22);
            this->fcgCXVppNnediErrorType->TabIndex = 11;
            this->fcgCXVppNnediErrorType->Tag = L"reCmd";
            // 
            // fcgLBVppNnediPrescreen
            // 
            this->fcgLBVppNnediPrescreen->AutoSize = true;
            this->fcgLBVppNnediPrescreen->Location = System::Drawing::Point(14, 124);
            this->fcgLBVppNnediPrescreen->Name = L"fcgLBVppNnediPrescreen";
            this->fcgLBVppNnediPrescreen->Size = System::Drawing::Size(60, 14);
            this->fcgLBVppNnediPrescreen->TabIndex = 8;
            this->fcgLBVppNnediPrescreen->Text = L"prescreen";
            // 
            // fcgCXVppNnediPrescreen
            // 
            this->fcgCXVppNnediPrescreen->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
            this->fcgCXVppNnediPrescreen->FormattingEnabled = true;
            this->fcgCXVppNnediPrescreen->Location = System::Drawing::Point(81, 121);
            this->fcgCXVppNnediPrescreen->Name = L"fcgCXVppNnediPrescreen";
            this->fcgCXVppNnediPrescreen->Size = System::Drawing::Size(160, 22);
            this->fcgCXVppNnediPrescreen->TabIndex = 9;
            this->fcgCXVppNnediPrescreen->Tag = L"reCmd";
            // 
            // fcgLBVppNnediPrec
            // 
            this->fcgLBVppNnediPrec->AutoSize = true;
            this->fcgLBVppNnediPrec->Location = System::Drawing::Point(14, 96);
            this->fcgLBVppNnediPrec->Name = L"fcgLBVppNnediPrec";
            this->fcgLBVppNnediPrec->Size = System::Drawing::Size(31, 14);
            this->fcgLBVppNnediPrec->TabIndex = 6;
            this->fcgLBVppNnediPrec->Text = L"prec";
            // 
            // fcgCXVppNnediPrec
            // 
            this->fcgCXVppNnediPrec->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
            this->fcgCXVppNnediPrec->FormattingEnabled = true;
            this->fcgCXVppNnediPrec->Location = System::Drawing::Point(81, 93);
            this->fcgCXVppNnediPrec->Name = L"fcgCXVppNnediPrec";
            this->fcgCXVppNnediPrec->Size = System::Drawing::Size(160, 22);
            this->fcgCXVppNnediPrec->TabIndex = 7;
            this->fcgCXVppNnediPrec->Tag = L"reCmd";
            // 
            // fcgLBVppNnediQual
            // 
            this->fcgLBVppNnediQual->AutoSize = true;
            this->fcgLBVppNnediQual->Location = System::Drawing::Point(14, 68);
            this->fcgLBVppNnediQual->Name = L"fcgLBVppNnediQual";
            this->fcgLBVppNnediQual->Size = System::Drawing::Size(30, 14);
            this->fcgLBVppNnediQual->TabIndex = 4;
            this->fcgLBVppNnediQual->Text = L"qual";
            // 
            // fcgCXVppNnediQual
            // 
            this->fcgCXVppNnediQual->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
            this->fcgCXVppNnediQual->FormattingEnabled = true;
            this->fcgCXVppNnediQual->Location = System::Drawing::Point(81, 65);
            this->fcgCXVppNnediQual->Name = L"fcgCXVppNnediQual";
            this->fcgCXVppNnediQual->Size = System::Drawing::Size(160, 22);
            this->fcgCXVppNnediQual->TabIndex = 5;
            this->fcgCXVppNnediQual->Tag = L"reCmd";
            // 
            // fcgLBVppNnediNsize
            // 
            this->fcgLBVppNnediNsize->AutoSize = true;
            this->fcgLBVppNnediNsize->Location = System::Drawing::Point(14, 40);
            this->fcgLBVppNnediNsize->Name = L"fcgLBVppNnediNsize";
            this->fcgLBVppNnediNsize->Size = System::Drawing::Size(34, 14);
            this->fcgLBVppNnediNsize->TabIndex = 2;
            this->fcgLBVppNnediNsize->Text = L"nsize";
            // 
            // fcgCXVppNnediNsize
            // 
            this->fcgCXVppNnediNsize->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
            this->fcgCXVppNnediNsize->FormattingEnabled = true;
            this->fcgCXVppNnediNsize->Location = System::Drawing::Point(81, 37);
            this->fcgCXVppNnediNsize->Name = L"fcgCXVppNnediNsize";
            this->fcgCXVppNnediNsize->Size = System::Drawing::Size(160, 22);
            this->fcgCXVppNnediNsize->TabIndex = 3;
            this->fcgCXVppNnediNsize->Tag = L"reCmd";
            // 
            // fcgLBVppNnediNns
            // 
            this->fcgLBVppNnediNns->AutoSize = true;
            this->fcgLBVppNnediNns->Location = System::Drawing::Point(14, 12);
            this->fcgLBVppNnediNns->Name = L"fcgLBVppNnediNns";
            this->fcgLBVppNnediNns->Size = System::Drawing::Size(26, 14);
            this->fcgLBVppNnediNns->TabIndex = 0;
            this->fcgLBVppNnediNns->Text = L"nns";
            // 
            // fcgCXVppNnediNns
            // 
            this->fcgCXVppNnediNns->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
            this->fcgCXVppNnediNns->FormattingEnabled = true;
            this->fcgCXVppNnediNns->Location = System::Drawing::Point(81, 9);
            this->fcgCXVppNnediNns->Name = L"fcgCXVppNnediNns";
            this->fcgCXVppNnediNns->Size = System::Drawing::Size(160, 22);
            this->fcgCXVppNnediNns->TabIndex = 1;
            this->fcgCXVppNnediNns->Tag = L"reCmd";
            // 
            // fcgPNVppAfs
            // 
            this->fcgPNVppAfs->Controls->Add(this->fcgTBVppAfsThreCMotion);
            this->fcgPNVppAfs->Controls->Add(this->fcgLBVppAfsThreCMotion);
            this->fcgPNVppAfs->Controls->Add(this->fcgTBVppAfsThreYMotion);
            this->fcgPNVppAfs->Controls->Add(this->fcgLBVppAfsThreYmotion);
            this->fcgPNVppAfs->Controls->Add(this->fcgTBVppAfsThreDeint);
            this->fcgPNVppAfs->Controls->Add(this->fcgLBVppAfsThreDeint);
            this->fcgPNVppAfs->Controls->Add(this->fcgTBVppAfsThreShift);
            this->fcgPNVppAfs->Controls->Add(this->fcgLBVppAfsThreShift);
            this->fcgPNVppAfs->Controls->Add(this->fcgTBVppAfsCoeffShift);
            this->fcgPNVppAfs->Controls->Add(this->fcgLBVppAfsCoeffShift);
            this->fcgPNVppAfs->Controls->Add(this->label20);
            this->fcgPNVppAfs->Controls->Add(this->label19);
            this->fcgPNVppAfs->Controls->Add(this->label4);
            this->fcgPNVppAfs->Controls->Add(this->fcgLBVppAfsUp);
            this->fcgPNVppAfs->Controls->Add(this->fcgNUVppAfsRight);
            this->fcgPNVppAfs->Controls->Add(this->fcgNUVppAfsLeft);
            this->fcgPNVppAfs->Controls->Add(this->fcgNUVppAfsBottom);
            this->fcgPNVppAfs->Controls->Add(this->fcgNUVppAfsUp);
            this->fcgPNVppAfs->Controls->Add(this->fcgTBVppAfsMethodSwitch);
            this->fcgPNVppAfs->Controls->Add(this->fcgCBVppAfs24fps);
            this->fcgPNVppAfs->Controls->Add(this->fcgCBVppAfsTune);
            this->fcgPNVppAfs->Controls->Add(this->fcgCBVppAfsSmooth);
            this->fcgPNVppAfs->Controls->Add(this->fcgCBVppAfsDrop);
            this->fcgPNVppAfs->Controls->Add(this->fcgCBVppAfsShift);
            this->fcgPNVppAfs->Controls->Add(this->fcgLBVppAfsAnalyze);
            this->fcgPNVppAfs->Controls->Add(this->fcgNUVppAfsThreCMotion);
            this->fcgPNVppAfs->Controls->Add(this->fcgNUVppAfsThreShift);
            this->fcgPNVppAfs->Controls->Add(this->fcgNUVppAfsThreDeint);
            this->fcgPNVppAfs->Controls->Add(this->fcgNUVppAfsThreYMotion);
            this->fcgPNVppAfs->Controls->Add(this->fcgLBVppAfsMethodSwitch);
            this->fcgPNVppAfs->Controls->Add(this->fcgCXVppAfsAnalyze);
            this->fcgPNVppAfs->Controls->Add(this->fcgNUVppAfsCoeffShift);
            this->fcgPNVppAfs->Controls->Add(this->fcgNUVppAfsMethodSwitch);
            this->fcgPNVppAfs->Location = System::Drawing::Point(6, 41);
            this->fcgPNVppAfs->Name = L"fcgPNVppAfs";
            this->fcgPNVppAfs->Size = System::Drawing::Size(251, 294);
            this->fcgPNVppAfs->TabIndex = 2;
            // 
            // fcgTBVppAfsThreCMotion
            // 
            this->fcgTBVppAfsThreCMotion->AutoSize = false;
            this->fcgTBVppAfsThreCMotion->Location = System::Drawing::Point(60, 179);
            this->fcgTBVppAfsThreCMotion->Maximum = 1024;
            this->fcgTBVppAfsThreCMotion->Name = L"fcgTBVppAfsThreCMotion";
            this->fcgTBVppAfsThreCMotion->Size = System::Drawing::Size(115, 18);
            this->fcgTBVppAfsThreCMotion->TabIndex = 24;
            this->fcgTBVppAfsThreCMotion->TickStyle = System::Windows::Forms::TickStyle::None;
            this->fcgTBVppAfsThreCMotion->Scroll += gcnew System::EventHandler(this, &frmConfig::fcgTBVppAfsScroll);
            // 
            // fcgLBVppAfsThreCMotion
            // 
            this->fcgLBVppAfsThreCMotion->AutoSize = true;
            this->fcgLBVppAfsThreCMotion->Location = System::Drawing::Point(9, 179);
            this->fcgLBVppAfsThreCMotion->Name = L"fcgLBVppAfsThreCMotion";
            this->fcgLBVppAfsThreCMotion->Size = System::Drawing::Size(33, 14);
            this->fcgLBVppAfsThreCMotion->TabIndex = 23;
            this->fcgLBVppAfsThreCMotion->Text = L"C動き";
            // 
            // fcgTBVppAfsThreYMotion
            // 
            this->fcgTBVppAfsThreYMotion->AutoSize = false;
            this->fcgTBVppAfsThreYMotion->Location = System::Drawing::Point(60, 154);
            this->fcgTBVppAfsThreYMotion->Maximum = 1024;
            this->fcgTBVppAfsThreYMotion->Name = L"fcgTBVppAfsThreYMotion";
            this->fcgTBVppAfsThreYMotion->Size = System::Drawing::Size(115, 18);
            this->fcgTBVppAfsThreYMotion->TabIndex = 21;
            this->fcgTBVppAfsThreYMotion->TickStyle = System::Windows::Forms::TickStyle::None;
            this->fcgTBVppAfsThreYMotion->Scroll += gcnew System::EventHandler(this, &frmConfig::fcgTBVppAfsScroll);
            // 
            // fcgLBVppAfsThreYmotion
            // 
            this->fcgLBVppAfsThreYmotion->AutoSize = true;
            this->fcgLBVppAfsThreYmotion->Location = System::Drawing::Point(9, 154);
            this->fcgLBVppAfsThreYmotion->Name = L"fcgLBVppAfsThreYmotion";
            this->fcgLBVppAfsThreYmotion->Size = System::Drawing::Size(33, 14);
            this->fcgLBVppAfsThreYmotion->TabIndex = 20;
            this->fcgLBVppAfsThreYmotion->Text = L"Y動き";
            // 
            // fcgTBVppAfsThreDeint
            // 
            this->fcgTBVppAfsThreDeint->AutoSize = false;
            this->fcgTBVppAfsThreDeint->Location = System::Drawing::Point(60, 129);
            this->fcgTBVppAfsThreDeint->Maximum = 1024;
            this->fcgTBVppAfsThreDeint->Name = L"fcgTBVppAfsThreDeint";
            this->fcgTBVppAfsThreDeint->Size = System::Drawing::Size(115, 18);
            this->fcgTBVppAfsThreDeint->TabIndex = 18;
            this->fcgTBVppAfsThreDeint->TickStyle = System::Windows::Forms::TickStyle::None;
            this->fcgTBVppAfsThreDeint->Scroll += gcnew System::EventHandler(this, &frmConfig::fcgTBVppAfsScroll);
            // 
            // fcgLBVppAfsThreDeint
            // 
            this->fcgLBVppAfsThreDeint->AutoSize = true;
            this->fcgLBVppAfsThreDeint->Location = System::Drawing::Point(9, 129);
            this->fcgLBVppAfsThreDeint->Name = L"fcgLBVppAfsThreDeint";
            this->fcgLBVppAfsThreDeint->Size = System::Drawing::Size(50, 14);
            this->fcgLBVppAfsThreDeint->TabIndex = 17;
            this->fcgLBVppAfsThreDeint->Text = L"縞(解除)";
            // 
            // fcgTBVppAfsThreShift
            // 
            this->fcgTBVppAfsThreShift->AutoSize = false;
            this->fcgTBVppAfsThreShift->Location = System::Drawing::Point(60, 104);
            this->fcgTBVppAfsThreShift->Maximum = 1024;
            this->fcgTBVppAfsThreShift->Name = L"fcgTBVppAfsThreShift";
            this->fcgTBVppAfsThreShift->Size = System::Drawing::Size(115, 18);
            this->fcgTBVppAfsThreShift->TabIndex = 15;
            this->fcgTBVppAfsThreShift->TickStyle = System::Windows::Forms::TickStyle::None;
            this->fcgTBVppAfsThreShift->Scroll += gcnew System::EventHandler(this, &frmConfig::fcgTBVppAfsScroll);
            // 
            // fcgLBVppAfsThreShift
            // 
            this->fcgLBVppAfsThreShift->AutoSize = true;
            this->fcgLBVppAfsThreShift->Location = System::Drawing::Point(9, 104);
            this->fcgLBVppAfsThreShift->Name = L"fcgLBVppAfsThreShift";
            this->fcgLBVppAfsThreShift->Size = System::Drawing::Size(46, 14);
            this->fcgLBVppAfsThreShift->TabIndex = 14;
            this->fcgLBVppAfsThreShift->Text = L"縞(ｼﾌﾄ)";
            // 
            // fcgTBVppAfsCoeffShift
            // 
            this->fcgTBVppAfsCoeffShift->AutoSize = false;
            this->fcgTBVppAfsCoeffShift->Location = System::Drawing::Point(60, 79);
            this->fcgTBVppAfsCoeffShift->Maximum = 256;
            this->fcgTBVppAfsCoeffShift->Name = L"fcgTBVppAfsCoeffShift";
            this->fcgTBVppAfsCoeffShift->Size = System::Drawing::Size(115, 18);
            this->fcgTBVppAfsCoeffShift->TabIndex = 12;
            this->fcgTBVppAfsCoeffShift->TickStyle = System::Windows::Forms::TickStyle::None;
            this->fcgTBVppAfsCoeffShift->Scroll += gcnew System::EventHandler(this, &frmConfig::fcgTBVppAfsScroll);
            // 
            // fcgLBVppAfsCoeffShift
            // 
            this->fcgLBVppAfsCoeffShift->AutoSize = true;
            this->fcgLBVppAfsCoeffShift->Location = System::Drawing::Point(9, 79);
            this->fcgLBVppAfsCoeffShift->Name = L"fcgLBVppAfsCoeffShift";
            this->fcgLBVppAfsCoeffShift->Size = System::Drawing::Size(40, 14);
            this->fcgLBVppAfsCoeffShift->TabIndex = 11;
            this->fcgLBVppAfsCoeffShift->Text = L"判定比";
            // 
            // label20
            // 
            this->label20->AutoSize = true;
            this->label20->Location = System::Drawing::Point(163, 29);
            this->label20->Name = L"label20";
            this->label20->Size = System::Drawing::Size(18, 14);
            this->label20->TabIndex = 6;
            this->label20->Text = L"右";
            // 
            // label19
            // 
            this->label19->AutoSize = true;
            this->label19->Location = System::Drawing::Point(57, 29);
            this->label19->Name = L"label19";
            this->label19->Size = System::Drawing::Size(18, 14);
            this->label19->TabIndex = 4;
            this->label19->Text = L"左";
            // 
            // label4
            // 
            this->label4->AutoSize = true;
            this->label4->Location = System::Drawing::Point(163, 4);
            this->label4->Name = L"label4";
            this->label4->Size = System::Drawing::Size(18, 14);
            this->label4->TabIndex = 2;
            this->label4->Text = L"下";
            // 
            // fcgLBVppAfsUp
            // 
            this->fcgLBVppAfsUp->AutoSize = true;
            this->fcgLBVppAfsUp->Location = System::Drawing::Point(57, 4);
            this->fcgLBVppAfsUp->Name = L"fcgLBVppAfsUp";
            this->fcgLBVppAfsUp->Size = System::Drawing::Size(18, 14);
            this->fcgLBVppAfsUp->TabIndex = 0;
            this->fcgLBVppAfsUp->Text = L"上";
            // 
            // fcgNUVppAfsRight
            // 
            this->fcgNUVppAfsRight->Increment = System::Decimal(gcnew cli::array< System::Int32 >(4) { 8, 0, 0, 0 });
            this->fcgNUVppAfsRight->Location = System::Drawing::Point(181, 27);
            this->fcgNUVppAfsRight->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 8192, 0, 0, 0 });
            this->fcgNUVppAfsRight->Name = L"fcgNUVppAfsRight";
            this->fcgNUVppAfsRight->Size = System::Drawing::Size(60, 21);
            this->fcgNUVppAfsRight->TabIndex = 7;
            this->fcgNUVppAfsRight->Tag = L"reCmd";
            this->fcgNUVppAfsRight->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
            // 
            // fcgNUVppAfsLeft
            // 
            this->fcgNUVppAfsLeft->Increment = System::Decimal(gcnew cli::array< System::Int32 >(4) { 8, 0, 0, 0 });
            this->fcgNUVppAfsLeft->Location = System::Drawing::Point(77, 27);
            this->fcgNUVppAfsLeft->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 8192, 0, 0, 0 });
            this->fcgNUVppAfsLeft->Name = L"fcgNUVppAfsLeft";
            this->fcgNUVppAfsLeft->Size = System::Drawing::Size(60, 21);
            this->fcgNUVppAfsLeft->TabIndex = 5;
            this->fcgNUVppAfsLeft->Tag = L"reCmd";
            this->fcgNUVppAfsLeft->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
            // 
            // fcgNUVppAfsBottom
            // 
            this->fcgNUVppAfsBottom->Increment = System::Decimal(gcnew cli::array< System::Int32 >(4) { 8, 0, 0, 0 });
            this->fcgNUVppAfsBottom->Location = System::Drawing::Point(181, 2);
            this->fcgNUVppAfsBottom->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 4096, 0, 0, 0 });
            this->fcgNUVppAfsBottom->Name = L"fcgNUVppAfsBottom";
            this->fcgNUVppAfsBottom->Size = System::Drawing::Size(60, 21);
            this->fcgNUVppAfsBottom->TabIndex = 3;
            this->fcgNUVppAfsBottom->Tag = L"reCmd";
            this->fcgNUVppAfsBottom->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
            // 
            // fcgNUVppAfsUp
            // 
            this->fcgNUVppAfsUp->Increment = System::Decimal(gcnew cli::array< System::Int32 >(4) { 8, 0, 0, 0 });
            this->fcgNUVppAfsUp->Location = System::Drawing::Point(77, 2);
            this->fcgNUVppAfsUp->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 4096, 0, 0, 0 });
            this->fcgNUVppAfsUp->Name = L"fcgNUVppAfsUp";
            this->fcgNUVppAfsUp->Size = System::Drawing::Size(60, 21);
            this->fcgNUVppAfsUp->TabIndex = 1;
            this->fcgNUVppAfsUp->Tag = L"reCmd";
            this->fcgNUVppAfsUp->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
            // 
            // fcgTBVppAfsMethodSwitch
            // 
            this->fcgTBVppAfsMethodSwitch->AutoSize = false;
            this->fcgTBVppAfsMethodSwitch->Location = System::Drawing::Point(60, 54);
            this->fcgTBVppAfsMethodSwitch->Maximum = 256;
            this->fcgTBVppAfsMethodSwitch->Name = L"fcgTBVppAfsMethodSwitch";
            this->fcgTBVppAfsMethodSwitch->Size = System::Drawing::Size(115, 18);
            this->fcgTBVppAfsMethodSwitch->TabIndex = 9;
            this->fcgTBVppAfsMethodSwitch->TickStyle = System::Windows::Forms::TickStyle::None;
            this->fcgTBVppAfsMethodSwitch->Scroll += gcnew System::EventHandler(this, &frmConfig::fcgTBVppAfsScroll);
            // 
            // fcgCBVppAfs24fps
            // 
            this->fcgCBVppAfs24fps->AutoSize = true;
            this->fcgCBVppAfs24fps->Location = System::Drawing::Point(149, 231);
            this->fcgCBVppAfs24fps->Name = L"fcgCBVppAfs24fps";
            this->fcgCBVppAfs24fps->Size = System::Drawing::Size(67, 18);
            this->fcgCBVppAfs24fps->TabIndex = 29;
            this->fcgCBVppAfs24fps->Tag = L"reCmd";
            this->fcgCBVppAfs24fps->Text = L"24fps化";
            this->fcgCBVppAfs24fps->UseVisualStyleBackColor = true;
            // 
            // fcgCBVppAfsTune
            // 
            this->fcgCBVppAfsTune->AutoSize = true;
            this->fcgCBVppAfsTune->Location = System::Drawing::Point(149, 273);
            this->fcgCBVppAfsTune->Name = L"fcgCBVppAfsTune";
            this->fcgCBVppAfsTune->Size = System::Drawing::Size(73, 18);
            this->fcgCBVppAfsTune->TabIndex = 32;
            this->fcgCBVppAfsTune->Tag = L"reCmd";
            this->fcgCBVppAfsTune->Text = L"調整モード";
            this->fcgCBVppAfsTune->UseVisualStyleBackColor = true;
            // 
            // fcgCBVppAfsSmooth
            // 
            this->fcgCBVppAfsSmooth->AutoSize = true;
            this->fcgCBVppAfsSmooth->Location = System::Drawing::Point(11, 273);
            this->fcgCBVppAfsSmooth->Name = L"fcgCBVppAfsSmooth";
            this->fcgCBVppAfsSmooth->Size = System::Drawing::Size(77, 18);
            this->fcgCBVppAfsSmooth->TabIndex = 31;
            this->fcgCBVppAfsSmooth->Tag = L"reCmd";
            this->fcgCBVppAfsSmooth->Text = L"スムージング";
            this->fcgCBVppAfsSmooth->UseVisualStyleBackColor = true;
            // 
            // fcgCBVppAfsDrop
            // 
            this->fcgCBVppAfsDrop->AutoSize = true;
            this->fcgCBVppAfsDrop->Location = System::Drawing::Point(11, 252);
            this->fcgCBVppAfsDrop->Name = L"fcgCBVppAfsDrop";
            this->fcgCBVppAfsDrop->Size = System::Drawing::Size(56, 18);
            this->fcgCBVppAfsDrop->TabIndex = 30;
            this->fcgCBVppAfsDrop->Tag = L"reCmd";
            this->fcgCBVppAfsDrop->Text = L"間引き";
            this->fcgCBVppAfsDrop->UseVisualStyleBackColor = true;
            // 
            // fcgCBVppAfsShift
            // 
            this->fcgCBVppAfsShift->AutoSize = true;
            this->fcgCBVppAfsShift->Location = System::Drawing::Point(11, 231);
            this->fcgCBVppAfsShift->Name = L"fcgCBVppAfsShift";
            this->fcgCBVppAfsShift->Size = System::Drawing::Size(89, 18);
            this->fcgCBVppAfsShift->TabIndex = 28;
            this->fcgCBVppAfsShift->Tag = L"reCmd";
            this->fcgCBVppAfsShift->Text = L"フィールドシフト";
            this->fcgCBVppAfsShift->UseVisualStyleBackColor = true;
            // 
            // fcgLBVppAfsAnalyze
            // 
            this->fcgLBVppAfsAnalyze->AutoSize = true;
            this->fcgLBVppAfsAnalyze->Location = System::Drawing::Point(9, 207);
            this->fcgLBVppAfsAnalyze->Name = L"fcgLBVppAfsAnalyze";
            this->fcgLBVppAfsAnalyze->Size = System::Drawing::Size(41, 14);
            this->fcgLBVppAfsAnalyze->TabIndex = 26;
            this->fcgLBVppAfsAnalyze->Text = L"解除Lv";
            // 
            // fcgNUVppAfsThreCMotion
            // 
            this->fcgNUVppAfsThreCMotion->Location = System::Drawing::Point(181, 179);
            this->fcgNUVppAfsThreCMotion->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 1024, 0, 0, 0 });
            this->fcgNUVppAfsThreCMotion->Name = L"fcgNUVppAfsThreCMotion";
            this->fcgNUVppAfsThreCMotion->Size = System::Drawing::Size(60, 21);
            this->fcgNUVppAfsThreCMotion->TabIndex = 25;
            this->fcgNUVppAfsThreCMotion->Tag = L"reCmd";
            this->fcgNUVppAfsThreCMotion->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
            this->fcgNUVppAfsThreCMotion->ValueChanged += gcnew System::EventHandler(this, &frmConfig::fcgNUVppAfsValueChanged);
            // 
            // fcgNUVppAfsThreShift
            // 
            this->fcgNUVppAfsThreShift->Location = System::Drawing::Point(181, 104);
            this->fcgNUVppAfsThreShift->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 1024, 0, 0, 0 });
            this->fcgNUVppAfsThreShift->Name = L"fcgNUVppAfsThreShift";
            this->fcgNUVppAfsThreShift->Size = System::Drawing::Size(60, 21);
            this->fcgNUVppAfsThreShift->TabIndex = 16;
            this->fcgNUVppAfsThreShift->Tag = L"reCmd";
            this->fcgNUVppAfsThreShift->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
            this->fcgNUVppAfsThreShift->ValueChanged += gcnew System::EventHandler(this, &frmConfig::fcgNUVppAfsValueChanged);
            // 
            // fcgNUVppAfsThreDeint
            // 
            this->fcgNUVppAfsThreDeint->Location = System::Drawing::Point(181, 129);
            this->fcgNUVppAfsThreDeint->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 1024, 0, 0, 0 });
            this->fcgNUVppAfsThreDeint->Name = L"fcgNUVppAfsThreDeint";
            this->fcgNUVppAfsThreDeint->Size = System::Drawing::Size(60, 21);
            this->fcgNUVppAfsThreDeint->TabIndex = 19;
            this->fcgNUVppAfsThreDeint->Tag = L"reCmd";
            this->fcgNUVppAfsThreDeint->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
            this->fcgNUVppAfsThreDeint->ValueChanged += gcnew System::EventHandler(this, &frmConfig::fcgNUVppAfsValueChanged);
            // 
            // fcgNUVppAfsThreYMotion
            // 
            this->fcgNUVppAfsThreYMotion->Location = System::Drawing::Point(181, 154);
            this->fcgNUVppAfsThreYMotion->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 1024, 0, 0, 0 });
            this->fcgNUVppAfsThreYMotion->Name = L"fcgNUVppAfsThreYMotion";
            this->fcgNUVppAfsThreYMotion->Size = System::Drawing::Size(60, 21);
            this->fcgNUVppAfsThreYMotion->TabIndex = 22;
            this->fcgNUVppAfsThreYMotion->Tag = L"reCmd";
            this->fcgNUVppAfsThreYMotion->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
            this->fcgNUVppAfsThreYMotion->ValueChanged += gcnew System::EventHandler(this, &frmConfig::fcgNUVppAfsValueChanged);
            // 
            // fcgLBVppAfsMethodSwitch
            // 
            this->fcgLBVppAfsMethodSwitch->AutoSize = true;
            this->fcgLBVppAfsMethodSwitch->Location = System::Drawing::Point(9, 54);
            this->fcgLBVppAfsMethodSwitch->Name = L"fcgLBVppAfsMethodSwitch";
            this->fcgLBVppAfsMethodSwitch->Size = System::Drawing::Size(40, 14);
            this->fcgLBVppAfsMethodSwitch->TabIndex = 8;
            this->fcgLBVppAfsMethodSwitch->Text = L"切替点";
            // 
            // fcgCXVppAfsAnalyze
            // 
            this->fcgCXVppAfsAnalyze->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
            this->fcgCXVppAfsAnalyze->FormattingEnabled = true;
            this->fcgCXVppAfsAnalyze->Location = System::Drawing::Point(60, 204);
            this->fcgCXVppAfsAnalyze->Name = L"fcgCXVppAfsAnalyze";
            this->fcgCXVppAfsAnalyze->Size = System::Drawing::Size(181, 22);
            this->fcgCXVppAfsAnalyze->TabIndex = 27;
            this->fcgCXVppAfsAnalyze->Tag = L"reCmd";
            // 
            // fcgNUVppAfsCoeffShift
            // 
            this->fcgNUVppAfsCoeffShift->Location = System::Drawing::Point(181, 79);
            this->fcgNUVppAfsCoeffShift->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 256, 0, 0, 0 });
            this->fcgNUVppAfsCoeffShift->Name = L"fcgNUVppAfsCoeffShift";
            this->fcgNUVppAfsCoeffShift->Size = System::Drawing::Size(60, 21);
            this->fcgNUVppAfsCoeffShift->TabIndex = 13;
            this->fcgNUVppAfsCoeffShift->Tag = L"reCmd";
            this->fcgNUVppAfsCoeffShift->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
            this->fcgNUVppAfsCoeffShift->ValueChanged += gcnew System::EventHandler(this, &frmConfig::fcgNUVppAfsValueChanged);
            // 
            // fcgNUVppAfsMethodSwitch
            // 
            this->fcgNUVppAfsMethodSwitch->Location = System::Drawing::Point(181, 54);
            this->fcgNUVppAfsMethodSwitch->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 256, 0, 0, 0 });
            this->fcgNUVppAfsMethodSwitch->Name = L"fcgNUVppAfsMethodSwitch";
            this->fcgNUVppAfsMethodSwitch->Size = System::Drawing::Size(60, 21);
            this->fcgNUVppAfsMethodSwitch->TabIndex = 10;
            this->fcgNUVppAfsMethodSwitch->Tag = L"reCmd";
            this->fcgNUVppAfsMethodSwitch->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
            this->fcgNUVppAfsMethodSwitch->ValueChanged += gcnew System::EventHandler(this, &frmConfig::fcgNUVppAfsValueChanged);
            // 
            // fcgCBVppDebandEnable
            // 
            this->fcgCBVppDebandEnable->AutoSize = true;
            this->fcgCBVppDebandEnable->Location = System::Drawing::Point(12, 310);
            this->fcgCBVppDebandEnable->Name = L"fcgCBVppDebandEnable";
            this->fcgCBVppDebandEnable->Size = System::Drawing::Size(96, 18);
            this->fcgCBVppDebandEnable->TabIndex = 20;
            this->fcgCBVppDebandEnable->Tag = L"reCmd";
            this->fcgCBVppDebandEnable->Text = L"バンディング低減";
            this->fcgCBVppDebandEnable->UseVisualStyleBackColor = true;
            this->fcgCBVppDebandEnable->CheckedChanged += gcnew System::EventHandler(this, &frmConfig::fcgChangeEnabled);
            // 
            // fcggroupBoxVppDeband
            // 
            this->fcggroupBoxVppDeband->Controls->Add(this->fcgCBVppDebandRandEachFrame);
            this->fcggroupBoxVppDeband->Controls->Add(this->fcgCBVppDebandBlurFirst);
            this->fcggroupBoxVppDeband->Controls->Add(this->fcgLBVppDebandSample);
            this->fcggroupBoxVppDeband->Controls->Add(this->fcgLBVppDebandDitherC);
            this->fcggroupBoxVppDeband->Controls->Add(this->fcgLBVppDebandDitherY);
            this->fcggroupBoxVppDeband->Controls->Add(this->fcgLBVppDebandDither);
            this->fcggroupBoxVppDeband->Controls->Add(this->fcgLBVppDebandThreCr);
            this->fcggroupBoxVppDeband->Controls->Add(this->fcgLBVppDebandThreCb);
            this->fcggroupBoxVppDeband->Controls->Add(this->fcgLBVppDebandThreY);
            this->fcggroupBoxVppDeband->Controls->Add(this->fcgNUVppDebandDitherC);
            this->fcggroupBoxVppDeband->Controls->Add(this->fcgNUVppDebandDitherY);
            this->fcggroupBoxVppDeband->Controls->Add(this->fcgNUVppDebandThreCr);
            this->fcggroupBoxVppDeband->Controls->Add(this->fcgNUVppDebandThreCb);
            this->fcggroupBoxVppDeband->Controls->Add(this->fcgLBVppDebandThreshold);
            this->fcggroupBoxVppDeband->Controls->Add(this->fcgLBVppDebandRange);
            this->fcggroupBoxVppDeband->Controls->Add(this->fcgCXVppDebandSample);
            this->fcggroupBoxVppDeband->Controls->Add(this->fcgNUVppDebandThreY);
            this->fcggroupBoxVppDeband->Controls->Add(this->fcgNUVppDebandRange);
            this->fcggroupBoxVppDeband->Location = System::Drawing::Point(3, 309);
            this->fcggroupBoxVppDeband->Name = L"fcggroupBoxVppDeband";
            this->fcggroupBoxVppDeband->Size = System::Drawing::Size(320, 144);
            this->fcggroupBoxVppDeband->TabIndex = 21;
            this->fcggroupBoxVppDeband->TabStop = false;
            // 
            // fcgCBVppDebandRandEachFrame
            // 
            this->fcgCBVppDebandRandEachFrame->AutoSize = true;
            this->fcgCBVppDebandRandEachFrame->Location = System::Drawing::Point(165, 122);
            this->fcgCBVppDebandRandEachFrame->Name = L"fcgCBVppDebandRandEachFrame";
            this->fcgCBVppDebandRandEachFrame->Size = System::Drawing::Size(122, 18);
            this->fcgCBVppDebandRandEachFrame->TabIndex = 17;
            this->fcgCBVppDebandRandEachFrame->Tag = L"reCmd";
            this->fcgCBVppDebandRandEachFrame->Text = L"毎フレーム乱数を生成";
            this->fcgCBVppDebandRandEachFrame->UseVisualStyleBackColor = true;
            // 
            // fcgCBVppDebandBlurFirst
            // 
            this->fcgCBVppDebandBlurFirst->AutoSize = true;
            this->fcgCBVppDebandBlurFirst->Location = System::Drawing::Point(16, 122);
            this->fcgCBVppDebandBlurFirst->Name = L"fcgCBVppDebandBlurFirst";
            this->fcgCBVppDebandBlurFirst->Size = System::Drawing::Size(101, 18);
            this->fcgCBVppDebandBlurFirst->TabIndex = 16;
            this->fcgCBVppDebandBlurFirst->Tag = L"reCmd";
            this->fcgCBVppDebandBlurFirst->Text = L"ブラー処理を先に";
            this->fcgCBVppDebandBlurFirst->UseVisualStyleBackColor = true;
            // 
            // fcgLBVppDebandSample
            // 
            this->fcgLBVppDebandSample->AutoSize = true;
            this->fcgLBVppDebandSample->Location = System::Drawing::Point(13, 97);
            this->fcgLBVppDebandSample->Name = L"fcgLBVppDebandSample";
            this->fcgLBVppDebandSample->Size = System::Drawing::Size(45, 14);
            this->fcgLBVppDebandSample->TabIndex = 14;
            this->fcgLBVppDebandSample->Text = L"sample";
            // 
            // fcgLBVppDebandDitherC
            // 
            this->fcgLBVppDebandDitherC->AutoSize = true;
            this->fcgLBVppDebandDitherC->Location = System::Drawing::Point(147, 72);
            this->fcgLBVppDebandDitherC->Name = L"fcgLBVppDebandDitherC";
            this->fcgLBVppDebandDitherC->Size = System::Drawing::Size(14, 14);
            this->fcgLBVppDebandDitherC->TabIndex = 12;
            this->fcgLBVppDebandDitherC->Text = L"C";
            // 
            // fcgLBVppDebandDitherY
            // 
            this->fcgLBVppDebandDitherY->AutoSize = true;
            this->fcgLBVppDebandDitherY->Location = System::Drawing::Point(57, 72);
            this->fcgLBVppDebandDitherY->Name = L"fcgLBVppDebandDitherY";
            this->fcgLBVppDebandDitherY->Size = System::Drawing::Size(14, 14);
            this->fcgLBVppDebandDitherY->TabIndex = 10;
            this->fcgLBVppDebandDitherY->Text = L"Y";
            // 
            // fcgLBVppDebandDither
            // 
            this->fcgLBVppDebandDither->AutoSize = true;
            this->fcgLBVppDebandDither->Location = System::Drawing::Point(13, 72);
            this->fcgLBVppDebandDither->Name = L"fcgLBVppDebandDither";
            this->fcgLBVppDebandDither->Size = System::Drawing::Size(39, 14);
            this->fcgLBVppDebandDither->TabIndex = 9;
            this->fcgLBVppDebandDither->Text = L"dither";
            // 
            // fcgLBVppDebandThreCr
            // 
            this->fcgLBVppDebandThreCr->AutoSize = true;
            this->fcgLBVppDebandThreCr->Location = System::Drawing::Point(233, 49);
            this->fcgLBVppDebandThreCr->Name = L"fcgLBVppDebandThreCr";
            this->fcgLBVppDebandThreCr->Size = System::Drawing::Size(19, 14);
            this->fcgLBVppDebandThreCr->TabIndex = 7;
            this->fcgLBVppDebandThreCr->Text = L"Cr";
            // 
            // fcgLBVppDebandThreCb
            // 
            this->fcgLBVppDebandThreCb->AutoSize = true;
            this->fcgLBVppDebandThreCb->Location = System::Drawing::Point(142, 49);
            this->fcgLBVppDebandThreCb->Name = L"fcgLBVppDebandThreCb";
            this->fcgLBVppDebandThreCb->Size = System::Drawing::Size(21, 14);
            this->fcgLBVppDebandThreCb->TabIndex = 5;
            this->fcgLBVppDebandThreCb->Text = L"Cb";
            // 
            // fcgLBVppDebandThreY
            // 
            this->fcgLBVppDebandThreY->AutoSize = true;
            this->fcgLBVppDebandThreY->Location = System::Drawing::Point(57, 48);
            this->fcgLBVppDebandThreY->Name = L"fcgLBVppDebandThreY";
            this->fcgLBVppDebandThreY->Size = System::Drawing::Size(14, 14);
            this->fcgLBVppDebandThreY->TabIndex = 3;
            this->fcgLBVppDebandThreY->Text = L"Y";
            // 
            // fcgNUVppDebandDitherC
            // 
            this->fcgNUVppDebandDitherC->Location = System::Drawing::Point(165, 70);
            this->fcgNUVppDebandDitherC->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 31, 0, 0, 0 });
            this->fcgNUVppDebandDitherC->Name = L"fcgNUVppDebandDitherC";
            this->fcgNUVppDebandDitherC->Size = System::Drawing::Size(60, 21);
            this->fcgNUVppDebandDitherC->TabIndex = 13;
            this->fcgNUVppDebandDitherC->Tag = L"reCmd";
            this->fcgNUVppDebandDitherC->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
            // 
            // fcgNUVppDebandDitherY
            // 
            this->fcgNUVppDebandDitherY->Location = System::Drawing::Point(76, 70);
            this->fcgNUVppDebandDitherY->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 31, 0, 0, 0 });
            this->fcgNUVppDebandDitherY->Name = L"fcgNUVppDebandDitherY";
            this->fcgNUVppDebandDitherY->Size = System::Drawing::Size(60, 21);
            this->fcgNUVppDebandDitherY->TabIndex = 11;
            this->fcgNUVppDebandDitherY->Tag = L"reCmd";
            this->fcgNUVppDebandDitherY->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
            // 
            // fcgNUVppDebandThreCr
            // 
            this->fcgNUVppDebandThreCr->Location = System::Drawing::Point(254, 46);
            this->fcgNUVppDebandThreCr->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 31, 0, 0, 0 });
            this->fcgNUVppDebandThreCr->Name = L"fcgNUVppDebandThreCr";
            this->fcgNUVppDebandThreCr->Size = System::Drawing::Size(60, 21);
            this->fcgNUVppDebandThreCr->TabIndex = 8;
            this->fcgNUVppDebandThreCr->Tag = L"reCmd";
            this->fcgNUVppDebandThreCr->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
            // 
            // fcgNUVppDebandThreCb
            // 
            this->fcgNUVppDebandThreCb->Location = System::Drawing::Point(164, 46);
            this->fcgNUVppDebandThreCb->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 31, 0, 0, 0 });
            this->fcgNUVppDebandThreCb->Name = L"fcgNUVppDebandThreCb";
            this->fcgNUVppDebandThreCb->Size = System::Drawing::Size(60, 21);
            this->fcgNUVppDebandThreCb->TabIndex = 6;
            this->fcgNUVppDebandThreCb->Tag = L"reCmd";
            this->fcgNUVppDebandThreCb->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
            // 
            // fcgLBVppDebandThreshold
            // 
            this->fcgLBVppDebandThreshold->AutoSize = true;
            this->fcgLBVppDebandThreshold->Location = System::Drawing::Point(13, 48);
            this->fcgLBVppDebandThreshold->Name = L"fcgLBVppDebandThreshold";
            this->fcgLBVppDebandThreshold->Size = System::Drawing::Size(29, 14);
            this->fcgLBVppDebandThreshold->TabIndex = 2;
            this->fcgLBVppDebandThreshold->Text = L"閾値";
            // 
            // fcgLBVppDebandRange
            // 
            this->fcgLBVppDebandRange->AutoSize = true;
            this->fcgLBVppDebandRange->Location = System::Drawing::Point(13, 23);
            this->fcgLBVppDebandRange->Name = L"fcgLBVppDebandRange";
            this->fcgLBVppDebandRange->Size = System::Drawing::Size(38, 14);
            this->fcgLBVppDebandRange->TabIndex = 0;
            this->fcgLBVppDebandRange->Text = L"range";
            // 
            // fcgCXVppDebandSample
            // 
            this->fcgCXVppDebandSample->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
            this->fcgCXVppDebandSample->FormattingEnabled = true;
            this->fcgCXVppDebandSample->Location = System::Drawing::Point(76, 94);
            this->fcgCXVppDebandSample->Name = L"fcgCXVppDebandSample";
            this->fcgCXVppDebandSample->Size = System::Drawing::Size(148, 22);
            this->fcgCXVppDebandSample->TabIndex = 15;
            this->fcgCXVppDebandSample->Tag = L"reCmd";
            // 
            // fcgNUVppDebandThreY
            // 
            this->fcgNUVppDebandThreY->Location = System::Drawing::Point(76, 46);
            this->fcgNUVppDebandThreY->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 31, 0, 0, 0 });
            this->fcgNUVppDebandThreY->Name = L"fcgNUVppDebandThreY";
            this->fcgNUVppDebandThreY->Size = System::Drawing::Size(60, 21);
            this->fcgNUVppDebandThreY->TabIndex = 4;
            this->fcgNUVppDebandThreY->Tag = L"reCmd";
            this->fcgNUVppDebandThreY->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
            // 
            // fcgNUVppDebandRange
            // 
            this->fcgNUVppDebandRange->Location = System::Drawing::Point(76, 21);
            this->fcgNUVppDebandRange->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 127, 0, 0, 0 });
            this->fcgNUVppDebandRange->Name = L"fcgNUVppDebandRange";
            this->fcgNUVppDebandRange->Size = System::Drawing::Size(60, 21);
            this->fcgNUVppDebandRange->TabIndex = 1;
            this->fcgNUVppDebandRange->Tag = L"reCmd";
            this->fcgNUVppDebandRange->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
            // 
            // fcggroupBoxVppDenoise
            // 
            this->fcggroupBoxVppDenoise->Controls->Add(this->fcgPNVppDenoiseConv3D);
            this->fcggroupBoxVppDenoise->Controls->Add(this->fcgPNVppDenoiseSmooth);
            this->fcggroupBoxVppDenoise->Controls->Add(this->fcgPNVppDenoiseKnn);
            this->fcggroupBoxVppDenoise->Controls->Add(this->fcgCXVppDenoiseMethod);
            this->fcggroupBoxVppDenoise->Controls->Add(this->fcgPNVppDenoisePmd);
            this->fcggroupBoxVppDenoise->Location = System::Drawing::Point(3, 57);
            this->fcggroupBoxVppDenoise->Name = L"fcggroupBoxVppDenoise";
            this->fcggroupBoxVppDenoise->Size = System::Drawing::Size(320, 139);
            this->fcggroupBoxVppDenoise->TabIndex = 10;
            this->fcggroupBoxVppDenoise->TabStop = false;
            this->fcggroupBoxVppDenoise->Text = L"ノイズ除去";
            // 
            // fcgPNVppDenoiseConv3D
            // 
            this->fcgPNVppDenoiseConv3D->Controls->Add(this->fcgCXVppDenoiseConv3DMatrix);
            this->fcgPNVppDenoiseConv3D->Controls->Add(this->fcgLBVppDenoiseConv3DMatrix);
            this->fcgPNVppDenoiseConv3D->Controls->Add(this->fcgLBVppDenoiseConv3DThreshTemporal);
            this->fcgPNVppDenoiseConv3D->Controls->Add(this->fcgLBVppDenoiseConv3DThreshSpatial);
            this->fcgPNVppDenoiseConv3D->Controls->Add(this->fcgLBVppDenoiseConv3DThreshCTemporal);
            this->fcgPNVppDenoiseConv3D->Controls->Add(this->fcgLBVppDenoiseConv3DThreshCSpatial);
            this->fcgPNVppDenoiseConv3D->Controls->Add(this->fcgNUVppDenoiseConv3DThreshCTemporal);
            this->fcgPNVppDenoiseConv3D->Controls->Add(this->fcgNUVppDenoiseConv3DThreshCSpatial);
            this->fcgPNVppDenoiseConv3D->Controls->Add(this->fcgLBVppDenoiseConv3DThreshYTemporal);
            this->fcgPNVppDenoiseConv3D->Controls->Add(this->fcgLBVppDenoiseConv3DThreshYSpatial);
            this->fcgPNVppDenoiseConv3D->Controls->Add(this->fcgNUVppDenoiseConv3DThreshYTemporal);
            this->fcgPNVppDenoiseConv3D->Controls->Add(this->fcgNUVppDenoiseConv3DThreshYSpatial);
            this->fcgPNVppDenoiseConv3D->Location = System::Drawing::Point(3, 42);
            this->fcgPNVppDenoiseConv3D->Name = L"fcgPNVppDenoiseConv3D";
            this->fcgPNVppDenoiseConv3D->Size = System::Drawing::Size(310, 92);
            this->fcgPNVppDenoiseConv3D->TabIndex = 66;
            // 
            // fcgCXVppDenoiseConv3DMatrix
            // 
            this->fcgCXVppDenoiseConv3DMatrix->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
            this->fcgCXVppDenoiseConv3DMatrix->FormattingEnabled = true;
            this->fcgCXVppDenoiseConv3DMatrix->Location = System::Drawing::Point(95, 9);
            this->fcgCXVppDenoiseConv3DMatrix->Name = L"fcgCXVppDenoiseConv3DMatrix";
            this->fcgCXVppDenoiseConv3DMatrix->Size = System::Drawing::Size(119, 22);
            this->fcgCXVppDenoiseConv3DMatrix->TabIndex = 1;
            this->fcgCXVppDenoiseConv3DMatrix->Tag = L"reCmd";
            // 
            // fcgLBVppDenoiseConv3DMatrix
            // 
            this->fcgLBVppDenoiseConv3DMatrix->AutoSize = true;
            this->fcgLBVppDenoiseConv3DMatrix->Location = System::Drawing::Point(8, 12);
            this->fcgLBVppDenoiseConv3DMatrix->Name = L"fcgLBVppDenoiseConv3DMatrix";
            this->fcgLBVppDenoiseConv3DMatrix->Size = System::Drawing::Size(29, 14);
            this->fcgLBVppDenoiseConv3DMatrix->TabIndex = 12;
            this->fcgLBVppDenoiseConv3DMatrix->Text = L"行列";
            // 
            // fcgLBVppDenoiseConv3DThreshTemporal
            // 
            this->fcgLBVppDenoiseConv3DThreshTemporal->AutoSize = true;
            this->fcgLBVppDenoiseConv3DThreshTemporal->Location = System::Drawing::Point(5, 66);
            this->fcgLBVppDenoiseConv3DThreshTemporal->Name = L"fcgLBVppDenoiseConv3DThreshTemporal";
            this->fcgLBVppDenoiseConv3DThreshTemporal->Size = System::Drawing::Size(73, 14);
            this->fcgLBVppDenoiseConv3DThreshTemporal->TabIndex = 11;
            this->fcgLBVppDenoiseConv3DThreshTemporal->Text = L"時間方向閾値";
            // 
            // fcgLBVppDenoiseConv3DThreshSpatial
            // 
            this->fcgLBVppDenoiseConv3DThreshSpatial->AutoSize = true;
            this->fcgLBVppDenoiseConv3DThreshSpatial->Location = System::Drawing::Point(5, 40);
            this->fcgLBVppDenoiseConv3DThreshSpatial->Name = L"fcgLBVppDenoiseConv3DThreshSpatial";
            this->fcgLBVppDenoiseConv3DThreshSpatial->Size = System::Drawing::Size(73, 14);
            this->fcgLBVppDenoiseConv3DThreshSpatial->TabIndex = 10;
            this->fcgLBVppDenoiseConv3DThreshSpatial->Text = L"空間方向閾値";
            // 
            // fcgLBVppDenoiseConv3DThreshCTemporal
            // 
            this->fcgLBVppDenoiseConv3DThreshCTemporal->AutoSize = true;
            this->fcgLBVppDenoiseConv3DThreshCTemporal->Location = System::Drawing::Point(204, 67);
            this->fcgLBVppDenoiseConv3DThreshCTemporal->Name = L"fcgLBVppDenoiseConv3DThreshCTemporal";
            this->fcgLBVppDenoiseConv3DThreshCTemporal->Size = System::Drawing::Size(29, 14);
            this->fcgLBVppDenoiseConv3DThreshCTemporal->TabIndex = 8;
            this->fcgLBVppDenoiseConv3DThreshCTemporal->Text = L"色差";
            // 
            // fcgLBVppDenoiseConv3DThreshCSpatial
            // 
            this->fcgLBVppDenoiseConv3DThreshCSpatial->AutoSize = true;
            this->fcgLBVppDenoiseConv3DThreshCSpatial->Location = System::Drawing::Point(204, 40);
            this->fcgLBVppDenoiseConv3DThreshCSpatial->Name = L"fcgLBVppDenoiseConv3DThreshCSpatial";
            this->fcgLBVppDenoiseConv3DThreshCSpatial->Size = System::Drawing::Size(29, 14);
            this->fcgLBVppDenoiseConv3DThreshCSpatial->TabIndex = 6;
            this->fcgLBVppDenoiseConv3DThreshCSpatial->Text = L"色差";
            // 
            // fcgNUVppDenoiseConv3DThreshCTemporal
            // 
            this->fcgNUVppDenoiseConv3DThreshCTemporal->Location = System::Drawing::Point(239, 65);
            this->fcgNUVppDenoiseConv3DThreshCTemporal->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 255, 0, 0, 0 });
            this->fcgNUVppDenoiseConv3DThreshCTemporal->Minimum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 1, 0, 0, 0 });
            this->fcgNUVppDenoiseConv3DThreshCTemporal->Name = L"fcgNUVppDenoiseConv3DThreshCTemporal";
            this->fcgNUVppDenoiseConv3DThreshCTemporal->Size = System::Drawing::Size(60, 21);
            this->fcgNUVppDenoiseConv3DThreshCTemporal->TabIndex = 5;
            this->fcgNUVppDenoiseConv3DThreshCTemporal->Tag = L"reCmd";
            this->fcgNUVppDenoiseConv3DThreshCTemporal->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
            this->fcgNUVppDenoiseConv3DThreshCTemporal->Value = System::Decimal(gcnew cli::array< System::Int32 >(4) { 12, 0, 0, 0 });
            // 
            // fcgNUVppDenoiseConv3DThreshCSpatial
            // 
            this->fcgNUVppDenoiseConv3DThreshCSpatial->Location = System::Drawing::Point(239, 38);
            this->fcgNUVppDenoiseConv3DThreshCSpatial->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 255, 0, 0, 0 });
            this->fcgNUVppDenoiseConv3DThreshCSpatial->Minimum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 1, 0, 0, 0 });
            this->fcgNUVppDenoiseConv3DThreshCSpatial->Name = L"fcgNUVppDenoiseConv3DThreshCSpatial";
            this->fcgNUVppDenoiseConv3DThreshCSpatial->Size = System::Drawing::Size(60, 21);
            this->fcgNUVppDenoiseConv3DThreshCSpatial->TabIndex = 3;
            this->fcgNUVppDenoiseConv3DThreshCSpatial->Tag = L"reCmd";
            this->fcgNUVppDenoiseConv3DThreshCSpatial->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
            this->fcgNUVppDenoiseConv3DThreshCSpatial->Value = System::Decimal(gcnew cli::array< System::Int32 >(4) { 3, 0, 0, 0 });
            // 
            // fcgLBVppDenoiseConv3DThreshYTemporal
            // 
            this->fcgLBVppDenoiseConv3DThreshYTemporal->AutoSize = true;
            this->fcgLBVppDenoiseConv3DThreshYTemporal->Location = System::Drawing::Point(92, 67);
            this->fcgLBVppDenoiseConv3DThreshYTemporal->Name = L"fcgLBVppDenoiseConv3DThreshYTemporal";
            this->fcgLBVppDenoiseConv3DThreshYTemporal->Size = System::Drawing::Size(29, 14);
            this->fcgLBVppDenoiseConv3DThreshYTemporal->TabIndex = 4;
            this->fcgLBVppDenoiseConv3DThreshYTemporal->Text = L"輝度";
            // 
            // fcgLBVppDenoiseConv3DThreshYSpatial
            // 
            this->fcgLBVppDenoiseConv3DThreshYSpatial->AutoSize = true;
            this->fcgLBVppDenoiseConv3DThreshYSpatial->Location = System::Drawing::Point(92, 40);
            this->fcgLBVppDenoiseConv3DThreshYSpatial->Name = L"fcgLBVppDenoiseConv3DThreshYSpatial";
            this->fcgLBVppDenoiseConv3DThreshYSpatial->Size = System::Drawing::Size(29, 14);
            this->fcgLBVppDenoiseConv3DThreshYSpatial->TabIndex = 2;
            this->fcgLBVppDenoiseConv3DThreshYSpatial->Text = L"輝度";
            // 
            // fcgNUVppDenoiseConv3DThreshYTemporal
            // 
            this->fcgNUVppDenoiseConv3DThreshYTemporal->Location = System::Drawing::Point(128, 65);
            this->fcgNUVppDenoiseConv3DThreshYTemporal->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 255, 0, 0, 0 });
            this->fcgNUVppDenoiseConv3DThreshYTemporal->Minimum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 1, 0, 0, 0 });
            this->fcgNUVppDenoiseConv3DThreshYTemporal->Name = L"fcgNUVppDenoiseConv3DThreshYTemporal";
            this->fcgNUVppDenoiseConv3DThreshYTemporal->Size = System::Drawing::Size(60, 21);
            this->fcgNUVppDenoiseConv3DThreshYTemporal->TabIndex = 4;
            this->fcgNUVppDenoiseConv3DThreshYTemporal->Tag = L"reCmd";
            this->fcgNUVppDenoiseConv3DThreshYTemporal->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
            this->fcgNUVppDenoiseConv3DThreshYTemporal->Value = System::Decimal(gcnew cli::array< System::Int32 >(4) { 12, 0, 0, 0 });
            // 
            // fcgNUVppDenoiseConv3DThreshYSpatial
            // 
            this->fcgNUVppDenoiseConv3DThreshYSpatial->Location = System::Drawing::Point(128, 38);
            this->fcgNUVppDenoiseConv3DThreshYSpatial->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 255, 0, 0, 0 });
            this->fcgNUVppDenoiseConv3DThreshYSpatial->Minimum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 1, 0, 0, 0 });
            this->fcgNUVppDenoiseConv3DThreshYSpatial->Name = L"fcgNUVppDenoiseConv3DThreshYSpatial";
            this->fcgNUVppDenoiseConv3DThreshYSpatial->Size = System::Drawing::Size(60, 21);
            this->fcgNUVppDenoiseConv3DThreshYSpatial->TabIndex = 2;
            this->fcgNUVppDenoiseConv3DThreshYSpatial->Tag = L"reCmd";
            this->fcgNUVppDenoiseConv3DThreshYSpatial->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
            this->fcgNUVppDenoiseConv3DThreshYSpatial->Value = System::Decimal(gcnew cli::array< System::Int32 >(4) { 3, 0, 0, 0 });
            // 
            // fcgPNVppDenoiseSmooth
            // 
            this->fcgPNVppDenoiseSmooth->Controls->Add(this->fcgLBVppDenoiseSmoothQP);
            this->fcgPNVppDenoiseSmooth->Controls->Add(this->fcgLBVppDenoiseSmoothQuality);
            this->fcgPNVppDenoiseSmooth->Controls->Add(this->fcgNUVppDenoiseSmoothQP);
            this->fcgPNVppDenoiseSmooth->Controls->Add(this->fcgNUVppDenoiseSmoothQuality);
            this->fcgPNVppDenoiseSmooth->Location = System::Drawing::Point(3, 42);
            this->fcgPNVppDenoiseSmooth->Name = L"fcgPNVppDenoiseSmooth";
            this->fcgPNVppDenoiseSmooth->Size = System::Drawing::Size(310, 92);
            this->fcgPNVppDenoiseSmooth->TabIndex = 65;
            // 
            // fcgLBVppDenoiseSmoothQP
            // 
            this->fcgLBVppDenoiseSmoothQP->AutoSize = true;
            this->fcgLBVppDenoiseSmoothQP->Location = System::Drawing::Point(69, 39);
            this->fcgLBVppDenoiseSmoothQP->Name = L"fcgLBVppDenoiseSmoothQP";
            this->fcgLBVppDenoiseSmoothQP->Size = System::Drawing::Size(21, 14);
            this->fcgLBVppDenoiseSmoothQP->TabIndex = 4;
            this->fcgLBVppDenoiseSmoothQP->Text = L"qp";
            // 
            // fcgLBVppDenoiseSmoothQuality
            // 
            this->fcgLBVppDenoiseSmoothQuality->AutoSize = true;
            this->fcgLBVppDenoiseSmoothQuality->Location = System::Drawing::Point(69, 12);
            this->fcgLBVppDenoiseSmoothQuality->Name = L"fcgLBVppDenoiseSmoothQuality";
            this->fcgLBVppDenoiseSmoothQuality->Size = System::Drawing::Size(29, 14);
            this->fcgLBVppDenoiseSmoothQuality->TabIndex = 2;
            this->fcgLBVppDenoiseSmoothQuality->Text = L"品質";
            // 
            // fcgNUVppDenoiseSmoothQP
            // 
            this->fcgNUVppDenoiseSmoothQP->Location = System::Drawing::Point(129, 37);
            this->fcgNUVppDenoiseSmoothQP->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 63, 0, 0, 0 });
            this->fcgNUVppDenoiseSmoothQP->Minimum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 1, 0, 0, 0 });
            this->fcgNUVppDenoiseSmoothQP->Name = L"fcgNUVppDenoiseSmoothQP";
            this->fcgNUVppDenoiseSmoothQP->Size = System::Drawing::Size(60, 21);
            this->fcgNUVppDenoiseSmoothQP->TabIndex = 5;
            this->fcgNUVppDenoiseSmoothQP->Tag = L"reCmd";
            this->fcgNUVppDenoiseSmoothQP->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
            this->fcgNUVppDenoiseSmoothQP->Value = System::Decimal(gcnew cli::array< System::Int32 >(4) { 12, 0, 0, 0 });
            // 
            // fcgNUVppDenoiseSmoothQuality
            // 
            this->fcgNUVppDenoiseSmoothQuality->Location = System::Drawing::Point(129, 10);
            this->fcgNUVppDenoiseSmoothQuality->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 6, 0, 0, 0 });
            this->fcgNUVppDenoiseSmoothQuality->Minimum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 1, 0, 0, 0 });
            this->fcgNUVppDenoiseSmoothQuality->Name = L"fcgNUVppDenoiseSmoothQuality";
            this->fcgNUVppDenoiseSmoothQuality->Size = System::Drawing::Size(60, 21);
            this->fcgNUVppDenoiseSmoothQuality->TabIndex = 3;
            this->fcgNUVppDenoiseSmoothQuality->Tag = L"reCmd";
            this->fcgNUVppDenoiseSmoothQuality->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
            this->fcgNUVppDenoiseSmoothQuality->Value = System::Decimal(gcnew cli::array< System::Int32 >(4) { 3, 0, 0, 0 });
            // 
            // fcgPNVppDenoiseKnn
            // 
            this->fcgPNVppDenoiseKnn->Controls->Add(this->fcgLBVppDenoiseKnnThreshold);
            this->fcgPNVppDenoiseKnn->Controls->Add(this->fcgLBVppDenoiseKnnStrength);
            this->fcgPNVppDenoiseKnn->Controls->Add(this->fcgLBVppDenoiseKnnRadius);
            this->fcgPNVppDenoiseKnn->Controls->Add(this->fcgNUVppDenoiseKnnThreshold);
            this->fcgPNVppDenoiseKnn->Controls->Add(this->fcgNUVppDenoiseKnnStrength);
            this->fcgPNVppDenoiseKnn->Controls->Add(this->fcgNUVppDenoiseKnnRadius);
            this->fcgPNVppDenoiseKnn->Location = System::Drawing::Point(3, 42);
            this->fcgPNVppDenoiseKnn->Name = L"fcgPNVppDenoiseKnn";
            this->fcgPNVppDenoiseKnn->Size = System::Drawing::Size(310, 92);
            this->fcgPNVppDenoiseKnn->TabIndex = 1;
            // 
            // fcgLBVppDenoiseKnnThreshold
            // 
            this->fcgLBVppDenoiseKnnThreshold->AutoSize = true;
            this->fcgLBVppDenoiseKnnThreshold->Location = System::Drawing::Point(69, 66);
            this->fcgLBVppDenoiseKnnThreshold->Name = L"fcgLBVppDenoiseKnnThreshold";
            this->fcgLBVppDenoiseKnnThreshold->Size = System::Drawing::Size(29, 14);
            this->fcgLBVppDenoiseKnnThreshold->TabIndex = 6;
            this->fcgLBVppDenoiseKnnThreshold->Text = L"閾値";
            // 
            // fcgLBVppDenoiseKnnStrength
            // 
            this->fcgLBVppDenoiseKnnStrength->AutoSize = true;
            this->fcgLBVppDenoiseKnnStrength->Location = System::Drawing::Point(69, 39);
            this->fcgLBVppDenoiseKnnStrength->Name = L"fcgLBVppDenoiseKnnStrength";
            this->fcgLBVppDenoiseKnnStrength->Size = System::Drawing::Size(26, 14);
            this->fcgLBVppDenoiseKnnStrength->TabIndex = 4;
            this->fcgLBVppDenoiseKnnStrength->Text = L"強さ";
            // 
            // fcgLBVppDenoiseKnnRadius
            // 
            this->fcgLBVppDenoiseKnnRadius->AutoSize = true;
            this->fcgLBVppDenoiseKnnRadius->Location = System::Drawing::Point(69, 12);
            this->fcgLBVppDenoiseKnnRadius->Name = L"fcgLBVppDenoiseKnnRadius";
            this->fcgLBVppDenoiseKnnRadius->Size = System::Drawing::Size(29, 14);
            this->fcgLBVppDenoiseKnnRadius->TabIndex = 2;
            this->fcgLBVppDenoiseKnnRadius->Text = L"半径";
            // 
            // fcgNUVppDenoiseKnnThreshold
            // 
            this->fcgNUVppDenoiseKnnThreshold->DecimalPlaces = 2;
            this->fcgNUVppDenoiseKnnThreshold->Increment = System::Decimal(gcnew cli::array< System::Int32 >(4) { 5, 0, 0, 131072 });
            this->fcgNUVppDenoiseKnnThreshold->Location = System::Drawing::Point(129, 64);
            this->fcgNUVppDenoiseKnnThreshold->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 1, 0, 0, 0 });
            this->fcgNUVppDenoiseKnnThreshold->Name = L"fcgNUVppDenoiseKnnThreshold";
            this->fcgNUVppDenoiseKnnThreshold->Size = System::Drawing::Size(60, 21);
            this->fcgNUVppDenoiseKnnThreshold->TabIndex = 7;
            this->fcgNUVppDenoiseKnnThreshold->Tag = L"reCmd";
            this->fcgNUVppDenoiseKnnThreshold->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
            this->fcgNUVppDenoiseKnnThreshold->Value = System::Decimal(gcnew cli::array< System::Int32 >(4) { 1, 0, 0, 0 });
            // 
            // fcgNUVppDenoiseKnnStrength
            // 
            this->fcgNUVppDenoiseKnnStrength->DecimalPlaces = 2;
            this->fcgNUVppDenoiseKnnStrength->Increment = System::Decimal(gcnew cli::array< System::Int32 >(4) { 5, 0, 0, 131072 });
            this->fcgNUVppDenoiseKnnStrength->Location = System::Drawing::Point(129, 37);
            this->fcgNUVppDenoiseKnnStrength->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 1, 0, 0, 0 });
            this->fcgNUVppDenoiseKnnStrength->Name = L"fcgNUVppDenoiseKnnStrength";
            this->fcgNUVppDenoiseKnnStrength->Size = System::Drawing::Size(60, 21);
            this->fcgNUVppDenoiseKnnStrength->TabIndex = 5;
            this->fcgNUVppDenoiseKnnStrength->Tag = L"reCmd";
            this->fcgNUVppDenoiseKnnStrength->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
            this->fcgNUVppDenoiseKnnStrength->Value = System::Decimal(gcnew cli::array< System::Int32 >(4) { 1, 0, 0, 0 });
            // 
            // fcgNUVppDenoiseKnnRadius
            // 
            this->fcgNUVppDenoiseKnnRadius->Location = System::Drawing::Point(129, 10);
            this->fcgNUVppDenoiseKnnRadius->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 5, 0, 0, 0 });
            this->fcgNUVppDenoiseKnnRadius->Name = L"fcgNUVppDenoiseKnnRadius";
            this->fcgNUVppDenoiseKnnRadius->Size = System::Drawing::Size(60, 21);
            this->fcgNUVppDenoiseKnnRadius->TabIndex = 3;
            this->fcgNUVppDenoiseKnnRadius->Tag = L"reCmd";
            this->fcgNUVppDenoiseKnnRadius->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
            this->fcgNUVppDenoiseKnnRadius->Value = System::Decimal(gcnew cli::array< System::Int32 >(4) { 3, 0, 0, 0 });
            // 
            // fcgCXVppDenoiseMethod
            // 
            this->fcgCXVppDenoiseMethod->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
            this->fcgCXVppDenoiseMethod->FormattingEnabled = true;
            this->fcgCXVppDenoiseMethod->Location = System::Drawing::Point(26, 18);
            this->fcgCXVppDenoiseMethod->Name = L"fcgCXVppDenoiseMethod";
            this->fcgCXVppDenoiseMethod->Size = System::Drawing::Size(191, 22);
            this->fcgCXVppDenoiseMethod->TabIndex = 0;
            this->fcgCXVppDenoiseMethod->Tag = L"reCmd";
            this->fcgCXVppDenoiseMethod->SelectedIndexChanged += gcnew System::EventHandler(this, &frmConfig::fcgChangeEnabled);
            // 
            // fcgPNVppDenoisePmd
            // 
            this->fcgPNVppDenoisePmd->Controls->Add(this->fcgLBVppDenoisePmdThreshold);
            this->fcgPNVppDenoisePmd->Controls->Add(this->fcgLBVppDenoisePmdStrength);
            this->fcgPNVppDenoisePmd->Controls->Add(this->fcgLBVppDenoisePmdApplyCount);
            this->fcgPNVppDenoisePmd->Controls->Add(this->fcgNUVppDenoisePmdThreshold);
            this->fcgPNVppDenoisePmd->Controls->Add(this->fcgNUVppDenoisePmdStrength);
            this->fcgPNVppDenoisePmd->Controls->Add(this->fcgNUVppDenoisePmdApplyCount);
            this->fcgPNVppDenoisePmd->Location = System::Drawing::Point(3, 42);
            this->fcgPNVppDenoisePmd->Name = L"fcgPNVppDenoisePmd";
            this->fcgPNVppDenoisePmd->Size = System::Drawing::Size(310, 92);
            this->fcgPNVppDenoisePmd->TabIndex = 64;
            // 
            // fcgLBVppDenoisePmdThreshold
            // 
            this->fcgLBVppDenoisePmdThreshold->AutoSize = true;
            this->fcgLBVppDenoisePmdThreshold->Location = System::Drawing::Point(69, 66);
            this->fcgLBVppDenoisePmdThreshold->Name = L"fcgLBVppDenoisePmdThreshold";
            this->fcgLBVppDenoisePmdThreshold->Size = System::Drawing::Size(29, 14);
            this->fcgLBVppDenoisePmdThreshold->TabIndex = 11;
            this->fcgLBVppDenoisePmdThreshold->Text = L"閾値";
            // 
            // fcgLBVppDenoisePmdStrength
            // 
            this->fcgLBVppDenoisePmdStrength->AutoSize = true;
            this->fcgLBVppDenoisePmdStrength->Location = System::Drawing::Point(69, 39);
            this->fcgLBVppDenoisePmdStrength->Name = L"fcgLBVppDenoisePmdStrength";
            this->fcgLBVppDenoisePmdStrength->Size = System::Drawing::Size(26, 14);
            this->fcgLBVppDenoisePmdStrength->TabIndex = 10;
            this->fcgLBVppDenoisePmdStrength->Text = L"強さ";
            // 
            // fcgLBVppDenoisePmdApplyCount
            // 
            this->fcgLBVppDenoisePmdApplyCount->AutoSize = true;
            this->fcgLBVppDenoisePmdApplyCount->Location = System::Drawing::Point(69, 12);
            this->fcgLBVppDenoisePmdApplyCount->Name = L"fcgLBVppDenoisePmdApplyCount";
            this->fcgLBVppDenoisePmdApplyCount->Size = System::Drawing::Size(29, 14);
            this->fcgLBVppDenoisePmdApplyCount->TabIndex = 9;
            this->fcgLBVppDenoisePmdApplyCount->Text = L"回数";
            // 
            // fcgNUVppDenoisePmdThreshold
            // 
            this->fcgNUVppDenoisePmdThreshold->Location = System::Drawing::Point(129, 64);
            this->fcgNUVppDenoisePmdThreshold->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 255, 0, 0, 0 });
            this->fcgNUVppDenoisePmdThreshold->Name = L"fcgNUVppDenoisePmdThreshold";
            this->fcgNUVppDenoisePmdThreshold->Size = System::Drawing::Size(60, 21);
            this->fcgNUVppDenoisePmdThreshold->TabIndex = 8;
            this->fcgNUVppDenoisePmdThreshold->Tag = L"reCmd";
            this->fcgNUVppDenoisePmdThreshold->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
            this->fcgNUVppDenoisePmdThreshold->Value = System::Decimal(gcnew cli::array< System::Int32 >(4) { 100, 0, 0, 0 });
            // 
            // fcgNUVppDenoisePmdStrength
            // 
            this->fcgNUVppDenoisePmdStrength->Location = System::Drawing::Point(129, 37);
            this->fcgNUVppDenoisePmdStrength->Name = L"fcgNUVppDenoisePmdStrength";
            this->fcgNUVppDenoisePmdStrength->Size = System::Drawing::Size(60, 21);
            this->fcgNUVppDenoisePmdStrength->TabIndex = 7;
            this->fcgNUVppDenoisePmdStrength->Tag = L"reCmd";
            this->fcgNUVppDenoisePmdStrength->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
            this->fcgNUVppDenoisePmdStrength->Value = System::Decimal(gcnew cli::array< System::Int32 >(4) { 100, 0, 0, 0 });
            // 
            // fcgNUVppDenoisePmdApplyCount
            // 
            this->fcgNUVppDenoisePmdApplyCount->Location = System::Drawing::Point(129, 10);
            this->fcgNUVppDenoisePmdApplyCount->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 5, 0, 0, 0 });
            this->fcgNUVppDenoisePmdApplyCount->Name = L"fcgNUVppDenoisePmdApplyCount";
            this->fcgNUVppDenoisePmdApplyCount->Size = System::Drawing::Size(60, 21);
            this->fcgNUVppDenoisePmdApplyCount->TabIndex = 6;
            this->fcgNUVppDenoisePmdApplyCount->Tag = L"reCmd";
            this->fcgNUVppDenoisePmdApplyCount->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
            this->fcgNUVppDenoisePmdApplyCount->Value = System::Decimal(gcnew cli::array< System::Int32 >(4) { 2, 0, 0, 0 });
            // 
            // fcgCBVppResize
            // 
            this->fcgCBVppResize->AutoSize = true;
            this->fcgCBVppResize->Location = System::Drawing::Point(11, 4);
            this->fcgCBVppResize->Name = L"fcgCBVppResize";
            this->fcgCBVppResize->Size = System::Drawing::Size(58, 18);
            this->fcgCBVppResize->TabIndex = 0;
            this->fcgCBVppResize->Tag = L"reCmd";
            this->fcgCBVppResize->Text = L"リサイズ";
            this->fcgCBVppResize->UseVisualStyleBackColor = true;
            this->fcgCBVppResize->CheckedChanged += gcnew System::EventHandler(this, &frmConfig::fcgChangeEnabled);
            // 
            // fcggroupBoxResize
            // 
            this->fcggroupBoxResize->Controls->Add(this->fcgCXVppResizeAlg);
            this->fcggroupBoxResize->Controls->Add(this->fcgLBVppResize);
            this->fcggroupBoxResize->Controls->Add(this->fcgNUVppResizeHeight);
            this->fcggroupBoxResize->Controls->Add(this->fcgNUVppResizeWidth);
            this->fcggroupBoxResize->Location = System::Drawing::Point(3, 3);
            this->fcggroupBoxResize->Name = L"fcggroupBoxResize";
            this->fcggroupBoxResize->Size = System::Drawing::Size(320, 51);
            this->fcggroupBoxResize->TabIndex = 1;
            this->fcggroupBoxResize->TabStop = false;
            // 
            // fcgCXVppResizeAlg
            // 
            this->fcgCXVppResizeAlg->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
            this->fcgCXVppResizeAlg->FormattingEnabled = true;
            this->fcgCXVppResizeAlg->Location = System::Drawing::Point(170, 22);
            this->fcgCXVppResizeAlg->Name = L"fcgCXVppResizeAlg";
            this->fcgCXVppResizeAlg->Size = System::Drawing::Size(144, 22);
            this->fcgCXVppResizeAlg->TabIndex = 3;
            this->fcgCXVppResizeAlg->Tag = L"reCmd";
            // 
            // fcgLBVppResize
            // 
            this->fcgLBVppResize->AutoSize = true;
            this->fcgLBVppResize->Location = System::Drawing::Point(77, 26);
            this->fcgLBVppResize->Name = L"fcgLBVppResize";
            this->fcgLBVppResize->Size = System::Drawing::Size(13, 14);
            this->fcgLBVppResize->TabIndex = 1;
            this->fcgLBVppResize->Text = L"x";
            // 
            // fcgNUVppResizeHeight
            // 
            this->fcgNUVppResizeHeight->Location = System::Drawing::Point(95, 23);
            this->fcgNUVppResizeHeight->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 65535, 0, 0, 0 });
            this->fcgNUVppResizeHeight->Minimum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 65536, 0, 0, System::Int32::MinValue });
            this->fcgNUVppResizeHeight->Name = L"fcgNUVppResizeHeight";
            this->fcgNUVppResizeHeight->Size = System::Drawing::Size(60, 21);
            this->fcgNUVppResizeHeight->TabIndex = 2;
            this->fcgNUVppResizeHeight->Tag = L"reCmd";
            this->fcgNUVppResizeHeight->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
            // 
            // fcgNUVppResizeWidth
            // 
            this->fcgNUVppResizeWidth->Location = System::Drawing::Point(11, 23);
            this->fcgNUVppResizeWidth->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 65535, 0, 0, 0 });
            this->fcgNUVppResizeWidth->Minimum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 65536, 0, 0, System::Int32::MinValue });
            this->fcgNUVppResizeWidth->Name = L"fcgNUVppResizeWidth";
            this->fcgNUVppResizeWidth->Size = System::Drawing::Size(60, 21);
            this->fcgNUVppResizeWidth->TabIndex = 0;
            this->fcgNUVppResizeWidth->Tag = L"reCmd";
            this->fcgNUVppResizeWidth->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
            // 
            // tabPageOther
            // 
            this->tabPageOther->Controls->Add(this->fcggroupBoxCmdEx);
            this->tabPageOther->Controls->Add(this->fcgCBLogDebug);
            this->tabPageOther->Controls->Add(this->fcgCBAuoTcfileout);
            this->tabPageOther->Controls->Add(this->fcgLBTempDir);
            this->tabPageOther->Controls->Add(this->fcgBTCustomTempDir);
            this->tabPageOther->Controls->Add(this->fcgTXCustomTempDir);
            this->tabPageOther->Controls->Add(this->fcgCXTempDir);
            this->tabPageOther->Controls->Add(this->fcgCBPerfMonitor);
            this->tabPageOther->Location = System::Drawing::Point(4, 24);
            this->tabPageOther->Name = L"tabPageOther";
            this->tabPageOther->Size = System::Drawing::Size(608, 494);
            this->tabPageOther->TabIndex = 4;
            this->tabPageOther->Text = L"その他";
            this->tabPageOther->UseVisualStyleBackColor = true;
            // 
            // fcggroupBoxCmdEx
            // 
            this->fcggroupBoxCmdEx->Controls->Add(this->fcgTXCmdEx);
            this->fcggroupBoxCmdEx->Location = System::Drawing::Point(10, 254);
            this->fcggroupBoxCmdEx->Name = L"fcggroupBoxCmdEx";
            this->fcggroupBoxCmdEx->Size = System::Drawing::Size(589, 233);
            this->fcggroupBoxCmdEx->TabIndex = 60;
            this->fcggroupBoxCmdEx->TabStop = false;
            this->fcggroupBoxCmdEx->Text = L"追加コマンド";
            // 
            // fcgTXCmdEx
            // 
            this->fcgTXCmdEx->AllowDrop = true;
            this->fcgTXCmdEx->Font = (gcnew System::Drawing::Font(L"ＭＳ ゴシック", 9, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
                static_cast<System::Byte>(128)));
            this->fcgTXCmdEx->Location = System::Drawing::Point(6, 20);
            this->fcgTXCmdEx->Multiline = true;
            this->fcgTXCmdEx->Name = L"fcgTXCmdEx";
            this->fcgTXCmdEx->Size = System::Drawing::Size(577, 202);
            this->fcgTXCmdEx->TabIndex = 0;
            this->fcgTXCmdEx->Tag = L"chValue";
            // 
            // fcgCBLogDebug
            // 
            this->fcgCBLogDebug->AutoSize = true;
            this->fcgCBLogDebug->Location = System::Drawing::Point(197, 104);
            this->fcgCBLogDebug->Name = L"fcgCBLogDebug";
            this->fcgCBLogDebug->Size = System::Drawing::Size(104, 19);
            this->fcgCBLogDebug->TabIndex = 59;
            this->fcgCBLogDebug->Tag = L"reCmd";
            this->fcgCBLogDebug->Text = L"デバッグログ出力";
            this->fcgCBLogDebug->UseVisualStyleBackColor = true;
            // 
            // fcgCBAuoTcfileout
            // 
            this->fcgCBAuoTcfileout->AutoSize = true;
            this->fcgCBAuoTcfileout->Location = System::Drawing::Point(15, 78);
            this->fcgCBAuoTcfileout->Name = L"fcgCBAuoTcfileout";
            this->fcgCBAuoTcfileout->Size = System::Drawing::Size(103, 19);
            this->fcgCBAuoTcfileout->TabIndex = 57;
            this->fcgCBAuoTcfileout->Tag = L"chValue";
            this->fcgCBAuoTcfileout->Text = L"タイムコード出力";
            this->fcgCBAuoTcfileout->UseVisualStyleBackColor = true;
            // 
            // fcgLBTempDir
            // 
            this->fcgLBTempDir->AutoSize = true;
            this->fcgLBTempDir->Location = System::Drawing::Point(10, 15);
            this->fcgLBTempDir->Name = L"fcgLBTempDir";
            this->fcgLBTempDir->Size = System::Drawing::Size(66, 15);
            this->fcgLBTempDir->TabIndex = 53;
            this->fcgLBTempDir->Text = L"一時フォルダ";
            // 
            // fcgBTCustomTempDir
            // 
            this->fcgBTCustomTempDir->Location = System::Drawing::Point(273, 39);
            this->fcgBTCustomTempDir->Name = L"fcgBTCustomTempDir";
            this->fcgBTCustomTempDir->Size = System::Drawing::Size(29, 23);
            this->fcgBTCustomTempDir->TabIndex = 56;
            this->fcgBTCustomTempDir->Text = L"...";
            this->fcgBTCustomTempDir->UseVisualStyleBackColor = true;
            this->fcgBTCustomTempDir->Click += gcnew System::EventHandler(this, &frmConfig::fcgBTCustomTempDir_Click);
            // 
            // fcgTXCustomTempDir
            // 
            this->fcgTXCustomTempDir->Location = System::Drawing::Point(88, 40);
            this->fcgTXCustomTempDir->Name = L"fcgTXCustomTempDir";
            this->fcgTXCustomTempDir->Size = System::Drawing::Size(182, 23);
            this->fcgTXCustomTempDir->TabIndex = 55;
            this->fcgTXCustomTempDir->Tag = L"";
            this->fcgTXCustomTempDir->TextChanged += gcnew System::EventHandler(this, &frmConfig::fcgTXCustomTempDir_TextChanged);
            // 
            // fcgCXTempDir
            // 
            this->fcgCXTempDir->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
            this->fcgCXTempDir->FormattingEnabled = true;
            this->fcgCXTempDir->Location = System::Drawing::Point(76, 12);
            this->fcgCXTempDir->Name = L"fcgCXTempDir";
            this->fcgCXTempDir->Size = System::Drawing::Size(209, 23);
            this->fcgCXTempDir->TabIndex = 54;
            this->fcgCXTempDir->Tag = L"chValue";
            // 
            // fcgCBPerfMonitor
            // 
            this->fcgCBPerfMonitor->AutoSize = true;
            this->fcgCBPerfMonitor->Location = System::Drawing::Point(15, 104);
            this->fcgCBPerfMonitor->Name = L"fcgCBPerfMonitor";
            this->fcgCBPerfMonitor->Size = System::Drawing::Size(131, 19);
            this->fcgCBPerfMonitor->TabIndex = 58;
            this->fcgCBPerfMonitor->Tag = L"reCmd";
            this->fcgCBPerfMonitor->Text = L"パフォーマンスログ出力";
            this->fcgCBPerfMonitor->UseVisualStyleBackColor = true;
            // 
            // fcgCSExeFiles
            // 
            this->fcgCSExeFiles->ImageScalingSize = System::Drawing::Size(18, 18);
            this->fcgCSExeFiles->Items->AddRange(gcnew cli::array< System::Windows::Forms::ToolStripItem^  >(1) { this->fcgTSExeFileshelp });
            this->fcgCSExeFiles->Name = L"fcgCSx264";
            this->fcgCSExeFiles->Size = System::Drawing::Size(131, 26);
            // 
            // fcgTSExeFileshelp
            // 
            this->fcgTSExeFileshelp->Name = L"fcgTSExeFileshelp";
            this->fcgTSExeFileshelp->Size = System::Drawing::Size(130, 22);
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
            this->fcgLBguiExBlog->Location = System::Drawing::Point(639, 596);
            this->fcgLBguiExBlog->Name = L"fcgLBguiExBlog";
            this->fcgLBguiExBlog->Size = System::Drawing::Size(79, 14);
            this->fcgLBguiExBlog->TabIndex = 50;
            this->fcgLBguiExBlog->TabStop = true;
            this->fcgLBguiExBlog->Text = L"NVEncについて";
            this->fcgLBguiExBlog->VisitedLinkColor = System::Drawing::Color::Gray;
            this->fcgLBguiExBlog->LinkClicked += gcnew System::Windows::Forms::LinkLabelLinkClickedEventHandler(this, &frmConfig::fcgLBguiExBlog_LinkClicked);
            // 
            // fcgtabControlAudio
            // 
            this->fcgtabControlAudio->Controls->Add(this->fcgtabPageAudioMain);
            this->fcgtabControlAudio->Controls->Add(this->fcgtabPageAudioOther);
            this->fcgtabControlAudio->Font = (gcnew System::Drawing::Font(L"Meiryo UI", 8.25F));
            this->fcgtabControlAudio->Location = System::Drawing::Point(621, 31);
            this->fcgtabControlAudio->Name = L"fcgtabControlAudio";
            this->fcgtabControlAudio->SelectedIndex = 0;
            this->fcgtabControlAudio->Size = System::Drawing::Size(384, 296);
            this->fcgtabControlAudio->TabIndex = 51;
            // 
            // fcgtabPageAudioMain
            // 
            this->fcgtabPageAudioMain->Controls->Add(this->fcgCBAudioUseExt);
            this->fcgtabPageAudioMain->Controls->Add(this->fcgCBFAWCheck);
            this->fcgtabPageAudioMain->Controls->Add(this->fcgPNAudioExt);
            this->fcgtabPageAudioMain->Controls->Add(this->fcgPNAudioInternal);
            this->fcgtabPageAudioMain->Location = System::Drawing::Point(4, 23);
            this->fcgtabPageAudioMain->Name = L"fcgtabPageAudioMain";
            this->fcgtabPageAudioMain->Padding = System::Windows::Forms::Padding(3);
            this->fcgtabPageAudioMain->Size = System::Drawing::Size(376, 269);
            this->fcgtabPageAudioMain->TabIndex = 0;
            this->fcgtabPageAudioMain->Text = L"音声";
            this->fcgtabPageAudioMain->UseVisualStyleBackColor = true;
            // 
            // fcgCBAudioUseExt
            // 
            this->fcgCBAudioUseExt->AutoSize = true;
            this->fcgCBAudioUseExt->Location = System::Drawing::Point(11, 4);
            this->fcgCBAudioUseExt->Name = L"fcgCBAudioUseExt";
            this->fcgCBAudioUseExt->Size = System::Drawing::Size(136, 18);
            this->fcgCBAudioUseExt->TabIndex = 77;
            this->fcgCBAudioUseExt->Tag = L"chValue";
            this->fcgCBAudioUseExt->Text = L"外部エンコーダを使用する";
            this->fcgCBAudioUseExt->UseVisualStyleBackColor = true;
            this->fcgCBAudioUseExt->CheckedChanged += gcnew System::EventHandler(this, &frmConfig::fcgCBAudioUseExt_CheckedChanged);
            // 
            // fcgCBFAWCheck
            // 
            this->fcgCBFAWCheck->AutoSize = true;
            this->fcgCBFAWCheck->Location = System::Drawing::Point(271, 4);
            this->fcgCBFAWCheck->Name = L"fcgCBFAWCheck";
            this->fcgCBFAWCheck->Size = System::Drawing::Size(81, 18);
            this->fcgCBFAWCheck->TabIndex = 58;
            this->fcgCBFAWCheck->Tag = L"chValue";
            this->fcgCBFAWCheck->Text = L"FAWCheck";
            this->fcgCBFAWCheck->UseVisualStyleBackColor = true;
            // 
            // fcgPNAudioExt
            // 
            this->fcgPNAudioExt->Controls->Add(this->fcgCXAudioDelayCut);
            this->fcgPNAudioExt->Controls->Add(this->fcgLBAudioDelayCut);
            this->fcgPNAudioExt->Controls->Add(this->fcgCBAudioEncTiming);
            this->fcgPNAudioExt->Controls->Add(this->fcgCXAudioEncTiming);
            this->fcgPNAudioExt->Controls->Add(this->fcgCXAudioTempDir);
            this->fcgPNAudioExt->Controls->Add(this->fcgTXCustomAudioTempDir);
            this->fcgPNAudioExt->Controls->Add(this->fcgBTCustomAudioTempDir);
            this->fcgPNAudioExt->Controls->Add(this->fcgCBAudioUsePipe);
            this->fcgPNAudioExt->Controls->Add(this->fcgNUAudioBitrate);
            this->fcgPNAudioExt->Controls->Add(this->fcgCBAudio2pass);
            this->fcgPNAudioExt->Controls->Add(this->fcgCXAudioEncMode);
            this->fcgPNAudioExt->Controls->Add(this->fcgLBAudioEncMode);
            this->fcgPNAudioExt->Controls->Add(this->fcgBTAudioEncoderPath);
            this->fcgPNAudioExt->Controls->Add(this->fcgTXAudioEncoderPath);
            this->fcgPNAudioExt->Controls->Add(this->fcgLBAudioEncoderPath);
            this->fcgPNAudioExt->Controls->Add(this->fcgCBAudioOnly);
            this->fcgPNAudioExt->Controls->Add(this->fcgCXAudioEncoder);
            this->fcgPNAudioExt->Controls->Add(this->fcgLBAudioTemp);
            this->fcgPNAudioExt->Controls->Add(this->fcgLBAudioBitrate);
            this->fcgPNAudioExt->Location = System::Drawing::Point(0, 24);
            this->fcgPNAudioExt->Name = L"fcgPNAudioExt";
            this->fcgPNAudioExt->Size = System::Drawing::Size(374, 246);
            this->fcgPNAudioExt->TabIndex = 55;
            // 
            // fcgCXAudioDelayCut
            // 
            this->fcgCXAudioDelayCut->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
            this->fcgCXAudioDelayCut->FormattingEnabled = true;
            this->fcgCXAudioDelayCut->Location = System::Drawing::Point(292, 107);
            this->fcgCXAudioDelayCut->Name = L"fcgCXAudioDelayCut";
            this->fcgCXAudioDelayCut->Size = System::Drawing::Size(70, 22);
            this->fcgCXAudioDelayCut->TabIndex = 65;
            this->fcgCXAudioDelayCut->Tag = L"chValue";
            // 
            // fcgLBAudioDelayCut
            // 
            this->fcgLBAudioDelayCut->AutoSize = true;
            this->fcgLBAudioDelayCut->Location = System::Drawing::Point(225, 110);
            this->fcgLBAudioDelayCut->Name = L"fcgLBAudioDelayCut";
            this->fcgLBAudioDelayCut->Size = System::Drawing::Size(60, 14);
            this->fcgLBAudioDelayCut->TabIndex = 75;
            this->fcgLBAudioDelayCut->Text = L"ディレイカット";
            // 
            // fcgCBAudioEncTiming
            // 
            this->fcgCBAudioEncTiming->AutoSize = true;
            this->fcgCBAudioEncTiming->Location = System::Drawing::Point(243, 28);
            this->fcgCBAudioEncTiming->Name = L"fcgCBAudioEncTiming";
            this->fcgCBAudioEncTiming->Size = System::Drawing::Size(40, 14);
            this->fcgCBAudioEncTiming->TabIndex = 74;
            this->fcgCBAudioEncTiming->Text = L"処理順";
            // 
            // fcgCXAudioEncTiming
            // 
            this->fcgCXAudioEncTiming->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
            this->fcgCXAudioEncTiming->FormattingEnabled = true;
            this->fcgCXAudioEncTiming->Location = System::Drawing::Point(287, 25);
            this->fcgCXAudioEncTiming->Name = L"fcgCXAudioEncTiming";
            this->fcgCXAudioEncTiming->Size = System::Drawing::Size(68, 22);
            this->fcgCXAudioEncTiming->TabIndex = 73;
            this->fcgCXAudioEncTiming->Tag = L"chValue";
            // 
            // fcgCXAudioTempDir
            // 
            this->fcgCXAudioTempDir->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
            this->fcgCXAudioTempDir->FormattingEnabled = true;
            this->fcgCXAudioTempDir->Location = System::Drawing::Point(136, 182);
            this->fcgCXAudioTempDir->Name = L"fcgCXAudioTempDir";
            this->fcgCXAudioTempDir->Size = System::Drawing::Size(150, 22);
            this->fcgCXAudioTempDir->TabIndex = 67;
            this->fcgCXAudioTempDir->Tag = L"chValue";
            // 
            // fcgTXCustomAudioTempDir
            // 
            this->fcgTXCustomAudioTempDir->Location = System::Drawing::Point(65, 210);
            this->fcgTXCustomAudioTempDir->Name = L"fcgTXCustomAudioTempDir";
            this->fcgTXCustomAudioTempDir->Size = System::Drawing::Size(245, 21);
            this->fcgTXCustomAudioTempDir->TabIndex = 68;
            this->fcgTXCustomAudioTempDir->TextChanged += gcnew System::EventHandler(this, &frmConfig::fcgTXCustomAudioTempDir_TextChanged);
            // 
            // fcgBTCustomAudioTempDir
            // 
            this->fcgBTCustomAudioTempDir->Location = System::Drawing::Point(317, 208);
            this->fcgBTCustomAudioTempDir->Name = L"fcgBTCustomAudioTempDir";
            this->fcgBTCustomAudioTempDir->Size = System::Drawing::Size(29, 23);
            this->fcgBTCustomAudioTempDir->TabIndex = 70;
            this->fcgBTCustomAudioTempDir->Text = L"...";
            this->fcgBTCustomAudioTempDir->UseVisualStyleBackColor = true;
            this->fcgBTCustomAudioTempDir->Click += gcnew System::EventHandler(this, &frmConfig::fcgBTCustomAudioTempDir_Click);
            // 
            // fcgCBAudioUsePipe
            // 
            this->fcgCBAudioUsePipe->AutoSize = true;
            this->fcgCBAudioUsePipe->Location = System::Drawing::Point(131, 108);
            this->fcgCBAudioUsePipe->Name = L"fcgCBAudioUsePipe";
            this->fcgCBAudioUsePipe->Size = System::Drawing::Size(73, 18);
            this->fcgCBAudioUsePipe->TabIndex = 64;
            this->fcgCBAudioUsePipe->Tag = L"chValue";
            this->fcgCBAudioUsePipe->Text = L"パイプ処理";
            this->fcgCBAudioUsePipe->UseVisualStyleBackColor = true;
            // 
            // fcgNUAudioBitrate
            // 
            this->fcgNUAudioBitrate->Location = System::Drawing::Point(213, 131);
            this->fcgNUAudioBitrate->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 1536, 0, 0, 0 });
            this->fcgNUAudioBitrate->Name = L"fcgNUAudioBitrate";
            this->fcgNUAudioBitrate->Size = System::Drawing::Size(65, 21);
            this->fcgNUAudioBitrate->TabIndex = 62;
            this->fcgNUAudioBitrate->Tag = L"chValue";
            this->fcgNUAudioBitrate->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
            // 
            // fcgCBAudio2pass
            // 
            this->fcgCBAudio2pass->AutoSize = true;
            this->fcgCBAudio2pass->Location = System::Drawing::Point(60, 108);
            this->fcgCBAudio2pass->Name = L"fcgCBAudio2pass";
            this->fcgCBAudio2pass->Size = System::Drawing::Size(56, 18);
            this->fcgCBAudio2pass->TabIndex = 63;
            this->fcgCBAudio2pass->Tag = L"chValue";
            this->fcgCBAudio2pass->Text = L"2pass";
            this->fcgCBAudio2pass->UseVisualStyleBackColor = true;
            this->fcgCBAudio2pass->CheckedChanged += gcnew System::EventHandler(this, &frmConfig::fcgCBAudio2pass_CheckedChanged);
            // 
            // fcgCXAudioEncMode
            // 
            this->fcgCXAudioEncMode->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
            this->fcgCXAudioEncMode->FormattingEnabled = true;
            this->fcgCXAudioEncMode->Location = System::Drawing::Point(17, 130);
            this->fcgCXAudioEncMode->Name = L"fcgCXAudioEncMode";
            this->fcgCXAudioEncMode->Size = System::Drawing::Size(189, 22);
            this->fcgCXAudioEncMode->TabIndex = 61;
            this->fcgCXAudioEncMode->Tag = L"chValue";
            this->fcgCXAudioEncMode->SelectedIndexChanged += gcnew System::EventHandler(this, &frmConfig::fcgCXAudioEncMode_SelectedIndexChanged);
            // 
            // fcgLBAudioEncMode
            // 
            this->fcgLBAudioEncMode->AutoSize = true;
            this->fcgLBAudioEncMode->Location = System::Drawing::Point(5, 110);
            this->fcgLBAudioEncMode->Name = L"fcgLBAudioEncMode";
            this->fcgLBAudioEncMode->Size = System::Drawing::Size(32, 14);
            this->fcgLBAudioEncMode->TabIndex = 69;
            this->fcgLBAudioEncMode->Text = L"モード";
            // 
            // fcgBTAudioEncoderPath
            // 
            this->fcgBTAudioEncoderPath->Location = System::Drawing::Point(325, 64);
            this->fcgBTAudioEncoderPath->Name = L"fcgBTAudioEncoderPath";
            this->fcgBTAudioEncoderPath->Size = System::Drawing::Size(30, 23);
            this->fcgBTAudioEncoderPath->TabIndex = 60;
            this->fcgBTAudioEncoderPath->Text = L"...";
            this->fcgBTAudioEncoderPath->UseVisualStyleBackColor = true;
            this->fcgBTAudioEncoderPath->Click += gcnew System::EventHandler(this, &frmConfig::fcgBTAudioEncoderPath_Click);
            // 
            // fcgTXAudioEncoderPath
            // 
            this->fcgTXAudioEncoderPath->AllowDrop = true;
            this->fcgTXAudioEncoderPath->Location = System::Drawing::Point(17, 66);
            this->fcgTXAudioEncoderPath->Name = L"fcgTXAudioEncoderPath";
            this->fcgTXAudioEncoderPath->Size = System::Drawing::Size(303, 21);
            this->fcgTXAudioEncoderPath->TabIndex = 59;
            this->fcgTXAudioEncoderPath->TextChanged += gcnew System::EventHandler(this, &frmConfig::fcgTXAudioEncoderPath_TextChanged);
            this->fcgTXAudioEncoderPath->DragDrop += gcnew System::Windows::Forms::DragEventHandler(this, &frmConfig::fcgSetDragDropFilename_DragDrop);
            this->fcgTXAudioEncoderPath->DragEnter += gcnew System::Windows::Forms::DragEventHandler(this, &frmConfig::fcgSetDragDropFilename_Enter);
            this->fcgTXAudioEncoderPath->Enter += gcnew System::EventHandler(this, &frmConfig::fcgTXAudioEncoderPath_Enter);
            this->fcgTXAudioEncoderPath->Leave += gcnew System::EventHandler(this, &frmConfig::fcgTXAudioEncoderPath_Leave);
            // 
            // fcgLBAudioEncoderPath
            // 
            this->fcgLBAudioEncoderPath->AutoSize = true;
            this->fcgLBAudioEncoderPath->Location = System::Drawing::Point(13, 49);
            this->fcgLBAudioEncoderPath->Name = L"fcgLBAudioEncoderPath";
            this->fcgLBAudioEncoderPath->Size = System::Drawing::Size(49, 14);
            this->fcgLBAudioEncoderPath->TabIndex = 66;
            this->fcgLBAudioEncoderPath->Text = L"～の指定";
            // 
            // fcgCBAudioOnly
            // 
            this->fcgCBAudioOnly->AutoSize = true;
            this->fcgCBAudioOnly->ForeColor = System::Drawing::SystemColors::ControlDarkDark;
            this->fcgCBAudioOnly->Location = System::Drawing::Point(271, 3);
            this->fcgCBAudioOnly->Name = L"fcgCBAudioOnly";
            this->fcgCBAudioOnly->Size = System::Drawing::Size(89, 18);
            this->fcgCBAudioOnly->TabIndex = 57;
            this->fcgCBAudioOnly->Tag = L"chValue";
            this->fcgCBAudioOnly->Text = L"音声のみ出力";
            this->fcgCBAudioOnly->UseVisualStyleBackColor = true;
            // 
            // fcgCXAudioEncoder
            // 
            this->fcgCXAudioEncoder->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
            this->fcgCXAudioEncoder->FormattingEnabled = true;
            this->fcgCXAudioEncoder->Location = System::Drawing::Point(18, 8);
            this->fcgCXAudioEncoder->Name = L"fcgCXAudioEncoder";
            this->fcgCXAudioEncoder->Size = System::Drawing::Size(172, 22);
            this->fcgCXAudioEncoder->TabIndex = 55;
            this->fcgCXAudioEncoder->Tag = L"chValue";
            this->fcgCXAudioEncoder->SelectedIndexChanged += gcnew System::EventHandler(this, &frmConfig::fcgCXAudioEncoder_SelectedIndexChanged);
            // 
            // fcgLBAudioTemp
            // 
            this->fcgLBAudioTemp->AutoSize = true;
            this->fcgLBAudioTemp->Location = System::Drawing::Point(8, 185);
            this->fcgLBAudioTemp->Name = L"fcgLBAudioTemp";
            this->fcgLBAudioTemp->Size = System::Drawing::Size(114, 14);
            this->fcgLBAudioTemp->TabIndex = 72;
            this->fcgLBAudioTemp->Text = L"音声一時ファイル出力先";
            // 
            // fcgLBAudioBitrate
            // 
            this->fcgLBAudioBitrate->AutoSize = true;
            this->fcgLBAudioBitrate->Location = System::Drawing::Point(285, 135);
            this->fcgLBAudioBitrate->Name = L"fcgLBAudioBitrate";
            this->fcgLBAudioBitrate->Size = System::Drawing::Size(32, 14);
            this->fcgLBAudioBitrate->TabIndex = 71;
            this->fcgLBAudioBitrate->Text = L"kbps";
            // 
            // fcgPNAudioInternal
            // 
            this->fcgPNAudioInternal->Controls->Add(this->fcgLBAudioBitrateInternal);
            this->fcgPNAudioInternal->Controls->Add(this->fcgNUAudioBitrateInternal);
            this->fcgPNAudioInternal->Controls->Add(this->fcgCXAudioEncModeInternal);
            this->fcgPNAudioInternal->Controls->Add(this->label2);
            this->fcgPNAudioInternal->Controls->Add(this->fcgCXAudioEncoderInternal);
            this->fcgPNAudioInternal->Location = System::Drawing::Point(0, 24);
            this->fcgPNAudioInternal->Name = L"fcgPNAudioInternal";
            this->fcgPNAudioInternal->Size = System::Drawing::Size(374, 246);
            this->fcgPNAudioInternal->TabIndex = 76;
            // 
            // fcgLBAudioBitrateInternal
            // 
            this->fcgLBAudioBitrateInternal->AutoSize = true;
            this->fcgLBAudioBitrateInternal->Location = System::Drawing::Point(285, 69);
            this->fcgLBAudioBitrateInternal->Name = L"fcgLBAudioBitrateInternal";
            this->fcgLBAudioBitrateInternal->Size = System::Drawing::Size(32, 14);
            this->fcgLBAudioBitrateInternal->TabIndex = 76;
            this->fcgLBAudioBitrateInternal->Text = L"kbps";
            // 
            // fcgNUAudioBitrateInternal
            // 
            this->fcgNUAudioBitrateInternal->Location = System::Drawing::Point(214, 67);
            this->fcgNUAudioBitrateInternal->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 1536, 0, 0, 0 });
            this->fcgNUAudioBitrateInternal->Name = L"fcgNUAudioBitrateInternal";
            this->fcgNUAudioBitrateInternal->Size = System::Drawing::Size(65, 21);
            this->fcgNUAudioBitrateInternal->TabIndex = 74;
            this->fcgNUAudioBitrateInternal->Tag = L"chValue";
            this->fcgNUAudioBitrateInternal->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
            // 
            // fcgCXAudioEncModeInternal
            // 
            this->fcgCXAudioEncModeInternal->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
            this->fcgCXAudioEncModeInternal->FormattingEnabled = true;
            this->fcgCXAudioEncModeInternal->Location = System::Drawing::Point(18, 66);
            this->fcgCXAudioEncModeInternal->Name = L"fcgCXAudioEncModeInternal";
            this->fcgCXAudioEncModeInternal->Size = System::Drawing::Size(189, 22);
            this->fcgCXAudioEncModeInternal->TabIndex = 73;
            this->fcgCXAudioEncModeInternal->Tag = L"chValue";
            this->fcgCXAudioEncModeInternal->SelectedIndexChanged += gcnew System::EventHandler(this, &frmConfig::fcgCXAudioEncModeInternal_SelectedIndexChanged);
            // 
            // label2
            // 
            this->label2->AutoSize = true;
            this->label2->Location = System::Drawing::Point(9, 41);
            this->label2->Name = L"label2";
            this->label2->Size = System::Drawing::Size(32, 14);
            this->label2->TabIndex = 75;
            this->label2->Text = L"モード";
            // 
            // fcgCXAudioEncoderInternal
            // 
            this->fcgCXAudioEncoderInternal->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
            this->fcgCXAudioEncoderInternal->FormattingEnabled = true;
            this->fcgCXAudioEncoderInternal->Location = System::Drawing::Point(18, 8);
            this->fcgCXAudioEncoderInternal->Name = L"fcgCXAudioEncoderInternal";
            this->fcgCXAudioEncoderInternal->Size = System::Drawing::Size(172, 22);
            this->fcgCXAudioEncoderInternal->TabIndex = 70;
            this->fcgCXAudioEncoderInternal->Tag = L"chValue";
            this->fcgCXAudioEncoderInternal->SelectedIndexChanged += gcnew System::EventHandler(this, &frmConfig::fcgCXAudioEncoderInternal_SelectedIndexChanged);
            // 
            // fcgtabPageAudioOther
            // 
            this->fcgtabPageAudioOther->Controls->Add(this->panel2);
            this->fcgtabPageAudioOther->Controls->Add(this->fcgLBBatAfterAudioString);
            this->fcgtabPageAudioOther->Controls->Add(this->fcgLBBatBeforeAudioString);
            this->fcgtabPageAudioOther->Controls->Add(this->fcgBTBatAfterAudioPath);
            this->fcgtabPageAudioOther->Controls->Add(this->fcgTXBatAfterAudioPath);
            this->fcgtabPageAudioOther->Controls->Add(this->fcgLBBatAfterAudioPath);
            this->fcgtabPageAudioOther->Controls->Add(this->fcgCBRunBatAfterAudio);
            this->fcgtabPageAudioOther->Controls->Add(this->panel1);
            this->fcgtabPageAudioOther->Controls->Add(this->fcgBTBatBeforeAudioPath);
            this->fcgtabPageAudioOther->Controls->Add(this->fcgTXBatBeforeAudioPath);
            this->fcgtabPageAudioOther->Controls->Add(this->fcgLBBatBeforeAudioPath);
            this->fcgtabPageAudioOther->Controls->Add(this->fcgCBRunBatBeforeAudio);
            this->fcgtabPageAudioOther->Controls->Add(this->fcgCXAudioPriority);
            this->fcgtabPageAudioOther->Controls->Add(this->fcgLBAudioPriority);
            this->fcgtabPageAudioOther->Location = System::Drawing::Point(4, 23);
            this->fcgtabPageAudioOther->Name = L"fcgtabPageAudioOther";
            this->fcgtabPageAudioOther->Padding = System::Windows::Forms::Padding(3);
            this->fcgtabPageAudioOther->Size = System::Drawing::Size(376, 269);
            this->fcgtabPageAudioOther->TabIndex = 1;
            this->fcgtabPageAudioOther->Text = L"その他";
            this->fcgtabPageAudioOther->UseVisualStyleBackColor = true;
            // 
            // panel2
            // 
            this->panel2->BorderStyle = System::Windows::Forms::BorderStyle::FixedSingle;
            this->panel2->Location = System::Drawing::Point(18, 126);
            this->panel2->Name = L"panel2";
            this->panel2->Size = System::Drawing::Size(342, 1);
            this->panel2->TabIndex = 61;
            // 
            // fcgLBBatAfterAudioString
            // 
            this->fcgLBBatAfterAudioString->AutoSize = true;
            this->fcgLBBatAfterAudioString->Font = (gcnew System::Drawing::Font(L"Meiryo UI", 9, static_cast<System::Drawing::FontStyle>((System::Drawing::FontStyle::Italic | System::Drawing::FontStyle::Underline)),
                System::Drawing::GraphicsUnit::Point, static_cast<System::Byte>(128)));
            this->fcgLBBatAfterAudioString->ForeColor = System::Drawing::SystemColors::ControlDarkDark;
            this->fcgLBBatAfterAudioString->Location = System::Drawing::Point(304, 208);
            this->fcgLBBatAfterAudioString->Name = L"fcgLBBatAfterAudioString";
            this->fcgLBBatAfterAudioString->Size = System::Drawing::Size(27, 15);
            this->fcgLBBatAfterAudioString->TabIndex = 60;
            this->fcgLBBatAfterAudioString->Text = L" 後& ";
            this->fcgLBBatAfterAudioString->TextAlign = System::Drawing::ContentAlignment::TopCenter;
            // 
            // fcgLBBatBeforeAudioString
            // 
            this->fcgLBBatBeforeAudioString->AutoSize = true;
            this->fcgLBBatBeforeAudioString->Font = (gcnew System::Drawing::Font(L"Meiryo UI", 9, static_cast<System::Drawing::FontStyle>((System::Drawing::FontStyle::Italic | System::Drawing::FontStyle::Underline)),
                System::Drawing::GraphicsUnit::Point, static_cast<System::Byte>(128)));
            this->fcgLBBatBeforeAudioString->ForeColor = System::Drawing::SystemColors::ControlDarkDark;
            this->fcgLBBatBeforeAudioString->Location = System::Drawing::Point(304, 139);
            this->fcgLBBatBeforeAudioString->Name = L"fcgLBBatBeforeAudioString";
            this->fcgLBBatBeforeAudioString->Size = System::Drawing::Size(27, 15);
            this->fcgLBBatBeforeAudioString->TabIndex = 51;
            this->fcgLBBatBeforeAudioString->Text = L" 前& ";
            this->fcgLBBatBeforeAudioString->TextAlign = System::Drawing::ContentAlignment::TopCenter;
            // 
            // fcgBTBatAfterAudioPath
            // 
            this->fcgBTBatAfterAudioPath->Location = System::Drawing::Point(330, 231);
            this->fcgBTBatAfterAudioPath->Name = L"fcgBTBatAfterAudioPath";
            this->fcgBTBatAfterAudioPath->Size = System::Drawing::Size(30, 23);
            this->fcgBTBatAfterAudioPath->TabIndex = 59;
            this->fcgBTBatAfterAudioPath->Tag = L"chValue";
            this->fcgBTBatAfterAudioPath->Text = L"...";
            this->fcgBTBatAfterAudioPath->UseVisualStyleBackColor = true;
            this->fcgBTBatAfterAudioPath->Click += gcnew System::EventHandler(this, &frmConfig::fcgBTBatAfterAudioPath_Click);
            // 
            // fcgTXBatAfterAudioPath
            // 
            this->fcgTXBatAfterAudioPath->AllowDrop = true;
            this->fcgTXBatAfterAudioPath->Location = System::Drawing::Point(126, 232);
            this->fcgTXBatAfterAudioPath->Name = L"fcgTXBatAfterAudioPath";
            this->fcgTXBatAfterAudioPath->Size = System::Drawing::Size(202, 21);
            this->fcgTXBatAfterAudioPath->TabIndex = 58;
            this->fcgTXBatAfterAudioPath->Tag = L"chValue";
            this->fcgTXBatAfterAudioPath->DragDrop += gcnew System::Windows::Forms::DragEventHandler(this, &frmConfig::fcgSetDragDropFilename_DragDrop);
            this->fcgTXBatAfterAudioPath->DragEnter += gcnew System::Windows::Forms::DragEventHandler(this, &frmConfig::fcgSetDragDropFilename_Enter);
            // 
            // fcgLBBatAfterAudioPath
            // 
            this->fcgLBBatAfterAudioPath->AutoSize = true;
            this->fcgLBBatAfterAudioPath->Location = System::Drawing::Point(40, 236);
            this->fcgLBBatAfterAudioPath->Name = L"fcgLBBatAfterAudioPath";
            this->fcgLBBatAfterAudioPath->Size = System::Drawing::Size(61, 14);
            this->fcgLBBatAfterAudioPath->TabIndex = 57;
            this->fcgLBBatAfterAudioPath->Text = L"バッチファイル";
            // 
            // fcgCBRunBatAfterAudio
            // 
            this->fcgCBRunBatAfterAudio->AutoSize = true;
            this->fcgCBRunBatAfterAudio->Location = System::Drawing::Point(18, 207);
            this->fcgCBRunBatAfterAudio->Name = L"fcgCBRunBatAfterAudio";
            this->fcgCBRunBatAfterAudio->Size = System::Drawing::Size(201, 18);
            this->fcgCBRunBatAfterAudio->TabIndex = 55;
            this->fcgCBRunBatAfterAudio->Tag = L"chValue";
            this->fcgCBRunBatAfterAudio->Text = L"音声エンコード終了後、バッチ処理を行う";
            this->fcgCBRunBatAfterAudio->UseVisualStyleBackColor = true;
            // 
            // panel1
            // 
            this->panel1->BorderStyle = System::Windows::Forms::BorderStyle::FixedSingle;
            this->panel1->Location = System::Drawing::Point(18, 196);
            this->panel1->Name = L"panel1";
            this->panel1->Size = System::Drawing::Size(342, 1);
            this->panel1->TabIndex = 54;
            // 
            // fcgBTBatBeforeAudioPath
            // 
            this->fcgBTBatBeforeAudioPath->Location = System::Drawing::Point(330, 164);
            this->fcgBTBatBeforeAudioPath->Name = L"fcgBTBatBeforeAudioPath";
            this->fcgBTBatBeforeAudioPath->Size = System::Drawing::Size(30, 23);
            this->fcgBTBatBeforeAudioPath->TabIndex = 53;
            this->fcgBTBatBeforeAudioPath->Tag = L"chValue";
            this->fcgBTBatBeforeAudioPath->Text = L"...";
            this->fcgBTBatBeforeAudioPath->UseVisualStyleBackColor = true;
            this->fcgBTBatBeforeAudioPath->Click += gcnew System::EventHandler(this, &frmConfig::fcgBTBatBeforeAudioPath_Click);
            // 
            // fcgTXBatBeforeAudioPath
            // 
            this->fcgTXBatBeforeAudioPath->AllowDrop = true;
            this->fcgTXBatBeforeAudioPath->Location = System::Drawing::Point(126, 164);
            this->fcgTXBatBeforeAudioPath->Name = L"fcgTXBatBeforeAudioPath";
            this->fcgTXBatBeforeAudioPath->Size = System::Drawing::Size(202, 21);
            this->fcgTXBatBeforeAudioPath->TabIndex = 52;
            this->fcgTXBatBeforeAudioPath->Tag = L"chValue";
            this->fcgTXBatBeforeAudioPath->DragDrop += gcnew System::Windows::Forms::DragEventHandler(this, &frmConfig::fcgSetDragDropFilename_DragDrop);
            this->fcgTXBatBeforeAudioPath->DragEnter += gcnew System::Windows::Forms::DragEventHandler(this, &frmConfig::fcgSetDragDropFilename_Enter);
            // 
            // fcgLBBatBeforeAudioPath
            // 
            this->fcgLBBatBeforeAudioPath->AutoSize = true;
            this->fcgLBBatBeforeAudioPath->Location = System::Drawing::Point(40, 167);
            this->fcgLBBatBeforeAudioPath->Name = L"fcgLBBatBeforeAudioPath";
            this->fcgLBBatBeforeAudioPath->Size = System::Drawing::Size(61, 14);
            this->fcgLBBatBeforeAudioPath->TabIndex = 50;
            this->fcgLBBatBeforeAudioPath->Text = L"バッチファイル";
            // 
            // fcgCBRunBatBeforeAudio
            // 
            this->fcgCBRunBatBeforeAudio->AutoSize = true;
            this->fcgCBRunBatBeforeAudio->Location = System::Drawing::Point(18, 139);
            this->fcgCBRunBatBeforeAudio->Name = L"fcgCBRunBatBeforeAudio";
            this->fcgCBRunBatBeforeAudio->Size = System::Drawing::Size(201, 18);
            this->fcgCBRunBatBeforeAudio->TabIndex = 48;
            this->fcgCBRunBatBeforeAudio->Tag = L"chValue";
            this->fcgCBRunBatBeforeAudio->Text = L"音声エンコード開始前、バッチ処理を行う";
            this->fcgCBRunBatBeforeAudio->UseVisualStyleBackColor = true;
            // 
            // fcgCXAudioPriority
            // 
            this->fcgCXAudioPriority->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
            this->fcgCXAudioPriority->FormattingEnabled = true;
            this->fcgCXAudioPriority->Location = System::Drawing::Point(156, 20);
            this->fcgCXAudioPriority->Name = L"fcgCXAudioPriority";
            this->fcgCXAudioPriority->Size = System::Drawing::Size(136, 22);
            this->fcgCXAudioPriority->TabIndex = 47;
            this->fcgCXAudioPriority->Tag = L"chValue";
            // 
            // fcgLBAudioPriority
            // 
            this->fcgLBAudioPriority->AutoSize = true;
            this->fcgLBAudioPriority->Location = System::Drawing::Point(29, 23);
            this->fcgLBAudioPriority->Name = L"fcgLBAudioPriority";
            this->fcgLBAudioPriority->Size = System::Drawing::Size(62, 14);
            this->fcgLBAudioPriority->TabIndex = 46;
            this->fcgLBAudioPriority->Text = L"音声優先度";
            // 
            // fcgTXCmd
            // 
            this->fcgTXCmd->Anchor = static_cast<System::Windows::Forms::AnchorStyles>(((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Left)
                | System::Windows::Forms::AnchorStyles::Right));
            this->fcgTXCmd->Location = System::Drawing::Point(7, 558);
            this->fcgTXCmd->Margin = System::Windows::Forms::Padding(2);
            this->fcgTXCmd->Name = L"fcgTXCmd";
            this->fcgTXCmd->ReadOnly = true;
            this->fcgTXCmd->Size = System::Drawing::Size(997, 23);
            this->fcgTXCmd->TabIndex = 52;
            this->fcgTXCmd->DoubleClick += gcnew System::EventHandler(this, &frmConfig::fcgTXCmd_DoubleClick);
            // 
            // fcgPNHideToolStripBorder
            // 
            this->fcgPNHideToolStripBorder->Location = System::Drawing::Point(0, 22);
            this->fcgPNHideToolStripBorder->Name = L"fcgPNHideToolStripBorder";
            this->fcgPNHideToolStripBorder->Size = System::Drawing::Size(1020, 4);
            this->fcgPNHideToolStripBorder->TabIndex = 53;
            this->fcgPNHideToolStripBorder->Visible = false;
            // 
            // frmConfig
            // 
            this->AutoScaleDimensions = System::Drawing::SizeF(96, 96);
            this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Dpi;
            this->ClientSize = System::Drawing::Size(1008, 617);
            this->Controls->Add(this->fcgPNHideToolStripBorder);
            this->Controls->Add(this->fcgTXCmd);
            this->Controls->Add(this->fcgtabControlAudio);
            this->Controls->Add(this->fcgLBguiExBlog);
            this->Controls->Add(this->fcgtabControlMux);
            this->Controls->Add(this->fcgtabControlNVEnc);
            this->Controls->Add(this->fcgLBVersion);
            this->Controls->Add(this->fcgLBVersionDate);
            this->Controls->Add(this->fcgBTDefault);
            this->Controls->Add(this->fcgBTOK);
            this->Controls->Add(this->fcgBTCancel);
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
            this->fcgtabPageInternal->ResumeLayout(false);
            this->fcgtabPageInternal->PerformLayout();
            this->fcgtabControlNVEnc->ResumeLayout(false);
            this->tabPageVideoEnc->ResumeLayout(false);
            this->tabPageVideoEnc->PerformLayout();
            this->fcggroupBoxColor->ResumeLayout(false);
            this->fcggroupBoxColor->PerformLayout();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNULookaheadDepth))->EndInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUAQStrength))->EndInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUVBVBufsize))->EndInit();
            this->fcgGroupBoxAspectRatio->ResumeLayout(false);
            this->fcgGroupBoxAspectRatio->PerformLayout();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUAspectRatioY))->EndInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUAspectRatioX))->EndInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNURefFrames))->EndInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUBframes))->EndInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUGopLength))->EndInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgPBNVEncLogoEnabled))->EndInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgPBNVEncLogoDisabled))->EndInit();
            this->fcgPNH264->ResumeLayout(false);
            this->fcgPNH264->PerformLayout();
            this->fcgPNQP->ResumeLayout(false);
            this->fcgPNQP->PerformLayout();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUQPI))->EndInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUQPP))->EndInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUQPB))->EndInit();
            this->fcgPNBitrate->ResumeLayout(false);
            this->fcgPNBitrate->PerformLayout();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUVBRTragetQuality))->EndInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUBitrate))->EndInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUMaxkbps))->EndInit();
            this->fcgPNHEVC->ResumeLayout(false);
            this->fcgPNHEVC->PerformLayout();
            this->tabPageVideoDetail->ResumeLayout(false);
            this->tabPageVideoDetail->PerformLayout();
            this->groupBoxQPDetail->ResumeLayout(false);
            this->groupBoxQPDetail->PerformLayout();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUChromaQPOffset))->EndInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUQPInitB))->EndInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUQPInitP))->EndInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUQPInitI))->EndInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUQPMinB))->EndInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUQPMinP))->EndInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUQPMinI))->EndInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUQPMaxB))->EndInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUQPMaxP))->EndInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUQPMaxI))->EndInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUSlices))->EndInit();
            this->fcgPNH264Detail->ResumeLayout(false);
            this->fcgPNH264Detail->PerformLayout();
            this->fcgPNHEVCDetail->ResumeLayout(false);
            this->fcgPNHEVCDetail->PerformLayout();
            this->tabPageExOpt->ResumeLayout(false);
            this->tabPageExOpt->PerformLayout();
            this->fcggroupBoxVppTweak->ResumeLayout(false);
            this->fcggroupBoxVppTweak->PerformLayout();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgTBVppTweakHue))->EndInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgTBVppTweakSaturation))->EndInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgTBVppTweakGamma))->EndInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgTBVppTweakContrast))->EndInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgTBVppTweakBrightness))->EndInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUVppTweakGamma))->EndInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUVppTweakSaturation))->EndInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUVppTweakHue))->EndInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUVppTweakContrast))->EndInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUVppTweakBrightness))->EndInit();
            this->fcggroupBoxVppDetailEnahance->ResumeLayout(false);
            this->fcgPNVppWarpsharp->ResumeLayout(false);
            this->fcgPNVppWarpsharp->PerformLayout();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUVppWarpsharpBlurType))->EndInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUVppWarpsharpDepth))->EndInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUVppWarpsharpBlur))->EndInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUVppWarpsharpThreshold))->EndInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->numericUpDown6))->EndInit();
            this->fcgPNVppEdgelevel->ResumeLayout(false);
            this->fcgPNVppEdgelevel->PerformLayout();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUVppEdgelevelWhite))->EndInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUVppEdgelevelThreshold))->EndInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUVppEdgelevelBlack))->EndInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUVppEdgelevelStrength))->EndInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->numericUpDown4))->EndInit();
            this->fcgPNVppUnsharp->ResumeLayout(false);
            this->fcgPNVppUnsharp->PerformLayout();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUVppUnsharpThreshold))->EndInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUVppUnsharpWeight))->EndInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUVppUnsharpRadius))->EndInit();
            this->fcggroupBoxVppDeinterlace->ResumeLayout(false);
            this->fcggroupBoxVppDeinterlace->PerformLayout();
            this->fcgPNVppYadif->ResumeLayout(false);
            this->fcgPNVppYadif->PerformLayout();
            this->fcgPNVppNnedi->ResumeLayout(false);
            this->fcgPNVppNnedi->PerformLayout();
            this->fcgPNVppAfs->ResumeLayout(false);
            this->fcgPNVppAfs->PerformLayout();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgTBVppAfsThreCMotion))->EndInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgTBVppAfsThreYMotion))->EndInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgTBVppAfsThreDeint))->EndInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgTBVppAfsThreShift))->EndInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgTBVppAfsCoeffShift))->EndInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUVppAfsRight))->EndInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUVppAfsLeft))->EndInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUVppAfsBottom))->EndInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUVppAfsUp))->EndInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgTBVppAfsMethodSwitch))->EndInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUVppAfsThreCMotion))->EndInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUVppAfsThreShift))->EndInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUVppAfsThreDeint))->EndInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUVppAfsThreYMotion))->EndInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUVppAfsCoeffShift))->EndInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUVppAfsMethodSwitch))->EndInit();
            this->fcggroupBoxVppDeband->ResumeLayout(false);
            this->fcggroupBoxVppDeband->PerformLayout();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUVppDebandDitherC))->EndInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUVppDebandDitherY))->EndInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUVppDebandThreCr))->EndInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUVppDebandThreCb))->EndInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUVppDebandThreY))->EndInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUVppDebandRange))->EndInit();
            this->fcggroupBoxVppDenoise->ResumeLayout(false);
            this->fcgPNVppDenoiseConv3D->ResumeLayout(false);
            this->fcgPNVppDenoiseConv3D->PerformLayout();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUVppDenoiseConv3DThreshCTemporal))->EndInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUVppDenoiseConv3DThreshCSpatial))->EndInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUVppDenoiseConv3DThreshYTemporal))->EndInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUVppDenoiseConv3DThreshYSpatial))->EndInit();
            this->fcgPNVppDenoiseSmooth->ResumeLayout(false);
            this->fcgPNVppDenoiseSmooth->PerformLayout();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUVppDenoiseSmoothQP))->EndInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUVppDenoiseSmoothQuality))->EndInit();
            this->fcgPNVppDenoiseKnn->ResumeLayout(false);
            this->fcgPNVppDenoiseKnn->PerformLayout();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUVppDenoiseKnnThreshold))->EndInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUVppDenoiseKnnStrength))->EndInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUVppDenoiseKnnRadius))->EndInit();
            this->fcgPNVppDenoisePmd->ResumeLayout(false);
            this->fcgPNVppDenoisePmd->PerformLayout();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUVppDenoisePmdThreshold))->EndInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUVppDenoisePmdStrength))->EndInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUVppDenoisePmdApplyCount))->EndInit();
            this->fcggroupBoxResize->ResumeLayout(false);
            this->fcggroupBoxResize->PerformLayout();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUVppResizeHeight))->EndInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUVppResizeWidth))->EndInit();
            this->tabPageOther->ResumeLayout(false);
            this->tabPageOther->PerformLayout();
            this->fcggroupBoxCmdEx->ResumeLayout(false);
            this->fcggroupBoxCmdEx->PerformLayout();
            this->fcgCSExeFiles->ResumeLayout(false);
            this->fcgtabControlAudio->ResumeLayout(false);
            this->fcgtabPageAudioMain->ResumeLayout(false);
            this->fcgtabPageAudioMain->PerformLayout();
            this->fcgPNAudioExt->ResumeLayout(false);
            this->fcgPNAudioExt->PerformLayout();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUAudioBitrate))->EndInit();
            this->fcgPNAudioInternal->ResumeLayout(false);
            this->fcgPNAudioInternal->PerformLayout();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->fcgNUAudioBitrateInternal))->EndInit();
            this->fcgtabPageAudioOther->ResumeLayout(false);
            this->fcgtabPageAudioOther->PerformLayout();
            this->ResumeLayout(false);
            this->PerformLayout();

        }
#pragma endregion
    private:
        const SYSTEM_DATA *sys_dat;
        CONF_GUIEX *conf;
        LocalSettings LocalStg;
        DarkenWindowStgReader *dwStgReader;
        AuoTheme themeMode;
        bool CurrentPipeEnabled;
        bool stgChanged;
        String^ CurrentStgDir;
        ToolStripMenuItem^ CheckedStgMenuItem;
        CONF_GUIEX *cnf_stgSelected;
        String^ lastQualityStr;
        VidEncInfo nvencInfo;
        Task<VidEncInfo>^ taskNVEncInfo;
        CancellationTokenSource^ taskNVEncInfoCancell;
#ifdef HIDE_MPEG2
        TabPage^ tabPageMpgMux;
#endif
    private:
        System::Void CheckTheme();
        System::Void SetAllMouseMove(Control ^top, const AuoTheme themeTo);
        System::Void fcgMouseEnter_SetColor(System::Object^  sender, System::EventArgs^  e);
        System::Void fcgMouseLeave_SetColor(System::Object^  sender, System::EventArgs^  e);
        System::Void TabControl_DarkDrawItem(System::Object^ sender, DrawItemEventArgs^ e);

        System::Int32 GetCurrentAudioDefaultBitrate();
        delegate System::Void qualityTimerChangeDelegate();
        System::Void InitComboBox();
        System::Void setAudioExtDisplay();
        System::Void AudioExtEncodeModeChanged();
        System::Void setAudioIntDisplay();
        System::Void AudioIntEncodeModeChanged();
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
        System::String^ FrmToConf(CONF_GUIEX *cnf);
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
        System::Void fcgCXAudioEncoderInternal_SelectedIndexChanged(System::Object ^sender, System::EventArgs ^e);
        System::Void fcgCXAudioEncModeInternal_SelectedIndexChanged(System::Object ^ sender, System::EventArgs ^ e);
        System::Void fcgCBAudioUseExt_CheckedChanged(System::Object ^sender, System::EventArgs ^e);
        System::Void AdjustLocation();
        System::Void ActivateToolTip(bool Enable);
        System::Void SetStgEscKey(bool Enable);
        System::Void SetToolStripEvents(ToolStrip^ TS, System::Windows::Forms::MouseEventHandler^ _event);
        System::Void fcgCodecChanged(System::Object^  sender, System::EventArgs^  e);
        System::Void GetVidEncInfoAsync();
        VidEncInfo GetVidEncInfo();
        System::Void SetVidEncInfo(VidEncInfo info);
        delegate System::Void SetVidEncInfoDelegate(VidEncInfo info);
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
            for (int i = 0; list[i].desc; i++) {
                CX->Items->Add(String(list[i].desc).ToString());
            }
            CX->EndUpdate();
        }
    private:
        System::Void setComboBox(ComboBox ^CX, const CX_DESC *list, const TCHAR *ignore) {
            CX->BeginUpdate();
            CX->Items->Clear();
            for (int i = 0; list[i].desc; i++) {
                if (ignore && strcmp(ignore, list[i].desc) == 0) {
                    //インデックスの順番を保持するため、途中の場合は"-----"をいれておく
                    if (list[i + 1].desc) {
                        CX->Items->Add(String("-----").ToString());
                    }
                } else {
                    CX->Items->Add(String(list[i].desc).ToString());
                }
            }
            CX->EndUpdate();
        }
    private:
        template<size_t size>
        System::Void setComboBox(ComboBox^ CX, const guid_desc (&list)[size]) {
            CX->BeginUpdate();
            CX->Items->Clear();
            for (int i = 0; i < size; i++)
                CX->Items->Add(String(list[i].desc).ToString());
            CX->EndUpdate();
        }
    private:
        template<size_t size>
        System::Void setComboBox(ComboBox ^CX, const guid_desc(&list)[size], int setLength) {
            CX->BeginUpdate();
            CX->Items->Clear();
            for (int i = 0; i < setLength; i++)
                CX->Items->Add(String(list[i].desc).ToString());
            CX->EndUpdate();
        }
    private:
        System::Void setComboBox(ComboBox^ CX, const char * const * list) {
            CX->BeginUpdate();
            CX->Items->Clear();
            for (int i = 0; list[i]; i++)
                CX->Items->Add(String(list[i]).ToString());
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

            fcgCXAudioEncoderInternal->BeginUpdate();
            fcgCXAudioEncoderInternal->Items->Clear();
            for (int i = 0; i < sys_dat->exstg->s_aud_int_count; i++) {
                fcgCXAudioEncoderInternal->Items->Add(String(sys_dat->exstg->s_aud_int[i].dispname).ToString());
            }
            fcgCXAudioEncoderInternal->EndUpdate();
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
            bool ret = (ofd.ShowDialog() == System::Windows::Forms::DialogResult::OK);
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
        System::Void fcgBTVideoEncoderPath_Click(System::Object^  sender, System::EventArgs^  e) {
            openExeFile(fcgTXVideoEncoderPath, LocalStg.vidEncName);
        }
    private:
        System::Void fcgTXVideoEncoderPath_Enter(System::Object^  sender, System::EventArgs^  e) {
            if (fcgTXVideoEncoderPath->Text == String(use_default_exe_path).ToString()) {
                fcgTXVideoEncoderPath->Text = L"";
            }
        }
    private:
        System::Void fcgTXVideoEncoderPath_Leave(System::Object^  sender, System::EventArgs^  e) {
            if (fcgTXVideoEncoderPath->Text->Length == 0) {
                fcgTXVideoEncoderPath->Text = String(use_default_exe_path).ToString();
            }
        }
    private:
        System::Void fcgTXAudioEncoderPath_Enter(System::Object^  sender, System::EventArgs^  e) {
            if (fcgTXAudioEncoderPath->Text == String(use_default_exe_path).ToString()) {
                fcgTXAudioEncoderPath->Text = L"";
            }
        }
    private:
        System::Void fcgTXAudioEncoderPath_Leave(System::Object^  sender, System::EventArgs^  e) {
            if (fcgTXAudioEncoderPath->Text->Length == 0) {
                fcgTXAudioEncoderPath->Text = String(use_default_exe_path).ToString();
            }
        }
    private:
        System::Void fcgTXMP4MuxerPath_Enter(System::Object^  sender, System::EventArgs^  e) {
            if (fcgTXMP4MuxerPath->Text == String(use_default_exe_path).ToString()) {
                fcgTXMP4MuxerPath->Text = L"";
            }
        }
    private:
        System::Void fcgTXMP4MuxerPath_Leave(System::Object^  sender, System::EventArgs^  e) {
            if (fcgTXMP4MuxerPath->Text->Length == 0) {
                fcgTXMP4MuxerPath->Text = String(use_default_exe_path).ToString();
            }
        }
    private:
        System::Void fcgTXTC2MP4Path_Enter(System::Object^  sender, System::EventArgs^  e) {
            if (fcgTXTC2MP4Path->Text == String(use_default_exe_path).ToString()) {
                fcgTXTC2MP4Path->Text = L"";
            }
        }
    private:
        System::Void fcgTXTC2MP4Path_Leave(System::Object^  sender, System::EventArgs^  e) {
            if (fcgTXTC2MP4Path->Text->Length == 0) {
                fcgTXTC2MP4Path->Text = String(use_default_exe_path).ToString();
            }
        }
    private:
        System::Void fcgTXMP4RawPath_Enter(System::Object^  sender, System::EventArgs^  e) {
            if (fcgTXMP4RawPath->Text == String(use_default_exe_path).ToString()) {
                fcgTXMP4RawPath->Text = L"";
            }
        }
    private:
        System::Void fcgTXMP4RawPath_Leave(System::Object^  sender, System::EventArgs^  e) {
            if (fcgTXMP4RawPath->Text->Length == 0) {
                fcgTXMP4RawPath->Text = String(use_default_exe_path).ToString();
            }
        }
    private:
        System::Void fcgTXMKVMuxerPath_Enter(System::Object^  sender, System::EventArgs^  e) {
            if (fcgTXMKVMuxerPath->Text == String(use_default_exe_path).ToString()) {
                fcgTXMKVMuxerPath->Text = L"";
            }
        }
    private:
        System::Void fcgTXMKVMuxerPath_Leave(System::Object^  sender, System::EventArgs^  e) {
            if (fcgTXMKVMuxerPath->Text->Length == 0) {
                fcgTXMKVMuxerPath->Text = String(use_default_exe_path).ToString();
            }
        }
    private:
        System::Void fcgTXMPGMuxerPath_Enter(System::Object^  sender, System::EventArgs^  e) {
            if (fcgTXMPGMuxerPath->Text == String(use_default_exe_path).ToString()) {
                fcgTXMPGMuxerPath->Text = L"";
            }
        }
    private:
        System::Void fcgTXMPGMuxerPath_Leave(System::Object^  sender, System::EventArgs^  e) {
            if (fcgTXMPGMuxerPath->Text->Length == 0) {
                fcgTXMPGMuxerPath->Text = String(use_default_exe_path).ToString();
            }
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
            if (fbd->ShowDialog() == System::Windows::Forms::DialogResult::OK) {
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
            bool ret = (ofd->ShowDialog() == System::Windows::Forms::DialogResult::OK);
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
        System::Void fcgBTBatBeforeAudioPath_Click(System::Object^  sender, System::EventArgs^  e) {
            if (openAndSetFilePath(fcgTXBatBeforeAudioPath, L"バッチファイル", ".bat", LocalStg.LastBatDir))
                LocalStg.LastBatDir = Path::GetDirectoryName(fcgTXBatBeforeAudioPath->Text);
        }
    private:
        System::Void fcgBTBatAfterAudioPath_Click(System::Object^  sender, System::EventArgs^  e) {
            if (openAndSetFilePath(fcgTXBatAfterAudioPath, L"バッチファイル", ".bat", LocalStg.LastBatDir))
                LocalStg.LastBatDir = Path::GetDirectoryName(fcgTXBatAfterAudioPath->Text);
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
        System::Void fcgRebuildCmd(System::Object^  sender, System::EventArgs^  e) {
            CONF_GUIEX rebuild;
            init_CONF_GUIEX(&rebuild, FALSE);
            fcgTXCmd->Text = FrmToConf(&rebuild);
            if (CheckedStgMenuItem != nullptr)
                ChangePresetNameDisplay(memcmp(&rebuild, cnf_stgSelected, sizeof(CONF_GUIEX)) != 0);
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
        System::Void fcgTXVideoEncoderPath_TextChanged(System::Object^  sender, System::EventArgs^  e) {
            if (fcgTXVideoEncoderPath->Text == String(use_default_exe_path).ToString()) {
                LocalStg.vidEncPath = L"";
                fcgTXVideoEncoderPath->ForeColor = getTextBoxForeColor(themeMode, dwStgReader, DarkenWindowState::Disabled);
            } else {
                fcgTXVideoEncoderPath->ForeColor = getTextBoxForeColor(themeMode, dwStgReader, DarkenWindowState::Normal);
                LocalStg.vidEncPath = fcgTXVideoEncoderPath->Text;
                fcgTXVideoEncoderPath->ContextMenuStrip = (File::Exists(fcgTXVideoEncoderPath->Text)) ? fcgCSExeFiles : nullptr;
            }
            GetVidEncInfoAsync();
        }
    private:
        System::Void fcgTXAudioEncoderPath_TextChanged(System::Object^  sender, System::EventArgs^  e) {
            if (fcgTXAudioEncoderPath->Text == String(use_default_exe_path).ToString()) {
                LocalStg.audEncPath[fcgCXAudioEncoder->SelectedIndex] = L"";
                fcgTXAudioEncoderPath->ForeColor = getTextBoxForeColor(themeMode, dwStgReader, DarkenWindowState::Disabled);
            } else {
                fcgTXAudioEncoderPath->ForeColor = getTextBoxForeColor(themeMode, dwStgReader, DarkenWindowState::Normal);
                LocalStg.audEncPath[fcgCXAudioEncoder->SelectedIndex] = fcgTXAudioEncoderPath->Text;
                fcgBTAudioEncoderPath->ContextMenuStrip = (File::Exists(fcgTXAudioEncoderPath->Text)) ? fcgCSExeFiles : nullptr;
            }
        }
    private:
        System::Void fcgTXMP4MuxerPath_TextChanged(System::Object^  sender, System::EventArgs^  e) {
            if (fcgTXMP4MuxerPath->Text == String(use_default_exe_path).ToString()) {
                LocalStg.MP4MuxerPath = L"";
                fcgTXMP4MuxerPath->ForeColor = getTextBoxForeColor(themeMode, dwStgReader, DarkenWindowState::Disabled);
            } else {
                fcgTXMP4MuxerPath->ForeColor = getTextBoxForeColor(themeMode, dwStgReader, DarkenWindowState::Normal);
                LocalStg.MP4MuxerPath = fcgTXMP4MuxerPath->Text;
                fcgBTMP4MuxerPath->ContextMenuStrip = (File::Exists(fcgTXMP4MuxerPath->Text)) ? fcgCSExeFiles : nullptr;
            }
        }
    private:
        System::Void fcgTXTC2MP4Path_TextChanged(System::Object^  sender, System::EventArgs^  e) {
            if (fcgTXTC2MP4Path->Text == String(use_default_exe_path).ToString()) {
                LocalStg.TC2MP4Path = L"";
                fcgTXTC2MP4Path->ForeColor = getTextBoxForeColor(themeMode, dwStgReader, DarkenWindowState::Disabled);
            } else {
                fcgTXTC2MP4Path->ForeColor = getTextBoxForeColor(themeMode, dwStgReader, DarkenWindowState::Normal);
                LocalStg.TC2MP4Path = fcgTXTC2MP4Path->Text;
                fcgBTTC2MP4Path->ContextMenuStrip = (File::Exists(fcgTXTC2MP4Path->Text)) ? fcgCSExeFiles : nullptr;
            }
        }
    private:
        System::Void fcgTXMP4RawMuxerPath_TextChanged(System::Object^  sender, System::EventArgs^  e) {
            if (fcgTXMP4RawPath->Text == String(use_default_exe_path).ToString()) {
                LocalStg.MP4RawPath = L"";
                fcgTXMP4RawPath->ForeColor = getTextBoxForeColor(themeMode, dwStgReader, DarkenWindowState::Disabled);
            } else {
                fcgTXMP4RawPath->ForeColor = getTextBoxForeColor(themeMode, dwStgReader, DarkenWindowState::Normal);
                LocalStg.MP4RawPath = fcgTXMP4RawPath->Text;
                fcgBTMP4RawPath->ContextMenuStrip = (File::Exists(fcgTXMP4RawPath->Text)) ? fcgCSExeFiles : nullptr;
            }
        }
    private:
        System::Void fcgTXMKVMuxerPath_TextChanged(System::Object^  sender, System::EventArgs^  e) {
            if (fcgTXMKVMuxerPath->Text == String(use_default_exe_path).ToString()) {
                LocalStg.MKVMuxerPath = L"";
                fcgTXMKVMuxerPath->ForeColor = getTextBoxForeColor(themeMode, dwStgReader, DarkenWindowState::Disabled);
            } else {
                fcgTXMKVMuxerPath->ForeColor = getTextBoxForeColor(themeMode, dwStgReader, DarkenWindowState::Normal);
                LocalStg.MKVMuxerPath = fcgTXMKVMuxerPath->Text;
                fcgBTMKVMuxerPath->ContextMenuStrip = (File::Exists(fcgTXMKVMuxerPath->Text)) ? fcgCSExeFiles : nullptr;
            }
        }
    private:
        System::Void fcgTXMPGMuxerPath_TextChanged(System::Object^  sender, System::EventArgs^  e) {
            if (fcgTXMPGMuxerPath->Text == String(use_default_exe_path).ToString()) {
                LocalStg.MPGMuxerPath = L"";
                fcgTXMPGMuxerPath->ForeColor = getTextBoxForeColor(themeMode, dwStgReader, DarkenWindowState::Disabled);
            } else {
                fcgTXMPGMuxerPath->ForeColor = getTextBoxForeColor(themeMode, dwStgReader, DarkenWindowState::Normal);
                LocalStg.MPGMuxerPath = fcgTXMPGMuxerPath->Text;
                fcgBTMPGMuxerPath->ContextMenuStrip = (File::Exists(fcgTXMPGMuxerPath->Text)) ? fcgCSExeFiles : nullptr;
            }
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
                { fcgBTVideoEncoderPath->Name,   fcgTXVideoEncoderPath->Text,   sys_dat->exstg->s_vid.help_cmd },
                { fcgBTAudioEncoderPath->Name,   fcgTXAudioEncoderPath->Text,   sys_dat->exstg->s_aud_ext[fcgCXAudioEncoder->SelectedIndex].cmd_help },
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
            //CONF_GUIEX cnf;
            //FrmToConf(&cnf);
            //auto presetList = nvfeature_GetCachedNVEncCapability(featureCache);
            //memcpy(&cnf.enc.enc_config, &presetList[fcgCXEncCodec->SelectedIndex].presetConfigs[fcgCXQualityPreset->SelectedIndex].presetCfg, sizeof(cnf.enc.enc_config));
            //if (cnf.enc.enc_config.gopLength == UINT32_MAX) {
            //    cnf.enc.enc_config.gopLength = 0;
            //    cnf.enc.enc_config.encodeCodecConfig.h264Config.idrPeriod = 0;
            //    cnf.enc.enc_config.encodeCodecConfig.hevcConfig.idrPeriod = 0;
            //}
            //cnf.enc.codecConfig[fcgCXEncCodec->SelectedIndex] = cnf.enc.enc_config.encodeCodecConfig;
            //cnf.enc.enc_config.encodeCodecConfig.h264Config.sliceModeData = 1;
            //cnf.enc.enc_config.encodeCodecConfig.hevcConfig.sliceMode     = 0;
            //cnf.enc.enc_config.encodeCodecConfig.hevcConfig.sliceModeData = 0;
            //ConfToFrm(&cnf);
        }
    private:
        System::Void fcgTBVppAfsScroll(System::Object^  sender, System::EventArgs^  e) {
            System::Windows::Forms::TrackBar^ senderTB = dynamic_cast<System::Windows::Forms::TrackBar^>(sender);
            if (senderTB == nullptr) return;

            array<TrackBarNU>^ targetList = {
                { fcgTBVppAfsMethodSwitch, fcgNUVppAfsMethodSwitch },
                { fcgTBVppAfsCoeffShift,   fcgNUVppAfsCoeffShift },
                { fcgTBVppAfsThreShift,    fcgNUVppAfsThreShift },
                { fcgTBVppAfsThreDeint,    fcgNUVppAfsThreDeint },
                { fcgTBVppAfsThreYMotion, fcgNUVppAfsThreYMotion },
                { fcgTBVppAfsThreCMotion, fcgNUVppAfsThreCMotion }
            };
            for (int i = 0; i < targetList->Length; i++) {
                if (NULL == String::Compare(senderTB->Name, targetList[i].TB->Name)) {
                    SetNUValue(targetList[i].NU, senderTB->Value);
                    return;
                }
            }
        }
    private:
        System::Void fcgNUVppAfsValueChanged(System::Object^  sender, System::EventArgs^  e) {
            System::Windows::Forms::NumericUpDown^ senderNU = dynamic_cast<System::Windows::Forms::NumericUpDown^>(sender);
            if (senderNU == nullptr) return;

            array<TrackBarNU>^ targetList = {
                { fcgTBVppAfsMethodSwitch, fcgNUVppAfsMethodSwitch },
                { fcgTBVppAfsCoeffShift,   fcgNUVppAfsCoeffShift },
                { fcgTBVppAfsThreShift,    fcgNUVppAfsThreShift },
                { fcgTBVppAfsThreDeint,    fcgNUVppAfsThreDeint },
                { fcgTBVppAfsThreYMotion,  fcgNUVppAfsThreYMotion },
                { fcgTBVppAfsThreCMotion,  fcgNUVppAfsThreCMotion }
            };
            for (int i = 0; i < targetList->Length; i++) {
                if (NULL == String::Compare(senderNU->Name, targetList[i].NU->Name)) {
                    targetList[i].TB->Value = (int)senderNU->Value;
                    return;
                }
            }
        }
    private:
        System::Void fcgTBVppTweakScroll(System::Object^  sender, System::EventArgs^  e) {
            System::Windows::Forms::TrackBar^ senderTB = dynamic_cast<System::Windows::Forms::TrackBar^>(sender);
            if (senderTB == nullptr) return;

            array<TrackBarNU>^ targetList = {
                { fcgTBVppTweakBrightness, fcgNUVppTweakBrightness },
                { fcgTBVppTweakContrast,   fcgNUVppTweakContrast },
                { fcgTBVppTweakGamma,      fcgNUVppTweakGamma },
                { fcgTBVppTweakSaturation, fcgNUVppTweakSaturation },
                { fcgTBVppTweakHue,        fcgNUVppTweakHue }
            };
            for (int i = 0; i < targetList->Length; i++) {
                if (NULL == String::Compare(senderTB->Name, targetList[i].TB->Name)) {
                    SetNUValue(targetList[i].NU, senderTB->Value);
                    return;
                }
            }
        }
    private:
        System::Void fcgNUVppTweakValueChanged(System::Object^  sender, System::EventArgs^  e) {
            System::Windows::Forms::NumericUpDown^ senderNU = dynamic_cast<System::Windows::Forms::NumericUpDown^>(sender);
            if (senderNU == nullptr) return;

            array<TrackBarNU>^ targetList = {
                { fcgTBVppTweakBrightness, fcgNUVppTweakBrightness },
                { fcgTBVppTweakContrast,   fcgNUVppTweakContrast },
                { fcgTBVppTweakGamma,      fcgNUVppTweakGamma },
                { fcgTBVppTweakSaturation, fcgNUVppTweakSaturation },
                { fcgTBVppTweakHue,        fcgNUVppTweakHue }
            };
            for (int i = 0; i < targetList->Length; i++) {
                if (NULL == String::Compare(senderNU->Name, targetList[i].NU->Name)) {
                    targetList[i].TB->Value = (int)senderNU->Value;
                    return;
                }
            }
        }
};
}
