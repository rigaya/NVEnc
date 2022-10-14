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

//以下warning C4100を黙らせる
//C4100 : 引数は関数の本体部で 1 度も参照されません。
#pragma warning( push )
#pragma warning( disable: 4100 )

#include "auo_version.h"
#include "auo_frm.h"
#include "auo_faw2aac.h"
#include "frmConfig.h"
#include "frmSaveNewStg.h"
#include "frmOtherSettings.h"
#include "frmBitrateCalculator.h"

#include "cpu_info.h"
#include "gpu_info.h"
#include "NVEncUtil.h"

using namespace AUO_NAME_R;

/// -------------------------------------------------
///     設定画面の表示
/// -------------------------------------------------
[STAThreadAttribute]
void ShowfrmConfig(CONF_GUIEX *conf, const SYSTEM_DATA *sys_dat) {
    if (!sys_dat->exstg->s_local.disable_visual_styles)
        System::Windows::Forms::Application::EnableVisualStyles();
    System::IO::Directory::SetCurrentDirectory(String(sys_dat->aviutl_dir).ToString());
    frmConfig frmConf(conf, sys_dat);
    frmConf.ShowDialog();
}

/// -------------------------------------------------
///     frmSaveNewStg 関数
/// -------------------------------------------------
System::Boolean frmSaveNewStg::checkStgFileName(String^ stgName) {
    String^ fileName;
    if (stgName->Length == 0)
        return false;

    if (!ValidiateFileName(stgName)) {
        MessageBox::Show(LOAD_CLI_STRING(AUO_CONFIG_ERR_INVALID_CHAR), LOAD_CLI_STRING(AUO_GUIEX_ERROR), MessageBoxButtons::OK, MessageBoxIcon::Error);
        return false;
    }
    if (String::Compare(Path::GetExtension(stgName), L".stg", true))
        stgName += L".stg";
    if (File::Exists(fileName = Path::Combine(fsnCXFolderBrowser->GetSelectedFolder(), stgName)))
        if (MessageBox::Show(stgName + LOAD_CLI_STRING(AUO_CONFIG_ALREADY_EXISTS), LOAD_CLI_STRING(AUO_CONFIG_OVERWRITE_CHECK), MessageBoxButtons::YesNo, MessageBoxIcon::Question)
            != System::Windows::Forms::DialogResult::Yes)
            return false;
    StgFileName = fileName;
    return true;
}

System::Void frmSaveNewStg::setStgDir(String^ _stgDir) {
    StgDir = _stgDir;
    fsnCXFolderBrowser->SetRootDirAndReload(StgDir);
}


/// -------------------------------------------------
///     frmBitrateCalculator 関数
/// -------------------------------------------------
System::Void frmBitrateCalculator::Init(int VideoBitrate, int AudioBitrate, bool BTVBEnable, bool BTABEnable, int ab_max, const AuoTheme themeTo, const DarkenWindowStgReader *dwStg) {
    guiEx_settings exStg(true);
    exStg.load_fbc();
    enable_events = false;
    dwStgReader = dwStg;
    CheckTheme(themeTo);
    fbcTXSize->Text = exStg.s_fbc.initial_size.ToString("F2");
    fbcChangeTimeSetMode(exStg.s_fbc.calc_time_from_frame != 0);
    fbcRBCalcRate->Checked = exStg.s_fbc.calc_bitrate != 0;
    fbcRBCalcSize->Checked = !fbcRBCalcRate->Checked;
    fbcTXMovieFrameRate->Text = Convert::ToString(exStg.s_fbc.last_fps);
    fbcNUMovieFrames->Value = exStg.s_fbc.last_frame_num;
    fbcNULengthHour->Value = Convert::ToDecimal((int)exStg.s_fbc.last_time_in_sec / 3600);
    fbcNULengthMin->Value = Convert::ToDecimal((int)(exStg.s_fbc.last_time_in_sec % 3600) / 60);
    fbcNULengthSec->Value =  Convert::ToDecimal((int)exStg.s_fbc.last_time_in_sec % 60);
    SetBTVBEnabled(BTVBEnable);
    SetBTABEnabled(BTABEnable, ab_max);
    SetNUVideoBitrate(VideoBitrate);
    SetNUAudioBitrate(AudioBitrate);
    enable_events = true;
}
System::Void frmBitrateCalculator::CheckTheme(const AuoTheme themeTo) {
    //変更の必要がなければ終了
    if (themeTo == themeMode) return;

    //一度ウィンドウの再描画を完全に抑止する
    SendMessage(reinterpret_cast<HWND>(this->Handle.ToPointer()), WM_SETREDRAW, 0, 0);
    SetAllColor(this, themeTo, this->GetType(), dwStgReader);
    SetAllMouseMove(this, themeTo);
    //一度ウィンドウの再描画を再開し、強制的に再描画させる
    SendMessage(reinterpret_cast<HWND>(this->Handle.ToPointer()), WM_SETREDRAW, 1, 0);
    this->Refresh();
    themeMode = themeTo;
}
System::Void frmBitrateCalculator::frmBitrateCalculator_FormClosing(System::Object^  sender, System::Windows::Forms::FormClosingEventArgs^  e) {
    guiEx_settings exStg(true);
    exStg.load_fbc();
    exStg.s_fbc.calc_bitrate = fbcRBCalcRate->Checked;
    exStg.s_fbc.calc_time_from_frame = fbcPNMovieFrames->Visible;
    exStg.s_fbc.last_fps = Convert::ToDouble(fbcTXMovieFrameRate->Text);
    exStg.s_fbc.last_frame_num = Convert::ToInt32(fbcNUMovieFrames->Value);
    exStg.s_fbc.last_time_in_sec = Convert::ToInt32(fbcNULengthHour->Value) * 3600
                                 + Convert::ToInt32(fbcNULengthMin->Value) * 60
                                 + Convert::ToInt32(fbcNULengthSec->Value);
    if (fbcRBCalcRate->Checked)
        exStg.s_fbc.initial_size = Convert::ToDouble(fbcTXSize->Text);
    exStg.save_fbc();
    frmConfig^ fcg = dynamic_cast<frmConfig^>(this->Owner);
    if (fcg != nullptr)
        fcg->InformfbcClosed();
}
System::Void frmBitrateCalculator::fbcRBCalcRate_CheckedChanged(System::Object^  sender, System::EventArgs^  e) {
    if (fbcRBCalcRate->Checked && Convert::ToDouble(fbcTXSize->Text) <= 0.0) {
        guiEx_settings exStg(true);
        exStg.load_fbc();
        fbcTXSize->Text = exStg.s_fbc.initial_size.ToString("F2");
    }
}
System::Void frmBitrateCalculator::fbcBTVBApply_Click(System::Object^  sender, System::EventArgs^  e) {
    frmConfig^ fcg = dynamic_cast<frmConfig^>(this->Owner);
    if (fcg != nullptr)
        fcg->SetVideoBitrate((int)fbcNUBitrateVideo->Value);
}
System::Void frmBitrateCalculator::fbcBTABApply_Click(System::Object^  sender, System::EventArgs^  e) {
    frmConfig^ fcg = dynamic_cast<frmConfig^>(this->Owner);
    if (fcg != nullptr)
        fcg->SetAudioBitrate((int)fbcNUBitrateAudio->Value);
}
System::Void frmBitrateCalculator::fbcMouseEnter_SetColor(System::Object^  sender, System::EventArgs^  e) {
    fcgMouseEnterLeave_SetColor(sender, themeMode, DarkenWindowState::Hot, dwStgReader);
}
System::Void frmBitrateCalculator::fbcMouseLeave_SetColor(System::Object^  sender, System::EventArgs^  e) {
    fcgMouseEnterLeave_SetColor(sender, themeMode, DarkenWindowState::Normal, dwStgReader);
}
System::Void frmBitrateCalculator::SetAllMouseMove(Control ^top, const AuoTheme themeTo) {
    if (themeTo == themeMode) return;
    System::Type^ type = top->GetType();
    if (type == CheckBox::typeid) {
        top->MouseEnter += gcnew System::EventHandler(this, &frmBitrateCalculator::fbcMouseEnter_SetColor);
        top->MouseLeave += gcnew System::EventHandler(this, &frmBitrateCalculator::fbcMouseLeave_SetColor);
    }
    for (int i = 0; i < top->Controls->Count; i++) {
        SetAllMouseMove(top->Controls[i], themeTo);
    }
}

/// -------------------------------------------------
///     frmConfig 関数  (frmBitrateCalculator関連)
/// -------------------------------------------------
System::Void frmConfig::CloseBitrateCalc() {
    frmBitrateCalculator::Instance::get()->Close();
}
System::Void frmConfig::fcgTSBBitrateCalc_CheckedChanged(System::Object^  sender, System::EventArgs^  e) {
    if (fcgTSBBitrateCalc->Checked) {
        bool videoBitrateMode = true;

        frmBitrateCalculator::Instance::get()->Init(
            (int)fcgNUBitrate->Value,
            (fcgNUAudioBitrate->Visible) ? (int)fcgNUAudioBitrate->Value : 0,
            videoBitrateMode,
            fcgNUAudioBitrate->Visible,
            (int)fcgNUAudioBitrate->Maximum,
            themeMode,
            dwStgReader
            );
        frmBitrateCalculator::Instance::get()->Owner = this;
        frmBitrateCalculator::Instance::get()->Show();
    } else {
        frmBitrateCalculator::Instance::get()->Close();
    }
}
System::Void frmConfig::SetfbcBTABEnable(bool enable, int max) {
    frmBitrateCalculator::Instance::get()->SetBTABEnabled(fcgNUAudioBitrate->Visible, max);
}
System::Void frmConfig::SetfbcBTVBEnable(bool enable) {
    frmBitrateCalculator::Instance::get()->SetBTVBEnabled(enable);
}

System::Void frmConfig::SetVideoBitrate(int bitrate) {
    SetNUValue(fcgNUBitrate, bitrate);
}

System::Void frmConfig::SetAudioBitrate(int bitrate) {
    SetNUValue(fcgNUAudioBitrate, bitrate);
}
System::Void frmConfig::InformfbcClosed() {
    fcgTSBBitrateCalc->Checked = false;
}


/// -------------------------------------------------
///     frmConfig 関数
/// -------------------------------------------------
/////////////   LocalStg関連  //////////////////////
System::Void frmConfig::LoadLocalStg() {
    guiEx_settings *_ex_stg = sys_dat->exstg;
    _ex_stg->load_encode_stg();
    LocalStg.CustomTmpDir    = String(_ex_stg->s_local.custom_tmp_dir).ToString();
    LocalStg.CustomAudTmpDir = String(_ex_stg->s_local.custom_audio_tmp_dir).ToString();
    LocalStg.CustomMP4TmpDir = String(_ex_stg->s_local.custom_mp4box_tmp_dir).ToString();
    LocalStg.LastAppDir      = String(_ex_stg->s_local.app_dir).ToString();
    LocalStg.LastBatDir      = String(_ex_stg->s_local.bat_dir).ToString();
    LocalStg.vidEncName      = String(_ex_stg->s_vid.filename).ToString();
    LocalStg.vidEncPath      = String(_ex_stg->s_vid.fullpath).ToString();
    LocalStg.MP4MuxerExeName = String(_ex_stg->s_mux[MUXER_MP4].filename).ToString();
    LocalStg.MP4MuxerPath    = String(_ex_stg->s_mux[MUXER_MP4].fullpath).ToString();
    LocalStg.MKVMuxerExeName = String(_ex_stg->s_mux[MUXER_MKV].filename).ToString();
    LocalStg.MKVMuxerPath    = String(_ex_stg->s_mux[MUXER_MKV].fullpath).ToString();
    LocalStg.TC2MP4ExeName   = String(_ex_stg->s_mux[MUXER_TC2MP4].filename).ToString();
    LocalStg.TC2MP4Path      = String(_ex_stg->s_mux[MUXER_TC2MP4].fullpath).ToString();
    LocalStg.MPGMuxerExeName = String(_ex_stg->s_mux[MUXER_MPG].filename).ToString();
    LocalStg.MPGMuxerPath    = String(_ex_stg->s_mux[MUXER_MPG].fullpath).ToString();
    LocalStg.MP4RawExeName   = String(_ex_stg->s_mux[MUXER_MP4_RAW].filename).ToString();
    LocalStg.MP4RawPath      = String(_ex_stg->s_mux[MUXER_MP4_RAW].fullpath).ToString();

    LocalStg.audEncName->Clear();
    LocalStg.audEncExeName->Clear();
    LocalStg.audEncPath->Clear();
    for (int i = 0; i < _ex_stg->s_aud_ext_count; i++) {
        LocalStg.audEncName->Add(String(_ex_stg->s_aud_ext[i].dispname).ToString());
        LocalStg.audEncExeName->Add(String(_ex_stg->s_aud_ext[i].filename).ToString());
        LocalStg.audEncPath->Add(String(_ex_stg->s_aud_ext[i].fullpath).ToString());
    }
    if (_ex_stg->s_local.large_cmdbox)
        fcgTXCmd_DoubleClick(nullptr, nullptr); //初期状態は縮小なので、拡大
}

System::Boolean frmConfig::CheckLocalStg() {
    bool error = false;
    String^ err = "";
    //映像エンコーダのチェック
    if (LocalStg.vidEncPath->Length > 0
        && !File::Exists(LocalStg.vidEncPath)) {
        if (!error) err += L"\n\n";
        error = true;
        err += LOAD_CLI_STRING(AUO_CONFIG_VID_ENC_NOT_EXIST) + L"\n [ " + LocalStg.vidEncPath + L" ]\n";
    }
    //音声エンコーダのチェック (実行ファイル名がない場合はチェックしない)
    if (fcgCBAudioUseExt->Checked
        && LocalStg.audEncExeName[fcgCXAudioEncoder->SelectedIndex]->Length) {
        String^ AudioEncoderPath = LocalStg.audEncPath[fcgCXAudioEncoder->SelectedIndex];
        if (AudioEncoderPath->Length > 0
            && !File::Exists(AudioEncoderPath)
            && (fcgCXAudioEncoder->SelectedIndex != sys_dat->exstg->get_faw_index(!fcgCBAudioUseExt->Checked)
                || !check_if_faw2aac_exists()) ) {
            //音声実行ファイルがない かつ
            //選択された音声がfawでない または fawであってもfaw2aacがない
            if (!error) err += L"\n\n";
            error = true;
            err += LOAD_CLI_STRING(AUO_CONFIG_AUD_ENC_NOT_EXIST) + L"\n [ " + AudioEncoderPath + L" ]\n";
        }
    }
    //FAWのチェック
    if (fcgCBFAWCheck->Checked) {
        if (sys_dat->exstg->get_faw_index(!fcgCBAudioUseExt->Checked) == FAW_INDEX_ERROR) {
            if (!error) err += L"\n\n";
            error = true;
            err += LOAD_CLI_STRING(AUO_CONFIG_FAW_STG_NOT_FOUND_IN_INI1) + L"\n"
                +  LOAD_CLI_STRING(AUO_CONFIG_FAW_STG_NOT_FOUND_IN_INI2) + L"\n"
                +  LOAD_CLI_STRING(AUO_CONFIG_FAW_STG_NOT_FOUND_IN_INI3);
        } else if (fcgCBAudioUseExt->Checked
                   && !File::Exists(LocalStg.audEncPath[sys_dat->exstg->get_faw_index(!fcgCBAudioUseExt->Checked)])
                   && !check_if_faw2aac_exists()) {
            //fawの実行ファイルが存在しない かつ faw2aacも存在しない
            if (!error) err += L"\n\n";
            error = true;
            err += LOAD_CLI_STRING(AUO_CONFIG_FAW_PATH_UNSET1) + L"\n"
                +  LOAD_CLI_STRING(AUO_CONFIG_FAW_PATH_UNSET2);
        }
    }
    if (error)
        MessageBox::Show(this, err, LOAD_CLI_STRING(AUO_GUIEX_ERROR), MessageBoxButtons::OK, MessageBoxIcon::Error);
    return error;
}

System::Void frmConfig::SaveLocalStg() {
    guiEx_settings *_ex_stg = sys_dat->exstg;
    _ex_stg->load_encode_stg();
    _ex_stg->s_local.large_cmdbox = fcgTXCmd->Multiline;
    GetCHARfromString(_ex_stg->s_local.custom_tmp_dir,        sizeof(_ex_stg->s_local.custom_tmp_dir),        LocalStg.CustomTmpDir);
    GetCHARfromString(_ex_stg->s_local.custom_mp4box_tmp_dir, sizeof(_ex_stg->s_local.custom_mp4box_tmp_dir), LocalStg.CustomMP4TmpDir);
    GetCHARfromString(_ex_stg->s_local.custom_audio_tmp_dir,  sizeof(_ex_stg->s_local.custom_audio_tmp_dir),  LocalStg.CustomAudTmpDir);
    GetCHARfromString(_ex_stg->s_local.app_dir,               sizeof(_ex_stg->s_local.app_dir),               LocalStg.LastAppDir);
    GetCHARfromString(_ex_stg->s_local.bat_dir,               sizeof(_ex_stg->s_local.bat_dir),               LocalStg.LastBatDir);
    GetCHARfromString(_ex_stg->s_vid.fullpath,                sizeof(_ex_stg->s_vid.fullpath),                LocalStg.vidEncPath);
    GetCHARfromString(_ex_stg->s_mux[MUXER_MP4].fullpath,     sizeof(_ex_stg->s_mux[MUXER_MP4].fullpath),     LocalStg.MP4MuxerPath);
    GetCHARfromString(_ex_stg->s_mux[MUXER_MKV].fullpath,     sizeof(_ex_stg->s_mux[MUXER_MKV].fullpath),     LocalStg.MKVMuxerPath);
    GetCHARfromString(_ex_stg->s_mux[MUXER_TC2MP4].fullpath,  sizeof(_ex_stg->s_mux[MUXER_TC2MP4].fullpath),  LocalStg.TC2MP4Path);
    GetCHARfromString(_ex_stg->s_mux[MUXER_MPG].fullpath,     sizeof(_ex_stg->s_mux[MUXER_MPG].fullpath),     LocalStg.MPGMuxerPath);
    GetCHARfromString(_ex_stg->s_mux[MUXER_MP4_RAW].fullpath, sizeof(_ex_stg->s_mux[MUXER_MP4_RAW].fullpath), LocalStg.MP4RawPath);
    for (int i = 0; i < _ex_stg->s_aud_ext_count; i++)
        GetCHARfromString(_ex_stg->s_aud_ext[i].fullpath, sizeof(_ex_stg->s_aud_ext[i].fullpath), LocalStg.audEncPath[i]);
    _ex_stg->save_local();
}

System::Void frmConfig::SetLocalStg() {
    fcgTXVideoEncoderPath->Text   = LocalStg.vidEncPath;
    fcgTXMP4MuxerPath->Text       = LocalStg.MP4MuxerPath;
    fcgTXMKVMuxerPath->Text       = LocalStg.MKVMuxerPath;
    fcgTXTC2MP4Path->Text         = LocalStg.TC2MP4Path;
    fcgTXMPGMuxerPath->Text       = LocalStg.MPGMuxerPath;
    fcgTXMP4RawPath->Text         = LocalStg.MP4RawPath;
    fcgTXCustomAudioTempDir->Text = LocalStg.CustomAudTmpDir;
    fcgTXCustomTempDir->Text      = LocalStg.CustomTmpDir;
    fcgTXMP4BoxTempDir->Text      = LocalStg.CustomMP4TmpDir;
    fcgLBVideoEncoderPath->Text   = LocalStg.vidEncName      + LOAD_CLI_STRING(AUO_CONFIG_SPECIFY_EXE_PATH);
    fcgLBMP4MuxerPath->Text       = LocalStg.MP4MuxerExeName + LOAD_CLI_STRING(AUO_CONFIG_SPECIFY_EXE_PATH);
    fcgLBMKVMuxerPath->Text       = LocalStg.MKVMuxerExeName + LOAD_CLI_STRING(AUO_CONFIG_SPECIFY_EXE_PATH);
    fcgLBTC2MP4Path->Text         = LocalStg.TC2MP4ExeName   + LOAD_CLI_STRING(AUO_CONFIG_SPECIFY_EXE_PATH);
    fcgLBMPGMuxerPath->Text       = LocalStg.MPGMuxerExeName + LOAD_CLI_STRING(AUO_CONFIG_SPECIFY_EXE_PATH);
    fcgLBMP4RawPath->Text         = LocalStg.MP4RawExeName   + LOAD_CLI_STRING(AUO_CONFIG_SPECIFY_EXE_PATH);

    fcgTXVideoEncoderPath->SelectionStart = fcgTXVideoEncoderPath->Text->Length;
    fcgTXMP4MuxerPath->SelectionStart     = fcgTXMP4MuxerPath->Text->Length;
    fcgTXTC2MP4Path->SelectionStart       = fcgTXTC2MP4Path->Text->Length;
    fcgTXMKVMuxerPath->SelectionStart     = fcgTXMKVMuxerPath->Text->Length;
    fcgTXMPGMuxerPath->SelectionStart     = fcgTXMPGMuxerPath->Text->Length;
    fcgTXMP4RawPath->SelectionStart       = fcgTXMP4RawPath->Text->Length;
}

//////////////       その他イベント処理   ////////////////////////
System::Void frmConfig::ActivateToolTip(bool Enable) {
    fcgTTEx->Active = Enable;
}

System::Void frmConfig::fcgTSBOtherSettings_Click(System::Object^  sender, System::EventArgs^  e) {
    frmOtherSettings::Instance::get()->stgDir = String(sys_dat->exstg->s_local.stg_dir).ToString();
    frmOtherSettings::Instance::get()->SetTheme(themeMode, dwStgReader);
    frmOtherSettings::Instance::get()->ShowDialog();
    char buf[MAX_PATH_LEN];
    GetCHARfromString(buf, sizeof(buf), frmOtherSettings::Instance::get()->stgDir);
    if (_stricmp(buf, sys_dat->exstg->s_local.stg_dir)) {
        //変更があったら保存する
        strcpy_s(sys_dat->exstg->s_local.stg_dir, sizeof(sys_dat->exstg->s_local.stg_dir), buf);
        sys_dat->exstg->save_local();
        InitStgFileList();
    }
    //再読み込み
    guiEx_settings stg;
    stg.load_encode_stg();
    log_reload_settings();
    sys_dat->exstg->s_local.default_audenc_use_in = stg.s_local.default_audenc_use_in;
    sys_dat->exstg->s_local.default_audio_encoder_ext = stg.s_local.default_audio_encoder_ext;
    sys_dat->exstg->s_local.default_audio_encoder_in = stg.s_local.default_audio_encoder_in;
    sys_dat->exstg->s_local.get_relative_path = stg.s_local.get_relative_path;
    SetStgEscKey(stg.s_local.enable_stg_esc_key != 0);
    ActivateToolTip(stg.s_local.disable_tooltip_help == FALSE);
    if (str_has_char(stg.s_local.conf_font.name))
        SetFontFamilyToForm(this, gcnew FontFamily(String(stg.s_local.conf_font.name).ToString()), this->Font->FontFamily);
}
System::Boolean frmConfig::EnableSettingsNoteChange(bool Enable) {
    if (fcgTSTSettingsNotes->Visible == Enable &&
        fcgTSLSettingsNotes->Visible == !Enable)
        return true;
    if (CountStringBytes(fcgTSTSettingsNotes->Text) > fcgTSTSettingsNotes->MaxLength - 1) {
        MessageBox::Show(this, LOAD_CLI_STRING(AUO_CONFIG_TEXT_LIMIT_LENGTH), LOAD_CLI_STRING(AUO_GUIEX_ERROR), MessageBoxButtons::OK, MessageBoxIcon::Error);
        fcgTSTSettingsNotes->Focus();
        fcgTSTSettingsNotes->SelectionStart = fcgTSTSettingsNotes->Text->Length;
        return false;
    }
    fcgTSTSettingsNotes->Visible = Enable;
    fcgTSLSettingsNotes->Visible = !Enable;
    if (Enable) {
        fcgTSTSettingsNotes->Text = fcgTSLSettingsNotes->Text;
        fcgTSTSettingsNotes->Focus();
        bool isDefaultNote = fcgTSLSettingsNotes->Overflow != ToolStripItemOverflow::Never;
        fcgTSTSettingsNotes->Select((isDefaultNote) ? 0 : fcgTSTSettingsNotes->Text->Length, fcgTSTSettingsNotes->Text->Length);
    } else {
        SetfcgTSLSettingsNotes(fcgTSTSettingsNotes->Text);
        CheckOtherChanges(nullptr, nullptr);
    }
    return true;
}


///////////////////  メモ関連  ///////////////////////////////////////////////
System::Void frmConfig::fcgTSLSettingsNotes_DoubleClick(System::Object^  sender, System::EventArgs^  e) {
    EnableSettingsNoteChange(true);
}
System::Void frmConfig::fcgTSTSettingsNotes_Leave(System::Object^  sender, System::EventArgs^  e) {
    EnableSettingsNoteChange(false);
}
System::Void frmConfig::fcgTSTSettingsNotes_KeyDown(System::Object^  sender, System::Windows::Forms::KeyEventArgs^  e) {
    if (e->KeyCode == Keys::Return)
        EnableSettingsNoteChange(false);
}
System::Void frmConfig::fcgTSTSettingsNotes_TextChanged(System::Object^  sender, System::EventArgs^  e) {
    SetfcgTSLSettingsNotes(fcgTSTSettingsNotes->Text);
    CheckOtherChanges(nullptr, nullptr);
}

/////////////    音声設定関連の関数    ///////////////
System::Void frmConfig::fcgCBAudio2pass_CheckedChanged(System::Object^  sender, System::EventArgs^  e) {
    if (fcgCBAudio2pass->Checked) {
        fcgCBAudioUsePipe->Checked = false;
        fcgCBAudioUsePipe->Enabled = false;
    } else if (CurrentPipeEnabled) {
        fcgCBAudioUsePipe->Checked = true;
        fcgCBAudioUsePipe->Enabled = true;
    }
}

System::Void frmConfig::fcgCXAudioEncoder_SelectedIndexChanged(System::Object ^sender, System::EventArgs ^e) {
    setAudioExtDisplay();
}

System::Void frmConfig::fcgCXAudioEncMode_SelectedIndexChanged(System::Object ^sender, System::EventArgs ^e) {
    AudioExtEncodeModeChanged();
}

System::Int32 frmConfig::GetCurrentAudioDefaultBitrate() {
    AUDIO_SETTINGS *astg = (fcgCBAudioUseExt->Checked) ? &sys_dat->exstg->s_aud_ext[std::max(fcgCXAudioEncoder->SelectedIndex, 0)] : &sys_dat->exstg->s_aud_int[std::max(fcgCXAudioEncoderInternal->SelectedIndex, 0)];
    const int encMode = std::max((fcgCBAudioUseExt->Checked) ? fcgCXAudioEncMode->SelectedIndex : fcgCXAudioEncModeInternal->SelectedIndex, 0);
    return astg->mode[encMode].bitrate_default;
}

System::Void frmConfig::setAudioExtDisplay() {
    AUDIO_SETTINGS *astg = &sys_dat->exstg->s_aud_ext[fcgCXAudioEncoder->SelectedIndex];
    //～の指定
    if (str_has_char(astg->filename)) {
        fcgLBAudioEncoderPath->Text = String(astg->filename).ToString() + LOAD_CLI_STRING(AUO_CONFIG_SPECIFY_EXE_PATH);
        fcgTXAudioEncoderPath->Enabled = true;
        fcgTXAudioEncoderPath->Text = LocalStg.audEncPath[fcgCXAudioEncoder->SelectedIndex];
        fcgBTAudioEncoderPath->Enabled = true;
    } else {
        //filename空文字列(wav出力時)
        fcgLBAudioEncoderPath->Text = L"";
        fcgTXAudioEncoderPath->Enabled = false;
        fcgTXAudioEncoderPath->Text = L"";
        fcgBTAudioEncoderPath->Enabled = false;
    }
    fcgTXAudioEncoderPath->SelectionStart = fcgTXAudioEncoderPath->Text->Length;
    fcgCXAudioEncMode->BeginUpdate();
    fcgCXAudioEncMode->Items->Clear();
    for (int i = 0; i < astg->mode_count; i++) {
        fcgCXAudioEncMode->Items->Add(String(astg->mode[i].name).ToString());
    }
    fcgCXAudioEncMode->EndUpdate();
    bool pipe_enabled = (astg->pipe_input && (!(fcgCBAudio2pass->Checked && astg->mode[fcgCXAudioEncMode->SelectedIndex].enc_2pass != 0)));
    CurrentPipeEnabled = pipe_enabled;
    fcgCBAudioUsePipe->Enabled = pipe_enabled;
    fcgCBAudioUsePipe->Checked = pipe_enabled;
    if (fcgCXAudioEncMode->Items->Count > 0)
        fcgCXAudioEncMode->SelectedIndex = 0;
}

System::Void frmConfig::AudioExtEncodeModeChanged() {
    int index = fcgCXAudioEncMode->SelectedIndex;
    AUDIO_SETTINGS *astg = &sys_dat->exstg->s_aud_ext[fcgCXAudioEncoder->SelectedIndex];
    if (astg->mode[index].bitrate) {
        fcgCXAudioEncMode->Width = fcgCXAudioEncModeSmallWidth;
        fcgLBAudioBitrate->Visible = true;
        fcgNUAudioBitrate->Visible = true;
        fcgNUAudioBitrate->Minimum = astg->mode[index].bitrate_min;
        fcgNUAudioBitrate->Maximum = astg->mode[index].bitrate_max;
        fcgNUAudioBitrate->Increment = astg->mode[index].bitrate_step;
        SetNUValue(fcgNUAudioBitrate, (conf->aud.ext.bitrate != 0) ? conf->aud.ext.bitrate : astg->mode[index].bitrate_default);
    } else {
        fcgCXAudioEncMode->Width = fcgCXAudioEncModeLargeWidth;
        fcgLBAudioBitrate->Visible = false;
        fcgNUAudioBitrate->Visible = false;
        fcgNUAudioBitrate->Minimum = 0;
        fcgNUAudioBitrate->Maximum = 65536;
    }
    fcgCBAudio2pass->Enabled = astg->mode[index].enc_2pass != 0;
    if (!fcgCBAudio2pass->Enabled) fcgCBAudio2pass->Checked = false;
    SetfbcBTABEnable(fcgNUAudioBitrate->Visible, (int)fcgNUAudioBitrate->Maximum);

    bool delay_cut_available = astg->mode[index].delay > 0;
    fcgLBAudioDelayCut->Visible = delay_cut_available;
    fcgCXAudioDelayCut->Visible = delay_cut_available;
    if (delay_cut_available) {
        const bool delay_cut_edts_available = str_has_char(astg->cmd_raw) && str_has_char(sys_dat->exstg->s_mux[MUXER_MP4_RAW].delay_cmd);
        const int current_idx = fcgCXAudioDelayCut->SelectedIndex;
        const int items_to_set = _countof(AUDIO_DELAY_CUT_MODE) - 1 - ((delay_cut_edts_available) ? 0 : 1);
        fcgCXAudioDelayCut->BeginUpdate();
        fcgCXAudioDelayCut->Items->Clear();
        for (int i = 0; i < items_to_set; i++) {
            String^ string = nullptr;
            if (AUDIO_DELAY_CUT_MODE[i].mes != AUO_MES_UNKNOWN) {
                string = LOAD_CLI_STRING(AUDIO_DELAY_CUT_MODE[i].mes);
            }
            if (string == nullptr || string->Length == 0) {
                string = String(AUDIO_DELAY_CUT_MODE[i].desc).ToString();
            }
            fcgCXAudioDelayCut->Items->Add(string);
        }
        fcgCXAudioDelayCut->EndUpdate();
        fcgCXAudioDelayCut->SelectedIndex = (current_idx >= items_to_set) ? 0 : current_idx;
    } else {
        fcgCXAudioDelayCut->SelectedIndex = 0;
    }
}

System::Void frmConfig::fcgCBAudioUseExt_CheckedChanged(System::Object ^sender, System::EventArgs ^e) {
    fcgPNAudioExt->Visible = fcgCBAudioUseExt->Checked;
    fcgPNAudioInternal->Visible = !fcgCBAudioUseExt->Checked;

    //一度ウィンドウの再描画を完全に抑止する
    SendMessage(reinterpret_cast<HWND>(this->Handle.ToPointer()), WM_SETREDRAW, 0, 0);
    //なぜか知らんが、Visibleプロパティをfalseにするだけでは非表示にできない
    //しょうがないので参照の削除と挿入を行う
    fcgtabControlMux->TabPages->Clear();
    if (fcgCBAudioUseExt->Checked) {
        fcgtabControlMux->TabPages->Insert(0, fcgtabPageMP4);
        fcgtabControlMux->TabPages->Insert(1, fcgtabPageMKV);
        fcgtabControlMux->TabPages->Insert(2, fcgtabPageBat);
        fcgtabControlMux->TabPages->Insert(3, fcgtabPageMux);
    } else {
        fcgtabControlMux->TabPages->Insert(0, fcgtabPageInternal);
        fcgtabControlMux->TabPages->Insert(1, fcgtabPageBat);
        fcgtabControlMux->TabPages->Insert(2, fcgtabPageMux);
    }
    //一度ウィンドウの再描画を再開し、強制的に再描画させる
    SendMessage(reinterpret_cast<HWND>(this->Handle.ToPointer()), WM_SETREDRAW, 1, 0);
    this->Refresh();
}

System::Void frmConfig::fcgCXAudioEncoderInternal_SelectedIndexChanged(System::Object ^sender, System::EventArgs ^e) {
    setAudioIntDisplay();
}
System::Void frmConfig::fcgCXAudioEncModeInternal_SelectedIndexChanged(System::Object ^sender, System::EventArgs ^e) {
    AudioIntEncodeModeChanged();
}

System::Void frmConfig::setAudioIntDisplay() {
    AUDIO_SETTINGS *astg = &sys_dat->exstg->s_aud_int[fcgCXAudioEncoderInternal->SelectedIndex];
    fcgCXAudioEncModeInternal->BeginUpdate();
    fcgCXAudioEncModeInternal->Items->Clear();
    for (int i = 0; i < astg->mode_count; i++) {
        fcgCXAudioEncModeInternal->Items->Add(String(astg->mode[i].name).ToString());
    }
    fcgCXAudioEncModeInternal->EndUpdate();
    if (fcgCXAudioEncModeInternal->Items->Count > 0)
        fcgCXAudioEncModeInternal->SelectedIndex = 0;
}
System::Void frmConfig::AudioIntEncodeModeChanged() {
    const int imode = fcgCXAudioEncModeInternal->SelectedIndex;
    if (imode >= 0 && fcgCXAudioEncoderInternal->SelectedIndex >= 0) {
        AUDIO_SETTINGS *astg = &sys_dat->exstg->s_aud_int[fcgCXAudioEncoderInternal->SelectedIndex];
        if (astg->mode[imode].bitrate) {
            fcgCXAudioEncModeInternal->Width = fcgCXAudioEncModeSmallWidth;
            fcgLBAudioBitrateInternal->Visible = true;
            fcgNUAudioBitrateInternal->Visible = true;
            fcgNUAudioBitrateInternal->Minimum = astg->mode[imode].bitrate_min;
            fcgNUAudioBitrateInternal->Maximum = astg->mode[imode].bitrate_max;
            fcgNUAudioBitrateInternal->Increment = astg->mode[imode].bitrate_step;
            SetNUValue(fcgNUAudioBitrateInternal, (conf->aud.in.bitrate > 0) ? conf->aud.in.bitrate : astg->mode[imode].bitrate_default);
        } else {
            fcgCXAudioEncModeInternal->Width = fcgCXAudioEncModeLargeWidth;
            fcgLBAudioBitrateInternal->Visible = false;
            fcgNUAudioBitrateInternal->Visible = false;
            fcgNUAudioBitrateInternal->Minimum = 0;
            fcgNUAudioBitrateInternal->Maximum = 65536;
        }
    }
    SetfbcBTABEnable(fcgNUAudioBitrateInternal->Visible, (int)fcgNUAudioBitrateInternal->Maximum);
}

///////////////   設定ファイル関連   //////////////////////
System::Void frmConfig::CheckTSItemsEnabled(CONF_GUIEX *current_conf) {
    bool selected = (CheckedStgMenuItem != nullptr);
    fcgTSBSave->Enabled = (selected && memcmp(cnf_stgSelected, current_conf, sizeof(CONF_GUIEX)));
    fcgTSBDelete->Enabled = selected;
}

System::Void frmConfig::UncheckAllDropDownItem(ToolStripItem^ mItem) {
    ToolStripDropDownItem^ DropDownItem = dynamic_cast<ToolStripDropDownItem^>(mItem);
    if (DropDownItem == nullptr)
        return;
    for (int i = 0; i < DropDownItem->DropDownItems->Count; i++) {
        UncheckAllDropDownItem(DropDownItem->DropDownItems[i]);
        ToolStripMenuItem^ item = dynamic_cast<ToolStripMenuItem^>(DropDownItem->DropDownItems[i]);
        if (item != nullptr)
            item->Checked = false;
    }
}

System::Void frmConfig::CheckTSSettingsDropDownItem(ToolStripMenuItem^ mItem) {
    UncheckAllDropDownItem(fcgTSSettings);
    CheckedStgMenuItem = mItem;
    fcgTSSettings->Text = (mItem == nullptr) ? LOAD_CLI_STRING(AUO_CONFIG_PROFILE) : mItem->Text;
    if (mItem != nullptr)
        mItem->Checked = true;
    fcgTSBSave->Enabled = false;
    fcgTSBDelete->Enabled = (mItem != nullptr);
}

ToolStripMenuItem^ frmConfig::fcgTSSettingsSearchItem(String^ stgPath, ToolStripItem^ mItem) {
    if (stgPath == nullptr)
        return nullptr;
    ToolStripDropDownItem^ DropDownItem = dynamic_cast<ToolStripDropDownItem^>(mItem);
    if (DropDownItem == nullptr)
        return nullptr;
    for (int i = 0; i < DropDownItem->DropDownItems->Count; i++) {
        ToolStripMenuItem^ item = fcgTSSettingsSearchItem(stgPath, DropDownItem->DropDownItems[i]);
        if (item != nullptr)
            return item;
        item = dynamic_cast<ToolStripMenuItem^>(DropDownItem->DropDownItems[i]);
        if (item      != nullptr &&
            item->Tag != nullptr &&
            0 == String::Compare(item->Tag->ToString(), stgPath, true))
            return item;
    }
    return nullptr;
}

ToolStripMenuItem^ frmConfig::fcgTSSettingsSearchItem(String^ stgPath) {
    return fcgTSSettingsSearchItem((stgPath != nullptr && stgPath->Length > 0) ? Path::GetFullPath(stgPath) : nullptr, fcgTSSettings);
}

System::Void frmConfig::SaveToStgFile(String^ stgName) {
    size_t nameLen = CountStringBytes(stgName) + 1;
    char *stg_name = (char *)malloc(nameLen);
    GetCHARfromString(stg_name, nameLen, stgName);
    init_CONF_GUIEX(cnf_stgSelected, FALSE);
    FrmToConf(cnf_stgSelected);
    String^ stgDir = Path::GetDirectoryName(stgName);
    if (!Directory::Exists(stgDir))
        Directory::CreateDirectory(stgDir);
    int result = guiEx_config::save_guiEx_conf(cnf_stgSelected, stg_name);
    free(stg_name);
    switch (result) {
        case CONF_ERROR_FILE_OPEN:
            MessageBox::Show(LOAD_CLI_STRING(AUO_CONFIG_ERR_OPEN_STG_FILE), LOAD_CLI_STRING(AUO_GUIEX_ERROR), MessageBoxButtons::OK, MessageBoxIcon::Error);
            return;
        case CONF_ERROR_INVALID_FILENAME:
            MessageBox::Show(LOAD_CLI_STRING(AUO_CONFIG_ERR_INVALID_CHAR), LOAD_CLI_STRING(AUO_GUIEX_ERROR), MessageBoxButtons::OK, MessageBoxIcon::Error);
            return;
        default:
            break;
    }
    init_CONF_GUIEX(cnf_stgSelected, FALSE);
    FrmToConf(cnf_stgSelected);
}

System::Void frmConfig::fcgTSBSave_Click(System::Object^  sender, System::EventArgs^  e) {
    if (CheckedStgMenuItem != nullptr)
        SaveToStgFile(CheckedStgMenuItem->Tag->ToString());
    CheckTSSettingsDropDownItem(CheckedStgMenuItem);
}

System::Void frmConfig::fcgTSBSaveNew_Click(System::Object^  sender, System::EventArgs^  e) {
    frmSaveNewStg::Instance::get()->setStgDir(String(sys_dat->exstg->s_local.stg_dir).ToString());
    frmSaveNewStg::Instance::get()->SetTheme(themeMode, dwStgReader);
    if (CheckedStgMenuItem != nullptr)
        frmSaveNewStg::Instance::get()->setFilename(CheckedStgMenuItem->Text);
    frmSaveNewStg::Instance::get()->ShowDialog();
    String^ stgName = frmSaveNewStg::Instance::get()->StgFileName;
    if (stgName != nullptr && stgName->Length)
        SaveToStgFile(stgName);
    RebuildStgFileDropDown(nullptr);
    CheckTSSettingsDropDownItem(fcgTSSettingsSearchItem(stgName));
}

System::Void frmConfig::DeleteStgFile(ToolStripMenuItem^ mItem) {
    if (System::Windows::Forms::DialogResult::OK ==
        MessageBox::Show(LOAD_CLI_STRING(AUO_CONFIG_ASK_STG_FILE_DELETE) + L"[" + mItem->Text + L"]",
        LOAD_CLI_STRING(AUO_GUIEX_ERROR), MessageBoxButtons::OKCancel, MessageBoxIcon::Exclamation))
    {
        File::Delete(mItem->Tag->ToString());
        RebuildStgFileDropDown(nullptr);
        CheckTSSettingsDropDownItem(nullptr);
        SetfcgTSLSettingsNotes(L"");
    }
}

System::Void frmConfig::fcgTSBDelete_Click(System::Object^  sender, System::EventArgs^  e) {
    DeleteStgFile(CheckedStgMenuItem);
}

System::Void frmConfig::fcgTSSettings_DropDownItemClicked(System::Object^  sender, System::Windows::Forms::ToolStripItemClickedEventArgs^  e) {
    ToolStripMenuItem^ ClickedMenuItem = dynamic_cast<ToolStripMenuItem^>(e->ClickedItem);
    if (ClickedMenuItem == nullptr)
        return;
    if (ClickedMenuItem->Tag == nullptr || ClickedMenuItem->Tag->ToString()->Length == 0)
        return;
    CONF_GUIEX load_stg;
    char stg_path[MAX_PATH_LEN];
    GetCHARfromString(stg_path, sizeof(stg_path), ClickedMenuItem->Tag->ToString());
    if (guiEx_config::load_guiEx_conf(&load_stg, stg_path) == CONF_ERROR_FILE_OPEN) {
        if (MessageBox::Show(LOAD_CLI_STRING(AUO_CONFIG_ERR_OPEN_STG_FILE) + L"\n"
                           + LOAD_CLI_STRING(AUO_CONFIG_ASK_STG_FILE_DELETE),
                           LOAD_CLI_STRING(AUO_GUIEX_ERROR), MessageBoxButtons::YesNo, MessageBoxIcon::Error)
                           == System::Windows::Forms::DialogResult::Yes)
            DeleteStgFile(ClickedMenuItem);
        return;
    }
    ConfToFrm(&load_stg);
    CheckTSSettingsDropDownItem(ClickedMenuItem);
    memcpy(cnf_stgSelected, &load_stg, sizeof(CONF_GUIEX));
}

System::Void frmConfig::RebuildStgFileDropDown(ToolStripDropDownItem^ TS, String^ dir) {
    array<String^>^ subDirs = Directory::GetDirectories(dir);
    for (int i = 0; i < subDirs->Length; i++) {
        ToolStripMenuItem^ DDItem = gcnew ToolStripMenuItem(L"[ " + subDirs[i]->Substring(dir->Length+1) + L" ]");
        DDItem->DropDownItemClicked += gcnew System::Windows::Forms::ToolStripItemClickedEventHandler(this, &frmConfig::fcgTSSettings_DropDownItemClicked);
        DDItem->ForeColor = Color::Blue;
        DDItem->Tag = nullptr;
        RebuildStgFileDropDown(DDItem, subDirs[i]);
        TS->DropDownItems->Add(DDItem);
    }
    array<String^>^ stgList = Directory::GetFiles(dir, L"*.stg");
    for (int i = 0; i < stgList->Length; i++) {
        ToolStripMenuItem^ mItem = gcnew ToolStripMenuItem(Path::GetFileNameWithoutExtension(stgList[i]));
        mItem->Tag = stgList[i];
        TS->DropDownItems->Add(mItem);
    }
}

System::Void frmConfig::RebuildStgFileDropDown(String^ stgDir) {
    fcgTSSettings->DropDownItems->Clear();
    if (stgDir != nullptr)
        CurrentStgDir = stgDir;
    if (!Directory::Exists(CurrentStgDir))
        Directory::CreateDirectory(CurrentStgDir);
    RebuildStgFileDropDown(fcgTSSettings, Path::GetFullPath(CurrentStgDir));
}

///////////////   言語ファイル関連   //////////////////////

System::Void frmConfig::CheckTSLanguageDropDownItem(ToolStripMenuItem^ mItem) {
    UncheckAllDropDownItem(fcgTSLanguage);
    fcgTSLanguage->Text = (mItem == nullptr) ? LOAD_CLI_STRING(AuofcgTSSettings) : mItem->Text;
    if (mItem != nullptr)
        mItem->Checked = true;
}
System::Void frmConfig::SetSelectedLanguage(const char *language_text) {
    for (int i = 0; i < fcgTSLanguage->DropDownItems->Count; i++) {
        ToolStripMenuItem^ item = dynamic_cast<ToolStripMenuItem^>(fcgTSLanguage->DropDownItems[i]);
        char item_text[MAX_PATH_LEN];
        GetCHARfromString(item_text, sizeof(item_text), item->Tag->ToString());
        if (strncmp(item_text, language_text, strlen(language_text)) == 0) {
            CheckTSLanguageDropDownItem(item);
            break;
        }
    }
}

System::Void frmConfig::SaveSelectedLanguage(const char *language_text) {
    sys_dat->exstg->set_and_save_lang(language_text);
}

System::Void frmConfig::fcgTSLanguage_DropDownItemClicked(System::Object^  sender, System::Windows::Forms::ToolStripItemClickedEventArgs^  e) {
    ToolStripMenuItem^ ClickedMenuItem = dynamic_cast<ToolStripMenuItem^>(e->ClickedItem);
    if (ClickedMenuItem == nullptr)
        return;
    if (ClickedMenuItem->Tag == nullptr || ClickedMenuItem->Tag->ToString()->Length == 0)
        return;

    char language_text[MAX_PATH_LEN];
    GetCHARfromString(language_text, sizeof(language_text), ClickedMenuItem->Tag->ToString());
    SaveSelectedLanguage(language_text);
    load_lng(language_text);
    overwrite_aviutl_ini_auo_info();
    LoadLangText();
    CheckTSLanguageDropDownItem(ClickedMenuItem);
}

System::Void frmConfig::InitLangList() {
    if (list_lng != nullptr) {
        delete list_lng;
    }
#define ENABLE_LNG_FILE_DETECT 1
#if ENABLE_LNG_FILE_DETECT
    auto lnglist = find_lng_files();
    list_lng = new std::vector<std::string>();
    for (const auto& lang : lnglist) {
        list_lng->push_back(lang);
    }
#endif

    fcgTSLanguage->DropDownItems->Clear();

    for (const auto& auo_lang : list_auo_languages) {
        String^ label = String(auo_lang.code).ToString() + L" (" + String(auo_lang.name).ToString() + L")";
        ToolStripMenuItem^ mItem = gcnew ToolStripMenuItem(label);
        mItem->DropDownItemClicked += gcnew System::Windows::Forms::ToolStripItemClickedEventHandler(this, &frmConfig::fcgTSLanguage_DropDownItemClicked);
        mItem->Tag = String(auo_lang.code).ToString();
        fcgTSLanguage->DropDownItems->Add(mItem);
    }
#if ENABLE_LNG_FILE_DETECT
    for (size_t i = 0; i < list_lng->size(); i++) {
        auto filename = String(PathFindFileNameA((*list_lng)[i].c_str())).ToString();
        ToolStripMenuItem^ mItem = gcnew ToolStripMenuItem(filename);
        mItem->DropDownItemClicked += gcnew System::Windows::Forms::ToolStripItemClickedEventHandler(this, &frmConfig::fcgTSLanguage_DropDownItemClicked);
        mItem->Tag = filename;
        fcgTSLanguage->DropDownItems->Add(mItem);
    }
#endif
    SetSelectedLanguage(sys_dat->exstg->get_lang());
}

//////////////   初期化関連     ////////////////
System::Void frmConfig::InitData(CONF_GUIEX *set_config, const SYSTEM_DATA *system_data) {
    if (set_config->size_all != CONF_INITIALIZED) {
        //初期化されていなければ初期化する
        init_CONF_GUIEX(set_config, FALSE);
    }
    conf = set_config;
    sys_dat = system_data;
}

System::Void frmConfig::InitComboBox() {
    //コンボボックスに値を設定する
    setComboBox(fcgCXEncCodec,          list_nvenc_codecs);
    setComboBox(fcgCXEncMode,           list_encmode);
    setComboBox(fcgCXQualityPreset,     list_nvenc_preset_names_ver10, 3);
    setComboBox(fcgCXMultiPass,         list_nvenc_multipass_mode);
    setComboBox(fcgCXCodecLevel,        list_avc_level);
    setComboBox(fcgCXCodecProfile,      h264_profile_names);
    setComboBox(fcgCXInterlaced,        list_interlaced, "auto");
    setComboBox(fcgCXAspectRatio,       aspect_desc);
    setComboBox(fcgCXAdaptiveTransform, list_adapt_transform);
    setComboBox(fcgCXBDirectMode,       list_bdirect);
    setComboBox(fcgCXMVPrecision,       list_mv_presicion);
    setComboBox(fcgCXVideoFormat,       list_videoformat, "auto");
    setComboBox(fcgCXTransfer,          list_transfer, "auto");
    setComboBox(fcgCXColorMatrix,       list_colormatrix, "auto");
    setComboBox(fcgCXColorPrim,         list_colorprim, "auto");
    setComboBox(fcgCXHEVCTier,          h265_profile_names);
    setComboBox(fxgCXHEVCLevel,         list_hevc_level);
    setComboBox(fcgCXHEVCMaxCUSize,     list_hevc_cu_size);
    setComboBox(fcgCXHEVCMinCUSize,     list_hevc_cu_size);
    setComboBox(fcgCXHEVCOutBitDepth,   list_bitdepth);
    setComboBox(fcgCXCodecProfileAV1,   av1_profile_names);
    setComboBox(fcgCXCodecLevelAV1,     list_av1_level);
    setComboBox(fcgCXAQ,                list_aq);
    setComboBox(fcgCXBrefMode,          list_bref_mode);
    setComboBox(fcgCXCudaSchdule,       list_cuda_schedule);
    setComboBox(fcgCXVppResizeAlg,      list_vpp_resize);
    setComboBox(fcgCXVppDenoiseMethod,  list_vpp_denoise);
    setComboBox(fcgCXVppDetailEnhance,  list_vpp_detail_enahance);
    setComboBox(fcgCXVppDebandSample,   list_vpp_deband_gui);
    setComboBox(fcgCXVppDeinterlace,    list_deinterlace_gui);
    setComboBox(fcgCXVppDenoiseConv3DMatrix,   list_vpp_convolution3d_matrix);
    setComboBox(fcgCXVppAfsAnalyze,     list_vpp_afs_analyze);
    setComboBox(fcgCXVppNnediNsize,     list_vpp_nnedi_nsize);
    setComboBox(fcgCXVppNnediNns,       list_vpp_nnedi_nns);
    setComboBox(fcgCXVppNnediQual,      list_vpp_nnedi_quality);
    setComboBox(fcgCXVppNnediPrec,      list_vpp_fp_prec);
    setComboBox(fcgCXVppNnediPrescreen, list_vpp_nnedi_pre_screen_gui);
    setComboBox(fcgCXVppNnediErrorType, list_vpp_nnedi_error_type);
    setComboBox(fcgCXVppYadifMode,      list_vpp_yadif_mode_gui);

    setComboBox(fcgCXAudioTempDir,  audtempdir_desc);
    setComboBox(fcgCXMP4BoxTempDir, mp4boxtempdir_desc);
    setComboBox(fcgCXTempDir,       tempdir_desc);

    setComboBox(fcgCXAudioEncTiming, audio_enc_timing_desc);
    setComboBox(fcgCXAudioDelayCut,  AUDIO_DELAY_CUT_MODE);

    setMuxerCmdExNames(fcgCXMP4CmdEx, MUXER_MP4);
    setMuxerCmdExNames(fcgCXMKVCmdEx, MUXER_MKV);
    setMuxerCmdExNames(fcgCXInternalCmdEx, MUXER_INTERNAL);
#ifdef HIDE_MPEG2
    fcgCXMPGCmdEx->Items->Clear();
    fcgCXMPGCmdEx->Items->Add("");
#else
    setMuxerCmdExNames(fcgCXMPGCmdEx, MUXER_MPG);
#endif
    fcgCXDevice->Items->Clear();
    fcgCXDevice->Items->Add(String(_T("Auto")).ToString());
    //適当に追加しておく
    //あとでデバイス情報を追加してから修正する
    for (int i = 0; i < 8; i++) {
        fcgCXDevice->Items->Add(_T("------"));
    }

    setAudioEncoderNames();

    setPriorityList(fcgCXMuxPriority);
    setPriorityList(fcgCXAudioPriority);
}

System::Void frmConfig::SetTXMaxLen(TextBox^ TX, int max_len) {
    TX->MaxLength = max_len;
    TX->Validating += gcnew System::ComponentModel::CancelEventHandler(this, &frmConfig::TX_LimitbyBytes);
}

System::Void frmConfig::SetTXMaxLenAll() {
    //MaxLengthに最大文字数をセットし、それをもとにバイト数計算を行うイベントをセットする。
    SetTXMaxLen(fcgTXVideoEncoderPath,   sizeof(sys_dat->exstg->s_vid.fullpath) - 1);
    SetTXMaxLen(fcgTXCmdEx,              sizeof(CONF_ENC::cmdex) - 1);
    SetTXMaxLen(fcgTXAudioEncoderPath,   sizeof(sys_dat->exstg->s_aud_ext[0].fullpath) - 1);
    SetTXMaxLen(fcgTXMP4MuxerPath,       sizeof(sys_dat->exstg->s_mux[MUXER_MP4].fullpath) - 1);
    SetTXMaxLen(fcgTXMKVMuxerPath,       sizeof(sys_dat->exstg->s_mux[MUXER_MKV].fullpath) - 1);
    SetTXMaxLen(fcgTXTC2MP4Path,         sizeof(sys_dat->exstg->s_mux[MUXER_TC2MP4].fullpath) - 1);
    SetTXMaxLen(fcgTXMPGMuxerPath,       sizeof(sys_dat->exstg->s_mux[MUXER_MPG].fullpath) - 1);
    SetTXMaxLen(fcgTXMP4RawPath,         sizeof(sys_dat->exstg->s_mux[MUXER_MP4_RAW].fullpath) - 1);
    SetTXMaxLen(fcgTXCustomTempDir,      sizeof(sys_dat->exstg->s_local.custom_tmp_dir) - 1);
    SetTXMaxLen(fcgTXCustomAudioTempDir, sizeof(sys_dat->exstg->s_local.custom_audio_tmp_dir) - 1);
    SetTXMaxLen(fcgTXMP4BoxTempDir,      sizeof(sys_dat->exstg->s_local.custom_mp4box_tmp_dir) - 1);
    SetTXMaxLen(fcgTXBatBeforeAudioPath, sizeof(conf->oth.batfile.before_audio) - 1);
    SetTXMaxLen(fcgTXBatAfterAudioPath,  sizeof(conf->oth.batfile.after_audio) - 1);
    SetTXMaxLen(fcgTXBatBeforePath,      sizeof(conf->oth.batfile.before_process) - 1);
    SetTXMaxLen(fcgTXBatAfterPath,       sizeof(conf->oth.batfile.after_process) - 1);
    fcgTSTSettingsNotes->MaxLength     = sizeof(conf->oth.notes) - 1;
}

System::Void frmConfig::InitStgFileList() {
    RebuildStgFileDropDown(String(sys_dat->exstg->s_local.stg_dir).ToString());
    stgChanged = false;
    CheckTSSettingsDropDownItem(nullptr);
}

System::Void frmConfig::fcgChangeEnabled(System::Object^  sender, System::EventArgs^  e) {
    const bool h264_mode = fcgCXEncCodec->SelectedIndex == NV_ENC_H264;
    const bool hevc_mode = fcgCXEncCodec->SelectedIndex == NV_ENC_HEVC;
    const bool av1_mode  = fcgCXEncCodec->SelectedIndex == NV_ENC_AV1;

    bool qvbr_mode = false;
    int vce_rc_method = list_encmode[fcgCXEncMode->SelectedIndex].value;
    if (vce_rc_method == NV_ENC_PARAMS_RC_QVBR) {
        vce_rc_method = NV_ENC_PARAMS_RC_VBR;
        qvbr_mode = true;
    }
    bool cqp_mode = (vce_rc_method == NV_ENC_PARAMS_RC_CONSTQP);
    bool cbr_vbr_mode = (vce_rc_method == NV_ENC_PARAMS_RC_CBR || vce_rc_method == NV_ENC_PARAMS_RC_CBR_HQ || vce_rc_method == NV_ENC_PARAMS_RC_VBR || vce_rc_method == NV_ENC_PARAMS_RC_VBR_HQ);

    this->SuspendLayout();

    fcgPNH264->Visible = h264_mode;
    fcgPNHEVC->Visible = hevc_mode;
    fcgPNAV1->Visible = av1_mode;
    fcgPNH264Detail->Visible = h264_mode;
    fcgPNHEVCDetail->Visible = hevc_mode;

    fcgPNQP->Visible = cqp_mode;
    fcgNUQPI->Enabled = cqp_mode;
    fcgNUQPP->Enabled = cqp_mode;
    fcgNUQPB->Enabled = cqp_mode;
    fcgLBQPI->Enabled = cqp_mode;
    fcgLBQPP->Enabled = cqp_mode;
    fcgLBQPB->Enabled = cqp_mode;

    fcgPNBitrate->Visible = !cqp_mode;
    fcgNUBitrate->Enabled = !cqp_mode && !qvbr_mode;
    fcgLBBitrate->Enabled = !cqp_mode && !qvbr_mode;
    fcgNUMaxkbps->Enabled = cbr_vbr_mode;
    fcgLBMaxkbps->Enabled = cbr_vbr_mode;
    fcgLBMaxBitrate2->Enabled = cbr_vbr_mode;

    if (qvbr_mode) {
        fcgNUBitrate->Value = 0;
    }

    fcggroupBoxResize->Enabled = fcgCBVppResize->Checked;
    fcgPNVppDenoiseKnn->Visible = (fcgCXVppDenoiseMethod->SelectedIndex == get_cx_index(list_vpp_denoise, _T("knn")));
    fcgPNVppDenoisePmd->Visible = (fcgCXVppDenoiseMethod->SelectedIndex == get_cx_index(list_vpp_denoise, _T("pmd")));
    fcgPNVppDenoiseSmooth->Visible = (fcgCXVppDenoiseMethod->SelectedIndex == get_cx_index(list_vpp_denoise, _T("smooth")));
    fcgPNVppDenoiseConv3D->Visible = (fcgCXVppDenoiseMethod->SelectedIndex == get_cx_index(list_vpp_denoise, _T("convolution3d")));
    fcgPNVppUnsharp->Visible    = (fcgCXVppDetailEnhance->SelectedIndex == get_cx_index(list_vpp_detail_enahance, _T("unsharp")));
    fcgPNVppEdgelevel->Visible  = (fcgCXVppDetailEnhance->SelectedIndex == get_cx_index(list_vpp_detail_enahance, _T("edgelevel")));
    fcgPNVppWarpsharp->Visible = (fcgCXVppDetailEnhance->SelectedIndex == get_cx_index(list_vpp_detail_enahance, _T("warpsharp")));
    fcggroupBoxVppDeband->Enabled = fcgCBVppDebandEnable->Checked;
    fcgPNVppAfs->Visible = (fcgCXVppDeinterlace->SelectedIndex == get_cx_index(list_deinterlace_gui, L"自動フィールドシフト"));
    fcgPNVppNnedi->Visible = (fcgCXVppDeinterlace->SelectedIndex == get_cx_index(list_deinterlace_gui, L"nnedi"));
    fcgPNVppYadif->Visible = (fcgCXVppDeinterlace->SelectedIndex == get_cx_index(list_deinterlace_gui, L"yadif"));
    fcggroupBoxVppTweak->Enabled = fcgCBVppTweakEnable->Checked;

    this->ResumeLayout();
    this->PerformLayout();
}

System::Void frmConfig::fcgCodecChanged(System::Object^  sender, System::EventArgs^  e) {
    const bool h264_mode = fcgCXEncCodec->SelectedIndex == NV_ENC_H264;
    const bool hevc_mode = fcgCXEncCodec->SelectedIndex == NV_ENC_HEVC;
    const bool av1_mode  = fcgCXEncCodec->SelectedIndex == NV_ENC_AV1;
    if (hevc_mode && !nvencInfo.hevcEnc) {
        fcgCXEncCodec->SelectedIndex = NV_ENC_H264;
    }
    if (av1_mode && !nvencInfo.av1Enc) {
        fcgCXEncCodec->SelectedIndex = NV_ENC_H264;
    }
}

System::Void frmConfig::fcgChangeMuxerVisible(System::Object^  sender, System::EventArgs^  e) {
    //tc2mp4のチェック
    const bool enable_tc2mp4_muxer = (0 != str_has_char(sys_dat->exstg->s_mux[MUXER_TC2MP4].base_cmd));
    fcgTXTC2MP4Path->Visible = enable_tc2mp4_muxer;
    fcgLBTC2MP4Path->Visible = enable_tc2mp4_muxer;
    fcgBTTC2MP4Path->Visible = enable_tc2mp4_muxer;
    //mp4 rawのチェック
    const bool enable_mp4raw_muxer = (0 != str_has_char(sys_dat->exstg->s_mux[MUXER_MP4_RAW].base_cmd));
    fcgTXMP4RawPath->Visible = enable_mp4raw_muxer;
    fcgLBMP4RawPath->Visible = enable_mp4raw_muxer;
    fcgBTMP4RawPath->Visible = enable_mp4raw_muxer;
    //一時フォルダのチェック
    const bool enable_mp4_tmp = (0 != str_has_char(sys_dat->exstg->s_mux[MUXER_MP4].tmp_cmd));
    fcgCXMP4BoxTempDir->Visible = enable_mp4_tmp;
    fcgLBMP4BoxTempDir->Visible = enable_mp4_tmp;
    fcgTXMP4BoxTempDir->Visible = enable_mp4_tmp;
    fcgBTMP4BoxTempDir->Visible = enable_mp4_tmp;
    //Apple Chapterのチェック
    bool enable_mp4_apple_cmdex = false;
    for (int i = 0; i < sys_dat->exstg->s_mux[MUXER_MP4].ex_count; i++)
        enable_mp4_apple_cmdex |= (0 != str_has_char(sys_dat->exstg->s_mux[MUXER_MP4].ex_cmd[i].cmd_apple));
    fcgCBMP4MuxApple->Visible = enable_mp4_apple_cmdex;

    //位置の調整
    static const int HEIGHT = 31;
    fcgLBTC2MP4Path->Location = Point(fcgLBTC2MP4Path->Location.X, fcgLBMP4MuxerPath->Location.Y + HEIGHT * enable_tc2mp4_muxer);
    fcgTXTC2MP4Path->Location = Point(fcgTXTC2MP4Path->Location.X, fcgTXMP4MuxerPath->Location.Y + HEIGHT * enable_tc2mp4_muxer);
    fcgBTTC2MP4Path->Location = Point(fcgBTTC2MP4Path->Location.X, fcgBTMP4MuxerPath->Location.Y + HEIGHT * enable_tc2mp4_muxer);
    fcgLBMP4RawPath->Location = Point(fcgLBMP4RawPath->Location.X, fcgLBTC2MP4Path->Location.Y   + HEIGHT * enable_mp4raw_muxer);
    fcgTXMP4RawPath->Location = Point(fcgTXMP4RawPath->Location.X, fcgTXTC2MP4Path->Location.Y   + HEIGHT * enable_mp4raw_muxer);
    fcgBTMP4RawPath->Location = Point(fcgBTMP4RawPath->Location.X, fcgBTTC2MP4Path->Location.Y   + HEIGHT * enable_mp4raw_muxer);
}

System::Void frmConfig::SetStgEscKey(bool Enable) {
    if (this->KeyPreview == Enable)
        return;
    this->KeyPreview = Enable;
    if (Enable)
        this->KeyDown += gcnew System::Windows::Forms::KeyEventHandler(this, &frmConfig::frmConfig_KeyDown);
    else
        this->KeyDown -= gcnew System::Windows::Forms::KeyEventHandler(this, &frmConfig::frmConfig_KeyDown);
}

System::Void frmConfig::AdjustLocation() {
    //デスクトップ領域(タスクバー等除く)
    System::Drawing::Rectangle screen = System::Windows::Forms::Screen::GetWorkingArea(this);
    //現在のデスクトップ領域の座標
    Point CurrentDesktopLocation = this->DesktopLocation::get();
    //チェック開始
    bool ChangeLocation = false;
    if (CurrentDesktopLocation.X + this->Size.Width > screen.Width) {
        ChangeLocation = true;
        CurrentDesktopLocation.X = clamp(screen.X - this->Size.Width, 4, CurrentDesktopLocation.X);
    }
    if (CurrentDesktopLocation.Y + this->Size.Height > screen.Height) {
        ChangeLocation = true;
        CurrentDesktopLocation.Y = clamp(screen.Y - this->Size.Height, 4, CurrentDesktopLocation.Y);
    }
    if (ChangeLocation) {
        this->StartPosition = FormStartPosition::Manual;
        this->DesktopLocation::set(CurrentDesktopLocation);
    }
}

System::Void frmConfig::InitForm() {
    //UIテーマ切り替え
    CheckTheme();
    //言語設定ファイルのロード
    InitLangList();
    fcgPBNVEncLogoEnabled->Visible = false;
    GetVidEncInfoAsync();
    //設定ファイル集の初期化
    InitStgFileList();
    //言語表示
    LoadLangText();
    //パラメータセット
    ConfToFrm(conf);
    //イベントセット
    SetTXMaxLenAll(); //テキストボックスの最大文字数
    SetAllCheckChangedEvents(this); //変更の確認,ついでにNUのEnterEvent
    //フォームの変更可不可を更新
    fcgChangeMuxerVisible(nullptr, nullptr);
    fcgChangeEnabled(nullptr, nullptr);
    EnableSettingsNoteChange(false);
    fcgCBAudioUseExt_CheckedChanged(nullptr, nullptr);
    //表示位置の調整
    AdjustLocation();
    //キー設定
    SetStgEscKey(sys_dat->exstg->s_local.enable_stg_esc_key != 0);
    //フォントの設定
    if (str_has_char(sys_dat->exstg->s_local.conf_font.name))
        SetFontFamilyToForm(this, gcnew FontFamily(String(sys_dat->exstg->s_local.conf_font.name).ToString()), this->Font->FontFamily);
}



System::Void frmConfig::LoadLangText() {
    //一度ウィンドウの再描画を完全に抑止する
    SendMessage(reinterpret_cast<HWND>(this->Handle.ToPointer()), WM_SETREDRAW, 0, 0);
    //空白時にグレーで入れる文字列を言語変更のため一度空白に戻す
    ExeTXPathEnter();
    //言語更新開始
    LOAD_CLI_TEXT(fcgtoolStripSettings);
    LOAD_CLI_TEXT(fcgTSBSave);
    LOAD_CLI_TEXT(fcgTSBSaveNew);
    LOAD_CLI_TEXT(fcgTSBDelete);
    LOAD_CLI_TEXT(fcgTSSettings);
    LOAD_CLI_TEXT(fcgTSBBitrateCalc);
    LOAD_CLI_TEXT(fcgTSBOtherSettings);
    LOAD_CLI_TEXT(fcgTSLSettingsNotes);
    LOAD_CLI_TEXT(fcgTSTSettingsNotes);
    LOAD_CLI_TEXT(fcgtabPageMP4);
    LOAD_CLI_TEXT(fcgBTMP4RawPath);
    LOAD_CLI_TEXT(fcgLBMP4RawPath);
    LOAD_CLI_TEXT(fcgCBMP4MuxApple);
    LOAD_CLI_TEXT(fcgBTMP4BoxTempDir);
    LOAD_CLI_TEXT(fcgLBMP4BoxTempDir);
    LOAD_CLI_TEXT(fcgBTTC2MP4Path);
    LOAD_CLI_TEXT(fcgBTMP4MuxerPath);
    LOAD_CLI_TEXT(fcgLBTC2MP4Path);
    LOAD_CLI_TEXT(fcgLBMP4MuxerPath);
    LOAD_CLI_TEXT(fcgLBMP4CmdEx);
    LOAD_CLI_TEXT(fcgCBMP4MuxerExt);
    LOAD_CLI_TEXT(fcgtabPageMKV);
    LOAD_CLI_TEXT(fcgBTMKVMuxerPath);
    LOAD_CLI_TEXT(fcgLBMKVMuxerPath);
    LOAD_CLI_TEXT(fcgLBMKVMuxerCmdEx);
    LOAD_CLI_TEXT(fcgCBMKVMuxerExt);
    LOAD_CLI_TEXT(fcgtabPageMPG);
    LOAD_CLI_TEXT(fcgBTMPGMuxerPath);
    LOAD_CLI_TEXT(fcgLBMPGMuxerPath);
    LOAD_CLI_TEXT(fcgLBMPGMuxerCmdEx);
    LOAD_CLI_TEXT(fcgCBMPGMuxerExt);
    LOAD_CLI_TEXT(fcgtabPageMux);
    LOAD_CLI_TEXT(fcgLBMuxPriority);
    LOAD_CLI_TEXT(fcgCBMuxMinimize);
    LOAD_CLI_TEXT(fcgtabPageBat);
    LOAD_CLI_TEXT(fcgLBBatAfterString);
    LOAD_CLI_TEXT(fcgLBBatBeforeString);
    LOAD_CLI_TEXT(fcgBTBatBeforePath);
    LOAD_CLI_TEXT(fcgLBBatBeforePath);
    LOAD_CLI_TEXT(fcgCBWaitForBatBefore);
    LOAD_CLI_TEXT(fcgCBRunBatBefore);
    LOAD_CLI_TEXT(fcgBTBatAfterPath);
    LOAD_CLI_TEXT(fcgLBBatAfterPath);
    LOAD_CLI_TEXT(fcgCBWaitForBatAfter);
    LOAD_CLI_TEXT(fcgCBRunBatAfter);
    LOAD_CLI_TEXT(fcgtabPageInternal);
    LOAD_CLI_TEXT(fcgLBInternalCmdEx);
    LOAD_CLI_TEXT(fcgBTCancel);
    LOAD_CLI_TEXT(fcgBTOK);
    LOAD_CLI_TEXT(fcgBTDefault);
    LOAD_CLI_TEXT(fcgLBVersionDate);
    LOAD_CLI_TEXT(fcgLBVersion);
    LOAD_CLI_TEXT(tabPageVideoEnc);
    LOAD_CLI_TEXT(fcgLBVideoFormat);
    LOAD_CLI_TEXT(fcggroupBoxColorMatrix);
    LOAD_CLI_TEXT(fcgLBFullrange);
    LOAD_CLI_TEXT(fcgLBTransfer);
    LOAD_CLI_TEXT(fcgLBColorPrim);
    LOAD_CLI_TEXT(fcgLBColorMatrix);
    LOAD_CLI_TEXT(fcgGroupBoxAspectRatio);
    LOAD_CLI_TEXT(fcgLBAspectRatio);
    LOAD_CLI_TEXT(fcgLBInterlaced);
    LOAD_CLI_TEXT(fcgBTVideoEncoderPath);
    LOAD_CLI_TEXT(fcgLBVideoEncoderPath);
    LOAD_CLI_TEXT(fcgCBAFS);
    LOAD_CLI_TEXT(fcgLBBitrate);
    LOAD_CLI_TEXT(fcgLBBitrate2);
    LOAD_CLI_TEXT(fcgLBMaxkbps);
    LOAD_CLI_TEXT(fcgLBMaxBitrate2);
    LOAD_CLI_TEXT(fcgLBQPI);
    LOAD_CLI_TEXT(fcgLBQPP);
    LOAD_CLI_TEXT(fcgLBQPB);
    LOAD_CLI_TEXT(fcgLBCodecLevel);
    LOAD_CLI_TEXT(fcgLBCodecProfile);
    LOAD_CLI_TEXT(fcgLBBframes);
    LOAD_CLI_TEXT(fcgLBEncMode);
    LOAD_CLI_TEXT(fcgLBRefFrames);
    LOAD_CLI_TEXT(fcgLBGOPLengthAuto);
    LOAD_CLI_TEXT(fcgLBGOPLength);
    LOAD_CLI_TEXT(fcgLBQualityPreset);
    LOAD_CLI_TEXT(fcgLBEncCodec);
    LOAD_CLI_TEXT(fcgLBSlices);
    LOAD_CLI_TEXT(fcgLBHEVCOutBitDepth);
    LOAD_CLI_TEXT(fcgLBMultiPass);
    LOAD_CLI_TEXT(fcgLBBrefMode);
    LOAD_CLI_TEXT(fcgLBWeightP);
    LOAD_CLI_TEXT(fcgLBVBVBufsize2);
    LOAD_CLI_TEXT(fcgLBLookaheadDisable);
    LOAD_CLI_TEXT(fcgLBAQStrengthAuto);
    LOAD_CLI_TEXT(fcgLBAQStrength);
    LOAD_CLI_TEXT(fcgLBLookaheadDepth);
    LOAD_CLI_TEXT(fcgLBAQ);
    LOAD_CLI_TEXT(fcgLBVBVBufsize);
    LOAD_CLI_TEXT(fcgLBBluray);
    LOAD_CLI_TEXT(fcgLBVBRTragetQuality2);
    LOAD_CLI_TEXT(fcgLBVBRTragetQuality);
    LOAD_CLI_TEXT(fcgLBQPDetailB);
    LOAD_CLI_TEXT(fcgLBQPDetailP);
    LOAD_CLI_TEXT(fcgLBQPDetailI);
    LOAD_CLI_TEXT(fcgCBQPInit);
    LOAD_CLI_TEXT(fcgLBQPInit2);
    LOAD_CLI_TEXT(fcgLBQPInit1);
    LOAD_CLI_TEXT(fcgCBQPMin);
    LOAD_CLI_TEXT(fcgLBQPMin2);
    LOAD_CLI_TEXT(fcgLBQPMin1);
    LOAD_CLI_TEXT(fcgCBQPMax);
    LOAD_CLI_TEXT(fcgLBQPMax2);
    LOAD_CLI_TEXT(fcgLBQPMax1);
    LOAD_CLI_TEXT(fxgLBHEVCTier);
    LOAD_CLI_TEXT(fcgLBHEVCProfile);
    LOAD_CLI_TEXT(fcgLBCodecProfileAV1);
    LOAD_CLI_TEXT(fcgLBCodecLevelAV1);
    LOAD_CLI_TEXT(fcgLBOutBitDepthAV1);
    LOAD_CLI_TEXT(tabPageVideoDetail);
    LOAD_CLI_TEXT(fcgLBLossless);
    LOAD_CLI_TEXT(fcgLBCudaSchdule);
    LOAD_CLI_TEXT(groupBoxQPDetail);
    LOAD_CLI_TEXT(fcgLBChromaQPOffset);
    LOAD_CLI_TEXT(fcgLBDevice);
    LOAD_CLI_TEXT(fcgLBDeblock);
    LOAD_CLI_TEXT(fcgLBAdaptiveTransform);
    LOAD_CLI_TEXT(fcgLBBDirectMode);
    LOAD_CLI_TEXT(fcgLBMVPrecision);
    LOAD_CLI_TEXT(fcgLBCABAC);
    LOAD_CLI_TEXT(fcgLBHEVCMinCUSize);
    LOAD_CLI_TEXT(fcgLBHEVCMaxCUSize);
    LOAD_CLI_TEXT(tabPageVpp);
    LOAD_CLI_TEXT(fcgCBVppTweakEnable);
    LOAD_CLI_TEXT(fcgLBVppTweakHue);
    LOAD_CLI_TEXT(fcgLBVppTweakSaturation);
    LOAD_CLI_TEXT(fcgLBVppTweakGamma);
    LOAD_CLI_TEXT(fcgLBVppTweakContrast);
    LOAD_CLI_TEXT(fcgLBVppTweakBrightness);
    LOAD_CLI_TEXT(fcgCBVppPerfMonitor);
    LOAD_CLI_TEXT(fcgCBVppDebandEnable);
    LOAD_CLI_TEXT(fcgCBVppDebandRandEachFrame);
    LOAD_CLI_TEXT(fcgCBVppDebandBlurFirst);
    LOAD_CLI_TEXT(fcgLBVppDebandSample);
    LOAD_CLI_TEXT(fcgLBVppDebandDitherC);
    LOAD_CLI_TEXT(fcgLBVppDebandDitherY);
    LOAD_CLI_TEXT(fcgLBVppDebandDither);
    LOAD_CLI_TEXT(fcgLBVppDebandThreCr);
    LOAD_CLI_TEXT(fcgLBVppDebandThreCb);
    LOAD_CLI_TEXT(fcgLBVppDebandThreY);
    LOAD_CLI_TEXT(fcgLBVppDebandThreshold);
    LOAD_CLI_TEXT(fcgLBVppDebandRange);
    LOAD_CLI_TEXT(fcggroupBoxVppDetailEnahance);
    LOAD_CLI_TEXT(fcgLBVppWarpsharpDepth);
    LOAD_CLI_TEXT(fcgLBVppWarpsharpThreshold);
    LOAD_CLI_TEXT(fcgLBVppWarpsharpType);
    LOAD_CLI_TEXT(fcgLBVppWarpsharpBlur);
    LOAD_CLI_TEXT(fcgLBVppEdgelevelWhite);
    LOAD_CLI_TEXT(fcgLBVppEdgelevelThreshold);
    LOAD_CLI_TEXT(fcgLBVppEdgelevelBlack);
    LOAD_CLI_TEXT(fcgLBVppEdgelevelStrength);
    LOAD_CLI_TEXT(fcgLBVppUnsharpThreshold);
    LOAD_CLI_TEXT(fcgLBVppUnsharpWeight);
    LOAD_CLI_TEXT(fcgLBVppUnsharpRadius);
    LOAD_CLI_TEXT(fcggroupBoxVppDenoise);
    LOAD_CLI_TEXT(fcgLBVppDenoiseConv3DMatrix);
    LOAD_CLI_TEXT(fcgLBVppDenoiseConv3DThreshTemporal);
    LOAD_CLI_TEXT(fcgLBVppDenoiseConv3DThreshSpatial);
    LOAD_CLI_TEXT(fcgLBVppDenoiseConv3DThreshCTemporal);
    LOAD_CLI_TEXT(fcgLBVppDenoiseConv3DThreshCSpatial);
    LOAD_CLI_TEXT(fcgLBVppDenoiseConv3DThreshYTemporal);
    LOAD_CLI_TEXT(fcgLBVppDenoiseConv3DThreshYSpatial);
    LOAD_CLI_TEXT(fcgLBVppDenoiseSmoothQP);
    LOAD_CLI_TEXT(fcgLBVppDenoiseSmoothQuality);
    LOAD_CLI_TEXT(fcgLBVppDenoiseKnnThreshold);
    LOAD_CLI_TEXT(fcgLBVppDenoiseKnnStrength);
    LOAD_CLI_TEXT(fcgLBVppDenoiseKnnRadius);
    LOAD_CLI_TEXT(fcgLBVppDenoisePmdThreshold);
    LOAD_CLI_TEXT(fcgLBVppDenoisePmdStrength);
    LOAD_CLI_TEXT(fcgLBVppDenoisePmdApplyCount);
    LOAD_CLI_TEXT(fcggroupBoxVppDeinterlace);
    LOAD_CLI_TEXT(fcgLBVppDeinterlace);
    LOAD_CLI_TEXT(fcgLBVppAfsThreCMotion);
    LOAD_CLI_TEXT(fcgLBVppAfsThreYmotion);
    LOAD_CLI_TEXT(fcgLBVppAfsThreDeint);
    LOAD_CLI_TEXT(fcgLBVppAfsThreShift);
    LOAD_CLI_TEXT(fcgLBVppAfsCoeffShift);
    LOAD_CLI_TEXT(fcgLBVppAfsRight);
    LOAD_CLI_TEXT(fcgLBVppAfsLeft);
    LOAD_CLI_TEXT(fcgLBVppAfsBottom);
    LOAD_CLI_TEXT(fcgLBVppAfsUp);
    LOAD_CLI_TEXT(fcgCBVppAfs24fps);
    LOAD_CLI_TEXT(fcgCBVppAfsTune);
    LOAD_CLI_TEXT(fcgCBVppAfsSmooth);
    LOAD_CLI_TEXT(fcgCBVppAfsDrop);
    LOAD_CLI_TEXT(fcgCBVppAfsShift);
    LOAD_CLI_TEXT(fcgLBVppAfsAnalyze);
    LOAD_CLI_TEXT(fcgLBVppAfsMethodSwitch);
    LOAD_CLI_TEXT(fcgLBVppYadifMode);
    LOAD_CLI_TEXT(fcgLBVppNnediErrorType);
    LOAD_CLI_TEXT(fcgLBVppNnediPrescreen);
    LOAD_CLI_TEXT(fcgLBVppNnediPrec);
    LOAD_CLI_TEXT(fcgLBVppNnediQual);
    LOAD_CLI_TEXT(fcgLBVppNnediNsize);
    LOAD_CLI_TEXT(fcgLBVppNnediNns);
    LOAD_CLI_TEXT(fcgCBVppResize);
    LOAD_CLI_TEXT(fcgLBVppResize);
    LOAD_CLI_TEXT(fcgLBPSNR);
    LOAD_CLI_TEXT(fcgLBSSIM);
    LOAD_CLI_TEXT(tabPageExOpt);
    LOAD_CLI_TEXT(fcggroupBoxCmdEx);
    LOAD_CLI_TEXT(fcgCBLogDebug);
    LOAD_CLI_TEXT(fcgCBAuoTcfileout);
    LOAD_CLI_TEXT(fcgLBTempDir);
    LOAD_CLI_TEXT(fcgBTCustomTempDir);
    LOAD_CLI_TEXT(fcgCBPerfMonitor);
    LOAD_CLI_TEXT(fcgTSExeFileshelp);
    LOAD_CLI_TEXT(fcgLBguiExBlog);
    LOAD_CLI_TEXT(fcgtabPageAudioMain);
    LOAD_CLI_TEXT(fcgLBAudioBitrateInternal);
    LOAD_CLI_TEXT(fcgLBAudioEncModeInternal);
    LOAD_CLI_TEXT(fcgCBAudioUseExt);
    LOAD_CLI_TEXT(fcgCBFAWCheck);
    LOAD_CLI_TEXT(fcgLBAudioDelayCut);
    LOAD_CLI_TEXT(fcgCBAudioEncTiming);
    LOAD_CLI_TEXT(fcgBTCustomAudioTempDir);
    LOAD_CLI_TEXT(fcgCBAudioUsePipe);
    LOAD_CLI_TEXT(fcgCBAudio2pass);
    LOAD_CLI_TEXT(fcgLBAudioEncMode);
    LOAD_CLI_TEXT(fcgBTAudioEncoderPath);
    LOAD_CLI_TEXT(fcgLBAudioEncoderPath);
    LOAD_CLI_TEXT(fcgCBAudioOnly);
    LOAD_CLI_TEXT(fcgLBAudioTemp);
    LOAD_CLI_TEXT(fcgLBAudioBitrate);
    LOAD_CLI_TEXT(fcgtabPageAudioOther);
    LOAD_CLI_TEXT(fcgLBBatAfterAudioString);
    LOAD_CLI_TEXT(fcgLBBatBeforeAudioString);
    LOAD_CLI_TEXT(fcgBTBatAfterAudioPath);
    LOAD_CLI_TEXT(fcgLBBatAfterAudioPath);
    LOAD_CLI_TEXT(fcgCBRunBatAfterAudio);
    LOAD_CLI_TEXT(fcgBTBatBeforeAudioPath);
    LOAD_CLI_TEXT(fcgLBBatBeforeAudioPath);
    LOAD_CLI_TEXT(fcgCBRunBatBeforeAudio);
    LOAD_CLI_TEXT(fcgLBAudioPriority);

    //ローカル設定のロード(ini変更を反映)
    LoadLocalStg();
    //ローカル設定の反映
    SetLocalStg();
    //コンボボックスの値を設定
    InitComboBox();
    //ツールチップ
    SetHelpToolTips();
    ActivateToolTip(sys_dat->exstg->s_local.disable_tooltip_help == FALSE);
    //タイムコードのappendix(後付修飾子)を反映
    fcgCBAuoTcfileout->Text = LOAD_CLI_STRING(AUO_CONFIG_TC_FILE_OUT) + L" (" + String(sys_dat->exstg->s_append.tc).ToString() + L")";
    { //タイトル表示,バージョン情報,コンパイル日時
        auto auo_full_name = g_auo_mes.get(AUO_GUIEX_FULL_NAME);
        if (auo_full_name == nullptr || wcslen(auo_full_name) == 0) auo_full_name = AUO_FULL_NAME_W;
        this->Text = String(auo_full_name).ToString();
        fcgLBVersion->Text = String(auo_full_name).ToString() + L" " + String(AUO_VERSION_STR).ToString();
        fcgLBVersionDate->Text = L"build " + String(__DATE__).ToString() + L" " + String(__TIME__).ToString();
    }
    //空白時にグレーで入れる文字列を言語に即して復活させる
    ExeTXPathLeave();
    //一度ウィンドウの再描画を再開し、強制的に再描画させる
    SendMessage(reinterpret_cast<HWND>(this->Handle.ToPointer()), WM_SETREDRAW, 1, 0);
    this->Refresh();
}

/////////////         データ <-> GUI     /////////////
System::Void frmConfig::ConfToFrm(CONF_GUIEX *cnf) {
    this->SuspendLayout();

    InEncodeVideoParam encPrm;
    NV_ENC_CODEC_CONFIG codecPrm[NV_ENC_CODEC_MAX] = { 0 };
    codecPrm[NV_ENC_H264] = DefaultParamH264();
    codecPrm[NV_ENC_HEVC] = DefaultParamHEVC();
    codecPrm[NV_ENC_AV1]  = DefaultParamAV1();
    parse_cmd(&encPrm, codecPrm, cnf->enc.cmd);

    SetCXIndex(fcgCXEncCodec,          get_index_from_value(encPrm.codec, list_nvenc_codecs));
    SetCXIndex(fcgCXEncMode,           get_cx_index(list_encmode, (encPrm.encConfig.rcParams.averageBitRate == 0 && encPrm.encConfig.rcParams.rateControlMode == NV_ENC_PARAMS_RC_VBR) ? NV_ENC_PARAMS_RC_QVBR : encPrm.encConfig.rcParams.rateControlMode));
    SetNUValue(fcgNUBitrate,           encPrm.encConfig.rcParams.averageBitRate / 1000);
    SetNUValue(fcgNUMaxkbps,           encPrm.encConfig.rcParams.maxBitRate / 1000);
    SetNUValue(fcgNUQPI,               encPrm.encConfig.rcParams.constQP.qpIntra);
    SetNUValue(fcgNUQPP,               encPrm.encConfig.rcParams.constQP.qpInterP);
    SetNUValue(fcgNUQPB,               encPrm.encConfig.rcParams.constQP.qpInterB);
    SetNUValue(fcgNUVBRTragetQuality,  Decimal(encPrm.encConfig.rcParams.targetQuality + encPrm.encConfig.rcParams.targetQualityLSB / 256.0));
    SetNUValue(fcgNUGopLength,         encPrm.encConfig.gopLength);
    SetNUValue(fcgNUBframes,           encPrm.encConfig.frameIntervalP - 1);
    SetCXIndex(fcgCXQualityPreset,     get_index_from_value(encPrm.preset, list_nvenc_preset_names_ver10));
    SetCXIndex(fcgCXMultiPass,         get_cx_index(list_nvenc_multipass_mode, encPrm.encConfig.rcParams.multiPass));
    SetCXIndex(fcgCXMVPrecision,       get_cx_index(list_mv_presicion, encPrm.encConfig.mvPrecision));
    SetNUValue(fcgNUVBVBufsize, encPrm.encConfig.rcParams.vbvBufferSize / 1000);
    SetNUValue(fcgNULookaheadDepth,   (encPrm.encConfig.rcParams.enableLookahead) ? encPrm.encConfig.rcParams.lookaheadDepth : 0);
    SetCXIndex(fcgCXBrefMode,          codecPrm[NV_ENC_H264].h264Config.useBFramesAsRef);
    uint32_t nAQ = 0;
    nAQ |= encPrm.encConfig.rcParams.enableAQ ? NV_ENC_AQ_SPATIAL : 0x00;
    nAQ |= encPrm.encConfig.rcParams.enableTemporalAQ ? NV_ENC_AQ_TEMPORAL : 0x00;
    SetCXIndex(fcgCXAQ,                get_cx_index(list_aq, nAQ));
    SetNUValue(fcgNUAQStrength,        encPrm.encConfig.rcParams.aqStrength);
    fcgCBWeightP->Checked            = encPrm.nWeightP != 0;

    if (encPrm.par[0] * encPrm.par[1] <= 0)
        encPrm.par[0] = encPrm.par[1] = 0;
    SetCXIndex(fcgCXAspectRatio, (encPrm.par[0] < 0));
    SetNUValue(fcgNUAspectRatioX, abs(encPrm.par[0]));
    SetNUValue(fcgNUAspectRatioY, abs(encPrm.par[1]));

    SetCXIndex(fcgCXDevice,      encPrm.deviceID+1); //先頭は"Auto"なので+1
    SetCXIndex(fcgCXCudaSchdule, encPrm.cudaSchedule);
    fcgCBPerfMonitor->Checked = 0 != encPrm.ctrl.perfMonitorSelect;
    fcgCBLossless->Checked = 0 != encPrm.lossless;

    //QPDetail
    fcgCBQPMin->Checked = encPrm.encConfig.rcParams.enableMinQP != 0;
    fcgCBQPMax->Checked = encPrm.encConfig.rcParams.enableMaxQP != 0;
    fcgCBQPInit->Checked = encPrm.encConfig.rcParams.enableInitialRCQP != 0;
    SetNUValue(fcgNUQPMinI, encPrm.encConfig.rcParams.minQP.qpIntra);
    SetNUValue(fcgNUQPMinP, encPrm.encConfig.rcParams.minQP.qpInterP);
    SetNUValue(fcgNUQPMinB, encPrm.encConfig.rcParams.minQP.qpInterB);
    SetNUValue(fcgNUQPMaxI, encPrm.encConfig.rcParams.maxQP.qpIntra);
    SetNUValue(fcgNUQPMaxP, encPrm.encConfig.rcParams.maxQP.qpInterP);
    SetNUValue(fcgNUQPMaxB, encPrm.encConfig.rcParams.maxQP.qpInterB);
    SetNUValue(fcgNUQPInitI, encPrm.encConfig.rcParams.initialRCQP.qpIntra);
    SetNUValue(fcgNUQPInitP, encPrm.encConfig.rcParams.initialRCQP.qpInterP);
    SetNUValue(fcgNUQPInitB, encPrm.encConfig.rcParams.initialRCQP.qpInterB);
    fcgNUChromaQPOffset->Value = encPrm.chromaQPOffset;

    //H.264
    SetNUValue(fcgNURefFrames,         codecPrm[NV_ENC_H264].h264Config.maxNumRefFrames);
    SetCXIndex(fcgCXInterlaced,   get_cx_index(list_interlaced, encPrm.input.picstruct));
    SetCXIndex(fcgCXCodecLevel,   get_cx_index(list_avc_level,            codecPrm[NV_ENC_H264].h264Config.level));
    SetCXIndex(fcgCXCodecProfile, get_index_from_value(get_value_from_guid(encPrm.encConfig.profileGUID, h264_profile_names), h264_profile_names));
    SetNUValue(fcgNUSlices,       codecPrm[NV_ENC_H264].h264Config.sliceModeData);
    fcgCBBluray->Checked       = 0 != encPrm.bluray;
    fcgCBCABAC->Checked        = NV_ENC_H264_ENTROPY_CODING_MODE_CAVLC != codecPrm[NV_ENC_H264].h264Config.entropyCodingMode;
    fcgCBDeblock->Checked                                          = 0 == codecPrm[NV_ENC_H264].h264Config.disableDeblockingFilterIDC;
    SetCXIndex(fcgCXAdaptiveTransform, get_cx_index(list_adapt_transform, codecPrm[NV_ENC_H264].h264Config.adaptiveTransformMode));
    SetCXIndex(fcgCXBDirectMode,       get_cx_index(list_bdirect,         codecPrm[NV_ENC_H264].h264Config.bdirectMode));

    //HEVC
    SetCXIndex(fcgCXHEVCOutBitDepth,       get_cx_index(list_bitdepth, codecPrm[NV_ENC_HEVC].hevcConfig.pixelBitDepthMinus8));
    SetCXIndex(fcgCXHEVCTier,      get_index_from_value(codecPrm[NV_ENC_HEVC].hevcConfig.tier, h265_profile_names));
    SetCXIndex(fxgCXHEVCLevel,     get_cx_index(list_hevc_level,   codecPrm[NV_ENC_HEVC].hevcConfig.level));
    SetCXIndex(fcgCXHEVCMaxCUSize, get_cx_index(list_hevc_cu_size, codecPrm[NV_ENC_HEVC].hevcConfig.maxCUSize));
    SetCXIndex(fcgCXHEVCMinCUSize, get_cx_index(list_hevc_cu_size, codecPrm[NV_ENC_HEVC].hevcConfig.minCUSize));

    //AV1
    SetCXIndex(fcgCXOutBitDepthAV1,    get_cx_index(list_bitdepth, codecPrm[NV_ENC_AV1].av1Config.pixelBitDepthMinus8));
    SetCXIndex(fcgCXCodecProfileAV1,   get_index_from_value(codecPrm[NV_ENC_HEVC].av1Config.tier, av1_profile_names));
    SetCXIndex(fcgCXCodecLevelAV1,     get_cx_index(list_av1_level,   codecPrm[NV_ENC_HEVC].av1Config.level));

    SetCXIndex(fcgCXVideoFormat,       get_cx_index(list_videoformat, encPrm.common.out_vui.format));
    fcgCBFullrange->Checked                                         = encPrm.common.out_vui.colorrange == RGY_COLORRANGE_FULL;
    SetCXIndex(fcgCXTransfer,          get_cx_index(list_transfer,    encPrm.common.out_vui.transfer));
    SetCXIndex(fcgCXColorMatrix,       get_cx_index(list_colormatrix, encPrm.common.out_vui.matrix));
    SetCXIndex(fcgCXColorPrim,         get_cx_index(list_colorprim,   encPrm.common.out_vui.colorprim));

    //fcgCBShowPerformanceInfo->Checked = (CHECK_PERFORMANCE) ? cnf->vid.vce_ext_prm.show_performance_info != 0 : false;

        //SetCXIndex(fcgCXX264Priority,        cnf->vid.priority);
        fcgCBSSIM->Checked                 = encPrm.common.metric.ssim;
        fcgCBPSNR->Checked                 = encPrm.common.metric.psnr;
        SetCXIndex(fcgCXTempDir,             cnf->oth.temp_dir);
        fcgCBAFS->Checked                  = cnf->vid.afs != 0;
        fcgCBAuoTcfileout->Checked         = cnf->vid.auo_tcfile_out != 0;
        fcgCBLogDebug->Checked             = encPrm.ctrl.loglevel.get(RGY_LOGT_APP) == RGY_LOG_DEBUG;

        fcgCBVppPerfMonitor->Checked   = encPrm.vpp.checkPerformance != 0;
        fcgCBVppResize->Checked        = cnf->vid.resize_enable != 0;
        SetNUValue(fcgNUVppResizeWidth,  cnf->vid.resize_width);
        SetNUValue(fcgNUVppResizeHeight, cnf->vid.resize_height);
        SetCXIndex(fcgCXVppResizeAlg,    get_cx_index(list_vpp_resize, encPrm.vpp.resize_algo));
        int denoise_idx = 0;
        if (encPrm.vpp.knn.enable) {
            denoise_idx = get_cx_index(list_vpp_denoise, _T("knn"));
        } else if (encPrm.vpp.pmd.enable) {
            denoise_idx = get_cx_index(list_vpp_denoise, _T("pmd"));
        } else if (encPrm.vpp.smooth.enable) {
            denoise_idx = get_cx_index(list_vpp_denoise, _T("smooth"));
        } else if (encPrm.vpp.convolution3d.enable) {
            denoise_idx = get_cx_index(list_vpp_denoise, _T("convolution3d"));
        }
        SetCXIndex(fcgCXVppDenoiseMethod, denoise_idx);

        int detail_enahance_idx = 0;
        if (encPrm.vpp.unsharp.enable) {
            detail_enahance_idx = get_cx_index(list_vpp_detail_enahance, _T("unsharp"));
        } else if (encPrm.vpp.edgelevel.enable) {
            detail_enahance_idx = get_cx_index(list_vpp_detail_enahance, _T("edgelevel"));
        } else if (encPrm.vpp.warpsharp.enable) {
            detail_enahance_idx = get_cx_index(list_vpp_detail_enahance, _T("warpsharp"));
        }
        SetCXIndex(fcgCXVppDetailEnhance, detail_enahance_idx);

        int deinterlacer_idx = 0;
        if (encPrm.vpp.afs.enable) {
            deinterlacer_idx = get_cx_index(list_deinterlace_gui, L"自動フィールドシフト");
        } else if (encPrm.vpp.nnedi.enable) {
            deinterlacer_idx = get_cx_index(list_deinterlace_gui, L"nnedi");
        } else if (encPrm.vpp.yadif.enable) {
            deinterlacer_idx = get_cx_index(list_deinterlace_gui, L"yadif");
        }
        SetCXIndex(fcgCXVppDeinterlace, deinterlacer_idx);

        SetNUValue(fcgNUVppDenoiseKnnRadius,     encPrm.vpp.knn.radius);
        SetNUValue(fcgNUVppDenoiseKnnStrength,   encPrm.vpp.knn.strength);
        SetNUValue(fcgNUVppDenoiseKnnThreshold,  encPrm.vpp.knn.lerp_threshold);
        SetNUValue(fcgNUVppDenoisePmdApplyCount, encPrm.vpp.pmd.applyCount);
        SetNUValue(fcgNUVppDenoisePmdStrength,   encPrm.vpp.pmd.strength);
        SetNUValue(fcgNUVppDenoisePmdThreshold,  encPrm.vpp.pmd.threshold);
        SetNUValue(fcgNUVppDenoiseSmoothQuality, encPrm.vpp.smooth.quality);
        SetNUValue(fcgNUVppDenoiseSmoothQP,      encPrm.vpp.smooth.qp);
        SetCXIndex(fcgCXVppDenoiseConv3DMatrix,          get_cx_index(list_vpp_convolution3d_matrix, (int)encPrm.vpp.convolution3d.matrix));
        SetNUValue(fcgNUVppDenoiseConv3DThreshYSpatial,  encPrm.vpp.convolution3d.threshYspatial);
        SetNUValue(fcgNUVppDenoiseConv3DThreshCSpatial,  encPrm.vpp.convolution3d.threshCspatial);
        SetNUValue(fcgNUVppDenoiseConv3DThreshYTemporal, encPrm.vpp.convolution3d.threshYtemporal);
        SetNUValue(fcgNUVppDenoiseConv3DThreshCTemporal, encPrm.vpp.convolution3d.threshCtemporal);
        fcgCBVppDebandEnable->Checked          = encPrm.vpp.deband.enable;
        SetNUValue(fcgNUVppDebandRange,          encPrm.vpp.deband.range);
        SetNUValue(fcgNUVppDebandThreY,          encPrm.vpp.deband.threY);
        SetNUValue(fcgNUVppDebandThreCb,         encPrm.vpp.deband.threCb);
        SetNUValue(fcgNUVppDebandThreCr,         encPrm.vpp.deband.threCr);
        SetNUValue(fcgNUVppDebandDitherY,        encPrm.vpp.deband.ditherY);
        SetNUValue(fcgNUVppDebandDitherC,        encPrm.vpp.deband.ditherC);
        SetCXIndex(fcgCXVppDebandSample,         get_cx_index(list_vpp_deband_gui, encPrm.vpp.deband.sample));
        fcgCBVppDebandBlurFirst->Checked       = encPrm.vpp.deband.blurFirst;
        fcgCBVppDebandRandEachFrame->Checked   = encPrm.vpp.deband.randEachFrame;
        SetNUValue(fcgNUVppUnsharpRadius,        encPrm.vpp.unsharp.radius);
        SetNUValue(fcgNUVppUnsharpWeight,        encPrm.vpp.unsharp.weight);
        SetNUValue(fcgNUVppUnsharpThreshold,     encPrm.vpp.unsharp.threshold);
        SetNUValue(fcgNUVppEdgelevelStrength,    encPrm.vpp.edgelevel.strength);
        SetNUValue(fcgNUVppEdgelevelThreshold,   encPrm.vpp.edgelevel.threshold);
        SetNUValue(fcgNUVppEdgelevelBlack,       encPrm.vpp.edgelevel.black);
        SetNUValue(fcgNUVppEdgelevelWhite,       encPrm.vpp.edgelevel.white);
        SetNUValue(fcgNUVppWarpsharpThreshold,   encPrm.vpp.warpsharp.threshold);
        SetNUValue(fcgNUVppWarpsharpDepth,       encPrm.vpp.warpsharp.depth);
        SetNUValue(fcgNUVppWarpsharpBlur,        encPrm.vpp.warpsharp.blur);
        SetNUValue(fcgNUVppWarpsharpType,        encPrm.vpp.warpsharp.type);
        SetNUValue(fcgNUVppAfsUp,                encPrm.vpp.afs.clip.top);
        SetNUValue(fcgNUVppAfsBottom,            encPrm.vpp.afs.clip.bottom);
        SetNUValue(fcgNUVppAfsLeft,              encPrm.vpp.afs.clip.left);
        SetNUValue(fcgNUVppAfsRight,             encPrm.vpp.afs.clip.right);
        SetNUValue(fcgNUVppAfsMethodSwitch,      encPrm.vpp.afs.method_switch);
        SetNUValue(fcgNUVppAfsCoeffShift,        encPrm.vpp.afs.coeff_shift);
        SetNUValue(fcgNUVppAfsThreShift,         encPrm.vpp.afs.thre_shift);
        SetNUValue(fcgNUVppAfsThreDeint,         encPrm.vpp.afs.thre_deint);
        SetNUValue(fcgNUVppAfsThreYMotion,       encPrm.vpp.afs.thre_Ymotion);
        SetNUValue(fcgNUVppAfsThreCMotion,       encPrm.vpp.afs.thre_Cmotion);
        SetCXIndex(fcgCXVppAfsAnalyze,           encPrm.vpp.afs.analyze);
        fcgCBVppAfsShift->Checked              = encPrm.vpp.afs.shift != 0;
        fcgCBVppAfsDrop->Checked               = encPrm.vpp.afs.drop != 0;
        fcgCBVppAfsSmooth->Checked             = encPrm.vpp.afs.smooth != 0;
        fcgCBVppAfs24fps->Checked              = encPrm.vpp.afs.force24 != 0;
        fcgCBVppAfsTune->Checked               = encPrm.vpp.afs.tune != 0;
        SetCXIndex(fcgCXVppNnediNsize,           get_cx_index(list_vpp_nnedi_nsize, encPrm.vpp.nnedi.nsize));
        SetCXIndex(fcgCXVppNnediNns,             get_cx_index(list_vpp_nnedi_nns, encPrm.vpp.nnedi.nns));
        SetCXIndex(fcgCXVppNnediPrec,            get_cx_index(list_vpp_fp_prec, encPrm.vpp.nnedi.precision));
        SetCXIndex(fcgCXVppNnediPrescreen,       get_cx_index(list_vpp_nnedi_pre_screen_gui, encPrm.vpp.nnedi.pre_screen));
        SetCXIndex(fcgCXVppNnediQual,            get_cx_index(list_vpp_nnedi_quality, encPrm.vpp.nnedi.quality));
        SetCXIndex(fcgCXVppNnediErrorType,       get_cx_index(list_vpp_nnedi_error_type, encPrm.vpp.nnedi.errortype));
        SetCXIndex(fcgCXVppYadifMode,            get_cx_index(list_vpp_yadif_mode_gui, encPrm.vpp.yadif.mode));
        fcgCBVppTweakEnable->Checked           = encPrm.vpp.tweak.enable;
        SetNUValue(fcgNUVppTweakBrightness,      (int)(encPrm.vpp.tweak.brightness * 100.0f));
        SetNUValue(fcgNUVppTweakContrast,        (int)(encPrm.vpp.tweak.contrast * 100.0f));
        SetNUValue(fcgNUVppTweakGamma,           (int)(encPrm.vpp.tweak.gamma * 100.0f));
        SetNUValue(fcgNUVppTweakSaturation,      (int)(encPrm.vpp.tweak.saturation * 100.0f));
        SetNUValue(fcgNUVppTweakHue,             (int) encPrm.vpp.tweak.hue);

        //音声
        fcgCBAudioUseExt->Checked          = cnf->aud.use_internal == 0;
        //外部音声エンコーダ
        fcgCBAudioOnly->Checked            = cnf->oth.out_audio_only != 0;
        fcgCBFAWCheck->Checked             = cnf->aud.ext.faw_check != 0;
        SetCXIndex(fcgCXAudioEncoder,        cnf->aud.ext.encoder);
        fcgCBAudio2pass->Checked           = cnf->aud.ext.use_2pass != 0;
        fcgCBAudioUsePipe->Checked = (CurrentPipeEnabled && !cnf->aud.ext.use_wav);
        SetCXIndex(fcgCXAudioDelayCut,       cnf->aud.ext.delay_cut);
        SetCXIndex(fcgCXAudioEncMode,        cnf->aud.ext.enc_mode);
        SetNUValue(fcgNUAudioBitrate,       (cnf->aud.ext.bitrate != 0) ? cnf->aud.ext.bitrate : GetCurrentAudioDefaultBitrate());
        SetCXIndex(fcgCXAudioPriority,       cnf->aud.ext.priority);
        SetCXIndex(fcgCXAudioTempDir,        cnf->aud.ext.aud_temp_dir);
        SetCXIndex(fcgCXAudioEncTiming,      cnf->aud.ext.audio_encode_timing);
        fcgCBRunBatBeforeAudio->Checked    =(cnf->oth.run_bat & RUN_BAT_BEFORE_AUDIO) != 0;
        fcgCBRunBatAfterAudio->Checked     =(cnf->oth.run_bat & RUN_BAT_AFTER_AUDIO) != 0;
        fcgTXBatBeforeAudioPath->Text      = String(cnf->oth.batfile.before_audio).ToString();
        fcgTXBatAfterAudioPath->Text       = String(cnf->oth.batfile.after_audio).ToString();
        //内蔵音声エンコーダ
        SetCXIndex(fcgCXAudioEncoderInternal, cnf->aud.in.encoder);
        SetCXIndex(fcgCXAudioEncModeInternal, cnf->aud.in.enc_mode);
        SetNUValue(fcgNUAudioBitrateInternal, (cnf->aud.in.bitrate != 0) ? cnf->aud.in.bitrate : GetCurrentAudioDefaultBitrate());

        //mux
        fcgCBMP4MuxerExt->Checked          = cnf->mux.disable_mp4ext == 0;
        fcgCBMP4MuxApple->Checked          = cnf->mux.apple_mode != 0;
        SetCXIndex(fcgCXMP4CmdEx,            cnf->mux.mp4_mode);
        SetCXIndex(fcgCXMP4BoxTempDir,       cnf->mux.mp4_temp_dir);
        fcgCBMKVMuxerExt->Checked          = cnf->mux.disable_mkvext == 0;
        SetCXIndex(fcgCXMKVCmdEx,            cnf->mux.mkv_mode);
        fcgCBMPGMuxerExt->Checked          = cnf->mux.disable_mpgext == 0;
        SetCXIndex(fcgCXMPGCmdEx,            cnf->mux.mpg_mode);
        fcgCBMuxMinimize->Checked          = cnf->mux.minimized != 0;
        SetCXIndex(fcgCXMuxPriority,         cnf->mux.priority);
        SetCXIndex(fcgCXInternalCmdEx,       cnf->mux.internal_mode);

        fcgCBRunBatBefore->Checked         =(cnf->oth.run_bat & RUN_BAT_BEFORE_PROCESS) != 0;
        fcgCBRunBatAfter->Checked          =(cnf->oth.run_bat & RUN_BAT_AFTER_PROCESS)  != 0;
        fcgCBWaitForBatBefore->Checked     =(cnf->oth.dont_wait_bat_fin & RUN_BAT_BEFORE_PROCESS) == 0;
        fcgCBWaitForBatAfter->Checked      =(cnf->oth.dont_wait_bat_fin & RUN_BAT_AFTER_PROCESS)  == 0;
        fcgTXBatBeforePath->Text           = String(cnf->oth.batfile.before_process).ToString();
        fcgTXBatAfterPath->Text            = String(cnf->oth.batfile.after_process).ToString();

        fcgTXCmdEx->Text                   = String(cnf->enc.cmdex).ToString();

        SetfcgTSLSettingsNotes(cnf->oth.notes);

    this->ResumeLayout();
    this->PerformLayout();
}

System::String^ frmConfig::FrmToConf(CONF_GUIEX *cnf) {
    InEncodeVideoParam encPrm;
    NV_ENC_CODEC_CONFIG codecPrm[NV_ENC_CODEC_MAX] = { 0 };
    codecPrm[NV_ENC_H264] = DefaultParamH264();
    codecPrm[NV_ENC_HEVC] = DefaultParamHEVC();
    codecPrm[NV_ENC_AV1]  = DefaultParamAV1();

    //これもひたすら書くだけ。めんどい
    encPrm.codec = list_nvenc_codecs[fcgCXEncCodec->SelectedIndex].value;
    cnf->enc.codec = encPrm.codec;
    encPrm.encConfig.rcParams.rateControlMode = (NV_ENC_PARAMS_RC_MODE)list_encmode[fcgCXEncMode->SelectedIndex].value;
    if (encPrm.encConfig.rcParams.rateControlMode == NV_ENC_PARAMS_RC_QVBR) {
        encPrm.encConfig.rcParams.rateControlMode = NV_ENC_PARAMS_RC_VBR;
        encPrm.encConfig.rcParams.averageBitRate = 0;
    } else {
        encPrm.encConfig.rcParams.averageBitRate = (int)fcgNUBitrate->Value * 1000;
    }
    encPrm.encConfig.rcParams.maxBitRate = (int)fcgNUMaxkbps->Value * 1000;
    encPrm.encConfig.rcParams.constQP.qpIntra  = (int)fcgNUQPI->Value;
    encPrm.encConfig.rcParams.constQP.qpInterP = (int)fcgNUQPP->Value;
    encPrm.encConfig.rcParams.constQP.qpInterB = (int)fcgNUQPB->Value;
    encPrm.encConfig.gopLength = (int)fcgNUGopLength->Value;
    const double targetQualityDouble = (double)fcgNUVBRTragetQuality->Value;
    const int targetQualityInt = (int)targetQualityDouble;
    encPrm.encConfig.rcParams.targetQuality = (uint8_t)targetQualityInt;
    encPrm.encConfig.rcParams.targetQualityLSB = (uint8_t)clamp(((targetQualityDouble - targetQualityInt) * 256.0), 0, 255);
    encPrm.encConfig.frameIntervalP = (int)fcgNUBframes->Value + 1;
    encPrm.encConfig.mvPrecision = (NV_ENC_MV_PRECISION)list_mv_presicion[fcgCXMVPrecision->SelectedIndex].value;
    encPrm.encConfig.rcParams.vbvBufferSize = (int)fcgNUVBVBufsize->Value * 1000;
    encPrm.preset = list_nvenc_preset_names_ver10[fcgCXQualityPreset->SelectedIndex].value;
    encPrm.encConfig.rcParams.multiPass = (NV_ENC_MULTI_PASS)list_nvenc_multipass_mode[fcgCXMultiPass->SelectedIndex].value;
    codecPrm[NV_ENC_H264].h264Config.useBFramesAsRef = (NV_ENC_BFRAME_REF_MODE)list_bref_mode[fcgCXBrefMode->SelectedIndex].value;
    codecPrm[NV_ENC_HEVC].hevcConfig.useBFramesAsRef = (NV_ENC_BFRAME_REF_MODE)list_bref_mode[fcgCXBrefMode->SelectedIndex].value;

    int nLookaheadDepth = (int)fcgNULookaheadDepth->Value;
    if (nLookaheadDepth > 0) {
        encPrm.encConfig.rcParams.enableLookahead = 1;
        encPrm.encConfig.rcParams.lookaheadDepth = (uint16_t)clamp(nLookaheadDepth, 0, 32);
    } else {
        encPrm.encConfig.rcParams.enableLookahead = 0;
        encPrm.encConfig.rcParams.lookaheadDepth = 0;
    }
    uint32_t nAQ = list_aq[fcgCXAQ->SelectedIndex].value;
    encPrm.encConfig.rcParams.enableAQ         = (nAQ & NV_ENC_AQ_SPATIAL)  ? 1 : 0;
    encPrm.encConfig.rcParams.enableTemporalAQ = (nAQ & NV_ENC_AQ_TEMPORAL) ? 1 : 0;
    encPrm.encConfig.rcParams.aqStrength = (uint32_t)clamp(fcgNUAQStrength->Value, 0, 15);
    encPrm.nWeightP                        = (fcgCBWeightP->Checked) ? 1 : 0;


    encPrm.par[0]                = (int)fcgNUAspectRatioX->Value;
    encPrm.par[1]                = (int)fcgNUAspectRatioY->Value;
    if (fcgCXAspectRatio->SelectedIndex == 1) {
        encPrm.par[0] *= -1;
        encPrm.par[1] *= -1;
    }

    encPrm.deviceID = (int)fcgCXDevice->SelectedIndex-1; //先頭は"Auto"なので-1
    encPrm.cudaSchedule = list_cuda_schedule[fcgCXCudaSchdule->SelectedIndex].value;
    encPrm.lossless = fcgCBLossless->Checked;

    //QPDetail
    encPrm.encConfig.rcParams.enableMinQP = (fcgCBQPMin->Checked) ? 1 : 0;
    encPrm.encConfig.rcParams.enableMaxQP = (fcgCBQPMax->Checked) ? 1 : 0;
    encPrm.encConfig.rcParams.enableInitialRCQP = (fcgCBQPInit->Checked) ? 1 : 0;
    encPrm.encConfig.rcParams.minQP.qpIntra  = (int)fcgNUQPMinI->Value;
    encPrm.encConfig.rcParams.minQP.qpInterP = (int)fcgNUQPMinP->Value;
    encPrm.encConfig.rcParams.minQP.qpInterB = (int)fcgNUQPMinB->Value;
    encPrm.encConfig.rcParams.maxQP.qpIntra  = (int)fcgNUQPMaxI->Value;
    encPrm.encConfig.rcParams.maxQP.qpInterP = (int)fcgNUQPMaxP->Value;
    encPrm.encConfig.rcParams.maxQP.qpInterB = (int)fcgNUQPMaxB->Value;
    encPrm.encConfig.rcParams.initialRCQP.qpIntra  = (int)fcgNUQPInitI->Value;
    encPrm.encConfig.rcParams.initialRCQP.qpInterP = (int)fcgNUQPInitP->Value;
    encPrm.encConfig.rcParams.initialRCQP.qpInterB = (int)fcgNUQPInitB->Value;
    encPrm.chromaQPOffset = (int)fcgNUChromaQPOffset->Value;

    encPrm.ctrl.perfMonitorSelect = fcgCBPerfMonitor->Checked ? UINT_MAX : 0;

    //H.264
    codecPrm[NV_ENC_H264].h264Config.bdirectMode = (NV_ENC_H264_BDIRECT_MODE)list_bdirect[fcgCXBDirectMode->SelectedIndex].value;
    codecPrm[NV_ENC_H264].h264Config.maxNumRefFrames = (int)fcgNURefFrames->Value;
    encPrm.input.picstruct = (RGY_PICSTRUCT)list_interlaced[fcgCXInterlaced->SelectedIndex].value;
    encPrm.encConfig.profileGUID = get_guid_from_value(h264_profile_names[fcgCXCodecProfile->SelectedIndex].value, h264_profile_names);
    codecPrm[NV_ENC_H264].h264Config.adaptiveTransformMode = (NV_ENC_H264_ADAPTIVE_TRANSFORM_MODE)list_adapt_transform[fcgCXAdaptiveTransform->SelectedIndex].value;
    codecPrm[NV_ENC_H264].h264Config.level = list_avc_level[fcgCXCodecLevel->SelectedIndex].value;
    codecPrm[NV_ENC_H264].h264Config.sliceModeData = (int)fcgNUSlices->Value;
    encPrm.bluray = fcgCBBluray->Checked;
    codecPrm[NV_ENC_H264].h264Config.entropyCodingMode = (fcgCBCABAC->Checked) ? NV_ENC_H264_ENTROPY_CODING_MODE_CABAC : NV_ENC_H264_ENTROPY_CODING_MODE_CAVLC;
    codecPrm[NV_ENC_H264].h264Config.disableDeblockingFilterIDC = false == fcgCBDeblock->Checked;

    //HEVC
    codecPrm[NV_ENC_HEVC].hevcConfig.pixelBitDepthMinus8 = list_bitdepth[fcgCXHEVCOutBitDepth->SelectedIndex].value;
    codecPrm[NV_ENC_HEVC].hevcConfig.maxNumRefFramesInDPB = (int)fcgNURefFrames->Value;
    codecPrm[NV_ENC_HEVC].hevcConfig.level = list_hevc_level[fxgCXHEVCLevel->SelectedIndex].value;
    codecPrm[NV_ENC_HEVC].hevcConfig.tier  = h265_profile_names[fcgCXHEVCTier->SelectedIndex].value;
    codecPrm[NV_ENC_HEVC].hevcConfig.maxCUSize = (NV_ENC_HEVC_CUSIZE)list_hevc_cu_size[fcgCXHEVCMaxCUSize->SelectedIndex].value;
    codecPrm[NV_ENC_HEVC].hevcConfig.minCUSize = (NV_ENC_HEVC_CUSIZE)list_hevc_cu_size[fcgCXHEVCMinCUSize->SelectedIndex].value;

    //AV1
    codecPrm[NV_ENC_AV1].av1Config.pixelBitDepthMinus8 = list_bitdepth[fcgCXOutBitDepthAV1->SelectedIndex].value;
    codecPrm[NV_ENC_AV1].av1Config.maxNumRefFramesInDPB = (int)fcgNURefFrames->Value;
    codecPrm[NV_ENC_AV1].av1Config.level = list_av1_level[fcgCXCodecLevelAV1->SelectedIndex].value;
    codecPrm[NV_ENC_AV1].av1Config.tier = av1_profile_names[fcgCXCodecProfileAV1->SelectedIndex].value;

    encPrm.common.out_vui.format    = list_videoformat[fcgCXVideoFormat->SelectedIndex].value;
    encPrm.common.out_vui.colorrange = fcgCBFullrange->Checked ? RGY_COLORRANGE_FULL : RGY_COLORRANGE_UNSPECIFIED;
    encPrm.common.out_vui.matrix    = (CspMatrix)list_colormatrix[fcgCXColorMatrix->SelectedIndex].value;
    encPrm.common.out_vui.colorprim = (CspColorprim)list_colorprim[fcgCXColorPrim->SelectedIndex].value;
    encPrm.common.out_vui.transfer  = (CspTransfer)list_transfer[fcgCXTransfer->SelectedIndex].value;
    encPrm.common.out_vui.descriptpresent = (
           fcgCXTransfer->SelectedIndex    == get_cx_index(list_transfer,    "undef")
        || fcgCXColorMatrix->SelectedIndex == get_cx_index(list_colormatrix, "undef")
        || fcgCXColorPrim->SelectedIndex   == get_cx_index(list_colorprim,   "undef")) ? 1 : 0;

    //cnf->vid.vce_ext_prm.show_performance_info = fcgCBShowPerformanceInfo->Checked;

    encPrm.common.metric.ssim       = fcgCBSSIM->Checked;
    encPrm.common.metric.psnr       = fcgCBPSNR->Checked;
    cnf->vid.afs                    = FALSE;
    cnf->vid.auo_tcfile_out         = FALSE;
    /*cnf->qsv.nPAR[0]                = (int)fcgNUAspectRatioX->Value;
    cnf->qsv.nPAR[1]                = (int)fcgNUAspectRatioY->Value;
    if (fcgCXAspectRatio->SelectedIndex == 1) {
        cnf->qsv.nPAR[0] *= -1;
        cnf->qsv.nPAR[1] *= -1;
    }*/

    //拡張部
    cnf->oth.temp_dir               = fcgCXTempDir->SelectedIndex;
    cnf->vid.afs                    = fcgCBAFS->Checked;
    cnf->vid.auo_tcfile_out         = fcgCBAuoTcfileout->Checked;
    cnf->vid.resize_enable          = fcgCBVppResize->Checked;
    cnf->vid.resize_width           = (int)fcgNUVppResizeWidth->Value;
    cnf->vid.resize_height          = (int)fcgNUVppResizeHeight->Value;
    if (cnf->vid.resize_enable) {
        encPrm.input.dstWidth = cnf->vid.resize_width;
        encPrm.input.dstHeight = cnf->vid.resize_height;
    } else {
        encPrm.input.dstWidth = 0;
        encPrm.input.dstHeight = 0;
    }

    encPrm.ctrl.loglevel.set(fcgCBLogDebug->Checked ? RGY_LOG_DEBUG : RGY_LOG_INFO, RGY_LOGT_ALL);
    encPrm.vpp.checkPerformance       = fcgCBVppPerfMonitor->Checked;

    encPrm.vpp.resize_algo            = (RGY_VPP_RESIZE_ALGO)list_vpp_resize[fcgCXVppResizeAlg->SelectedIndex].value;

    encPrm.vpp.knn.enable             = fcgCXVppDenoiseMethod->SelectedIndex == get_cx_index(list_vpp_denoise, _T("knn"));
    encPrm.vpp.knn.radius             = (int)fcgNUVppDenoiseKnnRadius->Value;
    encPrm.vpp.knn.strength           = (float)fcgNUVppDenoiseKnnStrength->Value;
    encPrm.vpp.knn.lerp_threshold     = (float)fcgNUVppDenoiseKnnThreshold->Value;

    encPrm.vpp.pmd.enable             = fcgCXVppDenoiseMethod->SelectedIndex == get_cx_index(list_vpp_denoise, _T("pmd"));
    encPrm.vpp.pmd.applyCount         = (int)fcgNUVppDenoisePmdApplyCount->Value;
    encPrm.vpp.pmd.strength           = (float)fcgNUVppDenoisePmdStrength->Value;
    encPrm.vpp.pmd.threshold          = (float)fcgNUVppDenoisePmdThreshold->Value;

    encPrm.vpp.smooth.enable          = fcgCXVppDenoiseMethod->SelectedIndex == get_cx_index(list_vpp_denoise, _T("smooth"));
    encPrm.vpp.smooth.quality         = (int)fcgNUVppDenoiseSmoothQuality->Value;
    encPrm.vpp.smooth.qp              = (int)fcgNUVppDenoiseSmoothQP->Value;

    encPrm.vpp.convolution3d.enable          = fcgCXVppDenoiseMethod->SelectedIndex == get_cx_index(list_vpp_denoise, _T("convolution3d"));
    encPrm.vpp.convolution3d.matrix          = (VppConvolution3dMatrix)list_vpp_convolution3d_matrix[fcgCXVppDenoiseConv3DMatrix->SelectedIndex].value;
    encPrm.vpp.convolution3d.threshYspatial  = (int)fcgNUVppDenoiseConv3DThreshYSpatial->Value;
    encPrm.vpp.convolution3d.threshCspatial  = (int)fcgNUVppDenoiseConv3DThreshCSpatial->Value;
    encPrm.vpp.convolution3d.threshYtemporal = (int)fcgNUVppDenoiseConv3DThreshYTemporal->Value;
    encPrm.vpp.convolution3d.threshCtemporal = (int)fcgNUVppDenoiseConv3DThreshCTemporal->Value;

    encPrm.vpp.unsharp.enable         = fcgCXVppDetailEnhance->SelectedIndex == get_cx_index(list_vpp_detail_enahance, _T("unsharp"));
    encPrm.vpp.unsharp.radius         = (int)fcgNUVppUnsharpRadius->Value;
    encPrm.vpp.unsharp.weight         = (float)fcgNUVppUnsharpWeight->Value;
    encPrm.vpp.unsharp.threshold      = (float)fcgNUVppUnsharpThreshold->Value;

    encPrm.vpp.edgelevel.enable       = fcgCXVppDetailEnhance->SelectedIndex == get_cx_index(list_vpp_detail_enahance, _T("edgelevel"));
    encPrm.vpp.edgelevel.strength     = (float)fcgNUVppEdgelevelStrength->Value;
    encPrm.vpp.edgelevel.threshold    = (float)fcgNUVppEdgelevelThreshold->Value;
    encPrm.vpp.edgelevel.black        = (float)fcgNUVppEdgelevelBlack->Value;
    encPrm.vpp.edgelevel.white        = (float)fcgNUVppEdgelevelWhite->Value;

    encPrm.vpp.warpsharp.enable       = fcgCXVppDetailEnhance->SelectedIndex == get_cx_index(list_vpp_detail_enahance, _T("warpsharp"));
    encPrm.vpp.warpsharp.threshold    = (float)fcgNUVppWarpsharpThreshold->Value;
    encPrm.vpp.warpsharp.depth        = (float)fcgNUVppWarpsharpDepth->Value;
    encPrm.vpp.warpsharp.blur         = (int)fcgNUVppWarpsharpBlur->Value;
    encPrm.vpp.warpsharp.type         = (int)fcgNUVppWarpsharpType->Value;

    encPrm.vpp.deband.enable          = fcgCBVppDebandEnable->Checked;
    encPrm.vpp.deband.range           = (int)fcgNUVppDebandRange->Value;
    encPrm.vpp.deband.threY           = (int)fcgNUVppDebandThreY->Value;
    encPrm.vpp.deband.threCb          = (int)fcgNUVppDebandThreCb->Value;
    encPrm.vpp.deband.threCr          = (int)fcgNUVppDebandThreCr->Value;
    encPrm.vpp.deband.ditherY         = (int)fcgNUVppDebandDitherY->Value;
    encPrm.vpp.deband.ditherC         = (int)fcgNUVppDebandDitherC->Value;
    encPrm.vpp.deband.sample          = list_vpp_deband_gui[fcgCXVppDebandSample->SelectedIndex].value;
    encPrm.vpp.deband.blurFirst       = fcgCBVppDebandBlurFirst->Checked;
    encPrm.vpp.deband.randEachFrame   = fcgCBVppDebandRandEachFrame->Checked;

    encPrm.vpp.afs.enable             = (fcgCXVppDeinterlace->SelectedIndex == get_cx_index(list_deinterlace_gui, L"自動フィールドシフト"));
    encPrm.vpp.afs.timecode           = encPrm.vpp.afs.enable;
    encPrm.vpp.afs.clip.top           = (int)fcgNUVppAfsUp->Value;
    encPrm.vpp.afs.clip.bottom        = (int)fcgNUVppAfsBottom->Value;
    encPrm.vpp.afs.clip.left          = (int)fcgNUVppAfsLeft->Value;
    encPrm.vpp.afs.clip.right         = (int)fcgNUVppAfsRight->Value;
    encPrm.vpp.afs.method_switch      = (int)fcgNUVppAfsMethodSwitch->Value;
    encPrm.vpp.afs.coeff_shift        = (int)fcgNUVppAfsCoeffShift->Value;
    encPrm.vpp.afs.thre_shift         = (int)fcgNUVppAfsThreShift->Value;
    encPrm.vpp.afs.thre_deint         = (int)fcgNUVppAfsThreDeint->Value;
    encPrm.vpp.afs.thre_Ymotion       = (int)fcgNUVppAfsThreYMotion->Value;
    encPrm.vpp.afs.thre_Cmotion       = (int)fcgNUVppAfsThreCMotion->Value;
    encPrm.vpp.afs.analyze            = fcgCXVppAfsAnalyze->SelectedIndex;
    encPrm.vpp.afs.shift              = fcgCBVppAfsShift->Checked;
    encPrm.vpp.afs.drop               = fcgCBVppAfsDrop->Checked;
    encPrm.vpp.afs.smooth             = fcgCBVppAfsSmooth->Checked;
    encPrm.vpp.afs.force24            = fcgCBVppAfs24fps->Checked;
    encPrm.vpp.afs.tune               = fcgCBVppAfsTune->Checked;

    encPrm.vpp.nnedi.enable           = (fcgCXVppDeinterlace->SelectedIndex == get_cx_index(list_deinterlace_gui, L"nnedi"));
    encPrm.vpp.nnedi.nsize            = (VppNnediNSize)list_vpp_nnedi_nsize[fcgCXVppNnediNsize->SelectedIndex].value;
    encPrm.vpp.nnedi.nns              = list_vpp_nnedi_nns[fcgCXVppNnediNns->SelectedIndex].value;
    encPrm.vpp.nnedi.quality          = (VppNnediQuality)list_vpp_nnedi_quality[fcgCXVppNnediQual->SelectedIndex].value;
    encPrm.vpp.nnedi.precision        = (VppFpPrecision)list_vpp_fp_prec[fcgCXVppNnediPrec->SelectedIndex].value;
    encPrm.vpp.nnedi.pre_screen       = (VppNnediPreScreen)list_vpp_nnedi_pre_screen_gui[fcgCXVppNnediPrescreen->SelectedIndex].value;
    encPrm.vpp.nnedi.errortype        = (VppNnediErrorType)list_vpp_nnedi_error_type[fcgCXVppNnediErrorType->SelectedIndex].value;

    encPrm.vpp.yadif.enable = (fcgCXVppDeinterlace->SelectedIndex == get_cx_index(list_deinterlace_gui, L"yadif"));
    encPrm.vpp.yadif.mode = (VppYadifMode)list_vpp_yadif_mode_gui[fcgCXVppYadifMode->SelectedIndex].value;

    encPrm.vpp.tweak.enable           = fcgCBVppTweakEnable->Checked;
    encPrm.vpp.tweak.brightness       = (float)fcgNUVppTweakBrightness->Value * 0.01f;
    encPrm.vpp.tweak.contrast         = (float)fcgNUVppTweakContrast->Value * 0.01f;
    encPrm.vpp.tweak.gamma            = (float)fcgNUVppTweakGamma->Value * 0.01f;
    encPrm.vpp.tweak.saturation       = (float)fcgNUVppTweakSaturation->Value * 0.01f;
    encPrm.vpp.tweak.hue              = (float)fcgNUVppTweakHue->Value;

    //音声部
    cnf->oth.out_audio_only             = fcgCBAudioOnly->Checked;
    cnf->aud.ext.faw_check              = fcgCBFAWCheck->Checked;
    cnf->aud.use_internal               = !fcgCBAudioUseExt->Checked;
    cnf->aud.ext.encoder                = fcgCXAudioEncoder->SelectedIndex;
    cnf->aud.ext.faw_check              = fcgCBFAWCheck->Checked;
    cnf->aud.ext.enc_mode               = fcgCXAudioEncMode->SelectedIndex;
    cnf->aud.ext.bitrate                = (int)fcgNUAudioBitrate->Value;
    cnf->aud.ext.use_2pass              = fcgCBAudio2pass->Checked;
    cnf->aud.ext.use_wav                = !fcgCBAudioUsePipe->Checked;
    cnf->aud.ext.delay_cut              = fcgCXAudioDelayCut->SelectedIndex;
    cnf->aud.ext.priority               = fcgCXAudioPriority->SelectedIndex;
    cnf->aud.ext.audio_encode_timing    = fcgCXAudioEncTiming->SelectedIndex;
    cnf->aud.ext.aud_temp_dir           = fcgCXAudioTempDir->SelectedIndex;
    cnf->aud.in.encoder                 = fcgCXAudioEncoderInternal->SelectedIndex;
    cnf->aud.in.faw_check               = fcgCBFAWCheck->Checked;
    cnf->aud.in.enc_mode                = fcgCXAudioEncModeInternal->SelectedIndex;
    cnf->aud.in.bitrate                 = (int)fcgNUAudioBitrateInternal->Value;

    //mux部
    cnf->mux.use_internal           = !fcgCBAudioUseExt->Checked;
    cnf->mux.disable_mp4ext         = !fcgCBMP4MuxerExt->Checked;
    cnf->mux.apple_mode             = fcgCBMP4MuxApple->Checked;
    cnf->mux.mp4_mode               = fcgCXMP4CmdEx->SelectedIndex;
    cnf->mux.mp4_temp_dir           = fcgCXMP4BoxTempDir->SelectedIndex;
    cnf->mux.disable_mkvext         = !fcgCBMKVMuxerExt->Checked;
    cnf->mux.mkv_mode               = fcgCXMKVCmdEx->SelectedIndex;
    cnf->mux.disable_mpgext         = !fcgCBMPGMuxerExt->Checked;
    cnf->mux.mpg_mode               = fcgCXMPGCmdEx->SelectedIndex;
    cnf->mux.minimized              = fcgCBMuxMinimize->Checked;
    cnf->mux.priority               = fcgCXMuxPriority->SelectedIndex;
    cnf->mux.internal_mode          = fcgCXInternalCmdEx->SelectedIndex;

    cnf->oth.run_bat                = RUN_BAT_NONE;
    cnf->oth.run_bat               |= (fcgCBRunBatBeforeAudio->Checked) ? RUN_BAT_BEFORE_AUDIO   : NULL;
    cnf->oth.run_bat               |= (fcgCBRunBatAfterAudio->Checked)  ? RUN_BAT_AFTER_AUDIO    : NULL;
    cnf->oth.run_bat               |= (fcgCBRunBatBefore->Checked)      ? RUN_BAT_BEFORE_PROCESS : NULL;
    cnf->oth.run_bat               |= (fcgCBRunBatAfter->Checked)       ? RUN_BAT_AFTER_PROCESS  : NULL;
    cnf->oth.dont_wait_bat_fin      = RUN_BAT_NONE;
    cnf->oth.dont_wait_bat_fin     |= (!fcgCBWaitForBatBefore->Checked) ? RUN_BAT_BEFORE_PROCESS : NULL;
    cnf->oth.dont_wait_bat_fin     |= (!fcgCBWaitForBatAfter->Checked)  ? RUN_BAT_AFTER_PROCESS  : NULL;
    GetCHARfromString(cnf->oth.batfile.before_process, sizeof(cnf->oth.batfile.before_process), fcgTXBatBeforePath->Text);
    GetCHARfromString(cnf->oth.batfile.after_process,  sizeof(cnf->oth.batfile.after_process),  fcgTXBatAfterPath->Text);
    GetCHARfromString(cnf->oth.batfile.before_audio, sizeof(cnf->oth.batfile.before_audio), fcgTXBatBeforeAudioPath->Text);
    GetCHARfromString(cnf->oth.batfile.after_audio,  sizeof(cnf->oth.batfile.after_audio),  fcgTXBatAfterAudioPath->Text);


    GetfcgTSLSettingsNotes(cnf->oth.notes, sizeof(cnf->oth.notes));

    GetCHARfromString(cnf->enc.cmdex, sizeof(cnf->enc.cmdex), fcgTXCmdEx->Text);
    strcpy_s(cnf->enc.cmd, gen_cmd(&encPrm, codecPrm, true).c_str());

    return String(gen_cmd(&encPrm, codecPrm, false).c_str()).ToString();
}

System::Void frmConfig::GetfcgTSLSettingsNotes(char *notes, int nSize) {
    ZeroMemory(notes, nSize);
    if (fcgTSLSettingsNotes->Overflow != ToolStripItemOverflow::Never)
        GetCHARfromString(notes, nSize, fcgTSLSettingsNotes->Text);
}

System::Void frmConfig::SetfcgTSLSettingsNotes(const char *notes) {
    if (str_has_char(notes)) {
        fcgTSLSettingsNotes->ForeColor = Color::FromArgb(StgNotesColor[0][0], StgNotesColor[0][1], StgNotesColor[0][2]);
        fcgTSLSettingsNotes->Text = String(notes).ToString();
        fcgTSLSettingsNotes->Overflow = ToolStripItemOverflow::AsNeeded;
    } else {
        fcgTSLSettingsNotes->ForeColor = Color::FromArgb(StgNotesColor[1][0], StgNotesColor[1][1], StgNotesColor[1][2]);
        fcgTSLSettingsNotes->Text = LOAD_CLI_STRING(AuofcgTSTSettingsNotes);
        fcgTSLSettingsNotes->Overflow = ToolStripItemOverflow::Never;
    }
}

System::Void frmConfig::SetfcgTSLSettingsNotes(String^ notes) {
    if (notes->Length && fcgTSLSettingsNotes->Overflow != ToolStripItemOverflow::Never) {
        fcgTSLSettingsNotes->ForeColor = Color::FromArgb(StgNotesColor[0][0], StgNotesColor[0][1], StgNotesColor[0][2]);
        fcgTSLSettingsNotes->Text = notes;
        fcgTSLSettingsNotes->Overflow = ToolStripItemOverflow::AsNeeded;
    } else {
        fcgTSLSettingsNotes->ForeColor = Color::FromArgb(StgNotesColor[1][0], StgNotesColor[1][1], StgNotesColor[1][2]);
        fcgTSLSettingsNotes->Text = LOAD_CLI_STRING(AuofcgTSTSettingsNotes);
        fcgTSLSettingsNotes->Overflow = ToolStripItemOverflow::Never;
    }
}

System::Void frmConfig::SetChangedEvent(Control^ control, System::EventHandler^ _event) {
    System::Type^ ControlType = control->GetType();
    if (ControlType == NumericUpDown::typeid)
        ((NumericUpDown^)control)->ValueChanged += _event;
    else if (ControlType == ComboBox::typeid)
        ((ComboBox^)control)->SelectedIndexChanged += _event;
    else if (ControlType == CheckBox::typeid)
        ((CheckBox^)control)->CheckedChanged += _event;
    else if (ControlType == TextBox::typeid)
        ((TextBox^)control)->TextChanged += _event;
}

System::Void frmConfig::SetToolStripEvents(ToolStrip^ TS, System::Windows::Forms::MouseEventHandler^ _event) {
    for (int i = 0; i < TS->Items->Count; i++) {
        ToolStripButton^ TSB = dynamic_cast<ToolStripButton^>(TS->Items[i]);
        if (TSB != nullptr) TSB->MouseDown += _event;
    }
}

System::Void frmConfig::TabControl_DarkDrawItem(System::Object^ sender, DrawItemEventArgs^ e) {
    //対象のTabControlを取得
    TabControl^ tab = dynamic_cast<TabControl^>(sender);
    //タブページのテキストを取得
    System::String^ txt = tab->TabPages[e->Index]->Text;

    //タブのテキストと背景を描画するためのブラシを決定する
    SolidBrush^ foreBrush = gcnew System::Drawing::SolidBrush(ColorfromInt(DEFAULT_UI_COLOR_TEXT_DARK));
    SolidBrush^ backBrush = gcnew System::Drawing::SolidBrush(ColorfromInt(DEFAULT_UI_COLOR_BASE_DARK));

    //StringFormatを作成
    StringFormat^ sf = gcnew System::Drawing::StringFormat();
    //中央に表示する
    sf->Alignment = StringAlignment::Center;
    sf->LineAlignment = StringAlignment::Center;

    //背景の描画
    e->Graphics->FillRectangle(backBrush, e->Bounds);
    //Textの描画
    e->Graphics->DrawString(txt, e->Font, foreBrush, e->Bounds, sf);
}

System::Void frmConfig::fcgMouseEnter_SetColor(System::Object^  sender, System::EventArgs^  e) {
    fcgMouseEnterLeave_SetColor(sender, themeMode, DarkenWindowState::Hot, dwStgReader);
}
System::Void frmConfig::fcgMouseLeave_SetColor(System::Object^  sender, System::EventArgs^  e) {
    fcgMouseEnterLeave_SetColor(sender, themeMode, DarkenWindowState::Normal, dwStgReader);
}

System::Void frmConfig::SetAllMouseMove(Control ^top, const AuoTheme themeTo) {
    if (themeTo == themeMode) return;
    System::Type^ type = top->GetType();
    if (type == CheckBox::typeid /* || isToolStripItem(type)*/) {
        top->MouseEnter += gcnew System::EventHandler(this, &frmConfig::fcgMouseEnter_SetColor);
        top->MouseLeave += gcnew System::EventHandler(this, &frmConfig::fcgMouseLeave_SetColor);
    } else if (type == ToolStrip::typeid) {
        ToolStrip^ TS = dynamic_cast<ToolStrip^>(top);
        for (int i = 0; i < TS->Items->Count; i++) {
            auto item = TS->Items[i];
            item->MouseEnter += gcnew System::EventHandler(this, &frmConfig::fcgMouseEnter_SetColor);
            item->MouseLeave += gcnew System::EventHandler(this, &frmConfig::fcgMouseLeave_SetColor);
        }
    }
    for (int i = 0; i < top->Controls->Count; i++) {
        SetAllMouseMove(top->Controls[i], themeTo);
    }
}

System::Void frmConfig::CheckTheme() {
    //DarkenWindowが使用されていれば設定をロードする
    if (dwStgReader != nullptr) delete dwStgReader;
    const auto [themeTo, dwStg] = check_current_theme(sys_dat->aviutl_dir);
    dwStgReader = dwStg;

    //変更の必要がなければ終了
    if (themeTo == themeMode) return;

    //一度ウィンドウの再描画を完全に抑止する
    SendMessage(reinterpret_cast<HWND>(this->Handle.ToPointer()), WM_SETREDRAW, 0, 0);
#if 0
    //tabcontrolのborderを隠す
    SwitchComboBoxBorder(fcgtabControlVideo, fcgPNHideTabControlVideo, themeMode, themeTo, dwStgReader);
    SwitchComboBoxBorder(fcgtabControlAudio, fcgPNHideTabControlAudio, themeMode, themeTo, dwStgReader);
    SwitchComboBoxBorder(fcgtabControlMux,   fcgPNHideTabControlMux,   themeMode, themeTo, dwStgReader);
#endif
    //上部のtoolstripborderを隠すためのパネル
    fcgPNHideToolStripBorder->Visible = themeTo == AuoTheme::DarkenWindowDark;
#if 0
    //TabControlをオーナードローする
    fcgtabControlVideo->DrawMode = TabDrawMode::OwnerDrawFixed;
    fcgtabControlVideo->DrawItem += gcnew DrawItemEventHandler(this, &frmConfig::TabControl_DarkDrawItem);

    fcgtabControlAudio->DrawMode = TabDrawMode::OwnerDrawFixed;
    fcgtabControlAudio->DrawItem += gcnew DrawItemEventHandler(this, &frmConfig::TabControl_DarkDrawItem);

    fcgtabControlMux->DrawMode = TabDrawMode::OwnerDrawFixed;
    fcgtabControlMux->DrawItem += gcnew DrawItemEventHandler(this, &frmConfig::TabControl_DarkDrawItem);
#endif
    if (themeTo != themeMode) {
        SetAllColor(this, themeTo, this->GetType(), dwStgReader);
        SetAllMouseMove(this, themeTo);
    }
    //一度ウィンドウの再描画を再開し、強制的に再描画させる
    SendMessage(reinterpret_cast<HWND>(this->Handle.ToPointer()), WM_SETREDRAW, 1, 0);
    this->Refresh();
    themeMode = themeTo;
}

System::Void frmConfig::SetAllCheckChangedEvents(Control ^top) {
    //再帰を使用してすべてのコントロールのtagを調べ、イベントをセットする
    for (int i = 0; i < top->Controls->Count; i++) {
        System::Type^ type = top->Controls[i]->GetType();
        if (type == NumericUpDown::typeid)
            top->Controls[i]->Enter += gcnew System::EventHandler(this, &frmConfig::NUSelectAll);

        if (type == Label::typeid || type == Button::typeid)
            ;
        else if (type == ToolStrip::typeid)
            SetToolStripEvents((ToolStrip^)(top->Controls[i]), gcnew System::Windows::Forms::MouseEventHandler(this, &frmConfig::fcgTSItem_MouseDown));
        else if (top->Controls[i]->Tag == nullptr)
            SetAllCheckChangedEvents(top->Controls[i]);
        else if (String::Equals(top->Controls[i]->Tag->ToString(), L"reCmd"))
            SetChangedEvent(top->Controls[i], gcnew System::EventHandler(this, &frmConfig::fcgRebuildCmd));
        else if (String::Equals(top->Controls[i]->Tag->ToString(), L"chValue"))
            SetChangedEvent(top->Controls[i], gcnew System::EventHandler(this, &frmConfig::CheckOtherChanges));
        else
            SetAllCheckChangedEvents(top->Controls[i]);
    }
}

System::Void frmConfig::SetHelpToolTipsColorMatrix(Control^ control, const CX_DESC *list, const wchar_t *type) {
    fcgTTEx->SetToolTip(control, L"--" + String(type).ToString() + L"\n"
        + LOAD_CLI_STRING(AuofrmTTColorMatrix1) + L"\n"
        + LOAD_CLI_STRING(AuofrmTTColorMatrix2) + L"\n"
        + LOAD_CLI_STRING(AuofrmTTColorMatrix3) + L" " + HD_HEIGHT_THRESHOLD + L" " + LOAD_CLI_STRING(AuofrmTTColorMatrix4) + L" … " + String(COLOR_VALUE_AUTO_HD_NAME).ToString() + L"\n"
        + LOAD_CLI_STRING(AuofrmTTColorMatrix3) + L" " + HD_HEIGHT_THRESHOLD + L" " + LOAD_CLI_STRING(AuofrmTTColorMatrix5) + L" … " + String(COLOR_VALUE_AUTO_SD_NAME).ToString() + L"\n"
        + LOAD_CLI_STRING(AuofrmTTColorMatrix6)
    );
}

System::Void frmConfig::SetHelpToolTips() {

#define SET_TOOL_TIP_EX2(target, x) { fcgTTEx->SetToolTip(target, LOAD_CLI_STRING(AuofrmTT ## x)); }
#define SET_TOOL_TIP_EX(target) { fcgTTEx->SetToolTip(target, LOAD_CLI_STRING(AuofrmTT ## target)); }
#define SET_TOOL_TIP_EX_AUD_INTERNAL(target) { fcgTTEx->SetToolTip(target ## Internal, LOAD_CLI_STRING(AuofrmTT ## target)); }


    SET_TOOL_TIP_EX(fcgTXVideoEncoderPath);
    SET_TOOL_TIP_EX(fcgCXEncCodec);
    SET_TOOL_TIP_EX(fcgCXEncMode);
    SET_TOOL_TIP_EX(fcgCXQualityPreset);
    SET_TOOL_TIP_EX(fcgCXMultiPass);
    SET_TOOL_TIP_EX(fcgNUQPI);
    SET_TOOL_TIP_EX(fcgNUQPP);
    SET_TOOL_TIP_EX(fcgNUQPB);
    //SET_TOOL_TIP_EX(fcgNUVBVBufsize);
    SET_TOOL_TIP_EX(fcgNUGopLength);
    SET_TOOL_TIP_EX(fcgNUBframes);
    SET_TOOL_TIP_EX(fcgCXBrefMode);
    SET_TOOL_TIP_EX(fcgNURefFrames);
    SET_TOOL_TIP_EX(fcgNULookaheadDepth);
    SET_TOOL_TIP_EX(fcgCBWeightP);
    SET_TOOL_TIP_EX(fcgNUBitrate);
    SET_TOOL_TIP_EX(fcgNUMaxkbps);
    SET_TOOL_TIP_EX(fcgNUVBRTragetQuality);
    SET_TOOL_TIP_EX(fcgCXInterlaced);
    SET_TOOL_TIP_EX(fcgCXHEVCOutBitDepth);
    SET_TOOL_TIP_EX(fcgCXHEVCTier);
    SET_TOOL_TIP_EX(fxgCXHEVCLevel);
    SET_TOOL_TIP_EX(fcgCXOutBitDepthAV1);
    SET_TOOL_TIP_EX(fcgCXCodecProfileAV1);
    SET_TOOL_TIP_EX(fcgCXCodecLevelAV1);
    SET_TOOL_TIP_EX(fcgCBBluray);
    SET_TOOL_TIP_EX(fcgCXCodecProfile);
    SET_TOOL_TIP_EX(fcgCXCodecLevel);
    SET_TOOL_TIP_EX(fcgCXAQ);
    SET_TOOL_TIP_EX(fcgNUAQStrength);
    SET_TOOL_TIP_EX(fcgCXVideoFormat);
    SET_TOOL_TIP_EX(fcgCBFullrange);
    SET_TOOL_TIP_EX(fcgCXDevice);
    SET_TOOL_TIP_EX(fcgCXCudaSchdule);
    SET_TOOL_TIP_EX(fcgCBLossless);
    SET_TOOL_TIP_EX(fcgNUQPMinI);
    SET_TOOL_TIP_EX(fcgNUQPMinP);
    SET_TOOL_TIP_EX(fcgNUQPMinB);
    SET_TOOL_TIP_EX(fcgNUQPMaxI);
    SET_TOOL_TIP_EX(fcgNUQPMaxP);
    SET_TOOL_TIP_EX(fcgNUQPMaxB);
    SET_TOOL_TIP_EX(fcgNUQPInitI);
    SET_TOOL_TIP_EX(fcgNUQPInitP);
    SET_TOOL_TIP_EX(fcgNUQPInitB);
    SET_TOOL_TIP_EX(fcgCBQPMin);
    SET_TOOL_TIP_EX(fcgCBQPMax);
    SET_TOOL_TIP_EX(fcgCBQPInit);
    SET_TOOL_TIP_EX(fcgNUChromaQPOffset);
    SET_TOOL_TIP_EX(fcgNUSlices);
    SET_TOOL_TIP_EX(fcgCBSSIM);
    SET_TOOL_TIP_EX(fcgCBPSNR);
    SET_TOOL_TIP_EX(fcgCBCABAC);
    SET_TOOL_TIP_EX(fcgCBDeblock);

    SetHelpToolTipsColorMatrix(fcgCXColorMatrix, list_colormatrix, L"colormatrix");
    SetHelpToolTipsColorMatrix(fcgCXColorPrim, list_colorprim, L"colorprim");
    SetHelpToolTipsColorMatrix(fcgCXTransfer, list_transfer, L"transfer");

    fcgTTEx->SetToolTip(fcgCXAspectRatio, L""
        + LOAD_CLI_STRING(aspect_desc[0].mes) + L"\n"
        + L"   " + LOAD_CLI_STRING(AuofrmTTfcgCXAspectRatioSAR) + L"\n"
        + L"\n"
        + LOAD_CLI_STRING(aspect_desc[1].mes) + L"\n"
        + L"   " + LOAD_CLI_STRING(AuofrmTTfcgCXAspectRatioDAR) + L"\n"
    );
    SET_TOOL_TIP_EX(fcgNUAspectRatioX);
    SET_TOOL_TIP_EX(fcgNUAspectRatioY);

    //フィルタ
    SET_TOOL_TIP_EX(fcgCBVppResize);
    SET_TOOL_TIP_EX(fcgCXVppResizeAlg);
    SET_TOOL_TIP_EX(fcgCXVppDenoiseMethod);
    SET_TOOL_TIP_EX(fcgNUVppDenoisePmdApplyCount);
    SET_TOOL_TIP_EX(fcgNUVppDenoisePmdStrength);
    SET_TOOL_TIP_EX(fcgNUVppDenoisePmdThreshold);
    SET_TOOL_TIP_EX(fcgCXVppDenoiseConv3DMatrix);
    SET_TOOL_TIP_EX(fcgNUVppDenoiseConv3DThreshYSpatial);
    SET_TOOL_TIP_EX(fcgNUVppDenoiseConv3DThreshCSpatial);
    SET_TOOL_TIP_EX(fcgNUVppDenoiseConv3DThreshYTemporal);
    SET_TOOL_TIP_EX(fcgNUVppDenoiseConv3DThreshCTemporal);
    SET_TOOL_TIP_EX(fcgNUVppDenoiseSmoothQuality);
    SET_TOOL_TIP_EX(fcgNUVppDenoiseSmoothQP);
    SET_TOOL_TIP_EX(fcgNUVppDenoiseKnnRadius);
    SET_TOOL_TIP_EX(fcgNUVppDenoiseKnnStrength);
    SET_TOOL_TIP_EX(fcgNUVppDenoiseKnnThreshold);
    SET_TOOL_TIP_EX(fcgCXVppDetailEnhance);
    SET_TOOL_TIP_EX(fcgNUVppWarpsharpBlur);
    SET_TOOL_TIP_EX(fcgNUVppWarpsharpThreshold);
    SET_TOOL_TIP_EX(fcgNUVppWarpsharpType);
    SET_TOOL_TIP_EX(fcgNUVppWarpsharpDepth);
    SET_TOOL_TIP_EX(fcgNUVppEdgelevelStrength);
    SET_TOOL_TIP_EX(fcgNUVppEdgelevelThreshold);
    SET_TOOL_TIP_EX(fcgNUVppEdgelevelBlack);
    SET_TOOL_TIP_EX(fcgNUVppEdgelevelWhite);
    SET_TOOL_TIP_EX(fcgNUVppUnsharpRadius);
    SET_TOOL_TIP_EX(fcgNUVppUnsharpThreshold);
    SET_TOOL_TIP_EX(fcgNUVppUnsharpWeight);
    SET_TOOL_TIP_EX(fcgCBVppDebandEnable);
    SET_TOOL_TIP_EX(fcgNUVppDebandRange);
    SET_TOOL_TIP_EX(fcgNUVppDebandThreY);
    SET_TOOL_TIP_EX(fcgNUVppDebandThreCb);
    SET_TOOL_TIP_EX(fcgNUVppDebandThreCr);
    SET_TOOL_TIP_EX(fcgNUVppDebandDitherY);
    SET_TOOL_TIP_EX(fcgNUVppDebandDitherC);
    SET_TOOL_TIP_EX(fcgCXVppDebandSample);
    SET_TOOL_TIP_EX(fcgCBVppDebandBlurFirst);
    SET_TOOL_TIP_EX(fcgCBVppDebandRandEachFrame);
    SET_TOOL_TIP_EX(fcgCXVppDeinterlace);
    SET_TOOL_TIP_EX(fcgNUVppAfsUp);
    SET_TOOL_TIP_EX(fcgNUVppAfsBottom);
    SET_TOOL_TIP_EX(fcgNUVppAfsLeft);
    SET_TOOL_TIP_EX(fcgNUVppAfsRight);
    SET_TOOL_TIP_EX(fcgNUVppAfsMethodSwitch);
    SET_TOOL_TIP_EX(fcgTBVppAfsMethodSwitch);
    SET_TOOL_TIP_EX(fcgNUVppAfsCoeffShift);
    SET_TOOL_TIP_EX(fcgTBVppAfsCoeffShift);
    SET_TOOL_TIP_EX(fcgNUVppAfsThreShift);
    SET_TOOL_TIP_EX(fcgTBVppAfsThreShift);
    SET_TOOL_TIP_EX(fcgNUVppAfsThreDeint);
    SET_TOOL_TIP_EX(fcgTBVppAfsThreDeint);
    SET_TOOL_TIP_EX(fcgNUVppAfsThreYMotion);
    SET_TOOL_TIP_EX(fcgTBVppAfsThreYMotion);
    SET_TOOL_TIP_EX(fcgNUVppAfsThreCMotion);
    SET_TOOL_TIP_EX(fcgTBVppAfsThreCMotion);
    SET_TOOL_TIP_EX(fcgCXVppAfsAnalyze);
    SET_TOOL_TIP_EX(fcgCBVppAfsShift);
    SET_TOOL_TIP_EX(fcgCBVppAfs24fps);
    SET_TOOL_TIP_EX(fcgCBVppAfsDrop);
    SET_TOOL_TIP_EX(fcgCBVppAfsSmooth);
    SET_TOOL_TIP_EX(fcgCBVppAfsTune);
    SET_TOOL_TIP_EX(fcgCXVppYadifMode);
    SET_TOOL_TIP_EX(fcgCXVppNnediNns);
    SET_TOOL_TIP_EX(fcgCXVppNnediNsize);
    SET_TOOL_TIP_EX(fcgCXVppNnediQual);
    SET_TOOL_TIP_EX(fcgCXVppNnediPrec);
    SET_TOOL_TIP_EX(fcgCXVppNnediPrescreen);
    SET_TOOL_TIP_EX(fcgCXVppNnediErrorType);
    SET_TOOL_TIP_EX(fcgCBVppTweakEnable);
    SET_TOOL_TIP_EX(fcgNUVppTweakBrightness);
    SET_TOOL_TIP_EX(fcgTBVppTweakBrightness);
    SET_TOOL_TIP_EX(fcgNUVppTweakContrast);
    SET_TOOL_TIP_EX(fcgTBVppTweakContrast);
    SET_TOOL_TIP_EX(fcgNUVppTweakGamma);
    SET_TOOL_TIP_EX(fcgTBVppTweakGamma);
    SET_TOOL_TIP_EX(fcgNUVppTweakSaturation);
    SET_TOOL_TIP_EX(fcgTBVppTweakSaturation);
    SET_TOOL_TIP_EX(fcgNUVppTweakHue);
    SET_TOOL_TIP_EX(fcgTBVppTweakHue);

    //拡張
    SET_TOOL_TIP_EX(fcgCBAFS);
    SET_TOOL_TIP_EX(fcgCBAuoTcfileout);
    SET_TOOL_TIP_EX(fcgCXTempDir);
    SET_TOOL_TIP_EX(fcgBTCustomTempDir);

    //音声
    SET_TOOL_TIP_EX(fcgCBAudioUseExt);
    SET_TOOL_TIP_EX_AUD_INTERNAL(fcgCXAudioEncoder);
    SET_TOOL_TIP_EX_AUD_INTERNAL(fcgCXAudioEncMode);
    SET_TOOL_TIP_EX_AUD_INTERNAL(fcgNUAudioBitrate);
    SET_TOOL_TIP_EX(fcgCXAudioEncoder);
    SET_TOOL_TIP_EX(fcgCBAudioOnly);
    SET_TOOL_TIP_EX(fcgCBFAWCheck);
    SET_TOOL_TIP_EX(fcgBTAudioEncoderPath);
    SET_TOOL_TIP_EX(fcgCXAudioEncMode);
    SET_TOOL_TIP_EX(fcgCBAudio2pass);
    SET_TOOL_TIP_EX(fcgCBAudioUsePipe);
    SET_TOOL_TIP_EX(fcgNUAudioBitrate);
    SET_TOOL_TIP_EX(fcgCXAudioPriority);
    SET_TOOL_TIP_EX(fcgCXAudioEncTiming);
    SET_TOOL_TIP_EX(fcgCXAudioTempDir);
    SET_TOOL_TIP_EX(fcgBTCustomAudioTempDir);
    //音声バッチファイル実行
    SET_TOOL_TIP_EX(fcgCBRunBatBeforeAudio);
    SET_TOOL_TIP_EX(fcgCBRunBatAfterAudio);
    SET_TOOL_TIP_EX(fcgBTBatBeforeAudioPath);
    SET_TOOL_TIP_EX(fcgBTBatAfterAudioPath);
    //muxer
    SET_TOOL_TIP_EX(fcgCBMP4MuxerExt);
    SET_TOOL_TIP_EX(fcgCXMP4CmdEx);
    SET_TOOL_TIP_EX(fcgBTMP4MuxerPath);
    SET_TOOL_TIP_EX(fcgBTTC2MP4Path);
    SET_TOOL_TIP_EX(fcgBTMP4RawPath);
    SET_TOOL_TIP_EX(fcgCXMP4BoxTempDir);
    SET_TOOL_TIP_EX(fcgBTMP4BoxTempDir);
    SET_TOOL_TIP_EX(fcgCBMKVMuxerExt);
    SET_TOOL_TIP_EX(fcgCXMKVCmdEx);
    SET_TOOL_TIP_EX(fcgBTMKVMuxerPath);
    SET_TOOL_TIP_EX(fcgCBMPGMuxerExt);
    SET_TOOL_TIP_EX(fcgCXMPGCmdEx);
    SET_TOOL_TIP_EX(fcgBTMPGMuxerPath);
    SET_TOOL_TIP_EX(fcgCXMuxPriority);
    //バッチファイル実行
    SET_TOOL_TIP_EX(fcgCBRunBatBefore);
    SET_TOOL_TIP_EX(fcgCBRunBatAfter);
    SET_TOOL_TIP_EX(fcgCBWaitForBatBefore);
    SET_TOOL_TIP_EX(fcgCBWaitForBatAfter);
    SET_TOOL_TIP_EX(fcgBTBatBeforePath);
    SET_TOOL_TIP_EX(fcgBTBatAfterPath);
    //上部ツールストリップ
    fcgTSBDelete->ToolTipText = LOAD_CLI_STRING(AuofrmTTfcgTSBDelete);
    fcgTSBOtherSettings->ToolTipText = LOAD_CLI_STRING(AuofrmTTfcgTSBOtherSettings);
    fcgTSBSave->ToolTipText = LOAD_CLI_STRING(AuofrmTTfcgTSBSave);
    fcgTSBSaveNew->ToolTipText = LOAD_CLI_STRING(AuofrmTTfcgTSBSaveNew);

    //他
    SET_TOOL_TIP_EX(fcgTXCmd);
    SET_TOOL_TIP_EX(fcgBTDefault);
}
System::Void frmConfig::ShowExehelp(String^ ExePath, String^ args) {
    if (!File::Exists(ExePath)) {
        MessageBox::Show(L"指定された実行ファイルが存在しません。", L"エラー", MessageBoxButtons::OK, MessageBoxIcon::Error);
    } else {
        char exe_path[MAX_PATH_LEN];
        char file_path[MAX_PATH_LEN];
        char cmd[MAX_CMD_LEN];
        GetCHARfromString(exe_path, sizeof(exe_path), ExePath);
        apply_appendix(file_path, _countof(file_path), exe_path, "_fullhelp.txt");
        File::Delete(String(file_path).ToString());
        array<String^>^ arg_list = args->Split(L';');
        for (int i = 0; i < arg_list->Length; i++) {
            if (i) {
                System::IO::StreamWriter^ sw;
                try {
                    sw = gcnew System::IO::StreamWriter(String(file_path).ToString(), true, System::Text::Encoding::GetEncoding("shift_jis"));
                    sw->WriteLine();
                    sw->WriteLine();
                } catch (...) {
                    //ファイルオープンに失敗…初回のget_exe_message_to_fileでエラーとなるため、おそらく起こらない
                } finally {
                    if (sw != nullptr) { sw->Close(); }
                }
            }
            GetCHARfromString(cmd, sizeof(cmd), arg_list[i]);
            if (get_exe_message_to_file(exe_path, cmd, file_path, AUO_PIPE_MUXED, 5) != RP_SUCCESS) {
                File::Delete(String(file_path).ToString());
                MessageBox::Show(L"helpの取得に失敗しました。", L"エラー", MessageBoxButtons::OK, MessageBoxIcon::Error);
                return;
            }
        }
        try {
            System::Diagnostics::Process::Start(String(file_path).ToString());
        } catch (...) {
            MessageBox::Show(L"helpを開く際に不明なエラーが発生しました。", L"エラー", MessageBoxButtons::OK, MessageBoxIcon::Error);
        }
    }
}

System::Void frmConfig::SetVidEncInfo(VidEncInfo info) {
    if (this->InvokeRequired) {
        this->Invoke(gcnew SetVidEncInfoDelegate(this, &frmConfig::SetVidEncInfo), info);
    } else {
        int oldIdx = fcgCXDevice->SelectedIndex;
        fcgCXDevice->Items->Clear();
        fcgCXDevice->Items->Add(String(_T("Auto")).ToString());
        for (int i = 0; i < nvencInfo.devices->Count; i++) {
            fcgCXDevice->Items->Add(nvencInfo.devices[i]);
        }
        //取得できたデバイス数よりもインデックスが大きければ自動(0)に戻す
        SetCXIndex(fcgCXDevice, oldIdx >= nvencInfo.devices->Count ? oldIdx : 0);
        fcgPBNVEncLogoEnabled->Visible = nvencInfo.hwencAvail;
        //HEVCエンコができるか
        if (!nvencInfo.hevcEnc) {
            fcgCXEncCodec->Items[NV_ENC_HEVC] = L"-------------";
        }
        if (!nvencInfo.av1Enc) {
            fcgCXEncCodec->Items[NV_ENC_AV1] = L"-------------";
        }
        fcgCXEncCodec->SelectedIndexChanged += gcnew System::EventHandler(this, &frmConfig::fcgCodecChanged);
        fcgCodecChanged(nullptr, nullptr);
    }
}

VidEncInfo frmConfig::GetVidEncInfo() {
    char exe_path[MAX_PATH_LEN];
    std::vector<char> mes(16384, 0);

    VidEncInfo info;
    info.hwencAvail = false;
    info.h264Enc = false;
    info.hevcEnc = false;
    info.av1Enc = false;
    if (info.devices != nullptr) {
        info.devices->Clear();
    }
    if (File::Exists(LocalStg.vidEncPath)) {
        GetCHARfromString(exe_path, sizeof(exe_path), LocalStg.vidEncPath);
    } else {
        const auto defaultExePath = find_latest_videnc_for_frm();
        if (defaultExePath.length() > 0) {
            strcpy_s(exe_path, defaultExePath.c_str());
        }
    }
    if (get_exe_message(exe_path, "--check-device", mes.data(), mes.size(), AUO_PIPE_MUXED) == RP_SUCCESS) {
        info.devices = gcnew List<String^>();
        auto lines = String(mes.data()).ToString()->Split(String(L"\r\n").ToString()->ToCharArray(), System::StringSplitOptions::RemoveEmptyEntries);
        for (int i = 0; i < lines->Length; i++) {
            if (lines[i]->Contains("DeviceId")) {
                info.devices->Add(lines[i]->Replace("DeviceId ", ""));
            }
        }
    }

    if (get_exe_message(exe_path, "--check-hw", mes.data(), mes.size(), AUO_PIPE_MUXED) == RP_SUCCESS) {
        auto lines = String(mes.data()).ToString()->Split(String(L"\r\n").ToString()->ToCharArray(), System::StringSplitOptions::RemoveEmptyEntries);
        bool check_codecs = false;
        for (int i = 0; i < lines->Length; i++) {
            if (lines[i]->Contains("Avaliable Codec")) {
                check_codecs = true;
            } else if (check_codecs) {
                if (lines[i]->ToLower()->Contains("h.264")) {
                    info.h264Enc = true;
                } else if (lines[i]->ToLower()->Contains("hevc")) {
                    info.hevcEnc = true;
                } else if (lines[i]->ToLower()->Contains("av1")) {
                    info.av1Enc = true;
                }
            }
        }
    }
    if (info.h264Enc || info.hevcEnc || info.av1Enc) {
        info.hwencAvail = true;
    }
    nvencInfo = info;
    SetVidEncInfo(info);
    return info;
}

System::Void frmConfig::GetVidEncInfoAsync() {
    if (!File::Exists(LocalStg.vidEncPath)) {
        const auto defaultExePath = find_latest_videnc_for_frm();
        if (defaultExePath.length() == 0) {
            nvencInfo.hwencAvail = false;
            nvencInfo.h264Enc = false;
            nvencInfo.hevcEnc = false;
            nvencInfo.av1Enc = false;
            return;
        }
        String^ exePath = String(defaultExePath.c_str()).ToString();
        if (!File::Exists(exePath)) {
            nvencInfo.hwencAvail = false;
            nvencInfo.h264Enc = false;
            nvencInfo.hevcEnc = false;
            nvencInfo.av1Enc = false;
            return;
        }
    }
    if (taskNVEncInfo != nullptr && !taskNVEncInfo->IsCompleted) {
        taskNVEncInfoCancell->Cancel();
    }
    taskNVEncInfoCancell = gcnew CancellationTokenSource();
    taskNVEncInfo = Task<VidEncInfo>::Factory->StartNew(gcnew Func<VidEncInfo>(this, &frmConfig::GetVidEncInfo), taskNVEncInfoCancell->Token);
}
#pragma warning( pop )
