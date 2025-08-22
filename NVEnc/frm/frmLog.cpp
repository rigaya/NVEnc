﻿// -----------------------------------------------------------------------------------------
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

#include "frmLog.h"
#include "string.h"

#include "auo_mes.h"
#include "auo_version.h"

using namespace AUO_NAME_R;

//すべてfrmLog::Instanceを通じてアクセス

//ログウィンドウを表示させる
[STAThreadAttribute]
void show_log_window(const TCHAR *aviutl_dir, BOOL disable_visual_styles) {
    if (!disable_visual_styles)
        System::Windows::Forms::Application::EnableVisualStyles();
    System::IO::Directory::SetCurrentDirectory(String(aviutl_dir).ToString());
    frmLog::Instance::get()->Show();
    frmLog::Instance::get()->SetWindowTitle(g_auo_mes.get(AUO_GUIEX_FULL_NAME), PROGRESSBAR_DISABLED);
}
//ログウィンドウのタイトルを設定
[STAThreadAttribute]
void set_window_title(const wchar_t *chr) {
    if (!frmLog::Instance::get()->InvokeRequired)
        frmLog::Instance::get()->SetWindowTitle(chr);
}
[STAThreadAttribute]
void set_window_title(const wchar_t *chr, int progress_mode) {
    if (!frmLog::Instance::get()->InvokeRequired)
        frmLog::Instance::get()->SetWindowTitle(chr, progress_mode);
}
//メッセージをログウィンドウのタイトルに
[STAThreadAttribute]
void set_window_title_enc_mes(const wchar_t *chr, int total_drop, int frame_n) {
    frmLog::Instance::get()->SetWindowTitleX264Mes(chr, total_drop, frame_n);
}
//メッセージをログウィンドウに表示
[STAThreadAttribute]
void write_log_auo_line(int log_type_index, const wchar_t *chr) {
    frmLog::Instance::get()->WriteLogAuoLine(String(chr).ToString(), log_type_index);
}
//現在実行中の内容の設定
[STAThreadAttribute]
void set_task_name(const wchar_t *chr) {
    if (!frmLog::Instance::get()->InvokeRequired)
        frmLog::Instance::get()->SetTaskName(chr);
}
//進捗情報の表示
[STAThreadAttribute]
void set_log_progress(double progress) {
    if (!frmLog::Instance::get()->InvokeRequired)
        frmLog::Instance::get()->SetProgress(progress);
}
//メッセージを直接ログウィンドウに表示
[STAThreadAttribute]
void write_log_line(int log_type_index, const wchar_t *chr) {
    frmLog::Instance::get()->WriteLogLine(String(chr).ToString(), log_type_index);
}
//音声を並列に処理する際に、蓄えた音声のログを表示
//必ず音声処理が動いていないところで呼ぶこと!
void flush_audio_log() {
    frmLog::Instance::get()->FlushAudioLogCache();
}
//ログウィンドウからのx264制御を有効化
[STAThreadAttribute]
void enable_enc_control(DWORD *priority, bool *enc_pause, BOOL afs, BOOL add_progress, DWORD start_time, int _total_frame) {
    frmLog::Instance::get()->EnableEncControl(priority, enc_pause, afs, add_progress, start_time, _total_frame);
}
//ログウィンドウからのx264制御を無効化
[STAThreadAttribute]
void disable_enc_control() {
    frmLog::Instance::get()->DisableEncControl();
}
//ログウィンドウを閉じられるかどうかを設定
[STAThreadAttribute]
void set_prevent_log_close(BOOL prevent) {
    frmLog::Instance::get()->SetPreventLogWindowClosing(prevent);
}
//自動ログ保存を実行
[STAThreadAttribute]
void auto_save_log_file(const TCHAR *log_filepath) {
    frmLog::Instance::get()->AutoSaveLogFile(log_filepath);
}
//ログウィンドウに設定を再ロードさせる
[STAThreadAttribute]
void log_reload_settings() {
    frmLog::Instance::get()->ReloadLogWindowSettings();
}
//ログウィンドウにイベントを実行させる
[STAThreadAttribute]
void log_process_events() {
    if (!frmLog::Instance::get()->InvokeRequired)
        System::Windows::Forms::Application::DoEvents();
}
//現在のログの長さを返す
[STAThreadAttribute]
int get_current_log_len(bool first_pass) {
    return frmLog::Instance::get()->GetLogStringLen(first_pass);
}

[STAThreadAttribute]
void close_log_window() {
    frmLog::Instance::get()->CloseLogWindow();
}

[STAThreadAttribute]
bool is_log_window_closed() {
    return frmLog::Instance::get()->IsClosed();
}

#pragma warning( push )
#pragma warning( disable: 4100 )
////////////////////////////////////////////////////
//       frmSetTransparency 関連
////////////////////////////////////////////////////
System::Void frmSetTransparency::setTransparency(int value) {
    value = Convert::ToInt32(clamp(value, fstNUTransparency->Minimum, fstNUTransparency->Maximum));
    fstNUTransparency->Value = clamp(value, fstNUTransparency->Minimum, fstNUTransparency->Maximum);
    fstTBTransparency->Value = clamp(value, fstTBTransparency->Minimum, fstTBTransparency->Maximum);
    frmLog^ log = dynamic_cast<frmLog^>(this->Owner);
    if (log != nullptr) {
        log->frmTransparency = value;
        log->Opacity = (100 - value) * 0.01f;
        //frmSetLogColor::Instance::get()->SetOpacity(log->Opacity);
    }
}

System::Void frmSetTransparency::frmSetTransparency_FormClosed(System::Object^  sender, System::Windows::Forms::FormClosedEventArgs^  e) {
    frmLog^ log = dynamic_cast<frmLog^>(this->Owner);
    if (log != nullptr)
        log->EnableToolStripMenuItemTransparent();
}

System::Void frmSetTransparency::fstSetLastTransparency() {
    frmLog^ log = dynamic_cast<frmLog^>(this->Owner);
    if (log != nullptr)
        last_transparency = 100 - (int)(log->Opacity * 100 + 0.5);
}
////////////////////////////////////////////////////
//       frmSetLogColor 関連
////////////////////////////////////////////////////
System::Void frmSetLogColor::fscBTOK_Click(System::Object^  sender, System::EventArgs^  e) {
    frmLog^ log = dynamic_cast<frmLog^>(this->Owner);
    if (log != nullptr)
        log->SetNewLogColor();
    this->Close();
}
#pragma warning( pop )
