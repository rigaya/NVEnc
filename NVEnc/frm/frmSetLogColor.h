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

#include "auo_settings.h"
#include "auo_mes.h"

using namespace System;
using namespace System::ComponentModel;
using namespace System::Collections;
using namespace System::Windows::Forms;
using namespace System::Data;
using namespace System::Drawing;

namespace AUO_NAME_R {

    /// <summary>
    /// frmSetLogColor の概要
    ///
    /// 警告: このクラスの名前を変更する場合、このクラスが依存するすべての .resx ファイルに関連付けられた
    ///          マネージ リソース コンパイラ ツールに対して 'Resource File Name' プロパティを
    ///          変更する必要があります。この変更を行わないと、
    ///          デザイナと、このフォームに関連付けられたローカライズ済みリソースとが、
    ///          正しく相互に利用できなくなります。
    /// </summary>
    public ref class frmSetLogColor : public System::Windows::Forms::Form
    {
    public:
        frmSetLogColor(void)
        {
            InitializeComponent();
            //
            //TODO: ここにコンストラクタ コードを追加します
            //
            themeMode = AuoTheme::DefaultLight;
            dwStgReader = nullptr;
        }

    protected:
        /// <summary>
        /// 使用中のリソースをすべてクリーンアップします。
        /// </summary>
        ~frmSetLogColor()
        {
            if (components)
            {
                delete components;
            }
            if (dwStgReader != nullptr)
                delete dwStgReader;
        }
    //Instanceを介し、ひとつだけ生成
    private:
        static frmSetLogColor^ _instance;
    public:
        static property frmSetLogColor^ Instance {
            frmSetLogColor^ get() {
                if (_instance == nullptr || _instance->IsDisposed)
                    _instance = gcnew frmSetLogColor();
                return _instance;
            }
        }
    private: System::Windows::Forms::Button^  fscBTOK;
    protected: 
    private: System::Windows::Forms::Button^  fscBTCancel;
    private: System::Windows::Forms::Button^  fscBTDefault;
    private: System::Windows::Forms::TextBox^  fcsTXColorBackground;

    private: System::Windows::Forms::Label^  fscLBColorBackground;
    private: System::Windows::Forms::Label^  fcsLBColorText;
    private: System::Windows::Forms::TextBox^  fcsTXColorTextInfo;
    private: System::Windows::Forms::TextBox^  fcsTXColorTextWarning;
    private: System::Windows::Forms::TextBox^  fcsTXColorTextError;
    private: System::Windows::Forms::ColorDialog^  colorDialogLog;







    private:
        /// <summary>
        /// 必要なデザイナ変数です。
        /// </summary>
        System::ComponentModel::Container ^components;

#pragma region Windows Form Designer generated code
        /// <summary>
        /// デザイナ サポートに必要なメソッドです。このメソッドの内容を
        /// コード エディタで変更しないでください。
        /// </summary>
        void InitializeComponent(void)
        {
            this->fscBTOK = (gcnew System::Windows::Forms::Button());
            this->fscBTCancel = (gcnew System::Windows::Forms::Button());
            this->fscBTDefault = (gcnew System::Windows::Forms::Button());
            this->fcsTXColorBackground = (gcnew System::Windows::Forms::TextBox());
            this->fscLBColorBackground = (gcnew System::Windows::Forms::Label());
            this->fcsLBColorText = (gcnew System::Windows::Forms::Label());
            this->fcsTXColorTextInfo = (gcnew System::Windows::Forms::TextBox());
            this->fcsTXColorTextWarning = (gcnew System::Windows::Forms::TextBox());
            this->fcsTXColorTextError = (gcnew System::Windows::Forms::TextBox());
            this->colorDialogLog = (gcnew System::Windows::Forms::ColorDialog());
            this->SuspendLayout();
            // 
            // fscBTOK
            // 
            this->fscBTOK->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((System::Windows::Forms::AnchorStyles::Bottom | System::Windows::Forms::AnchorStyles::Right));
            this->fscBTOK->Location = System::Drawing::Point(166, 178);
            this->fscBTOK->Name = L"fscBTOK";
            this->fscBTOK->Size = System::Drawing::Size(79, 34);
            this->fscBTOK->TabIndex = 0;
            this->fscBTOK->Text = L"OK";
            this->fscBTOK->UseVisualStyleBackColor = true;
            this->fscBTOK->Click += gcnew System::EventHandler(this, &frmSetLogColor::fscBTOK_Click);
            // 
            // fscBTCancel
            // 
            this->fscBTCancel->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((System::Windows::Forms::AnchorStyles::Bottom | System::Windows::Forms::AnchorStyles::Right));
            this->fscBTCancel->DialogResult = System::Windows::Forms::DialogResult::Cancel;
            this->fscBTCancel->Location = System::Drawing::Point(261, 178);
            this->fscBTCancel->Name = L"fscBTCancel";
            this->fscBTCancel->Size = System::Drawing::Size(79, 34);
            this->fscBTCancel->TabIndex = 1;
            this->fscBTCancel->Text = L"キャンセル";
            this->fscBTCancel->UseVisualStyleBackColor = true;
            this->fscBTCancel->Click += gcnew System::EventHandler(this, &frmSetLogColor::fscBTCancel_Click);
            // 
            // fscBTDefault
            // 
            this->fscBTDefault->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((System::Windows::Forms::AnchorStyles::Bottom | System::Windows::Forms::AnchorStyles::Left));
            this->fscBTDefault->Location = System::Drawing::Point(12, 178);
            this->fscBTDefault->Name = L"fscBTDefault";
            this->fscBTDefault->Size = System::Drawing::Size(79, 34);
            this->fscBTDefault->TabIndex = 2;
            this->fscBTDefault->Text = L"デフォルト";
            this->fscBTDefault->UseVisualStyleBackColor = true;
            this->fscBTDefault->Click += gcnew System::EventHandler(this, &frmSetLogColor::fscBTDefault_Click);
            // 
            // fcsTXColorBackground
            // 
            this->fcsTXColorBackground->BorderStyle = System::Windows::Forms::BorderStyle::FixedSingle;
            this->fcsTXColorBackground->Location = System::Drawing::Point(137, 31);
            this->fcsTXColorBackground->Name = L"fcsTXColorBackground";
            this->fcsTXColorBackground->ReadOnly = true;
            this->fcsTXColorBackground->Size = System::Drawing::Size(95, 22);
            this->fcsTXColorBackground->TabIndex = 3;
            this->fcsTXColorBackground->TextAlign = System::Windows::Forms::HorizontalAlignment::Center;
            this->fcsTXColorBackground->Click += gcnew System::EventHandler(this, &frmSetLogColor::fcsTXColorBackground_Click);
            // 
            // fscLBColorBackground
            // 
            this->fscLBColorBackground->AutoSize = true;
            this->fscLBColorBackground->Location = System::Drawing::Point(75, 33);
            this->fscLBColorBackground->Name = L"fscLBColorBackground";
            this->fscLBColorBackground->Size = System::Drawing::Size(37, 14);
            this->fscLBColorBackground->TabIndex = 4;
            this->fscLBColorBackground->Text = L"背景色";
            // 
            // fcsLBColorText
            // 
            this->fcsLBColorText->AutoSize = true;
            this->fcsLBColorText->Location = System::Drawing::Point(75, 74);
            this->fcsLBColorText->Name = L"fcsLBColorText";
            this->fcsLBColorText->Size = System::Drawing::Size(37, 14);
            this->fcsLBColorText->TabIndex = 5;
            this->fcsLBColorText->Text = L"文字色";
            // 
            // fcsTXColorTextInfo
            // 
            this->fcsTXColorTextInfo->BorderStyle = System::Windows::Forms::BorderStyle::FixedSingle;
            this->fcsTXColorTextInfo->Location = System::Drawing::Point(137, 72);
            this->fcsTXColorTextInfo->Name = L"fcsTXColorTextInfo";
            this->fcsTXColorTextInfo->ReadOnly = true;
            this->fcsTXColorTextInfo->Size = System::Drawing::Size(95, 22);
            this->fcsTXColorTextInfo->TabIndex = 6;
            this->fcsTXColorTextInfo->Text = L"info";
            this->fcsTXColorTextInfo->TextAlign = System::Windows::Forms::HorizontalAlignment::Center;
            this->fcsTXColorTextInfo->Click += gcnew System::EventHandler(this, &frmSetLogColor::fcsTXColorTextInfo_Click);
            // 
            // fcsTXColorTextWarning
            // 
            this->fcsTXColorTextWarning->BorderStyle = System::Windows::Forms::BorderStyle::FixedSingle;
            this->fcsTXColorTextWarning->Location = System::Drawing::Point(137, 100);
            this->fcsTXColorTextWarning->Name = L"fcsTXColorTextWarning";
            this->fcsTXColorTextWarning->ReadOnly = true;
            this->fcsTXColorTextWarning->Size = System::Drawing::Size(95, 22);
            this->fcsTXColorTextWarning->TabIndex = 7;
            this->fcsTXColorTextWarning->Text = L"warning";
            this->fcsTXColorTextWarning->TextAlign = System::Windows::Forms::HorizontalAlignment::Center;
            this->fcsTXColorTextWarning->Click += gcnew System::EventHandler(this, &frmSetLogColor::fcsTXColorTextWarning_Click);
            // 
            // fcsTXColorTextError
            // 
            this->fcsTXColorTextError->BorderStyle = System::Windows::Forms::BorderStyle::FixedSingle;
            this->fcsTXColorTextError->Location = System::Drawing::Point(137, 132);
            this->fcsTXColorTextError->Name = L"fcsTXColorTextError";
            this->fcsTXColorTextError->ReadOnly = true;
            this->fcsTXColorTextError->Size = System::Drawing::Size(95, 22);
            this->fcsTXColorTextError->TabIndex = 8;
            this->fcsTXColorTextError->Text = L"error";
            this->fcsTXColorTextError->TextAlign = System::Windows::Forms::HorizontalAlignment::Center;
            this->fcsTXColorTextError->Click += gcnew System::EventHandler(this, &frmSetLogColor::fcsTXColorTextError_Click);
            // 
            // colorDialogLog
            // 
            this->colorDialogLog->AnyColor = true;
            this->colorDialogLog->FullOpen = true;
            // 
            // frmSetLogColor
            // 
            this->AcceptButton = this->fscBTOK;
            this->AutoScaleDimensions = System::Drawing::SizeF(96, 96);
            this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Dpi;
            this->CancelButton = this->fscBTCancel;
            this->ClientSize = System::Drawing::Size(352, 224);
            this->Controls->Add(this->fcsTXColorTextError);
            this->Controls->Add(this->fcsTXColorTextWarning);
            this->Controls->Add(this->fcsTXColorTextInfo);
            this->Controls->Add(this->fcsLBColorText);
            this->Controls->Add(this->fscLBColorBackground);
            this->Controls->Add(this->fcsTXColorBackground);
            this->Controls->Add(this->fscBTDefault);
            this->Controls->Add(this->fscBTCancel);
            this->Controls->Add(this->fscBTOK);
            this->Font = (gcnew System::Drawing::Font(L"Meiryo UI", 9, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point, 
                static_cast<System::Byte>(0)));
            this->FormBorderStyle = System::Windows::Forms::FormBorderStyle::FixedDialog;
            this->KeyPreview = true;
            this->MaximizeBox = false;
            this->MinimizeBox = false;
            this->Name = L"frmSetLogColor";
            this->ShowIcon = false;
            this->Text = L"各ボックスをクリックして色を変更...";
            this->Load += gcnew System::EventHandler(this, &frmSetLogColor::frmSetLogColor_Load);
            this->KeyDown += gcnew System::Windows::Forms::KeyEventHandler(this, &frmSetLogColor::frmSetLogColor_KeyDown);
            this->ResumeLayout(false);
            this->PerformLayout();

        }
#pragma endregion
    public:
        Color colorBackGround;
        Color colorInfo;
        Color colorWarning;
        Color colorError;
    private:
        AuoTheme themeMode;
        const DarkenWindowStgReader *dwStgReader;
    private:
        System::Void LoadLangText() {
            LOAD_CLI_TEXT(fscBTOK);
            LOAD_CLI_TEXT(fscBTCancel);
            LOAD_CLI_TEXT(fscBTDefault);
            LOAD_CLI_TEXT(fscLBColorBackground);
            LOAD_CLI_TEXT(fcsLBColorText);
            LOAD_CLI_TEXT(fcsTXColorTextInfo);
            LOAD_CLI_TEXT(fcsTXColorTextWarning);
            LOAD_CLI_TEXT(fcsTXColorTextError);
            LOAD_CLI_MAIN_TEXT(fcsMain);
        }
    private:
        System::Void fscBTOK_Click(System::Object^  sender, System::EventArgs^  e);
    private:
        System::Void SetColors() {
            fcsTXColorBackground->BackColor  = colorBackGround;
            fcsTXColorTextInfo->BackColor    = colorBackGround;
            fcsTXColorTextWarning->BackColor = colorBackGround;
            fcsTXColorTextError->BackColor   = colorBackGround;
            fcsTXColorTextInfo->ForeColor    = colorInfo;
            fcsTXColorTextWarning->ForeColor = colorWarning;
            fcsTXColorTextError->ForeColor   = colorError;
        }
    public:
        System::Void SetOpacity(double opacity) {
            this->Opacity = opacity;
        }
    private:
        System::Void frmSetLogColor_Load(System::Object^  sender, System::EventArgs^  e) {
            LoadLangText();
            SetColors();
        }
    private:
        System::Void fcsTXColorBackground_Click(System::Object^  sender, System::EventArgs^  e) {
            colorDialogLog->Color = colorBackGround;
            if (System::Windows::Forms::DialogResult::OK == colorDialogLog->ShowDialog()) {
                colorBackGround = colorDialogLog->Color;
                SetColors();
            }
        }
    private:
        System::Void fcsTXColorTextInfo_Click(System::Object^  sender, System::EventArgs^  e) {
            colorDialogLog->Color = colorInfo;
            if (System::Windows::Forms::DialogResult::OK == colorDialogLog->ShowDialog()) {
                colorInfo = colorDialogLog->Color;
                SetColors();
            }
        }
    private:
        System::Void fcsTXColorTextWarning_Click(System::Object^  sender, System::EventArgs^  e) {
            colorDialogLog->Color = colorWarning;
            if (System::Windows::Forms::DialogResult::OK == colorDialogLog->ShowDialog()) {
                colorWarning = colorDialogLog->Color;
                SetColors();
            }
        }
    private:
        System::Void fcsTXColorTextError_Click(System::Object^  sender, System::EventArgs^  e) {
            colorDialogLog->Color = colorError;
            if (System::Windows::Forms::DialogResult::OK == colorDialogLog->ShowDialog()) {
                colorError = colorDialogLog->Color;
                SetColors();
            }
        }
    private:
        System::Void GetDefaultColors() {
            colorBackGround = Color::FromArgb(DEFAULT_LOG_COLOR_BACKGROUND[0], DEFAULT_LOG_COLOR_BACKGROUND[1], DEFAULT_LOG_COLOR_BACKGROUND[2]);
            colorInfo       = Color::FromArgb(DEFAULT_LOG_COLOR_TEXT[0][0], DEFAULT_LOG_COLOR_TEXT[0][1], DEFAULT_LOG_COLOR_TEXT[0][2]);
            colorWarning    = Color::FromArgb(DEFAULT_LOG_COLOR_TEXT[1][0], DEFAULT_LOG_COLOR_TEXT[1][1], DEFAULT_LOG_COLOR_TEXT[1][2]);
            colorError      = Color::FromArgb(DEFAULT_LOG_COLOR_TEXT[2][0], DEFAULT_LOG_COLOR_TEXT[2][1], DEFAULT_LOG_COLOR_TEXT[2][2]);
        }
    private:
        System::Void fscBTDefault_Click(System::Object^  sender, System::EventArgs^  e) {
            GetDefaultColors();
            SetColors();
        }
    private:
        System::Void fscBTCancel_Click(System::Object^  sender, System::EventArgs^  e) {
            this->Close();
        }
    private:
        System::Void frmSetLogColor_KeyDown(System::Object^  sender, System::Windows::Forms::KeyEventArgs^  e) {
            if (e->KeyCode == Keys::Escape)
                this->Close();
        }
    public:
        System::Void InitTheme() {
            if (dwStgReader != nullptr) delete dwStgReader;
            char aviutl_dir[MAX_PATH_LEN];
            get_aviutl_dir(aviutl_dir, _countof(aviutl_dir));
            const auto [themeTo, dwStg] = check_current_theme(aviutl_dir);
            dwStgReader = dwStg;
            CheckTheme(themeTo);
        }
    private:
        System::Void CheckTheme(const AuoTheme themeTo) {
            //変更の必要がなければ終了
            if (themeTo == themeMode) return;

            //一度ウィンドウの再描画を完全に抑止する
            SendMessage(reinterpret_cast<HWND>(this->Handle.ToPointer()), WM_SETREDRAW, 0, 0);
            SetAllColor(this, themeTo, this->GetType(), dwStgReader);
            //一度ウィンドウの再描画を再開し、強制的に再描画させる
            SendMessage(reinterpret_cast<HWND>(this->Handle.ToPointer()), WM_SETREDRAW, 1, 0);
            this->Refresh();
            themeMode = themeTo;
        }
};
}
