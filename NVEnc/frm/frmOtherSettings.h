// -----------------------------------------------------------------------------------------
// NVEnc by rigaya
// -----------------------------------------------------------------------------------------
// The MIT License
//
// Copyright (c) 2014-2016 rigaya
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

#include "auo_version.h"
#include "auo_settings.h"

using namespace System;
using namespace System::ComponentModel;
using namespace System::Collections;
using namespace System::Windows::Forms;
using namespace System::Data;
using namespace System::Drawing;

namespace NVEnc {

    /// <summary>
    /// frmOtherSettings の概要
    ///
    /// 警告: このクラスの名前を変更する場合、このクラスが依存するすべての .resx ファイルに関連付けられた
    ///          マネージ リソース コンパイラ ツールに対して 'Resource File Name' プロパティを
    ///          変更する必要があります。この変更を行わないと、
    ///          デザイナと、このフォームに関連付けられたローカライズ済みリソースとが、
    ///          正しく相互に利用できなくなります。
    /// </summary>
    public ref class frmOtherSettings : public System::Windows::Forms::Form
    {
    public:
        frmOtherSettings(void)
        {
            fos_ex_stg = new guiEx_settings(TRUE);
            InitializeComponent();
            //
            //TODO: ここにコンストラクタ コードを追加します
            //
        }

    protected:
        /// <summary>
        /// 使用中のリソースをすべてクリーンアップします。
        /// </summary>
        ~frmOtherSettings()
        {
            if (components)
            {
                delete components;
            }
            delete fos_ex_stg;
        }
    private:
        guiEx_settings *fos_ex_stg;
        static frmOtherSettings^ _instance;

    protected:
    private: System::Windows::Forms::Button^  fosCBCancel;
    private: System::Windows::Forms::Button^  fosCBOK;
    private: System::Windows::Forms::TextBox^  fosTXStgDir;
    private: System::Windows::Forms::Label^  fosLBStgDir;
    private: System::Windows::Forms::Button^  fosBTStgDir;
    public:
        static String^ stgDir;
        static int useLastExt;
        //static bool DisableToolTipHelp;


    private: System::Windows::Forms::CheckBox^  fosCBDisableToolTip;
    private: System::Windows::Forms::CheckBox^  fosCBDisableVisualStyles;
    private: System::Windows::Forms::Label^  fosLBDisableVisualStyles;
    private: System::Windows::Forms::CheckBox^  fosCBLogStartMinimized;
    private: System::Windows::Forms::CheckBox^  fosCBLogDisableTransparency;
    private: System::Windows::Forms::CheckBox^  fosCBAutoDelChap;
    private: System::Windows::Forms::CheckBox^  fosCBStgEscKey;
    private: System::Windows::Forms::Button^  fosBTSetFont;
    private: System::Windows::Forms::FontDialog^  fosfontDialog;
    private: System::Windows::Forms::CheckBox^  fosCBGetRelativePath;
    private: System::Windows::Forms::CheckBox^  fosCBAutoAFSDisable;
    private: System::Windows::Forms::Label^  fosLBDefaultOutExt2;
    private: System::Windows::Forms::ComboBox^  fosCXDefaultOutExt;
    private: System::Windows::Forms::Label^  fosLBDefaultOutExt;
    private: System::Windows::Forms::CheckBox^  fosCBRunBatMinimized;
    private: System::Windows::Forms::CheckBox^  fosCBChapConvertToUTF8;
    private: System::Windows::Forms::CheckBox^  fosCBOutputMoreLog;
    private: System::Windows::Forms::TabControl ^tabControl1;
    private: System::Windows::Forms::TabPage ^tabPage1;
    private: System::Windows::Forms::ComboBox ^fosCXDefaultAudioEncoder;
    private: System::Windows::Forms::Label ^fosLBDefaultAudioEncoder;
    private: System::Windows::Forms::Label ^label1;
    private: System::Windows::Forms::Label ^label2;
    private: System::Windows::Forms::ComboBox ^comboBox1;
    private: System::Windows::Forms::TabPage ^tabPage2;



    public:
        static property frmOtherSettings^ Instance {
            frmOtherSettings^ get() {
                if (_instance == nullptr || _instance->IsDisposed)
                    _instance = gcnew frmOtherSettings();
                return _instance;
            }
        }


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
            this->fosCBCancel = (gcnew System::Windows::Forms::Button());
            this->fosCBOK = (gcnew System::Windows::Forms::Button());
            this->fosTXStgDir = (gcnew System::Windows::Forms::TextBox());
            this->fosLBStgDir = (gcnew System::Windows::Forms::Label());
            this->fosBTStgDir = (gcnew System::Windows::Forms::Button());
            this->fosCBDisableToolTip = (gcnew System::Windows::Forms::CheckBox());
            this->fosCBDisableVisualStyles = (gcnew System::Windows::Forms::CheckBox());
            this->fosLBDisableVisualStyles = (gcnew System::Windows::Forms::Label());
            this->fosCBLogStartMinimized = (gcnew System::Windows::Forms::CheckBox());
            this->fosCBLogDisableTransparency = (gcnew System::Windows::Forms::CheckBox());
            this->fosCBAutoDelChap = (gcnew System::Windows::Forms::CheckBox());
            this->fosCBStgEscKey = (gcnew System::Windows::Forms::CheckBox());
            this->fosBTSetFont = (gcnew System::Windows::Forms::Button());
            this->fosfontDialog = (gcnew System::Windows::Forms::FontDialog());
            this->fosCBGetRelativePath = (gcnew System::Windows::Forms::CheckBox());
            this->fosCBAutoAFSDisable = (gcnew System::Windows::Forms::CheckBox());
            this->fosLBDefaultOutExt2 = (gcnew System::Windows::Forms::Label());
            this->fosCXDefaultOutExt = (gcnew System::Windows::Forms::ComboBox());
            this->fosLBDefaultOutExt = (gcnew System::Windows::Forms::Label());
            this->fosCBRunBatMinimized = (gcnew System::Windows::Forms::CheckBox());
            this->fosCBChapConvertToUTF8 = (gcnew System::Windows::Forms::CheckBox());
            this->fosCBOutputMoreLog = (gcnew System::Windows::Forms::CheckBox());
            this->tabControl1 = (gcnew System::Windows::Forms::TabControl());
            this->tabPage1 = (gcnew System::Windows::Forms::TabPage());
            this->fosCXDefaultAudioEncoder = (gcnew System::Windows::Forms::ComboBox());
            this->fosLBDefaultAudioEncoder = (gcnew System::Windows::Forms::Label());
            this->label1 = (gcnew System::Windows::Forms::Label());
            this->label2 = (gcnew System::Windows::Forms::Label());
            this->comboBox1 = (gcnew System::Windows::Forms::ComboBox());
            this->tabPage2 = (gcnew System::Windows::Forms::TabPage());
            this->tabControl1->SuspendLayout();
            this->tabPage1->SuspendLayout();
            this->tabPage2->SuspendLayout();
            this->SuspendLayout();
            //
            // fosCBCancel
            //
            this->fosCBCancel->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((System::Windows::Forms::AnchorStyles::Bottom | System::Windows::Forms::AnchorStyles::Right));
            this->fosCBCancel->DialogResult = System::Windows::Forms::DialogResult::Cancel;
            this->fosCBCancel->Location = System::Drawing::Point(171, 385);
            this->fosCBCancel->Name = L"fosCBCancel";
            this->fosCBCancel->Size = System::Drawing::Size(84, 29);
            this->fosCBCancel->TabIndex = 1;
            this->fosCBCancel->Text = L"キャンセル";
            this->fosCBCancel->UseVisualStyleBackColor = true;
            this->fosCBCancel->Click += gcnew System::EventHandler(this, &frmOtherSettings::fosCBCancel_Click);
            //
            // fosCBOK
            //
            this->fosCBOK->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((System::Windows::Forms::AnchorStyles::Bottom | System::Windows::Forms::AnchorStyles::Right));
            this->fosCBOK->Location = System::Drawing::Point(283, 385);
            this->fosCBOK->Name = L"fosCBOK";
            this->fosCBOK->Size = System::Drawing::Size(84, 29);
            this->fosCBOK->TabIndex = 2;
            this->fosCBOK->Text = L"OK";
            this->fosCBOK->UseVisualStyleBackColor = true;
            this->fosCBOK->Click += gcnew System::EventHandler(this, &frmOtherSettings::fosCBOK_Click);
            //
            // fosTXStgDir
            //
            this->fosTXStgDir->Location = System::Drawing::Point(37, 37);
            this->fosTXStgDir->Name = L"fosTXStgDir";
            this->fosTXStgDir->Size = System::Drawing::Size(294, 23);
            this->fosTXStgDir->TabIndex = 3;
            //
            // fosLBStgDir
            //
            this->fosLBStgDir->AutoSize = true;
            this->fosLBStgDir->Location = System::Drawing::Point(10, 17);
            this->fosLBStgDir->Name = L"fosLBStgDir";
            this->fosLBStgDir->Size = System::Drawing::Size(123, 15);
            this->fosLBStgDir->TabIndex = 4;
            this->fosLBStgDir->Text = L"設定ファイルの保存場所";
            //
            // fosBTStgDir
            //
            this->fosBTStgDir->Location = System::Drawing::Point(337, 37);
            this->fosBTStgDir->Name = L"fosBTStgDir";
            this->fosBTStgDir->Size = System::Drawing::Size(35, 23);
            this->fosBTStgDir->TabIndex = 5;
            this->fosBTStgDir->Text = L"...";
            this->fosBTStgDir->UseVisualStyleBackColor = true;
            this->fosBTStgDir->Click += gcnew System::EventHandler(this, &frmOtherSettings::fosBTStgDir_Click);
            //
            // fosCBDisableToolTip
            //
            this->fosCBDisableToolTip->AutoSize = true;
            this->fosCBDisableToolTip->Location = System::Drawing::Point(20, 127);
            this->fosCBDisableToolTip->Name = L"fosCBDisableToolTip";
            this->fosCBDisableToolTip->Size = System::Drawing::Size(158, 19);
            this->fosCBDisableToolTip->TabIndex = 8;
            this->fosCBDisableToolTip->Text = L"ポップアップヘルプを抑制する";
            this->fosCBDisableToolTip->UseVisualStyleBackColor = true;
            //
            // fosCBDisableVisualStyles
            //
            this->fosCBDisableVisualStyles->AutoSize = true;
            this->fosCBDisableVisualStyles->Location = System::Drawing::Point(20, 157);
            this->fosCBDisableVisualStyles->Name = L"fosCBDisableVisualStyles";
            this->fosCBDisableVisualStyles->Size = System::Drawing::Size(128, 19);
            this->fosCBDisableVisualStyles->TabIndex = 9;
            this->fosCBDisableVisualStyles->Text = L"視覚効果をオフにする";
            this->fosCBDisableVisualStyles->UseVisualStyleBackColor = true;
            //
            // fosLBDisableVisualStyles
            //
            this->fosLBDisableVisualStyles->AutoSize = true;
            this->fosLBDisableVisualStyles->Font = (gcnew System::Drawing::Font(L"Meiryo UI", 8.25F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
                static_cast<System::Byte>(128)));
            this->fosLBDisableVisualStyles->ForeColor = System::Drawing::Color::OrangeRed;
            this->fosLBDisableVisualStyles->Location = System::Drawing::Point(41, 176);
            this->fosLBDisableVisualStyles->Name = L"fosLBDisableVisualStyles";
            this->fosLBDisableVisualStyles->Size = System::Drawing::Size(161, 14);
            this->fosLBDisableVisualStyles->TabIndex = 10;
            this->fosLBDisableVisualStyles->Text = L"※反映にはAviutlの再起動が必要";
            //
            // fosCBLogStartMinimized
            //
            this->fosCBLogStartMinimized->AutoSize = true;
            this->fosCBLogStartMinimized->Location = System::Drawing::Point(20, 203);
            this->fosCBLogStartMinimized->Name = L"fosCBLogStartMinimized";
            this->fosCBLogStartMinimized->Size = System::Drawing::Size(184, 19);
            this->fosCBLogStartMinimized->TabIndex = 11;
            this->fosCBLogStartMinimized->Text = L"ログウィンドウを最小化で開始する";
            this->fosCBLogStartMinimized->UseVisualStyleBackColor = true;
            //
            // fosCBLogDisableTransparency
            //
            this->fosCBLogDisableTransparency->AutoSize = true;
            this->fosCBLogDisableTransparency->Location = System::Drawing::Point(20, 233);
            this->fosCBLogDisableTransparency->Name = L"fosCBLogDisableTransparency";
            this->fosCBLogDisableTransparency->Size = System::Drawing::Size(174, 19);
            this->fosCBLogDisableTransparency->TabIndex = 12;
            this->fosCBLogDisableTransparency->Text = L"ログウィンドウの透過をオフにする";
            this->fosCBLogDisableTransparency->UseVisualStyleBackColor = true;
            //
            // fosCBAutoDelChap
            //
            this->fosCBAutoDelChap->AutoSize = true;
            this->fosCBAutoDelChap->Location = System::Drawing::Point(21, 201);
            this->fosCBAutoDelChap->Name = L"fosCBAutoDelChap";
            this->fosCBAutoDelChap->Size = System::Drawing::Size(295, 19);
            this->fosCBAutoDelChap->TabIndex = 13;
            this->fosCBAutoDelChap->Text = L"mux正常終了後、チャプターファイルを自動的に削除する";
            this->fosCBAutoDelChap->UseVisualStyleBackColor = true;
            //
            // fosCBStgEscKey
            //
            this->fosCBStgEscKey->AutoSize = true;
            this->fosCBStgEscKey->Location = System::Drawing::Point(20, 262);
            this->fosCBStgEscKey->Name = L"fosCBStgEscKey";
            this->fosCBStgEscKey->Size = System::Drawing::Size(168, 19);
            this->fosCBStgEscKey->TabIndex = 14;
            this->fosCBStgEscKey->Text = L"設定画面でEscキーを有効化";
            this->fosCBStgEscKey->UseVisualStyleBackColor = true;
            //
            // fosBTSetFont
            //
            this->fosBTSetFont->Location = System::Drawing::Point(249, 300);
            this->fosBTSetFont->Name = L"fosBTSetFont";
            this->fosBTSetFont->Size = System::Drawing::Size(124, 27);
            this->fosBTSetFont->TabIndex = 16;
            this->fosBTSetFont->Text = L"フォントの変更...";
            this->fosBTSetFont->UseVisualStyleBackColor = true;
            this->fosBTSetFont->Click += gcnew System::EventHandler(this, &frmOtherSettings::fosBTSetFont_Click);
            //
            // fosfontDialog
            //
            this->fosfontDialog->AllowVerticalFonts = false;
            this->fosfontDialog->Color = System::Drawing::SystemColors::ControlText;
            this->fosfontDialog->Font = (gcnew System::Drawing::Font(L"Meiryo UI", 9, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
                static_cast<System::Byte>(0)));
            this->fosfontDialog->FontMustExist = true;
            this->fosfontDialog->MaxSize = 9;
            this->fosfontDialog->MinSize = 9;
            this->fosfontDialog->ShowEffects = false;
            //
            // fosCBGetRelativePath
            //
            this->fosCBGetRelativePath->AutoSize = true;
            this->fosCBGetRelativePath->Location = System::Drawing::Point(20, 291);
            this->fosCBGetRelativePath->Name = L"fosCBGetRelativePath";
            this->fosCBGetRelativePath->Size = System::Drawing::Size(185, 19);
            this->fosCBGetRelativePath->TabIndex = 17;
            this->fosCBGetRelativePath->Text = L"ダイアログから相対パスで取得する";
            this->fosCBGetRelativePath->UseVisualStyleBackColor = true;
            //
            // fosCBAutoAFSDisable
            //
            this->fosCBAutoAFSDisable->Location = System::Drawing::Point(21, 142);
            this->fosCBAutoAFSDisable->Name = L"fosCBAutoAFSDisable";
            this->fosCBAutoAFSDisable->Size = System::Drawing::Size(308, 53);
            this->fosCBAutoAFSDisable->TabIndex = 18;
            this->fosCBAutoAFSDisable->Text = L"自動フィールドシフト(afs)オンで初期化に失敗した場合、afsをオフにしてエンコード続行を試みる";
            this->fosCBAutoAFSDisable->UseVisualStyleBackColor = true;
            //
            // fosLBDefaultOutExt2
            //
            this->fosLBDefaultOutExt2->AutoSize = true;
            this->fosLBDefaultOutExt2->Font = (gcnew System::Drawing::Font(L"Meiryo UI", 8.25F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
                static_cast<System::Byte>(128)));
            this->fosLBDefaultOutExt2->ForeColor = System::Drawing::Color::OrangeRed;
            this->fosLBDefaultOutExt2->Location = System::Drawing::Point(210, 23);
            this->fosLBDefaultOutExt2->Name = L"fosLBDefaultOutExt2";
            this->fosLBDefaultOutExt2->Size = System::Drawing::Size(161, 14);
            this->fosLBDefaultOutExt2->TabIndex = 22;
            this->fosLBDefaultOutExt2->Text = L"※反映にはAviutlの再起動が必要";
            //
            // fosCXDefaultOutExt
            //
            this->fosCXDefaultOutExt->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
            this->fosCXDefaultOutExt->FormattingEnabled = true;
            this->fosCXDefaultOutExt->Location = System::Drawing::Point(45, 44);
            this->fosCXDefaultOutExt->Name = L"fosCXDefaultOutExt";
            this->fosCXDefaultOutExt->Size = System::Drawing::Size(190, 23);
            this->fosCXDefaultOutExt->TabIndex = 21;
            //
            // fosLBDefaultOutExt
            //
            this->fosLBDefaultOutExt->AutoSize = true;
            this->fosLBDefaultOutExt->Location = System::Drawing::Point(18, 21);
            this->fosLBDefaultOutExt->Name = L"fosLBDefaultOutExt";
            this->fosLBDefaultOutExt->Size = System::Drawing::Size(172, 15);
            this->fosLBDefaultOutExt->TabIndex = 20;
            this->fosLBDefaultOutExt->Text = L"出力するファイルの種類のデフォルト";
            //
            // fosCBRunBatMinimized
            //
            this->fosCBRunBatMinimized->AutoSize = true;
            this->fosCBRunBatMinimized->Location = System::Drawing::Point(21, 272);
            this->fosCBRunBatMinimized->Name = L"fosCBRunBatMinimized";
            this->fosCBRunBatMinimized->Size = System::Drawing::Size(205, 19);
            this->fosCBRunBatMinimized->TabIndex = 23;
            this->fosCBRunBatMinimized->Text = L"エンコ前後バッチ処理を最小化で実行";
            this->fosCBRunBatMinimized->UseVisualStyleBackColor = true;
            //
            // fosCBChapConvertToUTF8
            //
            this->fosCBChapConvertToUTF8->AutoSize = true;
            this->fosCBChapConvertToUTF8->Location = System::Drawing::Point(21, 299);
            this->fosCBChapConvertToUTF8->Name = L"fosCBChapConvertToUTF8";
            this->fosCBChapConvertToUTF8->Size = System::Drawing::Size(204, 19);
            this->fosCBChapConvertToUTF8->TabIndex = 26;
            this->fosCBChapConvertToUTF8->Text = L"チャプターmux時、UTF-8に変換する";
            this->fosCBChapConvertToUTF8->UseVisualStyleBackColor = true;
            //
            // fosCBOutputMoreLog
            //
            this->fosCBOutputMoreLog->AutoSize = true;
            this->fosCBOutputMoreLog->Location = System::Drawing::Point(20, 316);
            this->fosCBOutputMoreLog->Name = L"fosCBOutputMoreLog";
            this->fosCBOutputMoreLog->Size = System::Drawing::Size(143, 19);
            this->fosCBOutputMoreLog->TabIndex = 28;
            this->fosCBOutputMoreLog->Text = L"音声・muxのログも表示";
            this->fosCBOutputMoreLog->UseVisualStyleBackColor = true;
            //
            // tabControl1
            //
            this->tabControl1->Controls->Add(this->tabPage1);
            this->tabControl1->Controls->Add(this->tabPage2);
            this->tabControl1->Location = System::Drawing::Point(2, 2);
            this->tabControl1->Name = L"tabControl1";
            this->tabControl1->SelectedIndex = 0;
            this->tabControl1->Size = System::Drawing::Size(390, 377);
            this->tabControl1->TabIndex = 29;
            //
            // tabPage1
            //
            this->tabPage1->Controls->Add(this->fosCXDefaultAudioEncoder);
            this->tabPage1->Controls->Add(this->fosLBDefaultAudioEncoder);
            this->tabPage1->Controls->Add(this->label1);
            this->tabPage1->Controls->Add(this->fosLBDefaultOutExt);
            this->tabPage1->Controls->Add(this->fosCBChapConvertToUTF8);
            this->tabPage1->Controls->Add(this->fosCXDefaultOutExt);
            this->tabPage1->Controls->Add(this->fosCBRunBatMinimized);
            this->tabPage1->Controls->Add(this->label2);
            this->tabPage1->Controls->Add(this->fosCBAutoAFSDisable);
            this->tabPage1->Controls->Add(this->fosLBDefaultOutExt2);
            this->tabPage1->Controls->Add(this->comboBox1);
            this->tabPage1->Controls->Add(this->fosCBAutoDelChap);
            this->tabPage1->Location = System::Drawing::Point(4, 24);
            this->tabPage1->Name = L"tabPage1";
            this->tabPage1->Padding = System::Windows::Forms::Padding(3);
            this->tabPage1->Size = System::Drawing::Size(382, 349);
            this->tabPage1->TabIndex = 0;
            this->tabPage1->Text = L"エンコード設定";
            this->tabPage1->UseVisualStyleBackColor = true;
            //
            // fosCXDefaultAudioEncoder
            //
            this->fosCXDefaultAudioEncoder->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
            this->fosCXDefaultAudioEncoder->FormattingEnabled = true;
            this->fosCXDefaultAudioEncoder->Location = System::Drawing::Point(45, 103);
            this->fosCXDefaultAudioEncoder->Name = L"fosCXDefaultAudioEncoder";
            this->fosCXDefaultAudioEncoder->Size = System::Drawing::Size(190, 23);
            this->fosCXDefaultAudioEncoder->TabIndex = 35;
            //
            // fosLBDefaultAudioEncoder
            //
            this->fosLBDefaultAudioEncoder->AutoSize = true;
            this->fosLBDefaultAudioEncoder->Location = System::Drawing::Point(18, 81);
            this->fosLBDefaultAudioEncoder->Name = L"fosLBDefaultAudioEncoder";
            this->fosLBDefaultAudioEncoder->Size = System::Drawing::Size(130, 15);
            this->fosLBDefaultAudioEncoder->TabIndex = 34;
            this->fosLBDefaultAudioEncoder->Text = L"デフォルトの音声エンコーダ";
            //
            // label1
            //
            this->label1->AutoSize = true;
            this->label1->Location = System::Drawing::Point(18, 21);
            this->label1->Name = L"label1";
            this->label1->Size = System::Drawing::Size(172, 15);
            this->label1->TabIndex = 20;
            this->label1->Text = L"出力するファイルの種類のデフォルト";
            //
            // label2
            //
            this->label2->AutoSize = true;
            this->label2->Font = (gcnew System::Drawing::Font(L"Meiryo UI", 8.25F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
                static_cast<System::Byte>(128)));
            this->label2->ForeColor = System::Drawing::Color::OrangeRed;
            this->label2->Location = System::Drawing::Point(210, 23);
            this->label2->Name = L"label2";
            this->label2->Size = System::Drawing::Size(161, 14);
            this->label2->TabIndex = 22;
            this->label2->Text = L"※反映にはAviutlの再起動が必要";
            //
            // comboBox1
            //
            this->comboBox1->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
            this->comboBox1->FormattingEnabled = true;
            this->comboBox1->Location = System::Drawing::Point(45, 44);
            this->comboBox1->Name = L"comboBox1";
            this->comboBox1->Size = System::Drawing::Size(190, 23);
            this->comboBox1->TabIndex = 21;
            //
            // tabPage2
            //
            this->tabPage2->Controls->Add(this->fosLBStgDir);
            this->tabPage2->Controls->Add(this->fosCBOutputMoreLog);
            this->tabPage2->Controls->Add(this->fosTXStgDir);
            this->tabPage2->Controls->Add(this->fosBTSetFont);
            this->tabPage2->Controls->Add(this->fosCBGetRelativePath);
            this->tabPage2->Controls->Add(this->fosBTStgDir);
            this->tabPage2->Controls->Add(this->fosCBDisableToolTip);
            this->tabPage2->Controls->Add(this->fosCBStgEscKey);
            this->tabPage2->Controls->Add(this->fosCBDisableVisualStyles);
            this->tabPage2->Controls->Add(this->fosCBLogDisableTransparency);
            this->tabPage2->Controls->Add(this->fosLBDisableVisualStyles);
            this->tabPage2->Controls->Add(this->fosCBLogStartMinimized);
            this->tabPage2->Location = System::Drawing::Point(4, 24);
            this->tabPage2->Name = L"tabPage2";
            this->tabPage2->Padding = System::Windows::Forms::Padding(3);
            this->tabPage2->Size = System::Drawing::Size(382, 349);
            this->tabPage2->TabIndex = 1;
            this->tabPage2->Text = L"ログ・設定画面";
            this->tabPage2->UseVisualStyleBackColor = true;
            //
            // frmOtherSettings
            //
            this->AcceptButton = this->fosCBOK;
            this->AutoScaleDimensions = System::Drawing::SizeF(96, 96);
            this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Dpi;
            this->CancelButton = this->fosCBCancel;
            this->ClientSize = System::Drawing::Size(392, 424);
            this->Controls->Add(this->tabControl1);
            this->Controls->Add(this->fosCBOK);
            this->Controls->Add(this->fosCBCancel);
            this->Font = (gcnew System::Drawing::Font(L"Meiryo UI", 9, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
                static_cast<System::Byte>(0)));
            this->FormBorderStyle = System::Windows::Forms::FormBorderStyle::FixedDialog;
            this->KeyPreview = true;
            this->MaximizeBox = false;
            this->Name = L"frmOtherSettings";
            this->ShowIcon = false;
            this->Text = L"frmOtherSettings";
            this->Load += gcnew System::EventHandler(this, &frmOtherSettings::frmOtherSettings_Load);
            this->KeyDown += gcnew System::Windows::Forms::KeyEventHandler(this, &frmOtherSettings::frmOtherSettings_KeyDown);
            this->tabControl1->ResumeLayout(false);
            this->tabPage1->ResumeLayout(false);
            this->tabPage1->PerformLayout();
            this->tabPage2->ResumeLayout(false);
            this->tabPage2->PerformLayout();
            this->ResumeLayout(false);

        }
#pragma endregion
    private:
        System::Void fosCBOK_Click(System::Object^  sender, System::EventArgs^  e) {
            //DisableToolTipHelp = fosCBDisableToolTip->Checked;
            make_file_filter(NULL, 0, fosCXDefaultOutExt->SelectedIndex);
            overwrite_aviutl_ini_file_filter(fosCXDefaultOutExt->SelectedIndex);

            stgDir = fosTXStgDir->Text;
            fos_ex_stg->load_encode_stg();
            fos_ex_stg->load_log_win();
            fos_ex_stg->s_local.auto_afs_disable          = fosCBAutoAFSDisable->Checked;
            fos_ex_stg->s_local.auto_del_chap             = fosCBAutoDelChap->Checked;
            fos_ex_stg->s_local.disable_tooltip_help      = fosCBDisableToolTip->Checked;
            fos_ex_stg->s_local.disable_visual_styles     = fosCBDisableVisualStyles->Checked;
            fos_ex_stg->s_local.enable_stg_esc_key        = fosCBStgEscKey->Checked;
            fos_ex_stg->s_local.chap_nero_convert_to_utf8 = fosCBChapConvertToUTF8->Checked;
            fos_ex_stg->s_log.minimized                   = fosCBLogStartMinimized->Checked;
            fos_ex_stg->s_log.transparent                 = !fosCBLogDisableTransparency->Checked;
            fos_ex_stg->s_log.log_level                    =(fosCBOutputMoreLog->Checked) ? LOG_MORE : LOG_INFO;
            fos_ex_stg->s_local.get_relative_path         = fosCBGetRelativePath->Checked;
            fos_ex_stg->s_local.default_output_ext        = fosCXDefaultOutExt->SelectedIndex;
            const int default_encoder = fosCXDefaultAudioEncoder->SelectedIndex;
            if (default_encoder >= fos_ex_stg->s_aud_int_count) {
                fos_ex_stg->s_local.default_audio_encoder_ext = default_encoder - fos_ex_stg->s_aud_int_count;
                fos_ex_stg->s_local.default_audenc_use_in = FALSE;
            } else {
                fos_ex_stg->s_local.default_audio_encoder_in = default_encoder;
                fos_ex_stg->s_local.default_audenc_use_in = TRUE;
            }
            fos_ex_stg->s_local.run_bat_minimized         = fosCBRunBatMinimized->Checked;
            fos_ex_stg->save_local();
            fos_ex_stg->save_log_win();
            this->Close();
        }
    private:
        System::Void fosSetComboBox() {
            fosCXDefaultOutExt->SuspendLayout();
            fosCXDefaultOutExt->Items->Clear();
            for (int i = 0; i < _countof(OUTPUT_FILE_EXT); i++)
                fosCXDefaultOutExt->Items->Add(String(OUTPUT_FILE_EXT_DESC[i]).ToString() + L" (" + String(OUTPUT_FILE_EXT[i]).ToString() + L")");
            fosCXDefaultOutExt->ResumeLayout();

            fosCXDefaultAudioEncoder->SuspendLayout();
            fosCXDefaultAudioEncoder->Items->Clear();
            for (int i = 0; i < fos_ex_stg->s_aud_int_count; i++)
                fosCXDefaultAudioEncoder->Items->Add(String(fos_ex_stg->s_aud_int[i].dispname).ToString());
            for (int i = 0; i < fos_ex_stg->s_aud_ext_count; i++)
                fosCXDefaultAudioEncoder->Items->Add(String(L"外部: ").ToString() + String(fos_ex_stg->s_aud_ext[i].dispname).ToString());
            fosCXDefaultAudioEncoder->ResumeLayout();
        }
    private:
        System::Void frmOtherSettings_Load(System::Object^  sender, System::EventArgs^  e) {
            fos_ex_stg->load_encode_stg();
            fos_ex_stg->load_log_win();
            fosSetComboBox();

            this->Text = String(AUO_FULL_NAME).ToString();
            fosTXStgDir->Text = stgDir;
            fosCBAutoAFSDisable->Checked         = fos_ex_stg->s_local.auto_afs_disable != 0;
            fosCBAutoDelChap->Checked            = fos_ex_stg->s_local.auto_del_chap != 0;
            fosCBDisableToolTip->Checked         = fos_ex_stg->s_local.disable_tooltip_help != 0;
            fosCBDisableVisualStyles->Checked    = fos_ex_stg->s_local.disable_visual_styles != 0;
            fosCBStgEscKey->Checked              = fos_ex_stg->s_local.enable_stg_esc_key != 0;
            fosCBChapConvertToUTF8->Checked      = fos_ex_stg->s_local.chap_nero_convert_to_utf8 != 0;
            fosCBLogStartMinimized->Checked      = fos_ex_stg->s_log.minimized != 0;
            fosCBLogDisableTransparency->Checked = fos_ex_stg->s_log.transparent == 0;
            fosCBOutputMoreLog->Checked          = fos_ex_stg->s_log.log_level != LOG_INFO;
            fosCBGetRelativePath->Checked        = fos_ex_stg->s_local.get_relative_path != 0;
            fosCXDefaultOutExt->SelectedIndex    = fos_ex_stg->s_local.default_output_ext;
            fosCBRunBatMinimized->Checked        = fos_ex_stg->s_local.run_bat_minimized != 0;
            if (fos_ex_stg->s_local.default_audenc_use_in) {
                fosCXDefaultAudioEncoder->SelectedIndex = clamp(fos_ex_stg->s_local.default_audio_encoder_in, 0, fos_ex_stg->s_aud_int_count-1);
            } else {
                fosCXDefaultAudioEncoder->SelectedIndex = clamp(fos_ex_stg->s_local.default_audio_encoder_ext, 0, fos_ex_stg->s_aud_ext_count-1) + fos_ex_stg->s_aud_int_count;
            }
            if (str_has_char(fos_ex_stg->s_local.conf_font.name))
                SetFontFamilyToForm(this, gcnew FontFamily(String(fos_ex_stg->s_local.conf_font.name).ToString()), this->Font->FontFamily);
        }
    private:
        System::Void fosBTStgDir_Click(System::Object^  sender, System::EventArgs^  e) {
            FolderBrowserDialog^ fbd = gcnew FolderBrowserDialog();
            if (System::IO::Directory::Exists(fosTXStgDir->Text)) {
                fbd->SelectedPath = fosTXStgDir->Text;
            }
            if (fbd->ShowDialog() == System::Windows::Forms::DialogResult::OK) {
                fosTXStgDir->Text = fbd->SelectedPath;
            }
        }
    private:
        System::Void fosCBCancel_Click(System::Object^  sender, System::EventArgs^  e) {
            this->Close();
        }
    private:
        System::Void frmOtherSettings_KeyDown(System::Object^  sender, System::Windows::Forms::KeyEventArgs^  e) {
            if (e->KeyCode == Keys::Escape)
                this->Close();
        }
    private:
        System::Void fosBTSetFont_Click(System::Object^  sender, System::EventArgs^  e) {
            fosfontDialog->Font = fosBTSetFont->Font;
            if (fosfontDialog->ShowDialog() != System::Windows::Forms::DialogResult::Cancel
                && String::Compare(fosfontDialog->Font->FontFamily->Name, this->Font->FontFamily->Name)) {
                guiEx_settings exstg(true);
                exstg.load_encode_stg();
                Set_AUO_FONT_INFO(&exstg.s_local.conf_font, fosfontDialog->Font, this->Font);
                exstg.s_local.conf_font.size = 0.0;
                exstg.s_local.conf_font.style = 0;
                exstg.save_local();
                SetFontFamilyToForm(this, fosfontDialog->Font->FontFamily, this->Font->FontFamily);
            }
        }
};
}
