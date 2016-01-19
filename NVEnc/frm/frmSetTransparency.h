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
using namespace System::ComponentModel;
using namespace System::Collections;
using namespace System::Windows::Forms;
using namespace System::Data;
using namespace System::Drawing;

#include "auo_settings.h"


namespace NVEnc {

    /// <summary>
    /// frmSetTransparency の概要
    ///
    /// 警告: このクラスの名前を変更する場合、このクラスが依存するすべての .resx ファイルに関連付けられた
    ///          マネージ リソース コンパイラ ツールに対して 'Resource File Name' プロパティを
    ///          変更する必要があります。この変更を行わないと、
    ///          デザイナと、このフォームに関連付けられたローカライズ済みリソースとが、
    ///          正しく相互に利用できなくなります。
    /// </summary>
    public ref class frmSetTransparency : public System::Windows::Forms::Form
    {
    public:
        frmSetTransparency()
        {
            InitializeComponent();
            //
            //TODO: ここにコンストラクタ コードを追加します
            //
        }

    protected:
        /// <summary>
        /// 使用中のリソースをすべてクリーンアップします。
        /// </summary>
        ~frmSetTransparency()
        {
            if (components)
            {
                delete components;
            }
        }
    //Instanceを介し、ひとつだけ生成
    private:
        static frmSetTransparency^ _instance;
    public:
        static property frmSetTransparency^ Instance {
            frmSetTransparency^ get() {
                if (_instance == nullptr || _instance->IsDisposed)
                    _instance = gcnew frmSetTransparency();
                return _instance;
            }
        }
    private: System::Windows::Forms::Button^  fstBTDefault;
    private: System::Windows::Forms::Button^  fstBTOK;
    private: System::Windows::Forms::Button^  fstBTCancel;
    private: System::Windows::Forms::TrackBar^  fstTBTransparency;
    private: System::Windows::Forms::NumericUpDown^  fstNUTransparency;
    private: System::Windows::Forms::Label^  fstLBTransparency;
    protected: 



    protected: 

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
            this->fstBTDefault = (gcnew System::Windows::Forms::Button());
            this->fstBTOK = (gcnew System::Windows::Forms::Button());
            this->fstBTCancel = (gcnew System::Windows::Forms::Button());
            this->fstTBTransparency = (gcnew System::Windows::Forms::TrackBar());
            this->fstNUTransparency = (gcnew System::Windows::Forms::NumericUpDown());
            this->fstLBTransparency = (gcnew System::Windows::Forms::Label());
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^  >(this->fstTBTransparency))->BeginInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^  >(this->fstNUTransparency))->BeginInit();
            this->SuspendLayout();
            // 
            // fstBTDefault
            // 
            this->fstBTDefault->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((System::Windows::Forms::AnchorStyles::Bottom | System::Windows::Forms::AnchorStyles::Left));
            this->fstBTDefault->Location = System::Drawing::Point(12, 74);
            this->fstBTDefault->Name = L"fstBTDefault";
            this->fstBTDefault->Size = System::Drawing::Size(80, 31);
            this->fstBTDefault->TabIndex = 2;
            this->fstBTDefault->Text = L"デフォルト";
            this->fstBTDefault->UseVisualStyleBackColor = true;
            this->fstBTDefault->Click += gcnew System::EventHandler(this, &frmSetTransparency::fstBTDefault_Click);
            // 
            // fstBTOK
            // 
            this->fstBTOK->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((System::Windows::Forms::AnchorStyles::Bottom | System::Windows::Forms::AnchorStyles::Right));
            this->fstBTOK->Location = System::Drawing::Point(154, 74);
            this->fstBTOK->Name = L"fstBTOK";
            this->fstBTOK->Size = System::Drawing::Size(80, 31);
            this->fstBTOK->TabIndex = 3;
            this->fstBTOK->Text = L"OK";
            this->fstBTOK->UseVisualStyleBackColor = true;
            this->fstBTOK->Click += gcnew System::EventHandler(this, &frmSetTransparency::fstBTOK_Click);
            // 
            // fstBTCancel
            // 
            this->fstBTCancel->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((System::Windows::Forms::AnchorStyles::Bottom | System::Windows::Forms::AnchorStyles::Right));
            this->fstBTCancel->DialogResult = System::Windows::Forms::DialogResult::Cancel;
            this->fstBTCancel->Location = System::Drawing::Point(240, 74);
            this->fstBTCancel->Name = L"fstBTCancel";
            this->fstBTCancel->Size = System::Drawing::Size(80, 31);
            this->fstBTCancel->TabIndex = 4;
            this->fstBTCancel->Text = L"キャンセル";
            this->fstBTCancel->UseVisualStyleBackColor = true;
            this->fstBTCancel->Click += gcnew System::EventHandler(this, &frmSetTransparency::fstBTCancel_Click);
            // 
            // fstTBTransparency
            // 
            this->fstTBTransparency->AutoSize = false;
            this->fstTBTransparency->Location = System::Drawing::Point(12, 27);
            this->fstTBTransparency->Maximum = 100;
            this->fstTBTransparency->Name = L"fstTBTransparency";
            this->fstTBTransparency->Size = System::Drawing::Size(222, 31);
            this->fstTBTransparency->TabIndex = 0;
            this->fstTBTransparency->TickStyle = System::Windows::Forms::TickStyle::None;
            this->fstTBTransparency->Scroll += gcnew System::EventHandler(this, &frmSetTransparency::fstTBTransparency_Scroll);
            // 
            // fstNUTransparency
            // 
            this->fstNUTransparency->Location = System::Drawing::Point(240, 27);
            this->fstNUTransparency->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) {90, 0, 0, 0});
            this->fstNUTransparency->Name = L"fstNUTransparency";
            this->fstNUTransparency->Size = System::Drawing::Size(62, 22);
            this->fstNUTransparency->TabIndex = 1;
            this->fstNUTransparency->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
            this->fstNUTransparency->TextChanged += gcnew System::EventHandler(this, &frmSetTransparency::fstNUTransparency_TextChanged);
            // 
            // fstLBTransparency
            // 
            this->fstLBTransparency->AutoSize = true;
            this->fstLBTransparency->Location = System::Drawing::Point(308, 29);
            this->fstLBTransparency->Name = L"fstLBTransparency";
            this->fstLBTransparency->Size = System::Drawing::Size(19, 14);
            this->fstLBTransparency->TabIndex = 5;
            this->fstLBTransparency->Text = L"%";
            // 
            // frmSetTransparency
            // 
            this->AutoScaleDimensions = System::Drawing::SizeF(96, 96);
            this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Dpi;
            this->AcceptButton = this->fstBTOK;
            this->CancelButton = this->fstBTCancel;
            this->ClientSize = System::Drawing::Size(332, 115);
            this->Controls->Add(this->fstLBTransparency);
            this->Controls->Add(this->fstNUTransparency);
            this->Controls->Add(this->fstTBTransparency);
            this->Controls->Add(this->fstBTCancel);
            this->Controls->Add(this->fstBTOK);
            this->Controls->Add(this->fstBTDefault);
            this->Font = (gcnew System::Drawing::Font(L"Meiryo UI", 9, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point, 
                static_cast<System::Byte>(0)));
            this->FormBorderStyle = System::Windows::Forms::FormBorderStyle::FixedDialog;
            this->KeyPreview = true;
            this->MaximizeBox = false;
            this->MinimizeBox = false;
            this->Name = L"frmSetTransparency";
            this->ShowIcon = false;
            this->Text = L"透過率の指定 (上限 90%)";
            this->Load += gcnew System::EventHandler(this, &frmSetTransparency::frmSetTransparency_Load);
            this->FormClosed += gcnew System::Windows::Forms::FormClosedEventHandler(this, &frmSetTransparency::frmSetTransparency_FormClosed);
            this->KeyDown += gcnew System::Windows::Forms::KeyEventHandler(this, &frmSetTransparency::frmSetTransparency_KeyDown);
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^  >(this->fstTBTransparency))->EndInit();
            (cli::safe_cast<System::ComponentModel::ISupportInitialize^  >(this->fstNUTransparency))->EndInit();
            this->ResumeLayout(false);
            this->PerformLayout();

        }
#pragma endregion
    private:
        int last_transparency;
    private:
        System::Void fstSetLastTransparency();
        System::Void setTransparency(int value);
        System::Void frmSetTransparency_FormClosed(System::Object^  sender, System::Windows::Forms::FormClosedEventArgs^  e);
    private: 
        System::Void frmSetTransparency_Load(System::Object^  sender, System::EventArgs^  e) {
            fstSetLastTransparency();
            setTransparency(last_transparency);
        }
    private:
        System::Void fstBTDefault_Click(System::Object^  sender, System::EventArgs^  e) {
            setTransparency(DEFAULT_LOG_TRANSPARENCY);
        }
    private:
        System::Void fstBTOK_Click(System::Object^  sender, System::EventArgs^  e) {
            this->Close();
        }
    private:
        System::Void fstBTCancel_Click(System::Object^  sender, System::EventArgs^  e) {
            setTransparency(last_transparency);
            this->Close();
        }
    private:
        System::Void fstNUTransparency_TextChanged(System::Object^  sender, System::EventArgs^  e) {
            int value = 0;
            if (Int32::TryParse(fstNUTransparency->Text, value))
                setTransparency(value);
            else
                setTransparency(fstTBTransparency->Value);
        }
    private:
        System::Void fstTBTransparency_Scroll(System::Object^  sender, System::EventArgs^  e) {
            setTransparency(fstTBTransparency->Value);
        }
    private:
        System::Void frmSetTransparency_KeyDown(System::Object^  sender, System::Windows::Forms::KeyEventArgs^  e) {
             if (e->KeyCode == Keys::Escape)
                this->Close();
        }
};
}
