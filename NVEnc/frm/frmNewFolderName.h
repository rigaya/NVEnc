#pragma once

#include "auo_settings.h"

using namespace System;
using namespace System::ComponentModel;
using namespace System::Collections;
using namespace System::Windows::Forms;
using namespace System::Data;
using namespace System::Drawing;


namespace NVEnc {

    /// <summary>
    /// frmNewFolderName の概要
    ///
    /// 警告: このクラスの名前を変更する場合、このクラスが依存するすべての .resx ファイルに関連付けられた
    ///          マネージ リソース コンパイラ ツールに対して 'Resource File Name' プロパティを
    ///          変更する必要があります。この変更を行わないと、
    ///          デザイナと、このフォームに関連付けられたローカライズ済みリソースとが、
    ///          正しく相互に利用できなくなります。
    /// </summary>
    public ref class frmNewFolderName : public System::Windows::Forms::Form
    {
    public:
        frmNewFolderName(void)
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
        ~frmNewFolderName()
        {
            if (components)
            {
                delete components;
            }
        }
    private: System::Windows::Forms::TextBox^  fnfTXNewFolderName;
    private: System::Windows::Forms::Button^  fnfBTOK;
    protected: 
    private: System::Windows::Forms::Button^  fnfBTCancel;

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
            this->fnfBTOK = (gcnew System::Windows::Forms::Button());
            this->fnfBTCancel = (gcnew System::Windows::Forms::Button());
            this->fnfTXNewFolderName = (gcnew System::Windows::Forms::TextBox());
            this->SuspendLayout();
            // 
            // fnfBTOK
            // 
            this->fnfBTOK->Location = System::Drawing::Point(239, 61);
            this->fnfBTOK->Name = L"fnfBTOK";
            this->fnfBTOK->Size = System::Drawing::Size(75, 32);
            this->fnfBTOK->TabIndex = 7;
            this->fnfBTOK->Text = L"OK";
            this->fnfBTOK->UseVisualStyleBackColor = true;
            this->fnfBTOK->Click += gcnew System::EventHandler(this, &frmNewFolderName::fnfBTOK_Click);
            // 
            // fnfBTCancel
            // 
            this->fnfBTCancel->DialogResult = System::Windows::Forms::DialogResult::Cancel;
            this->fnfBTCancel->Location = System::Drawing::Point(155, 61);
            this->fnfBTCancel->Name = L"fnfBTCancel";
            this->fnfBTCancel->Size = System::Drawing::Size(75, 32);
            this->fnfBTCancel->TabIndex = 6;
            this->fnfBTCancel->Text = L"キャンセル";
            this->fnfBTCancel->UseVisualStyleBackColor = true;
            this->fnfBTCancel->Click += gcnew System::EventHandler(this, &frmNewFolderName::fnfBTCancel_Click);
            // 
            // fnfTXNewFolderName
            // 
            this->fnfTXNewFolderName->Location = System::Drawing::Point(24, 23);
            this->fnfTXNewFolderName->Name = L"fnfTXNewFolderName";
            this->fnfTXNewFolderName->Size = System::Drawing::Size(290, 23);
            this->fnfTXNewFolderName->TabIndex = 8;
            // 
            // frmNewFolderName
            // 
            this->AutoScaleDimensions = System::Drawing::SizeF(96, 96);
            this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Dpi;
            this->AcceptButton = this->fnfBTOK;
            this->CancelButton = this->fnfBTCancel;
            this->ClientSize = System::Drawing::Size(327, 105);
            this->Controls->Add(this->fnfTXNewFolderName);
            this->Controls->Add(this->fnfBTOK);
            this->Controls->Add(this->fnfBTCancel);
            this->Font = (gcnew System::Drawing::Font(L"Meiryo UI", 9, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point, 
                static_cast<System::Byte>(0)));
            this->FormBorderStyle = System::Windows::Forms::FormBorderStyle::FixedDialog;
            this->KeyPreview = true;
            this->MaximizeBox = false;
            this->MinimizeBox = false;
            this->Name = L"frmNewFolderName";
            this->ShowIcon = false;
            this->Text = L"新しいフォルダ...";
            this->Load += gcnew System::EventHandler(this, &frmNewFolderName::frmNewFolderName_Load);
            this->KeyDown += gcnew System::Windows::Forms::KeyEventHandler(this, &frmNewFolderName::frmNewFolderName_KeyDown);
            this->ResumeLayout(false);
            this->PerformLayout();

        }
#pragma endregion
    public:
        String^ NewFolder;
    private: 
        System::Void frmNewFolderName_Load(System::Object^  sender, System::EventArgs^  e) {
            NewFolder = L"";
            fnfTXNewFolderName->Select();
            
            //フォントの設定
            guiEx_settings exstg;
            exstg.load_encode_stg();
            if (str_has_char(exstg.s_local.conf_font.name))
                SetFontFamilyToForm(this, gcnew FontFamily(String(exstg.s_local.conf_font.name).ToString()), this->Font->FontFamily);
        }
    private:
        System::Void fnfBTOK_Click(System::Object^  sender, System::EventArgs^  e) {
            if (!ValidiateFileName(fnfTXNewFolderName->Text)) {
                MessageBox::Show(L"フォルダ名に使用できない文字が含まれています。", L"エラー", MessageBoxButtons::OK, MessageBoxIcon::Error);
                return;
            }
            NewFolder = fnfTXNewFolderName->Text;
            this->Close();
        }
    private:
        System::Void fnfBTCancel_Click(System::Object^  sender, System::EventArgs^  e) {
            this->Close();
        }
    private:
        System::Void frmNewFolderName_KeyDown(System::Object^  sender, System::Windows::Forms::KeyEventArgs^  e) {
            if (e->KeyCode == Keys::Escape)
                this->Close();
        }
};
}
