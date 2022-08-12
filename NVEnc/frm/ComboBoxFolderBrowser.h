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

using namespace System;
using namespace System::IO;
using namespace System::ComponentModel;
using namespace System::Collections;
using namespace System::Diagnostics;
using namespace System::Drawing;
using namespace System::Collections::Generic;
using namespace System::Windows::Forms;

namespace NVEnc 
{
	//ComboBoxにRootDirからのサブフォルダ一覧を表示する
	//リンクオプションの「埋め込みマネージリソースファイル」に"folder_open.ico"を追加する
	public ref class ComboBoxFolderBrowser : public System::Windows::Forms::ComboBox
	{
	public:
		ComboBoxFolderBrowser(void)
		{
			DirPathList = gcnew List<String^>();
			DirLevelList = gcnew List<int>();
			Init();
		}
		//表示のもととなるフォルダを設定し、サブフォルダ一覧を更新する
		//RootDirを選択状態にする
		void SetRootDirAndReload(String^ rootDir)
		{
			RootDir = (Path::GetDirectoryName(rootDir)->Length + 1 == rootDir->Length) ? Path::GetDirectoryName(rootDir) : rootDir;
			ReLoad();
			this->SelectedIndex = 0;
		}
		//サブフォルダ一覧を更新する
		//現在選択中のフォルダが存在しなければ、RootDirを選択状態にする
		void ReLoad()
		{
			String^ SelectedDir = (this->SelectedIndex >= 0) ? DirPathList[this->SelectedIndex] : RootDir;
			DirPathList->Clear();
			DirLevelList->Clear();
			this->Items->Clear();

			BuildFolderList(RootDir, 0);
			SelectDirectory(SelectedDir);
		}
		//指定したフォルダを選択状態にする。
		//指定したフォルダが存在しない場合、RootDirを選択状態にする
		void SelectDirectory(String^ dir)
		{
			this->SelectedIndex = 0;
			for (int i = 0; i < this->Items->Count; i++) {
				if (String::Compare(DirPathList[i], dir, true) == 0) {
					this->SelectedIndex = i;
					break;
				}
			}
		}
		//選択されたフォルダのパスを返す
		String^ GetSelectedFolder()
		{
			return (this->SelectedIndex >= 0) ? DirPathList[this->SelectedIndex] : L"";
		}
	protected:
		~ComboBoxFolderBrowser(void)
		{
			delete DirPathList;
			delete DirLevelList;
		}
		//初期化、必要な設定を行う
		void Init()
		{
			this->DrawMode = System::Windows::Forms::DrawMode::OwnerDrawFixed;
			this->DrawItem += gcnew System::Windows::Forms::DrawItemEventHandler(this, &ComboBoxFolderBrowser::_DrawItem);
			this->DropDown += gcnew System::EventHandler(this, &ComboBoxFolderBrowser::_DropDown);
			this->DropDownClosed += gcnew System::EventHandler(this, &ComboBoxFolderBrowser::_DropDownClosed);
			this->DropDownStyle = ComboBoxStyle::DropDownList;
			DropDownOpened = false;

			System::Reflection::Assembly^ assem = System::Reflection::Assembly::GetExecutingAssembly();
			FolderIcon = Image::FromStream(assem->GetManifestResourceStream(L"folder_open.ico"));
		}
		//フォルダリストの構築
		void BuildFolderList(String^ dir, int level)
		{
			DirPathList->Add(dir);
			DirLevelList->Add(level);
			this->Items->Add(dir->Substring(Path::GetDirectoryName(dir)->Length + 1));
			array<String^>^ FolderList = Directory::GetDirectories(dir);
			for (int i = 0; i < FolderList->Length; i++)
				BuildFolderList(FolderList[i], level + 1);
		}
		//描画イベント群
		System::Void _DropDown(System::Object^  sender, System::EventArgs^  e) {
			DropDownOpened = true;
		}
		System::Void _DropDownClosed(System::Object^  sender, System::EventArgs^  e) {
			DropDownOpened = false;
			this->Refresh();
		}
		void _DrawItem(Object^ sender, DrawItemEventArgs^ e)
		{
			if (e->Index < 0)
				return;

			if (DropDownOpened)
				e->DrawBackground();

			e->Graphics->DrawImage(FolderIcon, (float)(e->Bounds.X + FolderIcon->Width * DirLevelList[e->Index]), (float)e->Bounds.Y);
			e->Graphics->DrawString(this->Items[e->Index]->ToString(),
				this->Font,
				gcnew SolidBrush(this->ForeColor),
				(float)(e->Bounds.X + FolderIcon->Width * DirLevelList[e->Index] + FolderIcon->Width),
				(float)e->Bounds.Y);
		}
	private:
		String^ RootDir; //RootDir
		List<String^>^ DirPathList; //パスの一覧、this->Itemsに対応
		List<int>^ DirLevelList; //RootDirからの階層のリスト、this->Itemsに対応 RootDir=0
		bool DropDownOpened; //ComboBoxのDropDownが開かれているか
		Image^ FolderIcon; //フォルダアイコン
	};
};