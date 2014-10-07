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
#include <mmsystem.h>

#include "auo.h"
#include "auo_frm.h"
#include "auo_util.h"
#include "auo_version.h"
#include "auo_settings.h"
#include "auo_win7_taskbar.h"


//以下部分的にwarning C4100を黙らせる
//C4100 : 引数は関数の本体部で 1 度も参照されません。
#pragma warning( push )
#pragma warning( disable: 4100 )

#include "auo_clrutil.h"
#include "frmAutoSaveLogSettings.h"
#include "frmSetTransparency.h"
#include "frmSetLogColor.h"

using namespace System;
using namespace System::ComponentModel;
using namespace System::Collections;
using namespace System::Windows::Forms;
using namespace System::Text;
using namespace System::Data;
using namespace System::Drawing;
using namespace System::IO;

namespace NVEnc {

	/// <summary>
	/// frmLog の概要
	///
	/// 警告: このクラスの名前を変更する場合、このクラスが依存するすべての .resx ファイルに関連付けられた
	///          マネージ リソース コンパイラ ツールに対して 'Resource File Name' プロパティを
	///          変更する必要があります。この変更を行わないと、
	///          デザイナと、このフォームに関連付けられたローカライズ済みリソースとが、
	///          正しく相互に利用できなくなります。
	/// </summary>
	public ref class frmLog : public System::Windows::Forms::Form
	{
	public:
		frmLog(void)
		{
			taskbar_progress = nullptr;
			timerResizeOrPos = nullptr;
			//これがfalseだとイベントで設定保存をするので、とりあえずtrue
			prevent_log_closing = true;

			_start_time = 0;

			//設定から情報を取得
			guiEx_settings exstg;
			exstg.load_log_win();
			if (exstg.s_log.minimized)
				this->WindowState = FormWindowState::Minimized;
			_enc_pause = NULL;
			LogTitle = String(AUO_FULL_NAME).ToString();

			InitializeComponent();
			//
			//TODO: ここにコンストラクタ コードを追加します
			//
			this->log_color_text = gcnew array<Color>(3) { ColorfromInt(exstg.s_log.log_color_text[0]), ColorfromInt(exstg.s_log.log_color_text[1]), ColorfromInt(exstg.s_log.log_color_text[2]) };
			this->richTextLog->BackColor = ColorfromInt(exstg.s_log.log_color_background);
			this->log_type = gcnew array<String^>(3) { L"info", L"warning", L"error" };
			this->richTextLog->LanguageOption = System::Windows::Forms::RichTextBoxLanguageOptions::UIFonts;

			//this->ToolStripMenuItemEncPause->Enabled = false;
			frmTransparency = exstg.s_log.transparency;
			this->ToolStripMenuItemTransparent->Checked = exstg.s_log.transparent != 0;
			this->toolStripMenuItemAutoSave->Checked = exstg.s_log.auto_save_log != 0;
			this->toolStripMenuItemShowStatus->Checked = exstg.s_log.show_status_bar != 0;
			this->ToolStripMenuItemStartMinimized->Checked = exstg.s_log.minimized != 0;
			this->toolStripMenuItemSaveLogSize->Checked = exstg.s_log.save_log_size != 0;
			bool check_win7later = check_OS_Win7orLater() != 0;
			this->toolStripMenuItemTaskBarProgress->Enabled = check_win7later;
			this->toolStripMenuItemTaskBarProgress->Checked = (exstg.s_log.taskbar_progress != 0 && check_win7later);
			//ウィンドウサイズ調整等(サイズ設定->最小化の設定の順に行うこと)
			if (exstg.s_log.save_log_size)
				SetWindowSize(exstg.s_log.log_width, exstg.s_log.log_height);
			lastWindowState = this->WindowState;
			//ウィンドウハンドルの取得
			hWnd = (HWND)this->Handle.ToPointer();
			//プログレスバーの初期化
			taskbar_progress = new taskbarProgress(hWnd);
			taskbar_progress->set_visible(FALSE != exstg.s_log.taskbar_progress);
			//ログフォントの設定
			richTextLog->Font = GetFontFrom_AUO_FONT_INFO(&exstg.s_log.log_font, richTextLog->Font);
			//wine互換モードの設定
			wine_compatible_mode = FALSE != exstg.s_log.wine_compat;
			//通常のステータスに戻す(false) -> 設定保存イベントで設定保存される
			prevent_log_closing = false;
			closed = true;
		}
	protected:
		/// <summary>
		/// 使用中のリソースをすべてクリーンアップします。
		/// </summary>
		~frmLog()
		{
			if (nullptr != taskbar_progress) {
				delete taskbar_progress;
			}
			if (components)
			{
				delete components;
			}
			delete log_type;

			frmAutoSaveLogSettings::Instance::get()->Close();
		}
	//Instanceを介し、ひとつだけ生成
	private:
		static frmLog^ _instance;
	public:
		static property frmLog^ Instance {
			frmLog^ get() {
				if (_instance == nullptr || _instance->IsDisposed)
					_instance = gcnew frmLog();
				return _instance;
			}
		}
	private:
		taskbarProgress *taskbar_progress; //タスクバーでの進捗表示
		HWND hWnd; //このウィンドウのハンドル
		BOOL *_enc_pause;      //エンコ一時停止へのポインタ
		DWORD _start_time;//x264エンコ開始時間
		bool closed; //このウィンドウが閉じているか、開いているか
		bool prevent_log_closing; //ログウィンドウを閉じるを無効化するか・設定保存イベントのフラグでもある
		bool wine_compatible_mode; //wine互換モード
		bool add_progress;
		array<String^>^ log_type;
		array<Color>^ log_color_text;
		int LastLogLen;  //ひとつ前のエンコードブロックの終わり
		bool using_afs; //afs使用時にオン
		int total_frame; //エンコ総フレーム数
		DWORD pause_start; //一時停止を開始した時間
		String^ LogTitle; //ログウィンドウのタイトル表示
		FormWindowState lastWindowState; //最終ウィンドウステータス(normal/最大化/最小化)
		System::Threading::Timer^ timerResizeOrPos; //ログウィンドウの位置・大きさ保存のイベントチェック用
		static const int timerResizeOrPosPeriod = 500;
		delegate System::Void timerResizeOrPosChangeDelegate();
	public:
		int frmTransparency; //透過率

	private: System::Windows::Forms::RichTextBox^  richTextLog;
	private: System::Windows::Forms::ContextMenuStrip^  contextMenuStripLog;


	private: System::Windows::Forms::ToolStripMenuItem^  ToolStripMenuItemTransparent;
	private: System::Windows::Forms::ToolStripMenuItem^  ToolStripMenuItemStartMinimized;

	private: System::Windows::Forms::ToolStripMenuItem^  toolStripMenuItemAutoSave;
	private: System::Windows::Forms::StatusStrip^  statusStripLog;
	private: System::Windows::Forms::ToolStripStatusLabel^  toolStripStatusCurrentTask;
	private: System::Windows::Forms::ToolStripProgressBar^  toolStripCurrentProgress;
	private: System::Windows::Forms::ToolStripStatusLabel^  toolStripStatusCurrentProgress;
	private: System::Windows::Forms::ToolStripMenuItem^  toolStripMenuItemShowStatus;
	private: System::Windows::Forms::ToolStripMenuItem^  toolStripMenuItemTaskBarProgress;
	private: System::Windows::Forms::ToolStripMenuItem^  toolStripMenuItemAutoSaveSettings;
	private: System::Windows::Forms::ToolStripMenuItem^  toolStripMenuItemSaveLogSize;
private: System::Windows::Forms::ToolStripMenuItem^  toolStripMenuItemWindowFont;
private: System::Windows::Forms::FontDialog^  fontDialogLog;
private: System::Windows::Forms::ToolStripMenuItem^  toolStripMenuItemTransparentValue;
private: System::Windows::Forms::ToolStripMenuItem^  toolStripMenuItemSetLogColor;
private: System::Windows::Forms::ToolStripMenuItem^  toolStripMenuItemFileOpen;
private: System::Windows::Forms::ToolStripMenuItem^  toolStripMenuItemFilePathOpen;
private: System::Windows::Forms::ToolStripMenuItem^  ToolStripMenuItemEncPause;
private: System::Windows::Forms::ToolStripStatusLabel^  toolStripStatusElapsedTime;
private: System::Windows::Forms::ToolStripMenuItem^  toolStripMenuItem1;


	private: System::ComponentModel::IContainer^  components;
	protected: 

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
			this->richTextLog = (gcnew System::Windows::Forms::RichTextBox());
			this->contextMenuStripLog = (gcnew System::Windows::Forms::ContextMenuStrip(this->components));
			this->toolStripMenuItemFileOpen = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->toolStripMenuItemFilePathOpen = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->ToolStripMenuItemEncPause = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->ToolStripMenuItemTransparent = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->toolStripMenuItemTransparentValue = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->toolStripMenuItemSetLogColor = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->ToolStripMenuItemStartMinimized = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->toolStripMenuItemSaveLogSize = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->toolStripMenuItemAutoSave = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->toolStripMenuItemAutoSaveSettings = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->toolStripMenuItemShowStatus = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->toolStripMenuItemTaskBarProgress = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->toolStripMenuItemWindowFont = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->statusStripLog = (gcnew System::Windows::Forms::StatusStrip());
			this->toolStripStatusCurrentTask = (gcnew System::Windows::Forms::ToolStripStatusLabel());
			this->toolStripStatusElapsedTime = (gcnew System::Windows::Forms::ToolStripStatusLabel());
			this->toolStripCurrentProgress = (gcnew System::Windows::Forms::ToolStripProgressBar());
			this->toolStripStatusCurrentProgress = (gcnew System::Windows::Forms::ToolStripStatusLabel());
			this->fontDialogLog = (gcnew System::Windows::Forms::FontDialog());
			this->contextMenuStripLog->SuspendLayout();
			this->statusStripLog->SuspendLayout();
			this->SuspendLayout();
			// 
			// richTextLog
			// 
			this->richTextLog->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom)
				| System::Windows::Forms::AnchorStyles::Left)
				| System::Windows::Forms::AnchorStyles::Right));
			this->richTextLog->BackColor = System::Drawing::Color::Black;
			this->richTextLog->ContextMenuStrip = this->contextMenuStripLog;
			this->richTextLog->Font = (gcnew System::Drawing::Font(L"ＭＳ ゴシック", 9, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(128)));
			this->richTextLog->Location = System::Drawing::Point(0, 0);
			this->richTextLog->Name = L"richTextLog";
			this->richTextLog->ReadOnly = true;
			this->richTextLog->Size = System::Drawing::Size(771, 442);
			this->richTextLog->TabIndex = 0;
			this->richTextLog->Text = L"";
			this->richTextLog->WordWrap = false;
			this->richTextLog->MouseDown += gcnew System::Windows::Forms::MouseEventHandler(this, &frmLog::richTextLog_MouseDown);
			// 
			// contextMenuStripLog
			// 
			this->contextMenuStripLog->Items->AddRange(gcnew cli::array< System::Windows::Forms::ToolStripItem^  >(13) {
				this->toolStripMenuItemFileOpen,
					this->toolStripMenuItemFilePathOpen, this->ToolStripMenuItemEncPause, this->ToolStripMenuItemTransparent, this->toolStripMenuItemTransparentValue,
					this->toolStripMenuItemSetLogColor, this->ToolStripMenuItemStartMinimized, this->toolStripMenuItemSaveLogSize, this->toolStripMenuItemAutoSave,
					this->toolStripMenuItemAutoSaveSettings, this->toolStripMenuItemShowStatus, this->toolStripMenuItemTaskBarProgress, this->toolStripMenuItemWindowFont
			});
			this->contextMenuStripLog->Name = L"contextMenuStrip1";
			this->contextMenuStripLog->Size = System::Drawing::Size(248, 290);
			// 
			// toolStripMenuItemFileOpen
			// 
			this->toolStripMenuItemFileOpen->ForeColor = System::Drawing::Color::Blue;
			this->toolStripMenuItemFileOpen->Name = L"toolStripMenuItemFileOpen";
			this->toolStripMenuItemFileOpen->Size = System::Drawing::Size(247, 22);
			this->toolStripMenuItemFileOpen->Text = L"この動画を再生...";
			this->toolStripMenuItemFileOpen->Click += gcnew System::EventHandler(this, &frmLog::toolStripMenuItemFileOpen_Click);
			// 
			// toolStripMenuItemFilePathOpen
			// 
			this->toolStripMenuItemFilePathOpen->ForeColor = System::Drawing::Color::Blue;
			this->toolStripMenuItemFilePathOpen->Name = L"toolStripMenuItemFilePathOpen";
			this->toolStripMenuItemFilePathOpen->Size = System::Drawing::Size(247, 22);
			this->toolStripMenuItemFilePathOpen->Text = L"この動画の場所を開く...";
			this->toolStripMenuItemFilePathOpen->Click += gcnew System::EventHandler(this, &frmLog::toolStripMenuItemFilePathOpen_Click);
			// 
			// ToolStripMenuItemEncPause
			// 
			this->ToolStripMenuItemEncPause->CheckOnClick = true;
			this->ToolStripMenuItemEncPause->Name = L"ToolStripMenuItemEncPause";
			this->ToolStripMenuItemEncPause->Size = System::Drawing::Size(247, 22);
			this->ToolStripMenuItemEncPause->Text = L"エンコード一時停止";
			this->ToolStripMenuItemEncPause->CheckedChanged += gcnew System::EventHandler(this, &frmLog::ToolStripMenuItemEncPause_CheckedChanged);
			// 
			// ToolStripMenuItemTransparent
			// 
			this->ToolStripMenuItemTransparent->CheckOnClick = true;
			this->ToolStripMenuItemTransparent->Name = L"ToolStripMenuItemTransparent";
			this->ToolStripMenuItemTransparent->Size = System::Drawing::Size(247, 22);
			this->ToolStripMenuItemTransparent->Text = L"ちょっと透過";
			this->ToolStripMenuItemTransparent->CheckedChanged += gcnew System::EventHandler(this, &frmLog::ToolStripMenuItemTransparent_CheckedChanged);
			// 
			// toolStripMenuItemTransparentValue
			// 
			this->toolStripMenuItemTransparentValue->Name = L"toolStripMenuItemTransparentValue";
			this->toolStripMenuItemTransparentValue->Size = System::Drawing::Size(247, 22);
			this->toolStripMenuItemTransparentValue->Text = L"透過率の指定...";
			this->toolStripMenuItemTransparentValue->Click += gcnew System::EventHandler(this, &frmLog::toolStripMenuItemTransparentValue_Click);
			// 
			// toolStripMenuItemSetLogColor
			// 
			this->toolStripMenuItemSetLogColor->Name = L"toolStripMenuItemSetLogColor";
			this->toolStripMenuItemSetLogColor->Size = System::Drawing::Size(247, 22);
			this->toolStripMenuItemSetLogColor->Text = L"ログの色の指定...";
			this->toolStripMenuItemSetLogColor->Click += gcnew System::EventHandler(this, &frmLog::toolStripMenuItemSetLogColor_Click);
			// 
			// ToolStripMenuItemStartMinimized
			// 
			this->ToolStripMenuItemStartMinimized->CheckOnClick = true;
			this->ToolStripMenuItemStartMinimized->Name = L"ToolStripMenuItemStartMinimized";
			this->ToolStripMenuItemStartMinimized->Size = System::Drawing::Size(247, 22);
			this->ToolStripMenuItemStartMinimized->Text = L"このウィンドウを最小化で開始";
			this->ToolStripMenuItemStartMinimized->CheckedChanged += gcnew System::EventHandler(this, &frmLog::ToolStripCheckItem_CheckedChanged);
			// 
			// toolStripMenuItemSaveLogSize
			// 
			this->toolStripMenuItemSaveLogSize->CheckOnClick = true;
			this->toolStripMenuItemSaveLogSize->Name = L"toolStripMenuItemSaveLogSize";
			this->toolStripMenuItemSaveLogSize->Size = System::Drawing::Size(247, 22);
			this->toolStripMenuItemSaveLogSize->Text = L"このウィンドウのサイズを保存";
			this->toolStripMenuItemSaveLogSize->CheckedChanged += gcnew System::EventHandler(this, &frmLog::toolStripMenuItemSaveLogSize_CheckedChanged);
			// 
			// toolStripMenuItemAutoSave
			// 
			this->toolStripMenuItemAutoSave->CheckOnClick = true;
			this->toolStripMenuItemAutoSave->Name = L"toolStripMenuItemAutoSave";
			this->toolStripMenuItemAutoSave->Size = System::Drawing::Size(247, 22);
			this->toolStripMenuItemAutoSave->Text = L"ログ自動保存を行う";
			this->toolStripMenuItemAutoSave->CheckedChanged += gcnew System::EventHandler(this, &frmLog::ToolStripCheckItem_CheckedChanged);
			// 
			// toolStripMenuItemAutoSaveSettings
			// 
			this->toolStripMenuItemAutoSaveSettings->Name = L"toolStripMenuItemAutoSaveSettings";
			this->toolStripMenuItemAutoSaveSettings->Size = System::Drawing::Size(247, 22);
			this->toolStripMenuItemAutoSaveSettings->Text = L"ログ自動保存の設定...";
			this->toolStripMenuItemAutoSaveSettings->Click += gcnew System::EventHandler(this, &frmLog::toolStripMenuItemAutoSaveSettings_Click);
			// 
			// toolStripMenuItemShowStatus
			// 
			this->toolStripMenuItemShowStatus->Checked = true;
			this->toolStripMenuItemShowStatus->CheckOnClick = true;
			this->toolStripMenuItemShowStatus->CheckState = System::Windows::Forms::CheckState::Checked;
			this->toolStripMenuItemShowStatus->Name = L"toolStripMenuItemShowStatus";
			this->toolStripMenuItemShowStatus->Size = System::Drawing::Size(247, 22);
			this->toolStripMenuItemShowStatus->Text = L"ステータスバーの表示";
			this->toolStripMenuItemShowStatus->CheckedChanged += gcnew System::EventHandler(this, &frmLog::toolStripMenuItemShowStatus_CheckedChanged);
			// 
			// toolStripMenuItemTaskBarProgress
			// 
			this->toolStripMenuItemTaskBarProgress->CheckOnClick = true;
			this->toolStripMenuItemTaskBarProgress->Name = L"toolStripMenuItemTaskBarProgress";
			this->toolStripMenuItemTaskBarProgress->Size = System::Drawing::Size(247, 22);
			this->toolStripMenuItemTaskBarProgress->Text = L"タスクバーに進捗を表示(Win7)";
			this->toolStripMenuItemTaskBarProgress->CheckedChanged += gcnew System::EventHandler(this, &frmLog::toolStripMenuItemTaskBarProgress_CheckedChanged);
			// 
			// toolStripMenuItemWindowFont
			// 
			this->toolStripMenuItemWindowFont->Name = L"toolStripMenuItemWindowFont";
			this->toolStripMenuItemWindowFont->Size = System::Drawing::Size(247, 22);
			this->toolStripMenuItemWindowFont->Text = L"表示フォント...";
			this->toolStripMenuItemWindowFont->Click += gcnew System::EventHandler(this, &frmLog::toolStripMenuItemWindowFont_Click);
			// 
			// statusStripLog
			// 
			this->statusStripLog->Items->AddRange(gcnew cli::array< System::Windows::Forms::ToolStripItem^  >(4) {
				this->toolStripStatusCurrentTask,
					this->toolStripStatusElapsedTime, this->toolStripCurrentProgress, this->toolStripStatusCurrentProgress
			});
			this->statusStripLog->Location = System::Drawing::Point(0, 445);
			this->statusStripLog->Name = L"statusStripLog";
			this->statusStripLog->Size = System::Drawing::Size(771, 23);
			this->statusStripLog->TabIndex = 1;
			this->statusStripLog->Text = L"statusStrip1";
			// 
			// toolStripStatusCurrentTask
			// 
			this->toolStripStatusCurrentTask->Name = L"toolStripStatusCurrentTask";
			this->toolStripStatusCurrentTask->Size = System::Drawing::Size(35, 18);
			this->toolStripStatusCurrentTask->Text = L"Task";
			this->toolStripStatusCurrentTask->TextAlign = System::Drawing::ContentAlignment::MiddleLeft;
			// 
			// toolStripStatusElapsedTime
			// 
			this->toolStripStatusElapsedTime->Font = (gcnew System::Drawing::Font(L"メイリオ", 8.25F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(128)));
			this->toolStripStatusElapsedTime->Name = L"toolStripStatusElapsedTime";
			this->toolStripStatusElapsedTime->Size = System::Drawing::Size(721, 18);
			this->toolStripStatusElapsedTime->Spring = true;
			this->toolStripStatusElapsedTime->Text = L"Elapsed Time";
			this->toolStripStatusElapsedTime->TextAlign = System::Drawing::ContentAlignment::BottomLeft;
			// 
			// toolStripCurrentProgress
			// 
			this->toolStripCurrentProgress->Maximum = 1000;
			this->toolStripCurrentProgress->Name = L"toolStripCurrentProgress";
			this->toolStripCurrentProgress->Size = System::Drawing::Size(180, 17);
			this->toolStripCurrentProgress->Style = System::Windows::Forms::ProgressBarStyle::Continuous;
			this->toolStripCurrentProgress->Visible = false;
			// 
			// toolStripStatusCurrentProgress
			// 
			this->toolStripStatusCurrentProgress->AutoSize = false;
			this->toolStripStatusCurrentProgress->Name = L"toolStripStatusCurrentProgress";
			this->toolStripStatusCurrentProgress->Size = System::Drawing::Size(64, 18);
			this->toolStripStatusCurrentProgress->Text = L"Progress";
			this->toolStripStatusCurrentProgress->TextAlign = System::Drawing::ContentAlignment::MiddleRight;
			this->toolStripStatusCurrentProgress->Visible = false;
			// 
			// fontDialogLog
			// 
			this->fontDialogLog->AllowVerticalFonts = false;
			this->fontDialogLog->Font = (gcnew System::Drawing::Font(L"Meiryo UI", 9, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(0)));
			this->fontDialogLog->FontMustExist = true;
			this->fontDialogLog->MaxSize = 9;
			this->fontDialogLog->MinSize = 9;
			this->fontDialogLog->ShowEffects = false;
			// 
			// frmLog
			// 
			this->AutoScaleDimensions = System::Drawing::SizeF(96, 96);
			this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Dpi;
			this->ClientSize = System::Drawing::Size(771, 468);
			this->Controls->Add(this->statusStripLog);
			this->Controls->Add(this->richTextLog);
			this->Font = (gcnew System::Drawing::Font(L"Meiryo UI", 9, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(0)));
			this->Name = L"frmLog";
			this->ShowIcon = false;
			this->Text = L"NVEnc Log";
			this->FormClosing += gcnew System::Windows::Forms::FormClosingEventHandler(this, &frmLog::frmLog_FormClosing);
			this->Load += gcnew System::EventHandler(this, &frmLog::frmLog_Load);
			this->ClientSizeChanged += gcnew System::EventHandler(this, &frmLog::frmLog_ClientSizeChanged);
			this->LocationChanged += gcnew System::EventHandler(this, &frmLog::frmLog_LocationChanged);
			this->contextMenuStripLog->ResumeLayout(false);
			this->statusStripLog->ResumeLayout(false);
			this->statusStripLog->PerformLayout();
			this->ResumeLayout(false);
			this->PerformLayout();

		}
#pragma endregion
	private: 
		System::Void frmLog_Load(System::Object^  sender, System::EventArgs^  e) {
			closed = false;
			pause_start = NULL;
			
			guiEx_settings exstg(true);
			exstg.load_log_win();
			SetWindowPos(exstg.s_log.log_pos[0], exstg.s_log.log_pos[1]);

			//timerの初期化
			timerResizeOrPos = gcnew System::Threading::Timer(
				gcnew System::Threading::TimerCallback(this, &frmLog::timerResizeOrPosChange),
				nullptr, System::Threading::Timeout::Infinite, timerResizeOrPosPeriod);
		}
	private:
		System::Void SetWindowPos(int x, int y) {
			//デフォルトのままにする
			if (x <= 0 || y <= 0)
				return;

			//有効な位置かどうかを確認
			array<System::Windows::Forms::Screen^>^ allScreens = System::Windows::Forms::Screen::AllScreens;
			for (int i = 0; i < allScreens->Length; i++) {
				if (   check_range(x, allScreens[i]->Bounds.X, allScreens[i]->Bounds.X + allScreens[i]->Bounds.Width)
					&& check_range(y, allScreens[i]->Bounds.Y, allScreens[i]->Bounds.Y + allScreens[i]->Bounds.Height)) {
					Point point;
					point.X = min(x, allScreens[i]->Bounds.X + allScreens[i]->Bounds.Width - 120);
					point.Y = min(y, allScreens[i]->Bounds.Y + allScreens[i]->Bounds.Height - 120);
					this->Location = point;
					return;
				}
			}
		}
	private:
		System::Void SetWindowSize(int width, int height) {
			//デフォルトのままにする
			if (width <= 0 || height <= 0)
				return;

			//デスクトップ領域(タスクバー等除く)
			System::Drawing::Rectangle screen = System::Windows::Forms::Screen::GetWorkingArea(this);
			this->ClientSize = System::Drawing::Size(min(width, screen.Width), min(height, screen.Height));
		}
	public:
		System::Void ReloadLogWindowSettings() {
			guiEx_settings exstg;
			exstg.load_log_win();
			wine_compatible_mode                     = exstg.s_log.wine_compat != 0;
			frmTransparency                          = exstg.s_log.transparency;
			ToolStripMenuItemTransparent->Checked    = exstg.s_log.transparent != 0;
			toolStripMenuItemAutoSave->Checked       = exstg.s_log.auto_save_log != 0;
			toolStripMenuItemShowStatus->Checked     = exstg.s_log.show_status_bar != 0;
			ToolStripMenuItemStartMinimized->Checked = exstg.s_log.minimized != 0;
			toolStripMenuItemSaveLogSize->Checked    = exstg.s_log.save_log_size != 0;
		}
	public:
		System::Void SetWindowTitle(const char *chr) {
			LogTitle = String(chr).ToString();
			this->Text = LogTitle;
		}
	public:
		System::Void SetWindowTitle(const char *chr, int progress_mode) {
			LogTitle = String(chr).ToString();
			this->Text = LogTitle;
			bool show_progress = (progress_mode != PROGRESSBAR_DISABLED);
			toolStripCurrentProgress->Visible = show_progress;
			toolStripStatusCurrentProgress->Visible = show_progress;
			toolStripCurrentProgress->Style = (progress_mode == PROGRESSBAR_MARQUEE) ? ProgressBarStyle::Marquee : ProgressBarStyle::Continuous;
			toolStripStatusCurrentProgress->Text = L"";
			toolStripStatusElapsedTime->Text = L"";
			toolStripStatusCurrentTask->Text = (show_progress) ? LogTitle : L"";
			if (!show_progress)
				toolStripCurrentProgress->Value = 0;
			taskbar_progress->set_progress(0.0);
			taskbar_progress->set_mode(progress_mode);
		}
	public:
		System::Void SetTaskName(const char *chr) {
			toolStripStatusCurrentTask->Text = String(chr).ToString();
		}
	public:
		System::Void SetProgress(double progress) {
			toolStripCurrentProgress->Value = clamp((int)(progress * toolStripCurrentProgress->Maximum + 0.5), toolStripCurrentProgress->Minimum, toolStripCurrentProgress->Maximum);
			toolStripStatusCurrentProgress->Text = (progress).ToString("P1");
			toolStripStatusElapsedTime->Text = L"";
			this->Text = L"[" + toolStripStatusCurrentProgress->Text + L"] " + LogTitle;
			taskbar_progress->set_progress(progress);
		}
	public:
		System::Void SetWindowTitleAndProgress(const char *chr, double progress) {
			toolStripCurrentProgress->Value = clamp((int)(progress * toolStripCurrentProgress->Maximum + 0.5), toolStripCurrentProgress->Minimum, toolStripCurrentProgress->Maximum);
			toolStripStatusCurrentProgress->Text = (progress).ToString("P1");
			
			if (_start_time) {
				DWORD time_elapsed = timeGetTime() - _start_time;

				time_elapsed /= 1000;
				int t = (int)(time_elapsed / 3600);
				StringBuilder^ SBE = gcnew StringBuilder();
				//SBE->Append(L"経過時間 ");
				SBE->Append(t.ToString("D2"));
				SBE->Append(L":");
				time_elapsed -= t * 3600;
				t = (int)(time_elapsed / 60);
				SBE->Append(t.ToString("D2"));
				SBE->Append(L":");
				time_elapsed -= t * 60;
				SBE->Append(time_elapsed.ToString("D2"));

				toolStripStatusElapsedTime->Text = SBE->ToString();
			} else {
				toolStripStatusElapsedTime->Text = L"";
			}

			this->Text = /*L"[" + toolStripStatusCurrentProgress->Text + L"] " + */String(chr).ToString();
			System::Windows::Forms::Application::DoEvents();
			taskbar_progress->set_progress(progress);
		}
	public:
		System::Void SetWindowTitleX264Mes(const char *chr, int total_drop, int frame_n) {
			String^ title = String(chr).ToString();
			double progress = frame_n / (double)total_frame;
			String^ ProgressPercent = (progress).ToString("P1");
			DWORD time_elapsed = timeGetTime() - _start_time;
			int t;
			if (using_afs) {
				StringBuilder^ SB = gcnew StringBuilder();
				SB->Append(title);
				SB->Append(L", current afs ");
				SB->Append(total_drop);
				SB->Append(L"/");
				SB->Append(frame_n);
				if (add_progress) {
					DWORD time_remain = (DWORD)(time_elapsed * ((double)(total_frame - frame_n) / (double)frame_n)) / 1000;
					SB->Insert(0, L"[" + ProgressPercent + "] ");

					SB->Append(", eta ");
					SB->Append((int)(time_remain / 3600));
					SB->Append(L":");
					SB->Append(((time_remain % 3600) / 60).ToString("D2"));
					SB->Append(L":");
					SB->Append((time_remain % 60).ToString("D2"));
				}
				title = SB->ToString();
			}
			toolStripCurrentProgress->Value = clamp((int)(progress * toolStripCurrentProgress->Maximum + 0.5), toolStripCurrentProgress->Minimum, toolStripCurrentProgress->Maximum);
			toolStripStatusCurrentProgress->Text = ProgressPercent;
			taskbar_progress->set_progress(progress);
			
			time_elapsed /= 1000;
			t = (int)(time_elapsed / 3600);
			StringBuilder^ SBE = gcnew StringBuilder();
			//SBE->Append(L"経過時間 ");
			SBE->Append(t.ToString("D2"));
			SBE->Append(L":");
			time_elapsed -= t * 3600;
			t = (int)(time_elapsed / 60);
			SBE->Append(t.ToString("D2"));
			SBE->Append(L":");
			time_elapsed -= t * 60;
			SBE->Append(time_elapsed.ToString("D2"));

			toolStripStatusElapsedTime->Text = SBE->ToString();

			this->Text = title;
		}
	public:
		System::Int32 GetLogStringLen(int current_pass) {
			if (current_pass == 1) {
				LastLogLen = (closed) ? 0 : this->richTextLog->Text->Length;
				return LastLogLen;
			} else {
				return (closed) ? 0 : this->richTextLog->Text->Length;
			}
		}
	public:
		value struct LogData {
			int type;
			String^ str;
			int log_type_index;
		};
	private:
		Generic::List<LogData> AudioParallelCache;
		//delegate void WriteLogAuoLineDelegate(String^ str, int log_type_index);
	public:
		System::Void WriteLogAuoLine(String^ str, int log_type_index) {
			if (this->InvokeRequired) {
				LogData dat;
				dat.type = 0;
				dat.str = str;
				dat.log_type_index = log_type_index;
				AudioParallelCache.Add(dat);
				//richTextLog->Invoke(gcnew WriteLogAuoLineDelegate(this, &frmLog::WriteLogAuoLine), arg_list);
			} else {
				log_type_index = clamp(log_type_index, LOG_INFO, LOG_ERROR);
				richTextLog->SuspendLayout();
				richTextLog->SelectionStart = richTextLog->Text->Length;
				richTextLog->SelectionLength = richTextLog->Text->Length;
				richTextLog->SelectionColor = log_color_text[log_type_index];
				richTextLog->AppendText(L"auo [" + log_type[log_type_index] + L"]: " + str + L"\n");
				richTextLog->SelectionStart = richTextLog->Text->Length;
				if (!wine_compatible_mode) {
					richTextLog->ScrollToCaret();
				}
				richTextLog->ResumeLayout();
			}
		}
	public:
		System::Void WriteLogLine(String^ str, int log_type_index) {
			if (this->InvokeRequired) {
				LogData dat;
				dat.type = 1;
				dat.str = str;
				dat.log_type_index = log_type_index;
				AudioParallelCache.Add(dat);
				//richTextLog->Invoke(gcnew WriteLogLineDelegate(this, &frmLog::WriteLogLine), arg_list);
			} else {
				log_type_index = clamp(log_type_index, LOG_INFO, LOG_ERROR);
				richTextLog->SuspendLayout();
				richTextLog->SelectionStart = richTextLog->Text->Length;
				richTextLog->SelectionLength = richTextLog->Text->Length;
				richTextLog->SelectionColor = log_color_text[log_type_index];
				richTextLog->AppendText(str + L"\n");
				richTextLog->SelectionStart = richTextLog->Text->Length;
				if (!wine_compatible_mode) {
					richTextLog->ScrollToCaret();
				}
				richTextLog->ResumeLayout();
			}
		}
	public:
		System::Void FlushAudioLogCache() {
			for (int i = 0; i < AudioParallelCache.Count; i++) {
				(AudioParallelCache[i].type) 
					? WriteLogLine(AudioParallelCache[i].str, AudioParallelCache[i].log_type_index)
					: WriteLogAuoLine(AudioParallelCache[i].str, AudioParallelCache[i].log_type_index);
			}
			AudioParallelCache.Clear();
		}
	private:
		System::Void SaveLog(String^ SaveLogName) {
			StreamWriter^ sw;
			try {
				sw = gcnew StreamWriter(SaveLogName, true, System::Text::Encoding::GetEncoding("shift_jis"));
				System::Text::StringBuilder^ sb = gcnew System::Text::StringBuilder(richTextLog->Text->Substring(LastLogLen));
				sb->Replace(L"\n", L"\r\n");//改行コード変換
				sw->WriteLine(sb->ToString());
				sw->WriteLine(DateTime::Now.ToString("yyyy年M月d日 H時mm分 エンコード終了"));
				sw->WriteLine(L"-------------------------------------------------------------------------------------");
				sw->WriteLine();
			} catch (IOException^ ex) {
				WriteLogAuoLine("自動ログ保存に失敗しました。", LOG_WARNING);
				WriteLogAuoLine(ex->Message, LOG_WARNING);
			}  catch (UnauthorizedAccessException^ ex) {
				WriteLogAuoLine("自動ログ保存に失敗しました。", LOG_WARNING);
				WriteLogAuoLine(ex->Message, LOG_WARNING);
			} catch (...) {
				WriteLogAuoLine("自動ログ保存に失敗しました。", LOG_WARNING);
			} finally {
				if (sw != nullptr) {
					sw->Close();
				}
			}
		}
	public:
		System::Void SetPreventLogWindowClosing(BOOL prevent) {
			prevent_log_closing = (prevent != 0);
			if (!prevent_log_closing) {
				SaveLogSettings();
				this->KeyDown += gcnew System::Windows::Forms::KeyEventHandler(this, &frmLog::frmLog_KeyDown);
			} else {
				this->KeyDown -= gcnew System::Windows::Forms::KeyEventHandler(this, &frmLog::frmLog_KeyDown);
			}
		}
	public:
		System::Void AutoSaveLogFile(const char *log_filename) {
			if (toolStripMenuItemAutoSave->Checked && !prevent_log_closing && log_filename != NULL)
				SaveLog(String(log_filename).ToString());
		}
	public:
		System::Void EnableEncControl(BOOL *enc_pause, BOOL afs, BOOL _add_progress, DWORD start_time, int _total_frame) {
			_enc_pause = enc_pause;
			add_progress = _add_progress != 0;
			using_afs = afs != 0;
			_start_time = start_time;
			total_frame = _total_frame;

			if (_enc_pause) {
				this->ToolStripMenuItemEncPause->Checked = *_enc_pause != 0;
				this->ToolStripMenuItemEncPause->Enabled = true;
			}
		}
	public:
		System::Void DisableEncControl() {
			this->ToolStripMenuItemEncPause->Enabled = false;
			_start_time = 0;
			_enc_pause = NULL;
		}
	private:
		System::Void frmLog_FormClosing(System::Object^  sender, System::Windows::Forms::FormClosingEventArgs^  e) {
			delete timerResizeOrPos;
			timerResizeOrPos = nullptr;
			if (prevent_log_closing && e->CloseReason == System::Windows::Forms::CloseReason::UserClosing) {
				e->Cancel = true;
				this->WindowState = FormWindowState::Minimized;
			} else
				closed = true;
		}
	private:
		System::Void SaveLogSettings() {
			guiEx_settings exstg(true);
			exstg.load_log_win();
			exstg.s_log.transparent      = ToolStripMenuItemTransparent->Checked;
			exstg.s_log.transparency     = frmTransparency;
			exstg.s_log.minimized        = ToolStripMenuItemStartMinimized->Checked;
			exstg.s_log.auto_save_log    = toolStripMenuItemAutoSave->Checked;
			exstg.s_log.show_status_bar  = toolStripMenuItemShowStatus->Checked;
			exstg.s_log.taskbar_progress = toolStripMenuItemTaskBarProgress->Checked;
			exstg.s_log.save_log_size    = toolStripMenuItemSaveLogSize->Checked;
			exstg.s_log.log_pos[0]       = this->Location.X;
			exstg.s_log.log_pos[1]       = this->Location.Y;
			//最大化・最小化中なら保存しない
			if (this->WindowState == FormWindowState::Normal) {
				if (exstg.s_log.save_log_size) {
					exstg.s_log.log_width    = this->ClientSize.Width;
					exstg.s_log.log_height   = this->ClientSize.Height;
				} else {
					//デフォルト
					exstg.s_log.log_width    = 0;
					exstg.s_log.log_height   = 0;
				}
			}
			exstg.save_log_win();
		}
	private:
		System::Void ToolStripCheckItem_CheckedChanged(System::Object^  sender, System::EventArgs^  e) {
			if (!prevent_log_closing)
				SaveLogSettings();
		}
	private: 
		System::Void ToolStripMenuItemTransparent_CheckedChanged(System::Object^  sender, System::EventArgs^  e) {
			this->Opacity = ((ToolStripMenuItemTransparent->Checked) ? 100 - frmTransparency : 100) * 0.01f;
			toolStripMenuItemTransparentValue->Enabled = ToolStripMenuItemTransparent->Checked;
			ToolStripCheckItem_CheckedChanged(sender, e);
		}
	private: 
		System::Void ToolStripMenuItemEncPause_CheckedChanged(System::Object^  sender, System::EventArgs^  e) {
			if (_enc_pause) {
				*_enc_pause = Convert::ToInt32(ToolStripMenuItemEncPause->Checked);
				if (nullptr != taskbar_progress) {
					(*_enc_pause) ? taskbar_progress->pause() : taskbar_progress->restart();
				}
				//if (*_enc_pause) {
				//	pause_start = timeGetTime(); //一時停止を開始した時間
				//} else {
				//	if (pause_start)
				//		*_x264_start_time += timeGetTime() - pause_start; //開始時間を修正し、一時停止後も正しい時間情報を維持
				//	pause_start = NULL;
				//}
			}
		}
	private: 
		System::Void toolStripMenuItemShowStatus_CheckedChanged(System::Object^  sender, System::EventArgs^  e) {
			statusStripLog->Visible = toolStripMenuItemShowStatus->Checked;
			if (statusStripLog->Visible)
				richTextLog->Height -= statusStripLog->Height;
			else
				richTextLog->Height += statusStripLog->Height;
			ToolStripCheckItem_CheckedChanged(sender, e);
		}
	private: 
		System::Void toolStripMenuItemTaskBarProgress_CheckedChanged(System::Object^  sender, System::EventArgs^  e) {
			if (nullptr != taskbar_progress) {
				taskbar_progress->set_visible(toolStripMenuItemTaskBarProgress->Checked);
			}
			ToolStripCheckItem_CheckedChanged(sender, e);
		 }
	private:
		System::Void timerResizeOrPosChange(Object^ state) {
			this->Invoke(gcnew timerResizeOrPosChangeDelegate(this, &frmLog::timerSaveSettings));
		}
	private:
		System::Void timerSaveSettings() {
			timerResizeOrPos->Change(System::Threading::Timeout::Infinite, timerResizeOrPosPeriod);
			SaveLogSettings();
		}
	private:
		System::Void frmLog_ClientSizeChanged(System::Object^  sender, System::EventArgs^  e) {
			//通常->通常でのサイズ変更以外、保存しないようにする
			//最小化/最大化->通常とか通常->最小化/最大化には興味がない
			if (this->WindowState == FormWindowState::Normal &&
				  lastWindowState == FormWindowState::Normal &&
				  timerResizeOrPos != nullptr) {
				timerResizeOrPos->Change(0, timerResizeOrPosPeriod);
			}
			lastWindowState = this->WindowState;
		}
	private:
		System::Void frmLog_LocationChanged(System::Object^  sender, System::EventArgs^  e) {
			//通常のウィンドウモード以外は気にしない
			if (this->WindowState == FormWindowState::Normal && timerResizeOrPos != nullptr) {
				timerResizeOrPos->Change(0, timerResizeOrPosPeriod);
			}
		}
	private:
		System::Void toolStripMenuItemSaveLogSize_CheckedChanged(System::Object^  sender, System::EventArgs^  e) {
			ToolStripCheckItem_CheckedChanged(sender, e);
		}
	private: 
		System::Void toolStripMenuItemAutoSaveSettings_Click(System::Object^  sender, System::EventArgs^  e) {
			frmAutoSaveLogSettings::Instance::get()->Owner = this;
			frmAutoSaveLogSettings::Instance::get()->Show();
		}
	private:
		System::Void frmLog_KeyDown(System::Object^  sender, System::Windows::Forms::KeyEventArgs^  e) {
			if (e->KeyCode == Keys::Escape)
				this->Close();
		}
	public:
		System::Void RewriteLogTextWithNewSettings(System::Drawing::Font^ newFont, cli::array<Color>^ newColorArray) {
			//フォントを変更するだけでは何故かうまく行かないので、
			//一度文字列を保存→テキストボックスクリア→フォント変更→文字列を一行ずつ追加
			//という力技を行う
			::SendMessage(hWnd, WM_SETREDRAW, false, 0); //描画を停止
			array<int>^ LogLineColorIndex = gcnew array<int>(richTextLog->Lines->Length); //各行の色のインデックス
			for (int i = 0, position = 0; i < richTextLog->Lines->Length; i++) {
				LogLineColorIndex[i] = 0;
				if (reinterpret_cast<String^>(richTextLog->Lines[i])->Length) {
					richTextLog->Select(position, 1);
					for (int i_col_idx = 0; i_col_idx < 3; i_col_idx++) {
						if (richTextLog->SelectionColor.Equals(log_color_text[i_col_idx])) {
							LogLineColorIndex[i] = i_col_idx;
							break;
						}
					}
				}
				position += reinterpret_cast<String^>(richTextLog->Lines[i])->Length + 1; //改行コード分追加
			}
			array<String^>^ LogLines = richTextLog->Lines; //各行の文字列
			//テキストボックスをクリア
			richTextLog->Clear();
			richTextLog->Text = String::Empty;
			//ここでフォントを変更
			if (newFont != nullptr)
				richTextLog->Font = newFont;
			//ここで色を変更
			if (newColorArray != nullptr)
				log_color_text = newColorArray;
			//文字を再登録
			for (int i = 0; i < LogLines->Length - 1; i++) {
				richTextLog->SelectionStart = richTextLog->Text->Length;
				richTextLog->SelectionLength = richTextLog->Text->Length;
				richTextLog->SelectionColor = log_color_text[LogLineColorIndex[i]];
				richTextLog->AppendText(LogLines[i] + L"\n");
			}
			richTextLog->SelectionStart = richTextLog->Text->Length;
			richTextLog->ScrollToCaret();
			::SendMessage(hWnd, WM_SETREDRAW, true, 0); //描画再開
			this->Refresh(); //強制再描画
		}
	private: 
		System::Void toolStripMenuItemWindowFont_Click(System::Object^  sender, System::EventArgs^  e) {
			System::Drawing::Font^ LastFont = richTextLog->Font;
			fontDialogLog->Font = richTextLog->Font;
			if (fontDialogLog->ShowDialog() != System::Windows::Forms::DialogResult::Cancel && LastFont != fontDialogLog->Font) {
				RewriteLogTextWithNewSettings(fontDialogLog->Font, nullptr);
				//設定を保存
				guiEx_settings exstg(true);
				exstg.load_log_win();
				Set_AUO_FONT_INFO(&exstg.s_log.log_font, fontDialogLog->Font, LastFont);
				exstg.save_log_win();
			}
		}
	private:
		System::Void toolStripMenuItemTransparentValue_Click(System::Object^  sender, System::EventArgs^  e) {
			ToolStripMenuItemTransparent->Enabled = false;
			frmSetTransparency::Instance::get()->Owner = this;
			frmSetTransparency::Instance::get()->Show();
		}
	public:
		System::Void EnableToolStripMenuItemTransparent() {
			ToolStripMenuItemTransparent->Enabled = true;
		}
	private:
		System::Void toolStripMenuItemSetLogColor_Click(System::Object^  sender, System::EventArgs^  e) {
			frmSetLogColor::Instance::get()->Owner = this;
			frmSetLogColor::Instance::get()->colorBackGround = this->richTextLog->BackColor;
			frmSetLogColor::Instance::get()->colorInfo       = log_color_text[0];
			frmSetLogColor::Instance::get()->colorWarning    = log_color_text[1];
			frmSetLogColor::Instance::get()->colorError      = log_color_text[2];
			//frmSetLogColor::Instance::get()->SetOpacity(this->Opacity);
			frmSetLogColor::Instance::get()->Show();
		}
	public:
		System::Void SetNewLogColor() {
			cli::array<Color>^ newColor = gcnew array<Color>(3) {
				frmSetLogColor::Instance::get()->colorInfo,
				frmSetLogColor::Instance::get()->colorWarning,
				frmSetLogColor::Instance::get()->colorError
			};
			this->richTextLog->BackColor = frmSetLogColor::Instance::get()->colorBackGround;
			RewriteLogTextWithNewSettings(nullptr, newColor);
			//設定を保存
			guiEx_settings exstg(true);
			exstg.load_log_win();
			ColortoInt(exstg.s_log.log_color_background, this->richTextLog->BackColor);
			for (int i = 0; i < 3; i++)
				ColortoInt(exstg.s_log.log_color_text[i], log_color_text[i]);
			exstg.save_log_win();
		}
	////  ファイル名をクリックして動画を再生・フォルダを開く /////////
	private:
		String^ selectedPathbyMouse;
	private:
		System::Void richTextLog_MouseDown(System::Object^  sender, System::Windows::Forms::MouseEventArgs^  e) {
			const int index = richTextLog->GetCharIndexFromPosition(e->Location);
			const int i_line = richTextLog->GetLineFromCharIndex(index);
			//前後1行も検索
			bool PathSelected = false;
			for (int j = -1; j <= 1; j++) {
				if (0 <= i_line + j && i_line + j < richTextLog->Lines->Length) {
					String^ strLine = reinterpret_cast<String^>(richTextLog->Lines[i_line + j]);
					const int startPos = strLine->IndexOf(L'[');
					const int finPos = strLine->LastIndexOf(L']');
					if (startPos >= 0 && finPos > startPos) {
						strLine = strLine->Substring(startPos + 1, finPos - startPos - 1);
						if (File::Exists(strLine)) {
							PathSelected = true;
							selectedPathbyMouse = strLine;
							break;
						}
					}
				}
			}

			bool FileEncondingFinished = (index < LastLogLen || !prevent_log_closing);
			toolStripMenuItemFileOpen->Enabled = PathSelected && FileEncondingFinished;
			toolStripMenuItemFilePathOpen->Enabled = PathSelected;
		}
	private:
		System::Void toolStripMenuItemFileOpen_Click(System::Object^  sender, System::EventArgs^  e) {
			if (File::Exists(selectedPathbyMouse)) {
				try {
					System::Diagnostics::Process::Start(selectedPathbyMouse);
				} catch (...) {
					MessageBox::Show(L"ファイルオープンでエラーが発生しました。", AUO_NAME_W, MessageBoxButtons::OK, MessageBoxIcon::Exclamation);
				}
			}
		}
	private:
		System::Void toolStripMenuItemFilePathOpen_Click(System::Object^  sender, System::EventArgs^  e) {
			if (File::Exists(selectedPathbyMouse)) {
				try {
					System::Diagnostics::Process::Start(L"explorer.exe", L"/select," + selectedPathbyMouse);
				} catch (...) {
					MessageBox::Show(L"ファイルの表示でエラーが発生しました。", AUO_NAME_W, MessageBoxButtons::OK, MessageBoxIcon::Exclamation);
				}
			}
		}
	////  ファイル名をクリックして動画を再生・フォルダを開く /////////
};
}

#pragma warning( pop ) //( disable: 4100 ) 終了
