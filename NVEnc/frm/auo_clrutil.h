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

#ifndef _AUO_CLRUTIL_H_
#define _AUO_CLRUTIL_H_

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>
#include <vcclr.h>
#include "auo.h"
#include "auo_util.h"
#include "ComboBoxFolderBrowser.h"

using namespace System;
using namespace System::IO;
using namespace System::Drawing;
using namespace System::Windows::Forms;

static size_t GetCHARfromString(char *chr, DWORD nSize, System::String^ str) {
    DWORD str_len;
    System::IntPtr ptr = System::Runtime::InteropServices::Marshal::StringToHGlobalAnsi(str);
    char *ch_ptr = (char *)ptr.ToPointer();
    if ((str_len = (DWORD)strlen(ch_ptr)) >= nSize)
        return 0; //バッファサイズを超えていたら何もしない
    memcpy(chr, ch_ptr, str_len+1);
    System::Runtime::InteropServices::Marshal::FreeHGlobal(ptr);
    return str_len;
}

template <size_t size>
int GetCHARfromString(char(&chr)[size], System::String^ str) {
    return GetCHARfromString(chr, size, str);
}

static std::string GetCHARfromString(System::String ^str) {
    System::IntPtr ptr = System::Runtime::InteropServices::Marshal::StringToHGlobalAnsi(str);
    char *ch_ptr = (char *)ptr.ToPointer();
    std::string result = ch_ptr;
    System::Runtime::InteropServices::Marshal::FreeHGlobal(ptr);
    return result;
}

static int CountStringBytes(System::String^ str) {
    return System::Text::Encoding::GetEncoding(L"Shift_JIS")->GetByteCount(str);
}

static String^ MakeExeFilter(System::String^ fileExeName) {
    array<String^>^ fileNames = fileExeName->Split(L';');
    String^ fileName;
    String^ filter;
    for (int i = 0; i < fileNames->Length; i++) {
        if (i) {
            fileName += L"/";
            filter += L";";
        }
        fileName += System::IO::Path::GetFileNameWithoutExtension(fileNames[i]);
        filter += System::IO::Path::GetFileNameWithoutExtension(fileNames[i]) + L"*" + System::IO::Path::GetExtension(fileNames[i]);
    }
    return fileName + L" (" + filter + L")|" + filter;
}

//使用できないファイル名かどうか確認する
static bool ValidiateFileName(System::String^ fileName) {
    array<wchar_t>^ InvalidChars = {L'\\', L'/', L':', L'*', L'?', L'\"', L'<', L'>', L'|'};
    array<String^>^ InvalidString = { L"CON", L"PRN", L"AUX", L"CLOCK$", L"NUL",
    L"COM0", L"COM1", L"COM2", L"COM3", L"COM4", L"COM5", L"COM6", L"COM7", L"COM8", L"COM9",
    L"LPT0", L"LPT1", L"LPT2", L"LPT3", L"LPT4", L"LPT5", L"LPT6", L"LPT7", L"LPT8", L"LPT9" };
    if (fileName->IndexOfAny(InvalidChars) >= 0)
        return false;
    for (int i = 0; i < InvalidString->Length; i++)
        if (String::Compare(fileName, InvalidString[i], true) == NULL)
            return false;
    return true;
}

//AUO_FONT_INFOからフォントを作成する
//情報がない場合、baseFontのものを使用する
static System::Drawing::Font^ GetFontFrom_AUO_FONT_INFO(const AUO_FONT_INFO *info, System::Drawing::Font^ baseFont) {
    if (info && (str_has_char(info->name) || info->size > 0.0)) {
        return gcnew System::Drawing::Font(
            (str_has_char(info->name)) ? String(info->name).ToString() : baseFont->FontFamily->ToString(),
            (info->size > 0.0) ? (float)info->size : baseFont->Size, 
            (System::Drawing::FontStyle)info->style);
    }
    return baseFont;
}

//DefaultFontと比較して、異なっていたらAUO_FONT_INFOに保存する
static void Set_AUO_FONT_INFO(AUO_FONT_INFO *info, System::Drawing::Font^ Font, System::Drawing::Font^ DefaultFont) {                
    if (String::Compare(DefaultFont->FontFamily->Name, Font->FontFamily->Name))
        GetCHARfromString(info->name, sizeof(info->name), Font->FontFamily->Name);
    if (DefaultFont->Size != Font->Size)
        info->size = Font->Size;
    info->style = (int)Font->Style;
}

//ToolStripへのフォントの適用
static void SetFontFamilyToToolStrip(ToolStrip^ TS, FontFamily^ NewFontFamily, FontFamily^ BaseFontFamily) {
    for (int i = 0; i < TS->Items->Count; i++) {
        TS->Items[i]->Font = gcnew Font(NewFontFamily, TS->Items[i]->Font->Size, TS->Items[i]->Font->Style);
    }
}

//再帰を用いて全コントロールにフォントを適用する
static void SetFontFamilyToControl(Control^ top, FontFamily^ NewFontFamily, FontFamily^ BaseFontFamily) {
    for (int i = 0; i < top->Controls->Count; i++) {
        System::Type^ type = top->Controls[i]->GetType();
        if (type == ToolStrip::typeid)
            SetFontFamilyToToolStrip((ToolStrip^)top->Controls[i], NewFontFamily, BaseFontFamily);
        else
            SetFontFamilyToControl(top->Controls[i], NewFontFamily, BaseFontFamily);
    }
    top->Font = gcnew Font(NewFontFamily, top->Font->Size, top->Font->Style);
}

//フォントが存在するかチェックする
static bool CheckFontFamilyExists(FontFamily^ targetFontFamily) {
    //新しくフォントを作ってみて、設定しようとしたフォントと違ったら諦める
    Font^ NewFont = gcnew Font(targetFontFamily, 12); //12は適当
    return (String::Compare(NewFont->FontFamily->Name, targetFontFamily->Name)) ? false : true;
}

//フォーム全体にフォントを適用する
static void SetFontFamilyToForm(Form^ form, FontFamily^ NewFontFamily, FontFamily^ BaseFontFamily) {
    if (!CheckFontFamilyExists(NewFontFamily))
        return;
    ::SendMessage((HWND)form->Handle.ToPointer(), WM_SETREDRAW, false, 0); //描画を停止
    SetFontFamilyToControl(form, NewFontFamily, BaseFontFamily);
    ::SendMessage((HWND)form->Handle.ToPointer(), WM_SETREDRAW, true, 0); //描画再開
    form->Refresh(); //強制再描画
}

static String^ AddBackSlash(String^ Dir) {
    if (Dir != nullptr && 
        Dir->Length > 0 && 
        L'\\' != Dir[Dir->Length - 1])
        Dir += L"\\";
    return Dir;
}

//相対パスに変換して返す
static String^ GetRelativePath(String^ path, String^ baseDir) { 
    //相対パスならそのまま返す
    if (path == nullptr || !Path::IsPathRooted(path))
        return path;
    //nullptrならCurrentDirを取得
    if (baseDir == nullptr)
        baseDir = Directory::GetCurrentDirectory();
    baseDir = AddBackSlash(baseDir);
    //同じルートでなければそのまま返す
    if (String::Compare(Directory::GetDirectoryRoot(path), Directory::GetDirectoryRoot(baseDir), true))
        return path;
    //相対パスへの変換
    pin_ptr<const WCHAR> base_dir = PtrToStringChars(baseDir);
    pin_ptr<const WCHAR> target_path = PtrToStringChars(path);
    WCHAR buf[MAX_PATH_LEN];
    GetRelativePathTo(buf, _countof(buf), target_path, NULL, base_dir);
    return String(buf).ToString();
}

//CurentDirectoryをベースフォルダとして相対パスに変換して返す
static String^ GetRelativePath(String^ path) {
    return GetRelativePath(path, Directory::GetCurrentDirectory());
}

static System::Void ColortoInt(int *color_dst, Color color_src) {
    color_dst[0] = color_src.R;
    color_dst[1] = color_src.G;
    color_dst[2] = color_src.B;
}

static Color ColorfromInt(const int *color_src) {
    return Color::FromArgb(color_src[0], color_src[1], color_src[2]);
}

static Color ColorfromInt(const ColorRGB c) {
    return Color::FromArgb(c.r, c.g, c.b);
}

static inline String^ Utf8toString(const char *str) {
    int length = strlen(str);
    if (CODE_PAGE_UTF8 != jpn_check(str, length)) {
        return String(str).ToString();
    }
    array<Byte>^ a = gcnew array<Byte>(length);
    for (int i = 0; i < length; i++)
        a[i] = str[i];
    return System::Text::Encoding::UTF8->GetString(a);
}

static System::Drawing::Color getTextBoxForeColor(const AuoTheme theme, const DarkenWindowStgReader *stgReader, const DarkenWindowState state) {
    System::Drawing::Color foreColor = (theme == AuoTheme::DarkenWindowDark) ? ColorfromInt(DEFAULT_UI_COLOR_TEXT_DARK) : System::Windows::Forms::Control::DefaultForeColor;
    if (state == DarkenWindowState::Disabled) {
        foreColor = (theme == AuoTheme::DarkenWindowDark) ? ColorfromInt(DEFAULT_UI_COLOR_TEXT_DARK_DISABLED) : System::Drawing::SystemColors::ControlDark;
    }
    if (theme != AuoTheme::DefaultLight && stgReader) {
        const DarkenWindowStgNamedColor *dwcolor = stgReader->getColorTextBox(state);
        if (dwcolor) {
            foreColor = ColorfromInt(dwcolor->textForeColor());
        }
    }
    return foreColor;
}

//tabcontrolのborderを隠す
//tabControlをPanelの(2,2)に配置
//panelのサイズをtabcontrolの width+4, height+4にする
//tabControlのanchorはLeft,Upのみにする
static System::Void SwitchComboBoxBorder(TabControl^ TB, Panel^ PN, const AuoTheme themeFrom, const AuoTheme themeTo, const DarkenWindowStgReader *stgReader) {
    if (themeTo == themeFrom) return;
    switch (themeFrom) {
    case AuoTheme::DarkenWindowDark:
        break;
    case AuoTheme::DefaultLight:
    case AuoTheme::DarkenWindowLight:
    default:
        // Light→Lightのときは調整不要
        if (themeTo == AuoTheme::DefaultLight || themeTo == AuoTheme::DarkenWindowLight) return;
        break;

    }
    const int offsetSizeX = (themeTo == AuoTheme::DarkenWindowDark) ? -12 : 12;
    const int offsetSizeY = (themeTo == AuoTheme::DarkenWindowDark) ? -6 : 6;
    //パネルの上に(2,2)の位置にtabControlが載っている
    //borderを隠す時は、パネルを縮小し、
    //tabControlをパネルの範囲外に移すことで、borderを見えないように調整する
    Point loc = PN->Location;
    loc.X -= offsetSizeX / 2;
    //loc.Y -= offsetSize / 2;
    PN->Location = loc;
    System::Drawing::Size size = PN->Size;
    size.Width += offsetSizeX;
    size.Height += offsetSizeY;
    PN->Size = size;

    loc = TB->Location;
    loc.X += offsetSizeX / 2;
    TB->Location = loc;
}

static bool isToolStripItem(System::Type^ type) {
    return type == ToolStrip::typeid
        || type == ToolStripButton::typeid
        || type == ToolStripComboBox::typeid
        || type == ToolStripContainer::typeid
        || type == ToolStripContentPanel::typeid
        || type == ToolStripDropDown::typeid
        || type == ToolStripDropDownButton::typeid
        || type == ToolStripDropDownItem::typeid
        || type == ToolStripDropDownMenu::typeid
        || type == ToolStripMenuItem::typeid
        || type == ToolStripLabel::typeid
        || type == ToolStripPanel::typeid
        || type == ToolStripSeparator::typeid
        || type == ToolStripSplitButton::typeid
        || type == ToolStripStatusLabel::typeid
        || type == ToolStripTextBox::typeid;
}

static System::Void SetAllColor(Control ^top, const AuoTheme themeTo, System::Type^ topType, const DarkenWindowStgReader *stgReader) {
    System::Type^ type = top->GetType();
    if (type == GroupBox::typeid) {
        GroupBox^ GB = dynamic_cast<GroupBox^>(top);
        GB->FlatStyle = (themeTo == AuoTheme::DefaultLight) ? System::Windows::Forms::FlatStyle::Standard : System::Windows::Forms::FlatStyle::System;
    } else if (type == TextBox::typeid) {
        TextBox^ TX = dynamic_cast<TextBox^>(top);
        TX->BorderStyle = (themeTo == AuoTheme::DefaultLight) ? System::Windows::Forms::BorderStyle::Fixed3D : System::Windows::Forms::BorderStyle::FixedSingle;
    } else if (type == ToolStrip::typeid) {
        ToolStrip^ TS = dynamic_cast<ToolStrip^>(top);
        TS->RenderMode = (themeTo == AuoTheme::DefaultLight) ? System::Windows::Forms::ToolStripRenderMode::ManagerRenderMode : System::Windows::Forms::ToolStripRenderMode::System;
    } else if (type == TabPage::typeid) {
        TabPage^ TC = dynamic_cast<TabPage^>(top);
        TC->BorderStyle = (themeTo == AuoTheme::DefaultLight) ? System::Windows::Forms::BorderStyle::None : System::Windows::Forms::BorderStyle::FixedSingle;
        TC->UseVisualStyleBackColor = (themeTo == AuoTheme::DefaultLight) ? true : false;
    } else if (type == Label::typeid) {
        //Label^ LB = dynamic_cast<Label^>(top->Controls[i]);
        //if (themeTo == AuoTheme::LightDefault) BT->FlatStyle = System::Windows::Forms::FlatStyle::Standard;
    } else if (type == TabControl::typeid) {
        //TabControl^ TC = dynamic_cast<TabControl^>(top->Controls[i]);
        //TabControlをオーナードローする -> タブ内はよくなってもタブ外がうまくいかない
        //TC->DrawMode = TabDrawMode::OwnerDrawFixed;
        //DrawItemイベントハンドラを追加
        //TC->DrawItem += gcnew System::Windows::Forms::DrawItemEventHandler(this, &frmConfig::TabControl_DarkDrawItem);
        //TC->Appearance = System::Windows::Forms::TabAppearance::FlatButtons;
    }
    //色の変更
    if (themeTo != AuoTheme::DefaultLight) {
        //DarkenWindow使用時
        //まず、値が取れなかった時に備えて、デフォルトの値を入れる
        System::Drawing::Color foreColor = (themeTo == AuoTheme::DarkenWindowDark) ? ColorfromInt(DEFAULT_UI_COLOR_TEXT_DARK) : System::Windows::Forms::Control::DefaultForeColor;
        System::Drawing::Color backColor = (themeTo == AuoTheme::DarkenWindowDark) ? ColorfromInt(DEFAULT_UI_COLOR_BASE_DARK) : System::Windows::Forms::Control::DefaultBackColor;
        //設定ファイルから値が取得できていれば、それを使用する
        bool setForeColor = true;
        bool setBackColor = true;
        const DarkenWindowStgNamedColor *dwcolor = nullptr;
        if (type == TextBox::typeid
            || type == ComboBox::typeid
            || type == NVEnc::ComboBoxFolderBrowser::typeid
            || type == NumericUpDown::typeid) {
            dwcolor = stgReader->getColorTextBox();
        } else if (type == Button::typeid) {
            dwcolor = stgReader->getColorButton();
            setBackColor = false;
        } else if (type == CheckBox::typeid) {
            dwcolor = stgReader->getColorCheckBox();
            setBackColor = false;
        } else if (type == ScrollBar::typeid
            ) {
            setForeColor = setBackColor = false;
        } else if (type == topType
            || type == ToolStrip::typeid
            || type == Label::typeid
            || type == TabControl::typeid
            || type == TabPage::typeid) {
            dwcolor = stgReader->getColorStatic();
        } else {
            setForeColor = setBackColor = false;
        }
        if (setForeColor || setBackColor) {
            if (dwcolor) {
                foreColor = ColorfromInt(dwcolor->textForeColor());
                backColor = ColorfromInt(dwcolor->fillColor());
            }
            if (setForeColor) {
                top->ForeColor = foreColor;
            }
            if (setBackColor && top->BackColor != System::Drawing::Color::Transparent) {
                top->BackColor = backColor;
            }
        }
    }
    for (int i = 0; i < top->Controls->Count; i++) {
        SetAllColor(top->Controls[i], themeTo, topType, stgReader);
    }
    //色を変更してから必要な処理
    if (type == Button::typeid) {
        Button^ BT = dynamic_cast<Button^>(top);
        // BackColor を変更すると自動的にoffになってしまうのを元に戻す
        if (themeTo == AuoTheme::DefaultLight) BT->UseVisualStyleBackColor = true;
    } else if (type == TabPage::typeid) {
        TabPage^ TC = dynamic_cast<TabPage^>(top);
        TC->UseVisualStyleBackColor = (themeTo == AuoTheme::DefaultLight) ? true : false;
    }
}

static System::Void fcgSetDataGridViewCellStyleHeader(DataGridViewCellStyle^ cellStyle, const AuoTheme themeMode, const DarkenWindowStgReader *stgReader) {
    if (stgReader == nullptr) return;
    System::Drawing::Color foreColor = (themeMode == AuoTheme::DarkenWindowDark) ? ColorfromInt(DEFAULT_UI_COLOR_TEXT_DARK) : System::Windows::Forms::Control::DefaultForeColor;
    System::Drawing::Color backColor = (themeMode == AuoTheme::DarkenWindowDark) ? ColorfromInt(DEFAULT_UI_COLOR_BASE_DARK) : System::Windows::Forms::Control::DefaultBackColor;
    bool setForeColor = false;
    bool setBackColor = false;
    const DarkenWindowStgNamedColor *dwcolor = stgReader->getColorStatic();
    if (dwcolor) {
        foreColor = ColorfromInt(dwcolor->textForeColor());
        backColor = ColorfromInt(dwcolor->fillColor());
        setForeColor = true;
        setBackColor = true;
    }
    if (setForeColor) {
        cellStyle->ForeColor = foreColor;
    }
    if (setBackColor) {
        cellStyle->BackColor = backColor;
    }
}

static System::Void fcgSetDataGridViewCellStyleHeader(DataGridView^ DG, const AuoTheme themeMode, const DarkenWindowStgReader *stgReader) {
    if (stgReader == nullptr) return;
    const DarkenWindowStgNamedColor *dwcolor = stgReader->getColorStatic();
    DG->EnableHeadersVisualStyles = dwcolor == nullptr;
    fcgSetDataGridViewCellStyleHeader(DG->ColumnHeadersDefaultCellStyle, themeMode, stgReader);
    fcgSetDataGridViewCellStyleHeader(DG->RowHeadersDefaultCellStyle, themeMode, stgReader);
}

static System::Void fcgMouseEnterLeave_SetColor(System::Object^ sender,  const AuoTheme themeMode, const DarkenWindowState state, const DarkenWindowStgReader *stgReader) {
    if (stgReader == nullptr) return;
    System::Type^ type = sender->GetType();
    bool setForeColor = false;
    bool setBackColor = false;
    System::Drawing::Color foreColor = (themeMode == AuoTheme::DarkenWindowDark) ? ColorfromInt(DEFAULT_UI_COLOR_TEXT_DARK) : System::Windows::Forms::Control::DefaultForeColor;
    System::Drawing::Color backColor = (themeMode == AuoTheme::DarkenWindowDark) ? ColorfromInt(DEFAULT_UI_COLOR_BASE_DARK) : System::Windows::Forms::Control::DefaultBackColor;
    if (type == CheckBox::typeid) {
        const DarkenWindowStgNamedColor *dwcolor = stgReader->getColorCheckBox(state);
        if (dwcolor) {
            foreColor = ColorfromInt(dwcolor->textForeColor());
            setForeColor = true;
        }
        Control^ control = dynamic_cast<Control^>(sender);
        if (setForeColor) {
            control->ForeColor = foreColor;
        }
        if (setBackColor) {
            control->BackColor = backColor;
        }
    } else if (isToolStripItem(type)) {
        const DarkenWindowStgNamedColor *dwcolor = stgReader->getColorStatic(state);
        if (dwcolor) {
            foreColor = ColorfromInt(dwcolor->textForeColor());
            backColor = ColorfromInt(dwcolor->fillColor());
            setForeColor = true;
            setBackColor = true;
        }
        ToolStripItem^ item = dynamic_cast<ToolStripItem^>(sender);
        if (setForeColor) {
            item->ForeColor = foreColor;
        }
        if (setBackColor) {
            item->BackColor = backColor;
        }
    }
}

//どうもうまくいかない
static System::Void SetToolTipColor(ToolTip ^TT, const AuoTheme themeTo, const DarkenWindowStgReader *stgReader) {
    TT->IsBalloon = (themeTo == AuoTheme::DefaultLight);

    System::Drawing::Color foreColor = (themeTo == AuoTheme::DarkenWindowDark) ? ColorfromInt(DEFAULT_UI_COLOR_TEXT_DARK) : System::Windows::Forms::Control::DefaultForeColor;
    System::Drawing::Color backColor = (themeTo == AuoTheme::DarkenWindowDark) ? ColorfromInt(DEFAULT_UI_COLOR_BASE_DARK) : System::Windows::Forms::Control::DefaultBackColor;
    if (stgReader) {
        const DarkenWindowStgNamedColor *dwcolor = stgReader->getColorToolTip(DarkenWindowState::Normal);
        if (dwcolor) {
            foreColor = ColorfromInt(dwcolor->textForeColor());
        }
    }
    TT->ForeColor = foreColor;
}

#endif //_AUO_CLRUTIL_H_
