// -----------------------------------------------------------------------------------------
// QSVEnc/NVEnc by rigaya
// -----------------------------------------------------------------------------------------
// The MIT License
//
// Copyright (c) 2011-2016 rigaya
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
#ifndef __RGY_FILESYSTEM_H__
#define __RGY_FILESYSTEM_H__

#include <vector>
#include "rgy_tchar.h"

#if defined(_WIN32) || defined(_WIN64)
std::wstring GetFullPathFrom(const wchar_t *path, const wchar_t *baseDir = nullptr);
std::wstring GetRelativePathFrom(const wchar_t *path, const wchar_t *baseDir = nullptr);
bool rgy_get_filesize(const wchar_t *filepath, uint64_t *filesize);
std::pair<int, std::wstring> PathRemoveFileSpecFixed(const std::wstring& path);
std::wstring PathRemoveExtensionS(const std::wstring& path);
std::wstring PathCombineS(const std::wstring& dir, const std::wstring& filename);
std::string PathCombineS(const std::string& dir, const std::string& filename);
bool CreateDirectoryRecursive(const wchar_t *dir);
std::vector<tstring> get_file_list(const tstring& pattern, const tstring& dir);
#endif //#if defined(_WIN32) || defined(_WIN64)
tstring getExePath();
tstring getExeDir();

std::string GetFullPathFrom(const char *path, const char *baseDir = nullptr);
std::string GetRelativePathFrom(const char *path, const char *baseDir = nullptr);
bool rgy_file_exists(const std::string& filepath);
bool rgy_file_exists(const std::wstring& filepath);
bool rgy_get_filesize(const char *filepath, uint64_t *filesize);
std::pair<int, std::string> PathRemoveFileSpecFixed(const std::string& path);
std::string PathRemoveExtensionS(const std::string& path);
bool CreateDirectoryRecursive(const char *dir);

bool check_ext(const TCHAR *filename, const std::vector<const char*>& ext_list);
bool check_ext(const tstring& filename, const std::vector<const char*>& ext_list);

//拡張子が一致するか確認する
bool _tcheck_ext(const TCHAR *filename, const TCHAR *ext);

bool rgy_path_is_same(const TCHAR *path1, const TCHAR *path2);
bool rgy_path_is_same(const tstring& path1, const tstring& path2);

#if defined(_WIN32) || defined(_WIN64)
std::vector<std::basic_string<TCHAR>> createProcessOpenedFileList(const std::vector<size_t>& list_pid);
#endif //#if defined(_WIN32) || defined(_WIN64)

#endif //__RGY_FILESYSTEM_H__
