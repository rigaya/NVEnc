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
#include <cstdint>
#include "rgy_tchar.h"

#if defined(_WIN32) || defined(_WIN64)
bool rgy_get_filesize(const wchar_t *filepath, uint64_t *filesize);
std::wstring PathRemoveExtensionS(const std::wstring& path);
std::wstring PathGetFilename(const std::wstring& path);
std::vector<tstring> get_file_list(const tstring& pattern, const tstring& dir);
void rgy_file_remove(const wchar_t *path);
#endif //#if defined(_WIN32) || defined(_WIN64)
std::wstring PathCombineS(const std::wstring& dir, const std::wstring& filename);
std::string PathCombineS(const std::string& dir, const std::string& filename);
std::pair<int, std::wstring> PathRemoveFileSpecFixed(const std::wstring& path);
std::string getModulePathA(void *module);
std::wstring getModulePathW(void *module);
tstring getModulePath(void *module);
std::string getExePathA();
std::wstring getExePathW();
tstring getExePath();
std::string getExeDirA();
std::wstring getExeDirW();
tstring getExeDir();
std::vector<std::wstring> get_file_list_with_filter(const std::wstring& dir, const std::wstring& filter_filename);
std::vector<std::string> get_file_list_with_filter(const std::string& dir, const std::string& filter_filename);

std::string GetFullPathFrom(const char *path, const char *baseDir = nullptr);
std::wstring GetFullPathFrom(const wchar_t *path, const wchar_t *baseDir = nullptr);
std::string GetRelativePathFrom(const char *path, const char *baseDir = nullptr);
std::wstring GetRelativePathFrom(const wchar_t *path, const wchar_t *baseDir = nullptr);
bool rgy_file_exists(const std::string& filepath);
bool rgy_file_exists(const std::wstring& filepath);
bool rgy_directory_exists(const std::string& directorypath);
bool rgy_directory_exists(const std::wstring& directorypath);
bool rgy_get_filesize(const char *filepath, uint64_t *filesize);
std::pair<int, std::string> PathRemoveFileSpecFixed(const std::string& path);
std::string PathRemoveExtensionS(const std::string& path);
bool CreateDirectoryRecursive(const char *dir, const bool errorIfAlreadyExists = false);
bool CreateDirectoryRecursive(const wchar_t *dir, const bool errorIfAlreadyExists = false);
std::string PathGetFilename(const std::string& path);
void rgy_file_remove(const char *path);
int rgy_directory_remove(const char *dir);
int rgy_directory_remove(const wchar_t *dir);

bool rgy_file_copy(const std::string& srcpath, const std::string& dstpath, const bool overwrite);
bool rgy_file_copy(const std::wstring& srcpath, const std::wstring& dstpath, const bool overwrite);

bool check_ext(const TCHAR *filename, const std::vector<const char*>& ext_list);
bool check_ext(const tstring& filename, const std::vector<const char*>& ext_list);

//拡張子が一致するか確認する
bool _tcheck_ext(const TCHAR *filename, const TCHAR *ext);

std::string rgy_get_extension(const std::string& filename);
std::wstring rgy_get_extension(const std::wstring& filename);

bool rgy_path_is_same(const TCHAR *path1, const TCHAR *path2);
bool rgy_path_is_same(const tstring& path1, const tstring& path2);

#if defined(_WIN32) || defined(_WIN64)
std::vector<std::basic_string<TCHAR>> createProcessOpenedFileList(const std::vector<size_t>& list_pid);
#endif //#if defined(_WIN32) || defined(_WIN64)

std::string find_executable_in_path(const std::string& name);
std::wstring find_executable_in_path(const std::wstring& name);

#endif //__RGY_FILESYSTEM_H__
