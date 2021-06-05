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
// ------------------------------------------------------------------------------------------

#include <filesystem>
#include <cstdint>
#include "rgy_util.h"
#include "rgy_codepage.h"
#include "rgy_filesystem.h"

std::string GetFullPath(const char *path) {
    return std::filesystem::absolute(std::filesystem::path(strlen(path) ? path : ".")).lexically_normal().string();
}
#if defined(_WIN32) || defined(_WIN64)
std::wstring GetFullPath(const wchar_t *path) {
    return std::filesystem::absolute(std::filesystem::path(wcslen(path) ? path : L".")).lexically_normal().wstring();
}
//ルートディレクトリを取得
std::string PathGetRoot(const char *path) {
    return std::filesystem::path(GetFullPath(path)).root_name().string();
}
std::wstring PathGetRoot(const wchar_t *path) {
    return std::filesystem::path(GetFullPath(path)).root_name().wstring();
}

//パスのルートが存在するかどうか
static bool PathRootExists(const char *path) {
    if (path == nullptr)
        return false;
    return std::filesystem::exists(PathGetRoot(path));
}
static bool PathRootExists(const wchar_t *path) {
    if (path == nullptr)
        return false;
    return std::filesystem::exists(PathGetRoot(path));
}
#endif //#if defined(_WIN32) || defined(_WIN64)
std::pair<int, std::string> PathRemoveFileSpecFixed(const std::string& path) {
    const auto newPath = std::filesystem::path(path).remove_filename().string();
    return std::make_pair((int)(path.length() - newPath.length()), newPath);
}
#if defined(_WIN32) || defined(_WIN64)
std::pair<int, std::wstring> PathRemoveFileSpecFixed(const std::wstring& path) {
    const auto newPath = std::filesystem::path(path).remove_filename().wstring();
    return std::make_pair((int)(path.length() - newPath.length()), newPath);
}
#endif //#if defined(_WIN32) || defined(_WIN64)
std::string PathRemoveExtensionS(const std::string& path) {
    const auto lastdot = path.find_last_of(".");
    if (lastdot == std::string::npos) return path;
    return path.substr(0, lastdot);
}
#if defined(_WIN32) || defined(_WIN64)
std::wstring PathRemoveExtensionS(const std::wstring& path) {
    const auto lastdot = path.find_last_of(L".");
    if (lastdot == std::string::npos) return path;
    return path.substr(0, lastdot);
}
std::string PathCombineS(const std::string& dir, const std::string& filename) {
    return std::filesystem::path(dir).append(filename).string();
}
std::wstring PathCombineS(const std::wstring& dir, const std::wstring& filename) {
    return std::filesystem::path(dir).append(filename).wstring();
}
#endif //#if defined(_WIN32) || defined(_WIN64)
//フォルダがあればOK、なければ作成する
bool CreateDirectoryRecursive(const char *dir) {
    auto targetDir = std::filesystem::path(strlen(dir) ? dir : ".");
    if (std::filesystem::exists(targetDir)) {
        return true;
    }
    return std::filesystem::create_directories(targetDir);
}
#if defined(_WIN32) || defined(_WIN64)
bool CreateDirectoryRecursive(const wchar_t *dir) {
    auto targetDir = std::filesystem::path(wcslen(dir) ? dir : L".");
    if (std::filesystem::exists(targetDir)) {
        return true;
    }
    return std::filesystem::create_directories(targetDir);
}
#endif //#if defined(_WIN32) || defined(_WIN64)

bool check_ext(const TCHAR *filename, const std::vector<const char*>& ext_list) {
    const auto target = tolowercase(std::filesystem::path(filename).extension().string());
    if (target.length() > 0) {
        for (auto ext : ext_list) {
            if (target == tolowercase(ext)) {
                return true;
            }
        }
    }
    return false;
}

bool check_ext(const tstring& filename, const std::vector<const char*>& ext_list) {
    return check_ext(filename.c_str(), ext_list);
}

bool _tcheck_ext(const TCHAR *filename, const TCHAR *ext) {
    return tolowercase(std::filesystem::path(filename).extension().string()) == tolowercase(tchar_to_string(ext));
}

bool rgy_file_exists(const std::string& filepath) {
    return std::filesystem::exists(filepath) && std::filesystem::is_regular_file(filepath);
}

bool rgy_file_exists(const std::wstring& filepath) {
    return std::filesystem::exists(filepath) && std::filesystem::is_regular_file(filepath);
}

bool rgy_get_filesize(const char *filepath, uint64_t *filesize) {
#if defined(_WIN32) || defined(_WIN64)
    const auto filepathw = char_to_wstring(filepath);
    return rgy_get_filesize(filepathw.c_str(), filesize);
#else //#if defined(_WIN32) || defined(_WIN64)
    struct stat stat;
    FILE *fp = fopen(filepath, "rb");
    if (fp == NULL || fstat(fileno(fp), &stat)) {
        *filesize = 0;
        return 1;
    }
    if (fp) {
        fclose(fp);
    }
    *filesize = stat.st_size;
    return 0;
#endif //#if defined(_WIN32) || defined(_WIN64)
}

#if defined(_WIN32) || defined(_WIN64)
bool rgy_get_filesize(const wchar_t *filepath, uint64_t *filesize) {
    WIN32_FILE_ATTRIBUTE_DATA fd = { 0 };
    bool ret = (GetFileAttributesExW(filepath, GetFileExInfoStandard, &fd)) ? true : false; // No MAX_PATH Limitation
    *filesize = (ret) ? (((UINT64)fd.nFileSizeHigh) << 32) + (UINT64)fd.nFileSizeLow : NULL;
    return ret;
}

std::vector<tstring> get_file_list(const tstring& pattern, const tstring& dir) {
    std::vector<tstring> list;

    auto buf = wstring_to_tstring(std::filesystem::path(GetFullPath(dir.c_str())).append(pattern).wstring());

    WIN32_FIND_DATA win32fd;
    HANDLE hFind = FindFirstFile(buf.c_str(), &win32fd); // FindFirstFileW No MAX_PATH Limitation

    if (hFind == INVALID_HANDLE_VALUE) {
        return list;
    }

    do {
        if ((win32fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)
            && _tcscmp(win32fd.cFileName, _T("..")) != 0
            && _tcscmp(win32fd.cFileName, _T(".")) != 0) {
            const auto buf2 = wstring_to_tstring(std::filesystem::path(GetFullPath(dir.c_str())).append(win32fd.cFileName).wstring());
            vector_cat(list, get_file_list(pattern, buf2));
        } else {
            buf = wstring_to_tstring(std::filesystem::path(GetFullPath(dir.c_str())).append(win32fd.cFileName).wstring());
            list.push_back(buf);
        }
    } while (FindNextFile(hFind, &win32fd));
    FindClose(hFind);
    return list;
}

bool PathFileExistsA(const char *filename) {
    auto path = std::filesystem::path(filename);
    return std::filesystem::exists(path) && std::filesystem::is_regular_file(path);
}

bool PathFileExistsW(const wchar_t *filename) {
    auto path = std::filesystem::path(filename);
    return std::filesystem::exists(path) && std::filesystem::is_regular_file(path);
}

tstring getExePath() {
    TCHAR exePath[16384];
    memset(exePath, 0, sizeof(exePath));
    GetModuleFileName(NULL, exePath, _countof(exePath));
    return exePath;
}

#else
tstring getExePath() {
    char prg_path[16384];
    auto ret = readlink("/proc/self/exe", prg_path, sizeof(prg_path));
    if (ret <= 0) {
        prg_path[0] = '\0';
    }
    return prg_path;
}

#endif //#if defined(_WIN32) || defined(_WIN64)
tstring getExeDir() {
    return PathRemoveFileSpecFixed(getExePath()).second;
}

bool rgy_path_is_same(const TCHAR *path1, const TCHAR *path2) {
    const auto p1 = std::filesystem::path(path1);
    const auto p2 = std::filesystem::path(path2);
    std::error_code ec;
    return std::filesystem::equivalent(p1, p2, ec);
}

bool rgy_path_is_same(const tstring& path1, const tstring& path2) {
    return rgy_path_is_same(path1.c_str(), path2.c_str());
}
