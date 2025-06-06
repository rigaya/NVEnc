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
#include "rgy_env.h"
#include "rgy_codepage.h"
#include "rgy_filesystem.h"
#if !(defined(_WIN32) || defined(_WIN64))
#include <dlfcn.h>  // dladdr関数用
#endif

std::string GetFullPathFrom(const char *path, const char *baseDir) {
    if (auto p = std::filesystem::path(path); p.is_absolute()) {
        return path;
    }
    path = (path && strlen(path)) ? path : ".";
    const auto p = (baseDir) ? std::filesystem::path(baseDir).append(path) : std::filesystem::absolute(std::filesystem::path(path));
    return p.lexically_normal().string();
}
std::string GetRelativePathFrom(const char *path, const char *baseDir) {
    if (path == nullptr || strlen(path) == 0) {
        return ".";
    }
    const auto p = std::filesystem::path(path);
    if (p.is_relative()) {
        return path;
    }
    const auto basePath = (baseDir) ? std::filesystem::path(baseDir) : std::filesystem::current_path();
    std::error_code ec;
    return std::filesystem::proximate(p, basePath, ec).string();
}
std::wstring GetFullPathFrom(const wchar_t *path, const wchar_t *baseDir) {
    if (auto p = std::filesystem::path(path); p.is_absolute()) {
        return path;
    }
    path = (path && wcslen(path)) ? path : L".";
    const auto p = (baseDir) ? std::filesystem::path(baseDir).append(path) : std::filesystem::absolute(std::filesystem::path(path));
    return p.lexically_normal().wstring();
}
std::wstring GetRelativePathFrom(const wchar_t *path, const wchar_t *baseDir) {
    if (path == nullptr || wcslen(path) == 0) {
        return L".";
    }
    const auto p = std::filesystem::path(path);
    if (p.is_relative()) {
        return path;
    }
    const auto basePath = (baseDir) ? std::filesystem::path(baseDir) : std::filesystem::current_path();
    std::error_code ec;
    return std::filesystem::proximate(p, basePath, ec).wstring();
}
#if defined(_WIN32) || defined(_WIN64)
//ルートディレクトリを取得
std::string PathGetRoot(const char *path) {
    return std::filesystem::path(GetFullPathFrom(path)).root_name().string();
}
std::wstring PathGetRoot(const wchar_t *path) {
    return std::filesystem::path(GetFullPathFrom(path)).root_name().wstring();
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
std::pair<int, std::wstring> PathRemoveFileSpecFixed(const std::wstring& path) {
    const auto newPath = std::filesystem::path(path).remove_filename().wstring();
    return std::make_pair((int)(path.length() - newPath.length()), newPath);
}
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
#endif //#if defined(_WIN32) || defined(_WIN64)
std::string PathCombineS(const std::string& dir, const std::string& filename) {
    return std::filesystem::path(dir).append(filename).string();
}
std::wstring PathCombineS(const std::wstring& dir, const std::wstring& filename) {
    return std::filesystem::path(dir).append(filename).wstring();
}
//フォルダがあればOK、なければ作成する
bool CreateDirectoryRecursive(const char *dir, const bool errorIfAlreadyExists) {
    auto targetDir = std::filesystem::path(strlen(dir) ? dir : ".");
    if (std::filesystem::exists(targetDir)) {
        return (errorIfAlreadyExists) ? false : true;
    }
    try {
        return std::filesystem::create_directories(targetDir);
    } catch (...) {
        return false;
    }
}
bool CreateDirectoryRecursive(const wchar_t *dir, const bool errorIfAlreadyExists) {
    auto targetDir = std::filesystem::path(wcslen(dir) ? dir : L".");
    if (std::filesystem::exists(targetDir)) {
        return (errorIfAlreadyExists) ? false : true;
    }
    try {
        return std::filesystem::create_directories(targetDir);
    } catch (...) {
        return false;
    }
}


std::string PathGetFilename(const std::string& path) {
    return std::filesystem::path(path).filename().string();
}
#if defined(_WIN32) || defined(_WIN64)
std::wstring PathGetFilename(const std::wstring& path) {
    return std::filesystem::path(path).filename().wstring();
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

std::string rgy_get_extension(const std::string& filename) {
    return std::filesystem::path(filename).extension().string();
}
std::wstring rgy_get_extension(const std::wstring& filename) {
    return std::filesystem::path(filename).extension().wstring();
}

bool rgy_file_exists(const std::string& filepath) {
    return std::filesystem::exists(filepath) && std::filesystem::is_regular_file(filepath);
}

bool rgy_file_exists(const std::wstring& filepath) {
    return std::filesystem::exists(filepath) && std::filesystem::is_regular_file(filepath);
}

bool rgy_directory_exists(const std::string& directorypath) {
    return std::filesystem::exists(directorypath) && std::filesystem::is_directory(directorypath);
}

bool rgy_directory_exists(const std::wstring& directorypath) {
    return std::filesystem::exists(directorypath) && std::filesystem::is_directory(directorypath);
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

std::vector<std::wstring> get_file_list_with_filter(const std::wstring& dir, const std::wstring& filter_filename) {
    std::vector<std::wstring> list;
    for (const auto& x : std::filesystem::recursive_directory_iterator(dir)) {
        if (filter_filename.length() == 0 || x.path().filename().wstring().find(filter_filename) != std::string::npos) {
            list.push_back(x.path().wstring());
        }
    }
    return list;
}

std::vector<std::string> get_file_list_with_filter(const std::string& dir, const std::string& filter_filename) {
    std::vector<std::string> list;
    for (const auto& x : std::filesystem::recursive_directory_iterator(dir)) {
        if (filter_filename.length() == 0 || x.path().filename().string().find(filter_filename) != std::string::npos) {
            list.push_back(x.path().string());
        }
    }
    return list;
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

    auto buf = wstring_to_tstring(std::filesystem::path(GetFullPathFrom(dir.c_str())).append(pattern).wstring());

    WIN32_FIND_DATA win32fd;
    HANDLE hFind = FindFirstFile(buf.c_str(), &win32fd); // FindFirstFileW No MAX_PATH Limitation

    if (hFind == INVALID_HANDLE_VALUE) {
        return list;
    }

    do {
        if ((win32fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)
            && _tcscmp(win32fd.cFileName, _T("..")) != 0
            && _tcscmp(win32fd.cFileName, _T(".")) != 0) {
            const auto buf2 = wstring_to_tstring(std::filesystem::path(GetFullPathFrom(dir.c_str())).append(win32fd.cFileName).wstring());
            vector_cat(list, get_file_list(pattern, buf2));
        } else {
            buf = wstring_to_tstring(std::filesystem::path(GetFullPathFrom(dir.c_str())).append(win32fd.cFileName).wstring());
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

std::wstring getExePathW() {
    WCHAR exePath[16384];
    memset(exePath, 0, sizeof(exePath));
    GetModuleFileNameW(NULL, exePath, _countof(exePath));
    return exePath;
}

std::string getExePathA() {
    char exePath[16384];
    memset(exePath, 0, sizeof(exePath));
    GetModuleFileNameA(NULL, exePath, _countof(exePath));
    return exePath;
}

tstring getModulePath(void *module) {
    TCHAR dllPath[16384];
    memset(dllPath, 0, sizeof(dllPath));
    GetModuleFileName((HMODULE)module, dllPath, _countof(dllPath));
    return dllPath;
}

std::wstring getModulePathW(void *module) {
    WCHAR dllPath[16384];
    memset(dllPath, 0, sizeof(dllPath));
    GetModuleFileNameW((HMODULE)module, dllPath, _countof(dllPath));
    return dllPath;
}

std::wstring getExeDirW() {
    return PathRemoveFileSpecFixed(getExePathW()).second;
}
#else

std::string getExePathA() {
    char prg_path[16384];
    auto ret = readlink("/proc/self/exe", prg_path, sizeof(prg_path));
    if (ret <= 0) {
        prg_path[0] = '\0';
    }
    return prg_path;
}

std::wstring getExePathW() {
    return char_to_wstring(getExePathA());
}

tstring getExePath() {
    return getExePathA();
}

std::string getModulePathA(void *module) {
    if (module == nullptr) {
        return getExePath();
    }
    // Linux実装 - 共有ライブラリ(.so)のパスを取得
    Dl_info dl_info;
    // 現在の実行コードのアドレスを使用して.soファイルの情報を取得
    // この関数ポインタ自体のアドレスを使用
    if (dladdr(module, &dl_info) != 0) {
        if (dl_info.dli_fname) {
            const char* sopath = dl_info.dli_fname;
            return sopath;
        }
    }
    return "";
}

std::wstring getModulePathW(void *module) {
    return char_to_wstring(getModulePathA(module));
}

tstring getModulePath(void *module) {
    return getModulePathA(module);
}

#endif //#if defined(_WIN32) || defined(_WIN64)

void rgy_file_remove(const char *path) {
    ::remove(path);
}
#if defined(_WIN32) || defined(_WIN64)
void rgy_file_remove(const wchar_t *path) {
    _wremove(path);
}
#endif


int rgy_directory_remove(const char *dirname) {
    try {
        std::filesystem::remove_all(dirname);
    } catch (...) {
        return 1;
    }
    return 0;
}

int rgy_directory_remove(const wchar_t *dirname) {
    try {
        std::filesystem::remove_all(dirname);
    } catch (...) {
        return 1;
    }
    return 0;
}


bool rgy_file_copy(const std::string& srcpath, const std::string& dstpath, const bool overwrite) {
    try {
        return std::filesystem::copy_file(srcpath, dstpath, overwrite ? std::filesystem::copy_options::overwrite_existing : std::filesystem::copy_options::none);
    } catch (...) {
        return false;
    }
}
bool rgy_file_copy(const std::wstring& srcpath, const std::wstring& dstpath, const bool overwrite) {
    try {
        return std::filesystem::copy_file(srcpath, dstpath, overwrite ? std::filesystem::copy_options::overwrite_existing : std::filesystem::copy_options::none);
    } catch (...) {
        return false;
    }
}

tstring getExeDir() {
    return PathRemoveFileSpecFixed(getExePath()).second;
}

std::string getExeDirA() {
    return PathRemoveFileSpecFixed(getExePathA()).second;
}

bool rgy_path_is_same(const TCHAR *path1, const TCHAR *path2) {
    try {
        const auto p1 = std::filesystem::path(path1);
        const auto p2 = std::filesystem::path(path2);
        std::error_code ec;
        return std::filesystem::equivalent(p1, p2, ec);
    } catch (...) {
        return false;
    }
}

bool rgy_path_is_same(const tstring& path1, const tstring& path2) {
    return rgy_path_is_same(path1.c_str(), path2.c_str());
}

#if defined(_WIN32) || defined(_WIN64)
std::vector<std::basic_string<TCHAR>> createProcessOpenedFileList(const std::vector<size_t>& list_pid) {
    const auto list_handle = createProcessHandleList(list_pid, L"File");
    std::vector<std::basic_string<TCHAR>> list_file;
    std::vector<TCHAR> filename(32768+1, 0);
    for (const auto& handle : list_handle) {
        const auto fileType = GetFileType(handle.get());
        if (fileType == FILE_TYPE_DISK) { //ハンドルがパイプだとGetFinalPathNameByHandleがフリーズするため使用不可
            memset(filename.data(), 0, sizeof(filename[0]) * filename.size());
            auto ret = GetFinalPathNameByHandle(handle.get(), filename.data(), (DWORD)filename.size(), FILE_NAME_NORMALIZED | VOLUME_NAME_DOS);
            if (ret != 0) {
                try {
                    auto f = std::filesystem::canonical(filename.data());
                    if (std::filesystem::is_regular_file(f)) {
                        list_file.push_back(f.string<TCHAR>());
                    }
                } catch (...) {}
            }
        }
    }
    // 重複を排除
    std::sort(list_file.begin(), list_file.end());
    auto result = std::unique(list_file.begin(), list_file.end());
    // 不要になった要素を削除
    list_file.erase(result, list_file.end());
    return list_file;
}
#endif //#if defined(_WIN32) || defined(_WIN64)

#if defined(_WIN32) || defined(_WIN64)
std::string find_executable_in_path(const std::string& name) {
    char path[1024];
    if (SearchPathA(NULL, name.c_str(), ".exe", _countof(path), path, NULL)) {
        return path;
    }
    return "";
}

std::wstring find_executable_in_path(const std::wstring& name) {
    wchar_t path[1024];
    if (SearchPathW(NULL, name.c_str(), L".exe", _countof(path), path, NULL)) {
        return path;
    }
    return L"";
}
#else
std::string find_executable_in_path(const std::string& name) {
    const char* path_env = getenv("PATH");
    if (!path_env) {
        return "";
    }

    std::string path_str(path_env);
    std::stringstream ss(path_str);
    std::string dir;
    
    while (std::getline(ss, dir, ':')) {
        std::string full_path = dir + "/" + name;
        if (access(full_path.c_str(), X_OK) == 0) {
            return full_path;
        }
    }
    return "";
}

std::wstring find_executable_in_path(const std::wstring& name) {
    return char_to_wstring(find_executable_in_path(wstring_to_string(name)));
}
#endif //#if defined(_WIN32) || defined(_WIN64)

