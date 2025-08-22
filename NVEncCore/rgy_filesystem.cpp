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
#if (defined(_WIN32) || defined(_WIN64))
#include <shlwapi.h>
#pragma comment(lib, "shlwapi.lib")
#else
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

void GetRelativePathTo(char *dst, size_t nSize, const char *path, const char *baseDir) {
    strcpy_s(dst, nSize, GetRelativePathFrom(path, baseDir).c_str());
}
void GetRelativePathTo(wchar_t *dst, size_t nSize, const wchar_t *path, const wchar_t *baseDir) {
    wcscpy_s(dst, nSize, GetRelativePathFrom(path, baseDir).c_str());
}

#if defined(_WIN32) || defined(_WIN64)
//ルートディレクトリを取得
std::string PathGetRoot(const char *path) {
    return std::filesystem::path(GetFullPathFrom(path)).root_name().string();
}
std::wstring PathGetRoot(const wchar_t *path) {
    return std::filesystem::path(GetFullPathFrom(path)).root_name().wstring();
}
bool PathGetRoot(const char *path, char *root, size_t nSize) {
    strcpy_s(root, nSize, PathGetRoot(path).c_str());
    return true;
}
bool PathGetRoot(const wchar_t *path, wchar_t *root, size_t nSize) {
    wcscpy_s(root, nSize, PathGetRoot(path).c_str());
    return true;
}

//パスのルートが存在するかどうか
bool PathRootExists(const char *path) {
    if (path == nullptr)
        return false;
    return std::filesystem::exists(PathGetRoot(path));
}
bool PathRootExists(const wchar_t *path) {
    if (path == nullptr)
        return false;
    return std::filesystem::exists(PathGetRoot(path));
}

bool GetPathRootFreeSpace(const char *path, uint64_t *freespace) {
    auto root = PathGetRoot(path);
    //ドライブの空き容量取得
    ULARGE_INTEGER drive_avail_space = { 0 };
    if (GetDiskFreeSpaceExA(root.c_str(), &drive_avail_space, NULL, NULL)) {
        *freespace = drive_avail_space.QuadPart;
        return TRUE;
    }
    return FALSE;
}
bool GetPathRootFreeSpace(const wchar_t *path, uint64_t *freespace) {
    auto root = PathGetRoot(path);
    //ドライブの空き容量取得
    ULARGE_INTEGER drive_avail_space = { 0 };
    if (GetDiskFreeSpaceExW(root.c_str(), &drive_avail_space, NULL, NULL)) {
        *freespace = drive_avail_space.QuadPart;
        return TRUE;
    }
    return FALSE;
}

//PathRemoveFileSpecFixedがVistaでは5C問題を発生させるため、その回避策
bool PathRemoveFileSpecFixed(char *path) {
    char *ptr = PathFindFileNameA(path);
    if (path == ptr)
        return FALSE;
    *(ptr - 1) = '\0';
    return TRUE;
}
bool PathRemoveFileSpecFixed(wchar_t *path) {
    wchar_t *ptr = PathFindFileNameW(path);
    if (path == ptr)
        return FALSE;
    *(ptr - 1) = L'\0';
    return TRUE;
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

bool check_ext(const char *filename, const char *ext) {
    return tolowercase(std::filesystem::path(filename).extension().string()) == tolowercase(ext);
}

#if defined(_WIN32) || defined(_WIN64)
bool check_ext(const wchar_t *filename, const wchar_t *ext) {
    return tolowercase(std::filesystem::path(filename).extension().wstring()) == tolowercase(ext);
}
#endif //#if defined(_WIN32) || defined(_WIN64)

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

#if !(defined(_WIN32) || defined(_WIN64))
bool PathFileExistsA(const char *filename) {
    auto path = std::filesystem::path(filename);
    return std::filesystem::exists(path) && std::filesystem::is_regular_file(path);
}

bool PathFileExistsW(const wchar_t *filename) {
    auto path = std::filesystem::path(filename);
    return std::filesystem::exists(path) && std::filesystem::is_regular_file(path);
}
#endif

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

std::string getModulePathA(void *module) {
    char dllPath[16384];
    memset(dllPath, 0, sizeof(dllPath));
    GetModuleFileNameA((HMODULE)module, dllPath, _countof(dllPath));
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

bool rgy_file_rename(const std::string& srcpath, const std::string& dstpath, const bool overwrite) {
    std::error_code ec;
    std::filesystem::rename(srcpath, dstpath, ec);
    return !ec;
}

bool rgy_file_rename(const std::wstring& srcpath, const std::wstring& dstpath, const bool overwrite) {
    std::error_code ec;
    std::filesystem::rename(srcpath, dstpath, ec);
    return !ec;
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

#if !(defined(_WIN32) || defined(_WIN64))
char *PathFindExtensionA(char *path) {
    return strrchr(path, '.');
}
wchar_t *PathFindExtensionW(wchar_t *path) {
    return wcsrchr(path, L'.');
}
const char *PathFindExtensionA(const char *path) {
    return strrchr(path, '.');
}
const wchar_t *PathFindExtensionW(const wchar_t *path) {
    return wcsrchr(path, L'.');
}
#endif

//ファイル名(拡張子除く)の後ろに文字列を追加する
void apply_appendix(char *new_filename, size_t new_filename_size, const char *orig_filename, const char *appendix) {
    if (new_filename != orig_filename)
        strcpy_s(new_filename, new_filename_size, orig_filename);
    strcpy_s(PathFindExtensionA(new_filename), new_filename_size - (PathFindExtensionA(new_filename) - new_filename), appendix);
}
void apply_appendix(wchar_t *new_filename, size_t new_filename_size, const wchar_t *orig_filename, const wchar_t *appendix) {
    if (new_filename != orig_filename)
        wcscpy_s(new_filename, new_filename_size, orig_filename);
    wcscpy_s(PathFindExtensionW(new_filename), new_filename_size - (PathFindExtensionW(new_filename) - new_filename), appendix);
}

void insert_before_ext(char *filename, size_t nSize, const char *insert_str) {
    char *ext = PathFindExtensionA(filename);
    if (ext == NULL)
        strcat_s(filename, nSize, insert_str);
    else {
        const size_t insert_len = strlen(insert_str);
        const size_t filename_len = strlen(filename);
        if (nSize > filename_len + insert_len) {
            memmove(ext + insert_len, ext, sizeof(insert_str[0]) * (strlen(ext)+1));
            memcpy(ext, insert_str, sizeof(insert_str[0]) * insert_len);
        }
    }
}
void insert_before_ext(wchar_t *filename, size_t nSize, const wchar_t *insert_str) {
    wchar_t *ext = PathFindExtensionW(filename);
    if (ext == NULL)
        wcscat_s(filename, nSize, insert_str);
    else {
        const size_t insert_len = wcslen(insert_str);
        const size_t filename_len = wcslen(filename);
        if (nSize > filename_len + insert_len) {
            memmove(ext + insert_len, ext, sizeof(insert_str[0]) * (wcslen(ext)+1));
            memcpy(ext, insert_str, sizeof(insert_str[0]) * insert_len);
        }
    }
}
void insert_before_ext(char *filename, size_t nSize, int insert_num) {
    char tmp[22];
    sprintf_s(tmp, _countof(tmp), "%d", insert_num);
    insert_before_ext(filename, nSize, tmp);
}
#if defined(_WIN32) || defined(_WIN64)
void insert_before_ext(wchar_t *filename, size_t nSize, int insert_num) {
    wchar_t tmp[22];
    swprintf_s(tmp, _countof(tmp), L"%d", insert_num);
    insert_before_ext(filename, nSize, tmp);
}
#endif //#if defined(_WIN32) || defined(_WIN64)

//パスの拡張子を変更する
void change_ext(char *filename, size_t nSize, const char *ext) {
    size_t len_to_ext;
    char *ext_ptr = PathFindExtensionA(filename);
    len_to_ext = (ext_ptr) ? ext_ptr - filename : strlen(filename);
    strcpy_s(filename + len_to_ext, nSize - len_to_ext, ext);
}
void change_ext(wchar_t *filename, size_t nSize, const wchar_t *ext) {
    size_t len_to_ext;
    wchar_t *ext_ptr = PathFindExtensionW(filename);
    len_to_ext = (ext_ptr) ? ext_ptr - filename : wcslen(filename);
    wcscpy_s(filename + len_to_ext, nSize - len_to_ext, ext);
}

#if defined(_WIN32) || defined(_WIN64)
//フォルダがあればOK、なければ作成する
bool DirectoryExistsOrCreate(const char *dir) {
    if (rgy_directory_exists(dir))
        return TRUE;
    return (PathRootExists(dir) && CreateDirectoryA(dir, NULL) != NULL) ? TRUE : FALSE;
}
bool DirectoryExistsOrCreate(const wchar_t *dir) {
    if (rgy_directory_exists(dir))
        return TRUE;
    return (PathRootExists(dir) && CreateDirectoryW(dir, NULL) != NULL) ? TRUE : FALSE;
}
#endif //#if defined(_WIN32) || defined(_WIN64)

//ファイルの存在と0byteより大きいかを確認
bool FileExistsAndHasSize(const char *path) {
    uint64_t filesize = 0;
    return rgy_file_exists(path) && rgy_get_filesize(path, &filesize) && filesize > 0;
}
#if defined(_WIN32) || defined(_WIN64)
bool FileExistsAndHasSize(const wchar_t *path) {
    uint64_t filesize = 0;
    return rgy_file_exists(path) && rgy_get_filesize(path, &filesize) && filesize > 0;
}
#endif //#if defined(_WIN32) || defined(_WIN64)

void PathGetDirectory(char *dir, size_t nSize, const char *path) {
#if defined(_WIN32) || defined(_WIN64)
    strcpy_s(dir, nSize, path);
    PathRemoveFileSpecFixed(dir);
#else
    strcpy_s(dir, nSize, PathRemoveFileSpecFixed(path).second.c_str());
#endif
}
void PathGetDirectory(wchar_t *dir, size_t nSize, const wchar_t *path) {
#if defined(_WIN32) || defined(_WIN64)
    wcscpy_s(dir, nSize, path);
    PathRemoveFileSpecFixed(dir);
#else
    wcscpy_s(dir, nSize, PathRemoveFileSpecFixed(path).second.c_str());
#endif
}

#if defined(_WIN32) || defined(_WIN64)
uint64_t GetFileLastUpdate(const char *filepath) {
    WIN32_FILE_ATTRIBUTE_DATA fd = { 0 };
    GetFileAttributesExA(filepath, GetFileExInfoStandard, &fd);
    return ((uint64_t)fd.ftLastWriteTime.dwHighDateTime << 32) + (uint64_t)fd.ftLastWriteTime.dwLowDateTime;
}
uint64_t GetFileLastUpdate(const wchar_t *filepath) {
    WIN32_FILE_ATTRIBUTE_DATA fd = { 0 };
    GetFileAttributesExW(filepath, GetFileExInfoStandard, &fd);
    return ((uint64_t)fd.ftLastWriteTime.dwHighDateTime << 32) + (uint64_t)fd.ftLastWriteTime.dwLowDateTime;
}
#endif //#if defined(_WIN32) || defined(_WIN64)

size_t append_str(char **dst, size_t *nSize, const char *append) {
    size_t len = strlen(append);
    if (*nSize - 1 <= len)
        return 0;
    memcpy(*dst, append, (len + 1) * sizeof(dst[0][0]));
    *dst += len;
    *nSize -= len;
    return len;
}
size_t append_str(wchar_t **dst, size_t *nSize, const wchar_t *append) {
    size_t len = wcslen(append);
    if (*nSize - 1 <= len)
        return 0;
    memcpy(*dst, append, (len + 1) * sizeof(dst[0][0]));
    *dst += len;
    *nSize -= len;
    return len;
}

//多くのPath～関数はMAX_LEN(260)以上でもOKだが、一部は不可
//これもそのひとつ
bool PathAddBackSlashLong(char *dir) {
    size_t len = strlen(dir);
    if (dir[len-1] != '\\') {
        dir[len] = '\\';
        dir[len+1] = '\0';
        return TRUE;
    }
    return FALSE;
}
bool PathAddBackSlashLong(wchar_t *dir) {
    size_t len = wcslen(dir);
    if (dir[len-1] != L'\\') {
        dir[len] = L'\\';
        dir[len+1] = L'\0';
        return TRUE;
    }
    return FALSE;
}
//PathCombineもMAX_LEN(260)以上不可
bool PathCombineLong(char *path, size_t nSize, const char *dir, const char *filename) {
    size_t dir_len;
    if (path == dir) {
        dir_len = strlen(path);
    } else {
        dir_len = strlen(dir);
        if (nSize <= dir_len)
            return FALSE;

        memcpy(path, dir, (dir_len+1) * sizeof(path[0]));
    }
    dir_len += PathAddBackSlashLong(path);

    size_t filename_len = strlen(filename);
    if (nSize - dir_len <= filename_len)
        return FALSE;
    memcpy(path + dir_len, filename, (filename_len+1) * sizeof(path[0]));
    return TRUE;
}
bool PathCombineLong(wchar_t *path, size_t nSize, const wchar_t *dir, const wchar_t *filename) {
    size_t dir_len;
    if (path == dir) {
        dir_len = wcslen(path);
    } else {
        dir_len = wcslen(dir);
        if (nSize <= dir_len)
            return FALSE;

        memcpy(path, dir, (dir_len+1) * sizeof(path[0]));
    }
    dir_len += PathAddBackSlashLong(path);

    size_t filename_len = wcslen(filename);
    if (nSize - dir_len <= filename_len)
        return FALSE;
    memcpy(path + dir_len, filename, (filename_len+1) * sizeof(path[0]));
    return TRUE;
}

bool PathForceRemoveBackSlash(char *path) {
    size_t len = strlen(path);
    int ret = FALSE;
    if (path != NULL && len) {
        char *ptr = path + len - 1;
        if (*ptr == '\\') {
            *ptr = '\0';
            ret = TRUE;
        }
    }
    return ret;
}
bool PathForceRemoveBackSlash(wchar_t *path) {
    size_t len = wcslen(path);
    int ret = FALSE;
    if (path != NULL && len) {
        wchar_t *ptr = path + len - 1;
        if (*ptr == L'\\') {
            *ptr = L'\0';
            ret = TRUE;
        }
    }
    return ret;
}

bool swap_file(const char *fileA, const char *fileB) {
    if (!rgy_file_exists(fileA) || !rgy_file_exists(fileB))
        return FALSE;

    std::string filetemp;
    for (int i = 0; !i || rgy_file_exists(filetemp); i++) {
        filetemp = std::string(fileA) + ".swap" + std::to_string(i) + ".tmp";
    }
    if (rgy_file_rename(fileA, filetemp))
        return FALSE;
    if (rgy_file_rename(fileB, fileA))
        return FALSE;
    if (rgy_file_rename(filetemp, fileB))
        return FALSE;
    return TRUE;
}
bool swap_file(const wchar_t *fileA, const wchar_t *fileB) {
    if (!rgy_file_exists(fileA) || !rgy_file_exists(fileB))
        return FALSE;

    std::wstring filetemp;
    for (int i = 0; !i || rgy_file_exists(filetemp); i++) {
        filetemp = std::wstring(fileA) + L".swap" + std::to_wstring(i) + L".tmp";
    }
    if (rgy_file_rename(fileA, filetemp))
        return FALSE;
    if (rgy_file_rename(fileB, fileA))
        return FALSE;
    if (rgy_file_rename(filetemp, fileB))
        return FALSE;
    return TRUE;
}

