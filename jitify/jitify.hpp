/*
 * Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * * Redistributions of source code must retain the above copyright
 *   notice, this list of conditions and the following disclaimer.
 * * Redistributions in binary form must reproduce the above copyright
 *   notice, this list of conditions and the following disclaimer in the
 *   documentation and/or other materials provided with the distribution.
 * * Neither the name of NVIDIA CORPORATION nor the names of its
 *   contributors may be used to endorse or promote products derived
 *   from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*
  -----------
  Jitify 0.9
  -----------
  A C++ library for easy integration of CUDA runtime compilation into
  existing codes.

  --------------
  How to compile
  --------------
  Compiler dependencies: <jitify.hpp>, -std=c++11
  Linker dependencies:   dl cuda nvrtc

  --------------------------------------
  Embedding source files into executable
  --------------------------------------
  g++  ... -ldl -rdynamic
  -Wl,-b,binary,my_kernel.cu,include/my_header.cuh,-b,default nvcc ... -ldl
  -Xcompiler "-rdynamic
  -Wl\,-b\,binary\,my_kernel.cu\,include/my_header.cuh\,-b\,default"
  JITIFY_INCLUDE_EMBEDDED_FILE(my_kernel_cu);
  JITIFY_INCLUDE_EMBEDDED_FILE(include_my_header_cuh);

  ----
  TODO
  ----
  Extract valid compile options and pass the rest to cuModuleLoadDataEx
  See if can have stringified headers automatically looked-up
    by having stringify add them to a (static) global map.
    The global map can be updated by creating a static class instance
      whose constructor performs the registration.
    Can then remove all headers from JitCache constructor in example code
  See other TODOs in code
*/

/*! \file jitify.hpp
 *  \brief The Jitify library header
 */

/*! \mainpage Jitify - A C++ library that simplifies the use of NVRTC
 *  \p Use class jitify::JitCache to manage and launch JIT-compiled CUDA
 *    kernels.
 *
 *  \p Use namespace jitify::reflection to reflect types and values into
 *    code-strings.
 *
 *  \p Use JITIFY_INCLUDE_EMBEDDED_FILE() to declare files that have been
 *  embedded into the executable using the GCC linker.
 *
 *  \p Use jitify::parallel_for and JITIFY_LAMBDA() to generate and launch
 *  simple kernels.
 */

#pragma once

#ifndef JITIFY_THREAD_SAFE
#define JITIFY_THREAD_SAFE 1
#endif

// WAR for MSVC not correctly defining __cplusplus (before MSVC 2017)
#ifdef _MSVC_LANG
#pragma push_macro("__cplusplus")
#undef __cplusplus
#define __cplusplus _MSVC_LANG
#endif

#if defined(_WIN32) || defined(_WIN64)
// WAR for strtok_r being called strtok_s on Windows
#pragma push_macro("strtok_r")
#undef strtok_r
#define strtok_r strtok_s
#endif

#if !DISABLE_DLFCN
#include <dlfcn.h>
#endif
#include <stdint.h>
#include <algorithm>
#include <cstring>  // For strtok_r etc.
#include <deque>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <vector>
#if JITIFY_THREAD_SAFE
#include <mutex>
#endif

#include <cuda.h>
#include <cuda_runtime_api.h>  // For dim3, cudaStream_t
#if CUDA_VERSION >= 8000
#define NVRTC_GET_TYPE_NAME 1
#endif
#include "rgy_nvrtc.h"

#ifndef JITIFY_PRINT_LOG
#define JITIFY_PRINT_LOG 1
#endif

#if JITIFY_PRINT_ALL
#define JITIFY_PRINT_INSTANTIATION 1
#define JITIFY_PRINT_SOURCE 1
#define JITIFY_PRINT_LOG 1
#define JITIFY_PRINT_PTX 1
#define JITIFY_PRINT_LAUNCH 1
#endif

#define JITIFY_FORCE_UNDEFINED_SYMBOL(x) void* x##_forced = (void*)&x
/*! Include a source file that has been embedded into the executable using the
 *    GCC linker.
 * \param name The name of the source file (<b>not</b> as a string), which must
 * be sanitized by replacing non-alpha-numeric characters with underscores.
 * E.g., \code{.cpp}JITIFY_INCLUDE_EMBEDDED_FILE(my_header_h)\endcode will
 * include the embedded file "my_header.h".
 * \note Files declared with this macro * can be referenced using
 * their original (unsanitized) filenames when creating * a \p
 * jitify::Program instance.
 */
#define JITIFY_INCLUDE_EMBEDDED_FILE(name)                                \
  extern "C" uint8_t _jitify_binary_##name##_start[] asm("_binary_" #name \
                                                         "_start");       \
  extern "C" uint8_t _jitify_binary_##name##_end[] asm("_binary_" #name   \
                                                       "_end");           \
  JITIFY_FORCE_UNDEFINED_SYMBOL(_jitify_binary_##name##_start);           \
  JITIFY_FORCE_UNDEFINED_SYMBOL(_jitify_binary_##name##_end)

/*! Jitify library namespace
 */
namespace jitify {

/*! Source-file load callback.
 *
 *  \param filename The name of the requested source file.
 *  \param tmp_stream A temporary stream that can be used to hold source code.
 *  \return A pointer to an input stream containing the source code, or NULL
 *  to defer loading of the file to Jitify's file-loading mechanisms.
 */
typedef std::istream* (*file_callback_type)(std::string filename,
                                            std::iostream& tmp_stream);

// Exclude from Doxygen
//! \cond

class JitCache;

// Simple cache using LRU discard policy
template <typename KeyType, typename ValueType>
class ObjectCache {
 public:
  typedef KeyType key_type;
  typedef ValueType value_type;

 private:
  typedef std::map<key_type, value_type> object_map;
  typedef std::deque<key_type> key_rank;
  typedef typename key_rank::iterator rank_iterator;
  object_map _objects;
  key_rank _ranked_keys;
  size_t _capacity;

  inline void discard_old(size_t n = 0) {
    if (n > _capacity) {
      throw std::runtime_error("Insufficient capacity in cache");
    }
    while (_objects.size() > _capacity - n) {
      key_type discard_key = _ranked_keys.back();
      _ranked_keys.pop_back();
      _objects.erase(discard_key);
    }
  }

 public:
  inline ObjectCache(size_t capacity = 8) : _capacity(capacity) {}
  inline void resize(size_t capacity) {
    _capacity = capacity;
    this->discard_old();
  }
  inline bool contains(const key_type& k) const { return _objects.count(k); }
  inline void touch(const key_type& k) {
    if (!this->contains(k)) {
      throw std::runtime_error("Key not found in cache");
    }
    rank_iterator rank = std::find(_ranked_keys.begin(), _ranked_keys.end(), k);
    if (rank != _ranked_keys.begin()) {
      // Move key to front of ranks
      _ranked_keys.erase(rank);
      _ranked_keys.push_front(k);
    }
  }
  inline value_type& get(const key_type& k) {
    if (!this->contains(k)) {
      throw std::runtime_error("Key not found in cache");
    }
    this->touch(k);
    return _objects[k];
  }
  inline value_type& insert(const key_type& k,
                            const value_type& v = value_type()) {
    this->discard_old(1);
    _ranked_keys.push_front(k);
    return _objects.insert(std::make_pair(k, v)).first->second;
  }
  template <typename... Args>
  inline value_type& emplace(const key_type& k, Args&&... args) {
    this->discard_old(1);
    // Note: Use of piecewise_construct allows non-movable non-copyable types
    auto iter = _objects
                    .emplace(std::piecewise_construct, std::forward_as_tuple(k),
                             std::forward_as_tuple(args...))
                    .first;
    _ranked_keys.push_front(iter->first);
    return iter->second;
  }
};

namespace detail {

// Convenience wrapper for std::vector that provides handy constructors
template <typename T>
class vector : public std::vector<T> {
  typedef std::vector<T> super_type;

 public:
  vector() : super_type() {}
  vector(size_t n) : super_type(n) {}  // Note: Not explicit, allows =0
  vector(std::vector<T> const& vals) : super_type(vals) {}
  template <int N>
  vector(T const (&vals)[N]) : super_type(vals, vals + N) {}
#if defined __cplusplus && __cplusplus >= 201103L
  vector(std::vector<T>&& vals) : super_type(vals) {}
  vector(std::initializer_list<T> vals) : super_type(vals) {}
#endif
};

// Helper functions for parsing/manipulating source code

inline std::string replace_characters(std::string str,
                                      std::string const& oldchars,
                                      char newchar) {
  size_t i = str.find_first_of(oldchars);
  while (i != std::string::npos) {
    str[i] = newchar;
    i = str.find_first_of(oldchars, i + 1);
  }
  return str;
}
inline std::string sanitize_filename(std::string name) {
  return replace_characters(name, "/\\.-: ?%*|\"<>", '_');
}
#if !DISABLE_DLFCN
class EmbeddedData {
  void* _app;
  EmbeddedData(EmbeddedData const&);
  EmbeddedData& operator=(EmbeddedData const&);

 public:
  EmbeddedData() {
    _app = dlopen(NULL, RTLD_LAZY);
    if (!_app) {
      throw std::runtime_error(std::string("dlopen failed: ") + dlerror());
    }
    dlerror();  // Clear any existing error
  }
  ~EmbeddedData() {
    if (_app) {
      dlclose(_app);
    }
  }
  const uint8_t* operator[](std::string key) const {
    key = sanitize_filename(key);
    key = "_binary_" + key;
    uint8_t const* data = (uint8_t const*)dlsym(_app, key.c_str());
    if (!data) {
      throw std::runtime_error(std::string("dlsym failed: ") + dlerror());
    }
    return data;
  }
  const uint8_t* begin(std::string key) const {
    return (*this)[key + "_start"];
  }
  const uint8_t* end(std::string key) const { return (*this)[key + "_end"]; }
};
#endif
inline bool is_tokenchar(char c) {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
         (c >= '0' && c <= '9') || c == '_';
}
inline std::string replace_token(std::string src, std::string token,
                                 std::string replacement) {
  size_t i = src.find(token);
  while (i != std::string::npos) {
    if (i == 0 || i == src.size() - token.size() ||
        (!is_tokenchar(src[i - 1]) && !is_tokenchar(src[i + token.size()]))) {
      src.replace(i, token.size(), replacement);
      i += replacement.size();
    } else {
      i += token.size();
    }
    i = src.find(token, i);
  }
  return src;
}
inline std::string path_base(std::string p) {
  // "/usr/local/myfile.dat" -> "/usr/local"
  // "foo/bar"  -> "foo"
  // "foo/bar/" -> "foo/bar"
#if defined _WIN32 || defined _WIN64
  char sep = '\\';
#else
  char sep = '/';
#endif
  size_t i = p.find_last_of(sep);
  if (i != std::string::npos) {
    return p.substr(0, i);
  } else {
    return "";
  }
}
inline std::string path_join(std::string p1, std::string p2) {
#ifdef _WIN32
  char sep = '\\';
#else
  char sep = '/';
#endif
  if (p1.size() && p2.size() && p2[0] == sep) {
    throw std::invalid_argument("Cannot join to absolute path");
  }
  if (p1.size() && p1[p1.size() - 1] != sep) {
    p1 += sep;
  }
  return p1 + p2;
}
inline unsigned long long hash_larson64(const char* s,
                                        unsigned long long seed = 0) {
  unsigned long long hash = seed;
  while (*s) {
    hash = hash * 101 + *s++;
  }
  return hash;
}

inline uint64_t hash_combine(uint64_t a, uint64_t b) {
  // Note: The magic number comes from the golden ratio
  return a ^ (0x9E3779B97F4A7C17ull + b + (b >> 2) + (a << 6));
}

inline bool extract_include_info_from_compile_error(std::string log,
                                                    std::string& name,
                                                    std::string& parent,
                                                    int& line_num) {
  static const std::string pattern = "cannot open source file \"";
  size_t beg = log.find(pattern);
  if (beg == std::string::npos) {
    return false;
  }
  beg += pattern.size();
  size_t end = log.find("\"", beg);
  name = log.substr(beg, end - beg);

  size_t line_beg = log.rfind("\n", beg);
  if (line_beg == std::string::npos) {
    line_beg = 0;
  } else {
    line_beg += 1;
  }

  size_t split = log.find("(", line_beg);
  parent = log.substr(line_beg, split - line_beg);
  line_num = atoi(
      log.substr(split + 1, log.find(")", split + 1) - (split + 1)).c_str());
  return true;
}

inline std::string comment_out_code_line(int line_num, std::string source) {
  size_t beg = 0;
  for (int i = 1; i < line_num; ++i) {
    beg = source.find("\n", beg) + 1;
  }
  return (source.substr(0, beg) + "//" + source.substr(beg));
}

inline std::string print_with_line_numbers(std::string const& source) {
  int linenum = 1;
  std::stringstream temp;
  std::stringstream source_ss(source);
  for (std::string line; std::getline(source_ss, line); ++linenum) {
      temp << std::setfill(' ') << std::setw(3) << linenum << " " << line
              << std::endl;
  }
  return temp.str();
}

inline std::string print_compile_log(std::string program_name,
                              std::string const& log) {
  std::stringstream temp;
  temp << "---------------------------------------------------"
       << std::endl;
  temp << "--- JIT compile log for " << program_name << " ---"
       << std::endl;
  temp << "---------------------------------------------------"
       << std::endl;
  temp << log << std::endl;
  temp << "---------------------------------------------------"
       << std::endl;
  return temp.str();
}

inline std::vector<std::string> split_string(std::string str,
                                             long maxsplit = -1,
                                             std::string delims = " \t") {
  std::vector<std::string> results;
  if (maxsplit == 0) {
    results.push_back(str);
    return results;
  }
  // Note: +1 to include NULL-terminator
  std::vector<char> v_str(str.c_str(), str.c_str() + (str.size() + 1));
  char* c_str = v_str.data();
  char* saveptr = c_str;
  char* token = nullptr;
  for (long i = 0; i != maxsplit; ++i) {
    token = ::strtok_r(c_str, delims.c_str(), &saveptr);
    c_str = 0;
    if (!token) {
      return results;
    }
    results.push_back(token);
  }
  // Check if there's a final piece
  token += ::strlen(token) + 1;
  if (token - v_str.data() < (ptrdiff_t)str.size()) {
    // Find the start of the final piece
    token += ::strspn(token, delims.c_str());
    if (*token) {
      results.push_back(token);
    }
  }
  return results;
}

inline bool load_source(
    std::string filename, std::map<std::string, std::string>& sources,
    std::string current_dir = "",
    std::vector<std::string> include_paths = std::vector<std::string>(),
    file_callback_type file_callback = 0, bool remove_missing_headers = true) {
  std::istream* source_stream = 0;
  std::stringstream string_stream;
  std::ifstream file_stream;
  // First detect direct source-code string ("my_program\nprogram_code...")
  size_t newline_pos = filename.find("\n");
  if (newline_pos != std::string::npos) {
    std::string source = filename.substr(newline_pos + 1);
    filename = filename.substr(0, newline_pos);
    string_stream << source;
    source_stream = &string_stream;
  }
  if (sources.count(filename)) {
    // Already got this one
    return true;
  }
  if (!source_stream) {
    std::string fullpath = path_join(current_dir, filename);
    // Try loading from callback
    if (!file_callback ||
        !(source_stream = file_callback(fullpath, string_stream))) {
      // Try loading as embedded file
#if !DISABLE_DLFCN
      EmbeddedData embedded;
      std::string source;
      try {
        source.assign(embedded.begin(fullpath), embedded.end(fullpath));
        string_stream << source;
        source_stream = &string_stream;
      } catch (std::runtime_error) {
#endif
        // Finally, try loading from filesystem
        file_stream.open(fullpath.c_str());
        if (!file_stream) {
          bool found_file = false;
          for (int i = 0; i < (int)include_paths.size(); ++i) {
            fullpath = path_join(include_paths[i], filename);
            file_stream.open(fullpath.c_str());
            if (file_stream) {
              found_file = true;
              break;
            }
          }
          if (!found_file) {
            return false;
          }
        }
        source_stream = &file_stream;
#if !DISABLE_DLFCN
      }
#endif
    }
  }
  sources[filename] = std::string();
  std::string& source = sources[filename];
  std::string line;
  size_t linenum = 0;
  unsigned long long hash = 0;
  bool pragma_once = false;
  bool remove_next_blank_line = false;
  while (std::getline(*source_stream, line)) {
    ++linenum;

    // HACK WAR for static variables not allowed on the device (unless
    // __shared__)
    // TODO: This breaks static member variables
    // line = replace_token(line, "static const", "/*static*/ const");

    // TODO: Need to watch out for /* */ comments too
    std::string cleanline =
        line.substr(0, line.find("//"));  // Strip line comments
    // if( cleanline.back() == "\r" ) { // Remove Windows line ending
    //	cleanline = cleanline.substr(0, cleanline.size()-1);
    //}
    // TODO: Should trim whitespace before checking .empty()
    if (cleanline.empty() && remove_next_blank_line) {
      remove_next_blank_line = false;
      continue;
    }
    // Maintain a file hash for use in #pragma once WAR
    hash = hash_larson64(line.c_str(), hash);
    if (cleanline.find("#pragma once") != std::string::npos) {
      pragma_once = true;
      // Note: This is an attempt to recover the original line numbering,
      //         which otherwise gets off-by-one due to the include guard.
      remove_next_blank_line = true;
      // line = "//" + line; // Comment out the #pragma once line
      continue;
    }

    // HACK WAR for Thrust using "#define FOO #pragma bar"
    size_t pragma_beg = cleanline.find("#pragma ");
    if (pragma_beg != std::string::npos) {
      std::string line_after_pragma = line.substr(pragma_beg);
      std::vector<std::string> pragma_split =
          split_string(line_after_pragma, 2);
      line =
          (line.substr(0, pragma_beg) + "_Pragma(\"" + pragma_split[1] + "\")");
      if (pragma_split.size() == 3) {
        line += " " + pragma_split[2];
      }
    }

    source += line + "\n";
  }
  // HACK TESTING (WAR for cub)
  // source = "#define cudaDeviceSynchronize() cudaSuccess\n" + source;
  ////source = "cudaError_t cudaDeviceSynchronize() { return cudaSuccess; }\n" +
  /// source;

  // WAR for #pragma once causing problems when there are multiple inclusions
  //   of the same header from different paths.
  if (pragma_once) {
    std::stringstream ss;
    ss << std::uppercase << std::hex << std::setw(8) << std::setfill('0')
       << hash;
    std::string include_guard_name = "_JITIFY_INCLUDE_GUARD_" + ss.str() + "\n";
    std::string include_guard_header;
    include_guard_header += "#ifndef " + include_guard_name;
    include_guard_header += "#define " + include_guard_name;
    std::string include_guard_footer;
    include_guard_footer += "#endif // " + include_guard_name;
    source = include_guard_header + source + "\n" + include_guard_footer;
  }
  // return filename;
  return true;
}

}  // namespace detail

//! \endcond

/*! Jitify reflection utilities namespace
 */
namespace reflection {

//  Provides type and value reflection via a function 'reflect':
//    reflect<Type>()   -> "Type"
//    reflect(value)    -> "(T)value"
//    reflect<VAL>()    -> "VAL"
//    reflect<Type,VAL> -> "VAL"
//    reflect_template<float,NonType<int,7>,char>() -> "<float,7,char>"
//    reflect_template({"float", "7", "char"}) -> "<float,7,char>"

/*! A wrapper class for non-type template parameters.
 */
template <typename T, T VALUE_>
struct NonType {
#if defined __cplusplus && __cplusplus >= 201103L
  constexpr
#endif
      static T VALUE = VALUE_;
};

// Forward declaration
template <typename T>
inline std::string reflect(T const& value);

//! \cond

namespace detail {

template <typename T>
inline std::string value_string(const T& x) {
  std::stringstream ss;
  ss << x;
  return ss.str();
}
// WAR for non-printable characters
template <>
inline std::string value_string<char>(const char& x) {
  std::stringstream ss;
  ss << (int)x;
  return ss.str();
}
template <>
inline std::string value_string<signed char>(const signed char& x) {
  std::stringstream ss;
  ss << (int)x;
  return ss.str();
}
template <>
inline std::string value_string<unsigned char>(const unsigned char& x) {
  std::stringstream ss;
  ss << (int)x;
  return ss.str();
}
template <>
inline std::string value_string<wchar_t>(const wchar_t& x) {
  std::stringstream ss;
  ss << (long)x;
  return ss.str();
}
// Specialisation for bool true/false literals
template <>
inline std::string value_string<bool>(const bool& x) {
  return x ? "true" : "false";
}

//#if CUDA_VERSION < 8000
#ifdef _MSC_VER  // MSVC compiler
inline std::string demangle(const char* verbose_name) {
  // Strips annotations from the verbose name returned by typeid(X).name()
  std::string result = verbose_name;
  result = jitify::detail::replace_token(result, "__ptr64", "");
  result = jitify::detail::replace_token(result, "__cdecl", "");
  result = jitify::detail::replace_token(result, "class", "");
  result = jitify::detail::replace_token(result, "struct", "");
  return result;
}
#else  // not MSVC
#include <cxxabi.h>
inline std::string demangle(const char* mangled_name) {
  size_t bufsize = 1024;
  auto buf = std::unique_ptr<char, decltype(free)*>(
      reinterpret_cast<char*>(malloc(bufsize)), free);
  std::string demangled_name;
  int status;
  char* demangled_ptr =
      abi::__cxa_demangle(mangled_name, buf.get(), &bufsize, &status);
  if (status == 0) {
    demangled_name = demangled_ptr;  // all worked as expected
  } else if (status == -2) {
    demangled_name = mangled_name;  // we interpret this as plain C name
  } else if (status == -1) {
    throw std::runtime_error(
        std::string("memory allocation failure in __cxa_demangle"));
  } else if (status == -3) {
    throw std::runtime_error(std::string("invalid argument to __cxa_demangle"));
  }
  return demangled_name;
}
#endif  // not MSVC
//#endif // CUDA_VERSION < 8000

template <typename T>
struct type_reflection {
  inline static std::string name() {
    //#if CUDA_VERSION < 8000
    // WAR for typeid discarding cv qualifiers on value-types
    // We use a pointer type to maintain cv qualifiers, then strip out the '*'
    std::string no_cv_name = demangle(typeid(T).name());
    std::string ptr_name = demangle(typeid(T*).name());
    // Find the right '*' by diffing the type name and ptr name
    // Note that the '*' will also be prefixed with the cv qualifiers
    size_t diff_begin =
        std::mismatch(no_cv_name.begin(), no_cv_name.end(), ptr_name.begin())
            .first -
        no_cv_name.begin();
    size_t star_begin = ptr_name.find("*", diff_begin);
    if (star_begin == std::string::npos) {
      throw std::runtime_error("Type reflection failed: " + ptr_name);
    }
    std::string name =
        ptr_name.substr(0, star_begin) + ptr_name.substr(star_begin + 1);
    return name;
    //#else
    //		std::string ret;
    //		nvrtcResult status = nvrtcGetTypeName<T>(&ret);
    //		if( status != NVRTC_SUCCESS ) {
    //			throw std::runtime_error(std::string("nvrtcGetTypeName
    // failed:
    //")+ nvrtcGetErrorString(status));
    //		}
    //		return ret;
    //#endif
  }
};
template <typename T, T VALUE>
struct type_reflection<NonType<T, VALUE> > {
  inline static std::string name() {
    return jitify::reflection::reflect(VALUE);
  }
};

}  // namespace detail

//! \endcond

/*! Create an Instance object that contains a const reference to the
 *  value.  We use this to wrap abstract objects from which we want to extract
 *  their type at runtime (e.g., derived type).  This is used to facilitate
 *  templating on derived type when all we know at compile time is abstract
 * type.
 */
template <typename T>
struct Instance {
  const T& value;
  Instance(const T& value) : value(value) {}
};

/*! Create an Instance object from which we can extract the value's run-time
 * type.
 *  \param value The const value to be captured.
 */
template <typename T>
inline Instance<T const> instance_of(T const& value) {
  return Instance<T const>(value);
}

/*! A wrapper used for representing types as values.
 */
template <typename T>
struct Type {};

// Type reflection
// E.g., reflect<float>() -> "float"
// Note: This strips trailing const and volatile qualifiers
/*! Generate a code-string for a type.
 *  \code{.cpp}reflect<float>() --> "float"\endcode
 */
template <typename T>
inline std::string reflect() {
  return detail::type_reflection<T>::name();
}
// Value reflection
// E.g., reflect(3.14f) -> "(float)3.14"
/*! Generate a code-string for a value.
 *  \code{.cpp}reflect(3.14f) --> "(float)3.14"\endcode
 */
template <typename T>
inline std::string reflect(T const& value) {
  return "(" + reflect<T>() + ")" + detail::value_string(value);
}
// Non-type template arg reflection (implicit conversion to int64_t)
// E.g., reflect<7>() -> "(int64_t)7"
/*! Generate a code-string for an integer non-type template argument.
 *  \code{.cpp}reflect<7>() --> "(int64_t)7"\endcode
 */
template <long long N>
inline std::string reflect() {
  return reflect<NonType<int, N> >();
}
// Non-type template arg reflection (explicit type)
// E.g., reflect<int,7>() -> "(int)7"
/*! Generate a code-string for a generic non-type template argument.
 *  \code{.cpp} reflect<int,7>() --> "(int)7" \endcode
 */
template <typename T, T N>
inline std::string reflect() {
  return reflect<NonType<T, N> >();
}
// Type reflection via value
// E.g., reflect(Type<float>()) -> "float"
/*! Generate a code-string for a type wrapped as a Type instance.
 *  \code{.cpp}reflect(Type<float>()) --> "float"\endcode
 */
template <typename T>
inline std::string reflect(jitify::reflection::Type<T>) {
  return reflect<T>();
}

/*! Generate a code-string for a type wrapped as an Instance instance.
 *  \code{.cpp}reflect(Instance<float>(3.1f)) --> "float"\endcode
 *  or more simply when passed to a instance_of helper
 *  \code{.cpp}reflect(instance_of(3.1f)) --> "float"\endcodei
 *  This is specifically for the case where we want to extract the run-time
 * type, e.g., derived type, of an object pointer.
 */
template <typename T>
inline std::string reflect(jitify::reflection::Instance<T>& value) {
  return detail::demangle(typeid(value.value).name());
}

// Type from value
// E.g., type_of(3.14f) -> Type<float>()
/*! Create a Type object representing a value's type.
 *  \param value The value whose type is to be captured.
 */
template <typename T>
inline Type<T> type_of(T& value) {
  return Type<T>();
}
/*! Create a Type object representing a value's type.
 *  \param value The const value whose type is to be captured.
 */
template <typename T>
inline Type<T const> type_of(T const& value) {
  return Type<T const>();
}

#if __cplusplus >= 201103L
// Multiple value reflections one call, returning list of strings
template <typename... Args>
inline std::vector<std::string> reflect_all(Args... args) {
  return {reflect(args)...};
}
#endif  // __cplusplus >= 201103L

inline std::string reflect_list(jitify::detail::vector<std::string> const& args,
                                std::string opener = "",
                                std::string closer = "") {
  std::stringstream ss;
  ss << opener;
  for (int i = 0; i < (int)args.size(); ++i) {
    if (i > 0) ss << ",";
    ss << args[i];
  }
  ss << closer;
  return ss.str();
}

// Template instantiation reflection
// inline std::string reflect_template(std::vector<std::string> const& args) {
inline std::string reflect_template(
    jitify::detail::vector<std::string> const& args) {
  // Note: The space in " >" is a WAR to avoid '>>' appearing
  return reflect_list(args, "<", " >");
}
#if __cplusplus >= 201103L
// TODO: See if can make this evaluate completely at compile-time
template <typename... Ts>
inline std::string reflect_template() {
  return reflect_template({reflect<Ts>()...});
  // return reflect_template<sizeof...(Ts)>({reflect<Ts>()...});
}
#endif

}  // namespace reflection

//! \cond

namespace detail {

class CUDAKernel {
  std::vector<std::string> _link_files;
  std::vector<std::string> _link_paths;
  CUlinkState _link_state;
  CUmodule _module;
  CUfunction _kernel;
  std::string _func_name;
  std::string _ptx;
  std::vector<CUjit_option> _opts;

  inline void cuda_safe_call(CUresult res) const {
    if (res != CUDA_SUCCESS) {
      const char* msg;
      cuGetErrorName(res, &msg);
      throw std::runtime_error(msg);
    }
  }
  inline void create_module(std::vector<std::string> link_files,
                            std::vector<std::string> link_paths,
                            void** optvals = 0) {
    if (link_files.empty()) {
      cuda_safe_call(cuModuleLoadDataEx(&_module, _ptx.c_str(), _opts.size(),
                                        _opts.data(), optvals));
    } else {
      cuda_safe_call(
          cuLinkCreate(_opts.size(), _opts.data(), optvals, &_link_state));
      cuda_safe_call(cuLinkAddData(_link_state, CU_JIT_INPUT_PTX,
                                   (void*)_ptx.c_str(), _ptx.size(),
                                   "jitified_source.ptx", 0, 0, 0));
      for (int i = 0; i < (int)link_files.size(); ++i) {
        std::string link_file = link_files[i];
#if defined _WIN32 || defined _WIN64
        link_file = link_file + ".lib";
#else
        link_file = "lib" + link_file + ".a";
#endif
        CUresult result = cuLinkAddFile(_link_state, CU_JIT_INPUT_LIBRARY,
                                        link_file.c_str(), 0, 0, 0);
        int path_num = 0;
        while (result == CUDA_ERROR_FILE_NOT_FOUND &&
               path_num < (int)link_paths.size()) {
          std::string filename = path_join(link_paths[path_num++], link_file);
          result = cuLinkAddFile(_link_state, CU_JIT_INPUT_LIBRARY,
                                 filename.c_str(), 0, 0, 0);
        }
#if JITIFY_PRINT_LOG
        if (result == CUDA_ERROR_FILE_NOT_FOUND) {
          std::cout << "Error: Device library not found: " << link_file
                    << std::endl;
        }
#endif
        cuda_safe_call(result);
      }
      size_t cubin_size;
      void* cubin;
      cuda_safe_call(cuLinkComplete(_link_state, &cubin, &cubin_size));
      cuda_safe_call(cuModuleLoadData(&_module, cubin));
    }
    cuda_safe_call(cuModuleGetFunction(&_kernel, _module, _func_name.c_str()));
  }
  inline void destroy_module() {
    if (_link_state) {
      cuda_safe_call(cuLinkDestroy(_link_state));
    }
    _link_state = 0;
    if (_module) {
      cuModuleUnload(_module);
    }
    _module = 0;
  }

 public:
  inline CUDAKernel() : _link_state(0), _module(0), _kernel(0) {}
  inline CUDAKernel(const CUDAKernel& other) = delete;
  inline CUDAKernel& operator=(const CUDAKernel& other) = delete;
  inline CUDAKernel(CUDAKernel&& other) = delete;
  inline CUDAKernel& operator=(CUDAKernel&& other) = delete;
  inline CUDAKernel(const char* func_name, const char* ptx,
                    std::vector<std::string> link_files,
                    std::vector<std::string> link_paths, unsigned int nopts = 0,
                    CUjit_option* opts = 0, void** optvals = 0)
      : _link_files(link_files),
        _link_paths(link_paths),
        _link_state(0),
        _module(0),
        _kernel(0),
        _func_name(func_name),
        _ptx(ptx),
        _opts(opts, opts + nopts) {
    this->create_module(link_files, link_paths, optvals);
  }
  inline CUDAKernel& set(const char* func_name, const char* ptx,
                         std::vector<std::string> link_files,
                         std::vector<std::string> link_paths,
                         unsigned int nopts = 0, CUjit_option* opts = 0,
                         void** optvals = 0) {
    this->destroy_module();
    _func_name = func_name;
    _ptx = ptx;
    _link_files = link_files;
    _link_paths = link_paths;
    _opts.assign(opts, opts + nopts);
    this->create_module(link_files, link_paths, optvals);
    return *this;
  }
  inline ~CUDAKernel() { this->destroy_module(); }
  inline operator CUfunction() const { return _kernel; }

  inline CUresult launch(dim3 grid, dim3 block, unsigned int smem,
                         CUstream stream, std::vector<void*> arg_ptrs) const {
    return cuLaunchKernel(_kernel, grid.x, grid.y, grid.z, block.x, block.y,
                          block.z, smem, stream, arg_ptrs.data(), NULL);
  }

  std::string function_name() const { return _func_name; }
  std::string ptx() const { return _ptx; }
  std::vector<std::string> link_files() const { return _link_files; }
  std::vector<std::string> link_paths() const { return _link_paths; }
};

static const char* jitsafe_header_preinclude_h = R"(
// WAR for Thrust (which appears to have forgotten to include this in result_of_adaptable_function.h
#include <type_traits>

// WAR for Thrust (which appear to have forgotten to include this in error_code.h)
#include <string>

// WAR for Thrust (which only supports gnuc, clang or msvc)
#define __GNUC__ 4

// WAR for generics/shfl.h
#define THRUST_STATIC_ASSERT(x)

// WAR for CUB
#ifdef __host__
#undef __host__
#endif
#define __host__

// WAR to allow exceptions to be parsed
#define try
#define catch(...)
)";

static const char* jitsafe_header_float_h =
    "#pragma once\n"
    "\n"
    "inline __host__ __device__ float  jitify_int_as_float(int i)             "
    "{ union FloatInt { float f; int i; } fi; fi.i = i; return fi.f; }\n"
    "inline __host__ __device__ double jitify_longlong_as_double(long long i) "
    "{ union DoubleLongLong { double f; long long i; } fi; fi.i = i; return "
    "fi.f; }\n"
    "#define FLT_RADIX       2\n"
    "#define FLT_MANT_DIG    24\n"
    "#define DBL_MANT_DIG    53\n"
    "#define FLT_DIG         6\n"
    "#define DBL_DIG         15\n"
    "#define FLT_MIN_EXP     -125\n"
    "#define DBL_MIN_EXP     -1021\n"
    "#define FLT_MIN_10_EXP  -37\n"
    "#define DBL_MIN_10_EXP  -307\n"
    "#define FLT_MAX_EXP     128\n"
    "#define DBL_MAX_EXP     1024\n"
    "#define FLT_MAX_10_EXP  38\n"
    "#define DBL_MAX_10_EXP  308\n"
    "#define FLT_MAX         jitify_int_as_float(2139095039)\n"
    "#define DBL_MAX         jitify_longlong_as_double(9218868437227405311)\n"
    "#define FLT_EPSILON     jitify_int_as_float(872415232)\n"
    "#define DBL_EPSILON     jitify_longlong_as_double(4372995238176751616)\n"
    "#define FLT_MIN         jitify_int_as_float(8388608)\n"
    "#define DBL_MIN         jitify_longlong_as_double(4503599627370496)\n"
    "#define FLT_ROUNDS      1\n"
    "#if defined __cplusplus && __cplusplus >= 201103L\n"
    "#define FLT_EVAL_METHOD 0\n"
    "#define DECIMAL_DIG     21\n"
    "#endif\n";

static const char* jitsafe_header_limits_h =
    "#pragma once\n"
    "\n"
    "#if defined _WIN32 || defined _WIN64\n"
    " #define __WORDSIZE 32\n"
    "#else\n"
    " #if defined __x86_64__ && !defined __ILP32__\n"
    "  #define __WORDSIZE 64\n"
    " #else\n"
    "  #define __WORDSIZE 32\n"
    " #endif\n"
    "#endif\n"
    "#define MB_LEN_MAX  16\n"
    "#define CHAR_BIT    8\n"
    "#define SCHAR_MIN   (-128)\n"
    "#define SCHAR_MAX   127\n"
    "#define UCHAR_MAX   255\n"
    "#ifdef __CHAR_UNSIGNED__\n"
    " #define CHAR_MIN   0\n"
    " #define CHAR_MAX   UCHAR_MAX\n"
    "#else\n"
    " #define CHAR_MIN   SCHAR_MIN\n"
    " #define CHAR_MAX   SCHAR_MAX\n"
    "#endif\n"
    "#define SHRT_MIN    (-32768)\n"
    "#define SHRT_MAX    32767\n"
    "#define USHRT_MAX   65535\n"
    "#define INT_MIN     (-INT_MAX - 1)\n"
    "#define INT_MAX     2147483647\n"
    "#define UINT_MAX    4294967295U\n"
    "#if __WORDSIZE == 64\n"
    " # define LONG_MAX  9223372036854775807L\n"
    "#else\n"
    " # define LONG_MAX  2147483647L\n"
    "#endif\n"
    "#define LONG_MIN    (-LONG_MAX - 1L)\n"
    "#if __WORDSIZE == 64\n"
    " #define ULONG_MAX  18446744073709551615UL\n"
    "#else\n"
    " #define ULONG_MAX  4294967295UL\n"
    "#endif\n"
    "#define LLONG_MAX  9223372036854775807LL\n"
    "#define LLONG_MIN  (-LLONG_MAX - 1LL)\n"
    "#define ULLONG_MAX 18446744073709551615ULL\n";

static const char* jitsafe_header_iterator =
    "#pragma once\n"
    "\n"
    "namespace __jitify_iterator_ns {\n"
    "struct output_iterator_tag {};\n"
    "struct input_iterator_tag {};\n"
    "struct forward_iterator_tag {};\n"
    "struct bidirectional_iterator_tag {};\n"
    "struct random_access_iterator_tag {};\n"
    "template<class Iterator>\n"
    "struct iterator_traits {\n"
    "  typedef typename Iterator::iterator_category iterator_category;\n"
    "  typedef typename Iterator::value_type        value_type;\n"
    "  typedef typename Iterator::difference_type   difference_type;\n"
    "  typedef typename Iterator::pointer           pointer;\n"
    "  typedef typename Iterator::reference         reference;\n"
    "};\n"
    "template<class T>\n"
    "struct iterator_traits<T*> {\n"
    "  typedef random_access_iterator_tag iterator_category;\n"
    "  typedef T                          value_type;\n"
    "  typedef ptrdiff_t                  difference_type;\n"
    "  typedef T*                         pointer;\n"
    "  typedef T&                         reference;\n"
    "};\n"
    "template<class T>\n"
    "struct iterator_traits<T const*> {\n"
    "  typedef random_access_iterator_tag iterator_category;\n"
    "  typedef T                          value_type;\n"
    "  typedef ptrdiff_t                  difference_type;\n"
    "  typedef T const*                   pointer;\n"
    "  typedef T const&                   reference;\n"
    "};\n"
    "} // namespace __jitify_iterator_ns\n"
    "namespace std { using namespace __jitify_iterator_ns; }\n"
    "using namespace __jitify_iterator_ns;\n";

// TODO: This is incomplete; need floating point limits
static const char* jitsafe_header_limits =
    "#pragma once\n"
    "#include <climits>\n"
    "\n"
    "namespace __jitify_limits_ns {\n"
    "// TODO: Floating-point limits\n"
    "namespace __jitify_detail {\n"
    "template<class T, T Min, T Max, int Digits=-1>\n"
    "struct IntegerLimits {\n"
    "	static inline __host__ __device__ T min() { return Min; }\n"
    "	static inline __host__ __device__ T max() { return Max; }\n"
    "	enum {\n"
    "		digits     = (Digits == -1) ? (int)(sizeof(T)*8 - (Min != 0)) "
    ": Digits,\n"
    "		digits10   = (digits * 30103) / 100000,\n"
    "		is_signed  = ((T)(-1)<0),\n"
    "		is_integer = true,\n"
    "		is_exact   = true,\n"
    "		radix      = 2,\n"
    "		is_bounded = true,\n"
    "		is_modulo  = false\n"
    "	};\n"
    "};\n"
    "} // namespace detail\n"
    "template<typename T> struct numeric_limits {};\n"
    "template<> struct numeric_limits<bool>               : public "
    "__jitify_detail::IntegerLimits<bool,              false,    true,1> {};\n"
    "template<> struct numeric_limits<char>               : public "
    "__jitify_detail::IntegerLimits<char,              CHAR_MIN, CHAR_MAX> "
    "{};\n"
    "template<> struct numeric_limits<signed char>        : public "
    "__jitify_detail::IntegerLimits<signed char,       SCHAR_MIN,SCHAR_MAX> "
    "{};\n"
    "template<> struct numeric_limits<unsigned char>      : public "
    "__jitify_detail::IntegerLimits<unsigned char,     0,        UCHAR_MAX> "
    "{};\n"
    "template<> struct numeric_limits<wchar_t>            : public "
    "__jitify_detail::IntegerLimits<wchar_t,           INT_MIN,  INT_MAX> {};\n"
    "template<> struct numeric_limits<short>              : public "
    "__jitify_detail::IntegerLimits<short,             SHRT_MIN, SHRT_MAX> "
    "{};\n"
    "template<> struct numeric_limits<unsigned short>     : public "
    "__jitify_detail::IntegerLimits<unsigned short,    0,        USHRT_MAX> "
    "{};\n"
    "template<> struct numeric_limits<int>                : public "
    "__jitify_detail::IntegerLimits<int,               INT_MIN,  INT_MAX> {};\n"
    "template<> struct numeric_limits<unsigned int>       : public "
    "__jitify_detail::IntegerLimits<unsigned int,      0,        UINT_MAX> "
    "{};\n"
    "template<> struct numeric_limits<long>               : public "
    "__jitify_detail::IntegerLimits<long,              LONG_MIN, LONG_MAX> "
    "{};\n"
    "template<> struct numeric_limits<unsigned long>      : public "
    "__jitify_detail::IntegerLimits<unsigned long,     0,        ULONG_MAX> "
    "{};\n"
    "template<> struct numeric_limits<long long>          : public "
    "__jitify_detail::IntegerLimits<long long,         LLONG_MIN,LLONG_MAX> "
    "{};\n"
    "template<> struct numeric_limits<unsigned long long> : public "
    "__jitify_detail::IntegerLimits<unsigned long long,0,        ULLONG_MAX> "
    "{};\n"
    "//template<typename T> struct numeric_limits { static const bool "
    "is_signed = ((T)(-1)<0); };\n"
    "} // namespace __jitify_limits_ns\n"
    "namespace std { using namespace __jitify_limits_ns; }\n"
    "using namespace __jitify_limits_ns;\n";

// TODO: This is highly incomplete
static const char* jitsafe_header_type_traits = R"(
    #pragma once
    #if __cplusplus >= 201103L
    namespace __jitify_type_traits_ns {

    template<bool B, class T = void> struct enable_if {};
    template<class T>                struct enable_if<true, T> { typedef T type; };
    #if __cplusplus >= 201402L
    template< bool B, class T = void > using enable_if_t = typename enable_if<B,T>::type;
    #endif

    struct true_type  {
      enum { value = true };
      operator bool() const { return true; }
    };
    struct false_type {
      enum { value = false };
      operator bool() const { return false; }
    };

    template<typename T> struct is_floating_point    : false_type {};
    template<> struct is_floating_point<float>       :  true_type {};
    template<> struct is_floating_point<double>      :  true_type {};
    template<> struct is_floating_point<long double> :  true_type {};

    template<class T> struct is_integral              : false_type {};
    template<> struct is_integral<bool>               :  true_type {};
    template<> struct is_integral<char>               :  true_type {};
    template<> struct is_integral<signed char>        :  true_type {};
    template<> struct is_integral<unsigned char>      :  true_type {};
    template<> struct is_integral<short>              :  true_type {};
    template<> struct is_integral<unsigned short>     :  true_type {};
    template<> struct is_integral<int>                :  true_type {};
    template<> struct is_integral<unsigned int>       :  true_type {};
    template<> struct is_integral<long>               :  true_type {};
    template<> struct is_integral<unsigned long>      :  true_type {};
    template<> struct is_integral<long long>          :  true_type {};
    template<> struct is_integral<unsigned long long> :  true_type {};

    template<typename T> struct is_signed    : false_type {};
    template<> struct is_signed<float>       :  true_type {};
    template<> struct is_signed<double>      :  true_type {};
    template<> struct is_signed<long double> :  true_type {};
    template<> struct is_signed<signed char> :  true_type {};
    template<> struct is_signed<short>       :  true_type {};
    template<> struct is_signed<int>         :  true_type {};
    template<> struct is_signed<long>        :  true_type {};
    template<> struct is_signed<long long>   :  true_type {};

    template<typename T> struct is_unsigned             : false_type {};
    template<> struct is_unsigned<unsigned char>      :  true_type {};
    template<> struct is_unsigned<unsigned short>     :  true_type {};
    template<> struct is_unsigned<unsigned int>       :  true_type {};
    template<> struct is_unsigned<unsigned long>      :  true_type {};
    template<> struct is_unsigned<unsigned long long> :  true_type {};

    template<typename T, typename U> struct is_same      : false_type {};
    template<typename T>             struct is_same<T,T> :  true_type {};

    template<class T> struct is_array : false_type {};
    template<class T> struct is_array<T[]> : true_type {};
    template<class T, size_t N> struct is_array<T[N]> : true_type {};

    //partial implementation only of is_function
    template<class> struct is_function : false_type { };
    template<class Ret, class... Args> struct is_function<Ret(Args...)> : true_type {}; //regular
    template<class Ret, class... Args> struct is_function<Ret(Args......)> : true_type {}; // variadic

    template<class> struct result_of;
    template<class F, typename... Args>
    struct result_of<F(Args...)> {
    // TODO: This is a hack; a proper implem is quite complicated.
    typedef typename F::result_type type;
    };

    template <class T> struct remove_reference { typedef T type; };
    template <class T> struct remove_reference<T&> { typedef T type; };
    template <class T> struct remove_reference<T&&> { typedef T type; };
    #if __cplusplus >= 201402L
    template< class T > using remove_reference_t = typename remove_reference<T>::type;
    #endif

    template<class T> struct remove_extent { typedef T type; };
    template<class T> struct remove_extent<T[]> { typedef T type; };
    template<class T, size_t N> struct remove_extent<T[N]> { typedef T type; };
    #if __cplusplus >= 201402L
    template< class T > using remove_extent_t = typename remove_extent<T>::type;
    #endif

    template< class T > struct remove_const          { typedef T type; };
    template< class T > struct remove_const<const T> { typedef T type; };
    template< class T > struct remove_volatile             { typedef T type; };
    template< class T > struct remove_volatile<volatile T> { typedef T type; };
    template< class T > struct remove_cv { typedef typename remove_volatile<typename remove_const<T>::type>::type type; };
    #if __cplusplus >= 201402L
    template< class T > using remove_cv_t       = typename remove_cv<T>::type;
    template< class T > using remove_const_t    = typename remove_const<T>::type;
    template< class T > using remove_volatile_t = typename remove_volatile<T>::type;
    #endif

    template<bool B, class T, class F> struct conditional { typedef T type; };
    template<class T, class F> struct conditional<false, T, F> { typedef F type; };
    #if __cplusplus >= 201402L
    template< bool B, class T, class F > using conditional_t = typename conditional<B,T,F>::type;
    #endif

    namespace __jitify_detail {
    template< class T, bool is_function_type = false > struct add_pointer { using type = typename remove_reference<T>::type*; };
    template< class T > struct add_pointer<T, true> { using type = T; };
    template< class T, class... Args > struct add_pointer<T(Args...), true> { using type = T(*)(Args...); };
    template< class T, class... Args > struct add_pointer<T(Args..., ...), true> { using type = T(*)(Args..., ...); };
    }
    template< class T > struct add_pointer : __jitify_detail::add_pointer<T, is_function<T>::value> {};
    #if __cplusplus >= 201402L
    template< class T > using add_pointer_t = typename add_pointer<T>::type;
    #endif

    template< class T > struct decay {
    private:
      typedef typename remove_reference<T>::type U;
    public:
      typedef typename conditional<is_array<U>::value, typename remove_extent<U>::type*,
        typename conditional<is_function<U>::value,typename add_pointer<U>::type,typename remove_cv<U>::type
        >::type>::type type;
    };
    #if __cplusplus >= 201402L
    template< class T > using decay_t = typename decay<T>::type;
    #endif

    } // namespace __jtiify_type_traits_ns
    namespace std { using namespace __jitify_type_traits_ns; }
    using namespace __jitify_type_traits_ns;
    #endif // c++11
)";

// TODO: INT_FAST8_MAX et al. and a few other misc constants
static const char* jitsafe_header_stdint_h =
    "#pragma once\n"
    "#include <climits>\n"
    "namespace __jitify_stdint_ns {\n"
    "typedef signed char      int8_t;\n"
    "typedef signed short     int16_t;\n"
    "typedef signed int       int32_t;\n"
    "typedef signed long long int64_t;\n"
    "typedef signed char      int_fast8_t;\n"
    "typedef signed short     int_fast16_t;\n"
    "typedef signed int       int_fast32_t;\n"
    "typedef signed long long int_fast64_t;\n"
    "typedef signed char      int_least8_t;\n"
    "typedef signed short     int_least16_t;\n"
    "typedef signed int       int_least32_t;\n"
    "typedef signed long long int_least64_t;\n"
    "typedef signed long long intmax_t;\n"
    "typedef signed long      intptr_t; //optional\n"
    "typedef unsigned char      uint8_t;\n"
    "typedef unsigned short     uint16_t;\n"
    "typedef unsigned int       uint32_t;\n"
    "typedef unsigned long long uint64_t;\n"
    "typedef unsigned char      uint_fast8_t;\n"
    "typedef unsigned short     uint_fast16_t;\n"
    "typedef unsigned int       uint_fast32_t;\n"
    "typedef unsigned long long uint_fast64_t;\n"
    "typedef unsigned char      uint_least8_t;\n"
    "typedef unsigned short     uint_least16_t;\n"
    "typedef unsigned int       uint_least32_t;\n"
    "typedef unsigned long long uint_least64_t;\n"
    "typedef unsigned long long uintmax_t;\n"
    "typedef unsigned long      uintptr_t; //optional\n"
    "#define INT8_MIN    SCHAR_MIN\n"
    "#define INT16_MIN   SHRT_MIN\n"
    "#define INT32_MIN   INT_MIN\n"
    "#define INT64_MIN   LLONG_MIN\n"
    "#define INT8_MAX    SCHAR_MAX\n"
    "#define INT16_MAX   SHRT_MAX\n"
    "#define INT32_MAX   INT_MAX\n"
    "#define INT64_MAX   LLONG_MAX\n"
    "#define UINT8_MAX   UCHAR_MAX\n"
    "#define UINT16_MAX  USHRT_MAX\n"
    "#define UINT32_MAX  UINT_MAX\n"
    "#define UINT64_MAX  ULLONG_MAX\n"
    "#define INTPTR_MIN  LONG_MIN\n"
    "#define INTMAX_MIN  LLONG_MIN\n"
    "#define INTPTR_MAX  LONG_MAX\n"
    "#define INTMAX_MAX  LLONG_MAX\n"
    "#define UINTPTR_MAX ULONG_MAX\n"
    "#define UINTMAX_MAX ULLONG_MAX\n"
    "#define PTRDIFF_MIN INTPTR_MIN\n"
    "#define PTRDIFF_MAX INTPTR_MAX\n"
    "#define SIZE_MAX    UINT64_MAX\n"
    "} // namespace __jitify_stdint_ns\n"
    "namespace std { using namespace __jitify_stdint_ns; }\n"
    "using namespace __jitify_stdint_ns;\n";

// TODO: offsetof
static const char* jitsafe_header_stddef_h =
    "#pragma once\n"
    "#include <climits>\n"
    "namespace __jitify_stddef_ns {\n"
    //"enum { NULL = 0 };\n"
    "typedef unsigned long size_t;\n"
    "typedef   signed long ptrdiff_t;\n"
    "} // namespace __jitify_stddef_ns\n"
    "namespace std { using namespace __jitify_stddef_ns; }\n"
    "using namespace __jitify_stddef_ns;\n";

static const char* jitsafe_header_stdlib_h =
    "#pragma once\n"
    "#include <stddef.h>\n";
static const char* jitsafe_header_stdio_h =
    "#pragma once\n"
    "#include <stddef.h>\n"
    "#define FILE int\n"
    "int fflush ( FILE * stream );\n"
    "int fprintf ( FILE * stream, const char * format, ... );\n";

static const char* jitsafe_header_string_h =
    "#pragma once\n"
    "char* strcpy ( char * destination, const char * source );\n"
    "int strcmp ( const char * str1, const char * str2 );\n"
    "char* strerror( int errnum );\n";

static const char* jitsafe_header_cstring =
    "#pragma once\n"
    "\n"
    "namespace __jitify_cstring_ns {\n"
    "char* strcpy ( char * destination, const char * source );\n"
    "int strcmp ( const char * str1, const char * str2 );\n"
    "char* strerror( int errnum );\n"
    "} // namespace __jitify_cstring_ns\n"
    "namespace std { using namespace __jitify_cstring_ns; }\n"
    "using namespace __jitify_cstring_ns;\n";

// HACK TESTING (WAR for cub)
static const char* jitsafe_header_iostream =
    "#pragma once\n"
    "#include <ostream>\n"
    "#include <istream>\n";
// HACK TESTING (WAR for Thrust)
static const char* jitsafe_header_ostream =
    "#pragma once\n"
    "\n"
    "namespace __jitify_ostream_ns {\n"
    "template<class CharT,class Traits=void>\n"  // = std::char_traits<CharT>
                                                 // >\n"
    "struct basic_ostream {\n"
    "};\n"
    "typedef basic_ostream<char> ostream;\n"
    "ostream& endl(ostream& os);\n"
    "ostream& operator<<( ostream&, ostream& (*f)( ostream& ) );\n"
    "template< class CharT, class Traits > basic_ostream<CharT, Traits>& endl( "
    "basic_ostream<CharT, Traits>& os );\n"
    "template< class CharT, class Traits > basic_ostream<CharT, Traits>& "
    "operator<<( basic_ostream<CharT,Traits>& os, const char* c );\n"
    "template< class CharT, class Traits, class T > basic_ostream<CharT, "
    "Traits>& operator<<( basic_ostream<CharT,Traits>&& os, const T& value );\n"
    "} // namespace __jitify_ostream_ns\n"
    "namespace std { using namespace __jitify_ostream_ns; }\n"
    "using namespace __jitify_ostream_ns;\n";

static const char* jitsafe_header_istream =
    "#pragma once\n"
    "\n"
    "namespace __jitify_istream_ns {\n"
    "template<class CharT,class Traits=void>\n"  // = std::char_traits<CharT>
                                                 // >\n"
    "struct basic_istream {\n"
    "};\n"
    "typedef basic_istream<char> istream;\n"
    "} // namespace __jitify_istream_ns\n"
    "namespace std { using namespace __jitify_istream_ns; }\n"
    "using namespace __jitify_istream_ns;\n";

static const char* jitsafe_header_sstream =
    "#pragma once\n"
    "#include <ostream>\n"
    "#include <istream>\n";

static const char* jitsafe_header_utility =
    "#pragma once\n"
    "namespace __jitify_utility_ns {\n"
    "template<class T1, class T2>\n"
    "struct pair {\n"
    "	T1 first;\n"
    "	T2 second;\n"
    "	inline pair() {}\n"
    "	inline pair(T1 const& first_, T2 const& second_)\n"
    "		: first(first_), second(second_) {}\n"
    "	// TODO: Standard includes many more constructors...\n"
    "	// TODO: Comparison operators\n"
    "};\n"
    "template<class T1, class T2>\n"
    "pair<T1,T2> make_pair(T1 const& first, T2 const& second) {\n"
    "	return pair<T1,T2>(first, second);\n"
    "}\n"
    "} // namespace __jitify_utility_ns\n"
    "namespace std { using namespace __jitify_utility_ns; }\n"
    "using namespace __jitify_utility_ns;\n";

// TODO: incomplete
static const char* jitsafe_header_vector =
    "#pragma once\n"
    "namespace __jitify_vector_ns {\n"
    "template<class T, class Allocator=void>\n"  // = std::allocator> \n"
    "struct vector {\n"
    "};\n"
    "} // namespace __jitify_vector_ns\n"
    "namespace std { using namespace __jitify_vector_ns; }\n"
    "using namespace __jitify_vector_ns;\n";

// TODO: incomplete
static const char* jitsafe_header_string =
    "#pragma once\n"
    "namespace __jitify_string_ns {\n"
    "template<class CharT,class Traits=void,class Allocator=void>\n"
    "struct basic_string {\n"
    "basic_string();\n"
    "basic_string( const CharT* s );\n"  //, const Allocator& alloc =
                                         // Allocator() );\n"
    "const CharT* c_str() const;\n"
    "bool empty() const;\n"
    "void operator+=(const char *);\n"
    "void operator+=(const basic_string &);\n"
    "};\n"
    "typedef basic_string<char> string;\n"
    "} // namespace __jitify_string_ns\n"
    "namespace std { using namespace __jitify_string_ns; }\n"
    "using namespace __jitify_string_ns;\n";

// TODO: incomplete
static const char* jitsafe_header_stdexcept =
    "#pragma once\n"
    "namespace __jitify_stdexcept_ns {\n"
    "struct runtime_error {\n"
    "explicit runtime_error( const std::string& what_arg );"
    "explicit runtime_error( const char* what_arg );"
    "virtual const char* what() const;\n"
    "};\n"
    "} // namespace __jitify_stdexcept_ns\n"
    "namespace std { using namespace __jitify_stdexcept_ns; }\n"
    "using namespace __jitify_stdexcept_ns;\n";

// TODO: incomplete
static const char* jitsafe_header_complex =
    "#pragma once\n"
    "namespace __jitify_complex_ns {\n"
    "template<typename T>\n"
    "class complex {\n"
    "	T _real;\n"
    "	T _imag;\n"
    "public:\n"
    "	complex() : _real(0), _imag(0) {}\n"
    "	complex(T const& real, T const& imag)\n"
    "		: _real(real), _imag(imag) {}\n"
    "	complex(T const& real)\n"
    "               : _real(real), _imag(static_cast<T>(0)) {}\n"
    "	T const& real() const { return _real; }\n"
    "	T&       real()       { return _real; }\n"
    "	void real(const T &r) { _real = r; }\n"
    "	T const& imag() const { return _imag; }\n"
    "	T&       imag()       { return _imag; }\n"
    "	void imag(const T &i) { _imag = i; }\n"
    "       complex<T>& operator+=(const complex<T> z)\n"
    "         { _real += z.real(); _imag += z.imag(); return *this; }\n"
    "};\n"
    "template<typename T>\n"
    "complex<T> operator*(const complex<T>& lhs, const complex<T>& rhs)\n"
    "  { return complex<T>(lhs.real()*rhs.real()-lhs.imag()*rhs.imag(),\n"
    "                      lhs.real()*rhs.imag()+lhs.imag()*rhs.real()); }\n"
    "template<typename T>\n"
    "complex<T> operator*(const complex<T>& lhs, const T & rhs)\n"
    "  { return complexs<T>(lhs.real()*rhs,lhs.imag()*rhs); }\n"
    "template<typename T>\n"
    "complex<T> operator*(const T& lhs, const complex<T>& rhs)\n"
    "  { return complexs<T>(rhs.real()*lhs,rhs.imag()*lhs); }\n"
    "} // namespace __jitify_complex_ns\n"
    "namespace std { using namespace __jitify_complex_ns; }\n"
    "using namespace __jitify_complex_ns;\n";

// TODO: This is incomplete (missing binary and integer funcs, macros,
// constants, types)
static const char* jitsafe_header_math =
    "#pragma once\n"
    "namespace __jitify_math_ns {\n"
    "#if __cplusplus >= 201103L\n"
    "#define DEFINE_MATH_UNARY_FUNC_WRAPPER(f) \\\n"
    "	inline double      f(double x)         { return ::f(x); } \\\n"
    "	inline float       f##f(float x)       { return ::f(x); } \\\n"
    "	/*inline long double f##l(long double x) { return ::f(x); }*/ \\\n"
    "	inline float       f(float x)          { return ::f(x); } \\\n"
    "	/*inline long double f(long double x)    { return ::f(x); }*/\n"
    "#else\n"
    "#define DEFINE_MATH_UNARY_FUNC_WRAPPER(f) \\\n"
    "	inline double      f(double x)         { return ::f(x); } \\\n"
    "	inline float       f##f(float x)       { return ::f(x); } \\\n"
    "	/*inline long double f##l(long double x) { return ::f(x); }*/\n"
    "#endif\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(cos)\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(sin)\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(tan)\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(acos)\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(asin)\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(atan)\n"
    "template<typename T> inline T atan2(T y, T x) { return ::atan2(y, x); }\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(cosh)\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(sinh)\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(tanh)\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(exp)\n"
    "template<typename T> inline T frexp(T x, int* exp) { return ::frexp(x, "
    "exp); }\n"
    "template<typename T> inline T ldexp(T x, int  exp) { return ::ldexp(x, "
    "exp); }\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(log)\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(log10)\n"
    "template<typename T> inline T modf(T x, T* intpart) { return ::modf(x, "
    "intpart); }\n"
    "template<typename T> inline T pow(T x, T y) { return ::pow(x, y); }\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(sqrt)\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(ceil)\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(floor)\n"
    "template<typename T> inline T fmod(T n, T d) { return ::fmod(n, d); }\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(fabs)\n"
    "template<typename T> inline T abs(T x) { return ::abs(x); }\n"
    "#if __cplusplus >= 201103L\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(acosh)\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(asinh)\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(atanh)\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(exp2)\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(expm1)\n"
    "template<typename T> inline int ilogb(T x) { return ::ilogb(x); }\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(log1p)\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(log2)\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(logb)\n"
    "template<typename T> inline T scalbn (T x, int n)  { return ::scalbn(x, "
    "n); }\n"
    "template<typename T> inline T scalbln(T x, long n) { return ::scalbn(x, "
    "n); }\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(cbrt)\n"
    "template<typename T> inline T hypot(T x, T y) { return ::hypot(x, y); }\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(erf)\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(erfc)\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(tgamma)\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(lgamma)\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(trunc)\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(round)\n"
    "template<typename T> inline long lround(T x) { return ::lround(x); }\n"
    "template<typename T> inline long long llround(T x) { return ::llround(x); "
    "}\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(rint)\n"
    "template<typename T> inline long lrint(T x) { return ::lrint(x); }\n"
    "template<typename T> inline long long llrint(T x) { return ::llrint(x); "
    "}\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(nearbyint)\n"
    // TODO: remainder, remquo, copysign, nan, nextafter, nexttoward, fdim,
    // fmax, fmin, fma
    "#endif\n"
    "#undef DEFINE_MATH_UNARY_FUNC_WRAPPER\n"
    "} // namespace __jitify_math_ns\n"
    "namespace std { using namespace __jitify_math_ns; }\n"
    "#define M_PI 3.14159265358979323846\n"
    // Note: Global namespace already includes CUDA math funcs
    "//using namespace __jitify_math_ns;\n";

// TODO: incomplete
static const char* jitsafe_header_mutex = R"(
    #pragma once
    #if __cplusplus >= 201103L
    namespace __jitify_mutex_ns {
    class mutex {
    public:
    void lock();
    bool try_lock();
    void unlock();
    };
    // namespace __jitify_mutex_ns
    namespace std { using namespace __jitify_mutex_ns; }
    using namespace __jitify_mutex_ns;
    #endif
 )";

static const char* jitsafe_headers[] = {
    jitsafe_header_preinclude_h, jitsafe_header_float_h,
    jitsafe_header_float_h,      jitsafe_header_limits_h,
    jitsafe_header_limits_h,     jitsafe_header_stdint_h,
    jitsafe_header_stdint_h,     jitsafe_header_stddef_h,
    jitsafe_header_stddef_h,     jitsafe_header_stdlib_h,
    jitsafe_header_stdlib_h,     jitsafe_header_stdio_h,
    jitsafe_header_stdio_h,      jitsafe_header_string_h,
    jitsafe_header_cstring,      jitsafe_header_iterator,
    jitsafe_header_limits,       jitsafe_header_type_traits,
    jitsafe_header_utility,      jitsafe_header_math,
    jitsafe_header_math,         jitsafe_header_complex,
    jitsafe_header_iostream,     jitsafe_header_ostream,
    jitsafe_header_istream,      jitsafe_header_sstream,
    jitsafe_header_vector,       jitsafe_header_string,
    jitsafe_header_stdexcept,    jitsafe_header_mutex};
static const char* jitsafe_header_names[] = {"jitify_preinclude.h",
                                             "float.h",
                                             "cfloat",
                                             "limits.h",
                                             "climits",
                                             "stdint.h",
                                             "cstdint",
                                             "stddef.h",
                                             "cstddef",
                                             "stdlib.h",
                                             "cstdlib",
                                             "stdio.h",
                                             "cstdio",
                                             "string.h",
                                             "cstring",
                                             "iterator",
                                             "limits",
                                             "type_traits",
                                             "utility",
                                             "math",
                                             "cmath",
                                             "complex",
                                             "iostream",
                                             "ostream",
                                             "istream",
                                             "sstream",
                                             "vector",
                                             "string",
                                             "stdexcept",
                                             "mutex"};

template <class T, size_t N>
size_t array_size(T (&)[N]) {
  return N;
}
const int jitsafe_headers_count = array_size(jitsafe_headers);

inline void add_options_from_env(std::vector<std::string>& options) {
  // Add options from environment variable
  const char* env_options = std::getenv("JITIFY_OPTIONS");
  if (env_options) {
    std::stringstream ss;
    ss << env_options;
    std::string opt;
    while (!(ss >> opt).fail()) {
      options.push_back(opt);
    }
  }
  // Add options from JITIFY_OPTIONS macro
#ifdef JITIFY_OPTIONS
#define JITIFY_TOSTRING_IMPL(x) #x
#define JITIFY_TOSTRING(x) JITIFY_TOSTRING_IMPL(x)
  std::stringstream ss;
  ss << JITIFY_TOSTRING(JITIFY_OPTIONS);
  std::string opt;
  while (!(ss >> opt).fail()) {
    options.push_back(opt);
  }
#undef JITIFY_TOSTRING
#undef JITIFY_TOSTRING_IMPL
#endif
}

inline void detect_and_add_cuda_arch(std::vector<std::string>& options) {
  for (int i = 0; i < (int)options.size(); ++i) {
    if (options[i].find("-arch") != std::string::npos) {
      // Arch already specified in options
      return;
    }
  }
  // Use the compute capability of the current device
  // TODO: Check these API calls for errors
  int device;
  cudaGetDevice(&device);
  int cc_major;
  cudaDeviceGetAttribute(&cc_major, cudaDevAttrComputeCapabilityMajor, device);
  int cc_minor;
  cudaDeviceGetAttribute(&cc_minor, cudaDevAttrComputeCapabilityMinor, device);
  int cc = cc_major * 10 + cc_minor;
  // Note: We must limit the architecture to the max supported by the current
  //         version of NVRTC, otherwise newer hardware will cause errors
  //         on older versions of CUDA.
  const int cuda_major = std::min(10, CUDA_VERSION / 1000);
  switch (cuda_major) {
    case 11: cc = std::min(cc, 86); break; // Ampere
    case 10: cc = std::min(cc, 75); break; // Turing
    case  9: cc = std::min(cc, 70); break; // Volta
    case  8: cc = std::min(cc, 61); break; // Pascal
    case  7: cc = std::min(cc, 52); break; // Maxwell
    default:
      throw std::runtime_error("Unexpected CUDA major version " +
                             std::to_string(cuda_major));
  }
  std::stringstream ss;
  ss << cc;
  options.push_back("-arch=compute_" + ss.str());
}

inline void split_compiler_and_linker_options(
    std::vector<std::string> options,
    std::vector<std::string>* compiler_options,
    std::vector<std::string>* linker_files,
    std::vector<std::string>* linker_paths) {
  for (int i = 0; i < (int)options.size(); ++i) {
    std::string opt = options[i];
    std::string flag = opt.substr(0, 2);
    std::string value = opt.substr(2);
    if (flag == "-l") {
      linker_files->push_back(value);
    } else if (flag == "-L") {
      linker_paths->push_back(value);
    } else {
      compiler_options->push_back(opt);
    }
  }
}

inline nvrtcResult compile_kernel(std::string program_name,
                                  std::map<std::string, std::string> sources,
                                  std::vector<std::string> options,
                                  std::string instantiation = "",
                                  std::string* log = 0, std::string* ptx = 0,
                                  std::string* mangled_instantiation = 0) {
  std::string program_source = sources[program_name];
  // Build arrays of header names and sources
  std::vector<const char*> header_names_c;
  std::vector<const char*> header_sources_c;
  int num_headers = sources.size() - 1;
  header_names_c.reserve(num_headers);
  header_sources_c.reserve(num_headers);
  typedef std::map<std::string, std::string> source_map;
  for (source_map::const_iterator iter = sources.begin(); iter != sources.end();
       ++iter) {
    std::string const& name = iter->first;
    std::string const& code = iter->second;
    if (name == program_name) {
      continue;
    }
    header_names_c.push_back(name.c_str());
    header_sources_c.push_back(code.c_str());
  }

  std::vector<const char*> options_c(options.size() + 2);
  options_c[0] = "--device-as-default-execution-space";
  options_c[1] = "--pre-include=jitify_preinclude.h";
  for (int i = 0; i < (int)options.size(); ++i) {
    options_c[i + 2] = options[i].c_str();
  }

#if CUDA_VERSION < 8000
  std::string inst_dummy;
  if (!instantiation.empty()) {
    // WAR for no nvrtcAddNameExpression before CUDA 8.0
    // Force template instantiation by adding dummy reference to kernel
    inst_dummy = "__jitify_instantiation";
    program_source +=
        "\nvoid* " + inst_dummy + " = (void*)" + instantiation + ";\n";
  }
#endif

#define CHECK_NVRTC(call)       \
  do {                          \
    nvrtcResult ret = call;     \
    if (ret != NVRTC_SUCCESS) { \
      return ret;               \
    }                           \
  } while (0)

  nvrtcProgram nvrtc_program;
  CHECK_NVRTC(nvrtcCreateProgram(
      &nvrtc_program, program_source.c_str(), program_name.c_str(), num_headers,
      header_sources_c.data(), header_names_c.data()));

#if CUDA_VERSION >= 8000
  if (!instantiation.empty()) {
    CHECK_NVRTC(nvrtcAddNameExpression(nvrtc_program, instantiation.c_str()));
  }
#endif

  nvrtcResult ret =
      nvrtcCompileProgram(nvrtc_program, options_c.size(), options_c.data());
  if (log) {
    size_t logsize;
    CHECK_NVRTC(nvrtcGetProgramLogSize(nvrtc_program, &logsize));
    std::vector<char> vlog(logsize, 0);
    CHECK_NVRTC(nvrtcGetProgramLog(nvrtc_program, vlog.data()));
    log->assign(vlog.data(), logsize);
    if (ret != NVRTC_SUCCESS) {
      return ret;
    }
  }

  if (ptx) {
    size_t ptxsize;
    CHECK_NVRTC(nvrtcGetPTXSize(nvrtc_program, &ptxsize));
    std::vector<char> vptx(ptxsize);
    CHECK_NVRTC(nvrtcGetPTX(nvrtc_program, vptx.data()));
    ptx->assign(vptx.data(), ptxsize);
  }

  if (!instantiation.empty() && mangled_instantiation) {
#if CUDA_VERSION >= 8000
    const char* mangled_instantiation_cstr;
    // Note: The returned string pointer becomes invalid after
    //         nvrtcDestroyProgram has been called, so we save it.
    CHECK_NVRTC(nvrtcGetLoweredName(nvrtc_program, instantiation.c_str(),
                                    &mangled_instantiation_cstr));
    *mangled_instantiation = mangled_instantiation_cstr;
#else
    // Extract mangled kernel template instantiation from PTX
    inst_dummy += " = ";  // Note: This must match how the PTX is generated
    int mi_beg = ptx->find(inst_dummy) + inst_dummy.size();
    int mi_end = ptx->find(";", mi_beg);
    *mangled_instantiation = ptx->substr(mi_beg, mi_end - mi_beg);
#endif
  }

  CHECK_NVRTC(nvrtcDestroyProgram(&nvrtc_program));
#undef CHECK_NVRTC
  return NVRTC_SUCCESS;
}

}  // namespace detail

//! \endcond

class KernelInstantiation;
class Kernel;
class Program;
class JitCache;

struct ProgramConfig {
  std::vector<std::string> options;
  std::vector<std::string> include_paths;
  std::string name;
  typedef std::map<std::string, std::string> source_map;
  source_map sources;
};

class JitCache_impl {
  friend class Program_impl;
  friend class KernelInstantiation_impl;
  friend class KernelLauncher_impl;
  typedef uint64_t key_type;
  jitify::ObjectCache<key_type, detail::CUDAKernel> _kernel_cache;
  jitify::ObjectCache<key_type, ProgramConfig> _program_config_cache;
  std::vector<std::string> _options;
#if JITIFY_THREAD_SAFE
  std::mutex _kernel_cache_mutex;
  std::mutex _program_cache_mutex;
#endif
 public:
  inline JitCache_impl(size_t cache_size)
      : _kernel_cache(cache_size), _program_config_cache(cache_size) {
    detail::add_options_from_env(_options);

    // Bootstrap the cuda context to avoid errors
    cudaFree(0);
  }
};

class Program_impl {
  // A friendly class
  friend class Kernel_impl;
  friend class KernelLauncher_impl;
  friend class KernelInstantiation_impl;
  // TODO: This can become invalid if JitCache is destroyed before the
  //         Program object is. However, this can't happen if JitCache
  //           instances are static.
  JitCache_impl& _cache;
  uint64_t _hash;
  ProgramConfig* _config;
  std::string _compile_log;
  void load_sources(std::string source, std::vector<std::string> headers,
                    std::vector<std::string> options,
                    file_callback_type file_callback);

 public:
  inline Program_impl(JitCache_impl& cache, std::string source,
                      jitify::detail::vector<std::string> headers = 0,
                      jitify::detail::vector<std::string> options = 0,
                      file_callback_type file_callback = 0);
#if __cplusplus >= 201103L
  inline Program_impl(Program_impl const&) = default;
  inline Program_impl(Program_impl&&) = default;
#endif
  inline std::vector<std::string> const& options() const {
    return _config->options;
  }
  inline std::string const& name() const { return _config->name; }
  inline ProgramConfig::source_map const& sources() const {
    return _config->sources;
  }
  inline std::vector<std::string> const& include_paths() const {
    return _config->include_paths;
  }
  std::string getLog() const { return _compile_log; };
};

class Kernel_impl {
  friend class KernelLauncher_impl;
  friend class KernelInstantiation_impl;
  Program_impl _program;
  std::string _name;
  std::vector<std::string> _options;
  uint64_t _hash;

 public:
  inline Kernel_impl(Program_impl const& program, std::string name,
                     jitify::detail::vector<std::string> options = 0);
#if __cplusplus >= 201103L
  inline Kernel_impl(Kernel_impl const&) = default;
  inline Kernel_impl(Kernel_impl&&) = default;
#endif
};

class KernelInstantiation_impl {
  friend class KernelLauncher_impl;
  Kernel_impl _kernel;
  uint64_t _hash;
  std::string _template_inst;
  std::vector<std::string> _options;
  detail::CUDAKernel* _cuda_kernel;
  std::string _compile_log;
  inline std::string print() const;
  void build_kernel();

 public:
  inline KernelInstantiation_impl(
      Kernel_impl const& kernel, std::vector<std::string> const& template_args);
#if __cplusplus >= 201103L
  inline KernelInstantiation_impl(KernelInstantiation_impl const&) = default;
  inline KernelInstantiation_impl(KernelInstantiation_impl&&) = default;
#endif
  detail::CUDAKernel const& cuda_kernel() const { return *_cuda_kernel; }
  const std::string& getLog() const { return _compile_log; }
};

class KernelLauncher_impl {
  KernelInstantiation_impl _kernel_inst;
  dim3 _grid;
  dim3 _block;
  size_t _smem;
  cudaStream_t _stream;

 public:
  inline KernelLauncher_impl(KernelInstantiation_impl const& kernel_inst,
                             dim3 grid, dim3 block, size_t smem = 0,
                             cudaStream_t stream = 0)
      : _kernel_inst(kernel_inst),
        _grid(grid),
        _block(block),
        _smem(smem),
        _stream(stream) {}
#if __cplusplus >= 201103L
  inline KernelLauncher_impl(KernelLauncher_impl const&) = default;
  inline KernelLauncher_impl(KernelLauncher_impl&&) = default;
#endif
  inline CUresult launch(
      jitify::detail::vector<void*> arg_ptrs,
      jitify::detail::vector<std::string> arg_types = 0) const;
};

/*! An object representing a configured and instantiated kernel ready
 *    for launching.
 */
class KernelLauncher {
  std::unique_ptr<KernelLauncher_impl const> _impl;

 public:
  inline KernelLauncher(KernelInstantiation const& kernel_inst, dim3 grid,
                        dim3 block, size_t smem = 0, cudaStream_t stream = 0);

  // Note: It's important that there is no implicit conversion required
  //         for arg_ptrs, because otherwise the parameter pack version
  //         below gets called instead (probably resulting in a segfault).
  /*! Launch the kernel.
   *
   *  \param arg_ptrs  A vector of pointers to each function argument for the
   *    kernel.
   *  \param arg_types A vector of function argument types represented
   *    as code-strings. This parameter is optional and is only used to print
   *    out the function signature.
   */
  inline CUresult launch(
      std::vector<void*> arg_ptrs = std::vector<void*>(),
      jitify::detail::vector<std::string> arg_types = 0) const {
    return _impl->launch(arg_ptrs, arg_types);
  }
#if __cplusplus >= 201103L
  // Regular function call syntax
  /*! Launch the kernel.
   *
   *  \see launch
   */
  template <typename... ArgTypes>
  inline CUresult operator()(ArgTypes... args) const {
    return this->launch(args...);
  }
  /*! Launch the kernel.
   *
   *  \param args Function arguments for the kernel.
   */
  template <typename... ArgTypes>
  inline CUresult launch(ArgTypes... args) const {
      return this->launch(std::vector<void *>({ (void *)&args... }),
          { reflection::reflect<ArgTypes>()... });
  }
#endif
};

/*! An object representing a kernel instantiation made up of a Kernel and
 *    template arguments.
 */
class KernelInstantiation {
  friend class KernelLauncher;
  std::unique_ptr<KernelInstantiation_impl const> _impl;

 public:
  inline KernelInstantiation(Kernel const& kernel,
                             std::vector<std::string> const& template_args);

  /*! Configure the kernel launch.
   *
   *  \see configure
   */
  inline KernelLauncher operator()(dim3 grid, dim3 block, size_t smem = 0,
                                   cudaStream_t stream = 0) const {
    return this->configure(grid, block, smem, stream);
  }
  /*! Configure the kernel launch.
   *
   *  \param grid   The thread grid dimensions for the launch.
   *  \param block  The thread block dimensions for the launch.
   *  \param smem   The amount of shared memory to dynamically allocate, in
   * bytes.
   *  \param stream The CUDA stream to launch the kernel in.
   */
  inline KernelLauncher configure(dim3 grid, dim3 block, size_t smem = 0,
                                  cudaStream_t stream = 0) const {
    return KernelLauncher(*this, grid, block, smem, stream);
  }
  /*! Configure the kernel launch with a 1-dimensional block and grid chosen
   *  automatically to maximise occupancy.
   *
   * \param max_block_size  The upper limit on the block size, or 0 for no
   * limit.
   * \param smem  The amount of shared memory to dynamically allocate, in bytes.
   * \param smem_callback  A function returning smem for a given block size (overrides \p smem).
   * \param stream The CUDA stream to launch the kernel in.
   * \param flags The flags to pass to cuOccupancyMaxPotentialBlockSizeWithFlags.
   */
  inline KernelLauncher configure_1d_max_occupancy(
      int max_block_size = 0, size_t smem = 0,
      CUoccupancyB2DSize smem_callback = 0, cudaStream_t stream = 0,
      unsigned int flags = 0) const {
    int grid;
    int block;
    CUfunction func = _impl->cuda_kernel();
    if (!func) {
      throw std::runtime_error(
          "Kernel pointer is NULL; you may need to define JITIFY_THREAD_SAFE "
          "1");
    }
    CUresult res = cuOccupancyMaxPotentialBlockSizeWithFlags(
        &grid, &block, func, smem_callback, smem, max_block_size, flags);
    if (res != CUDA_SUCCESS) {
      const char* msg;
      cuGetErrorName(res, &msg);
      throw std::runtime_error(msg);
    }
    if (smem_callback) {
      smem = smem_callback(block);
    }
    return this->configure(grid, block, smem, stream);
  }
  const std::string getLog() const {
      return _impl->getLog();
  }
};

/*! An object representing a kernel made up of a Program, a name and options.
 */
class Kernel {
  friend class KernelInstantiation;
  std::unique_ptr<Kernel_impl const> _impl;

 public:
  Kernel(Program const& program, std::string name,
         jitify::detail::vector<std::string> options = 0);

  /*! Instantiate the kernel.
   *
   *  \param template_args A vector of template arguments represented as
   *    code-strings. These can be generated using
   *    \code{.cpp}jitify::reflection::reflect<type>()\endcode or
   *    \code{.cpp}jitify::reflection::reflect(value)\endcode
   *
   *  \note Template type deduction is not possible, so all types must be
   *    explicitly specified.
   */
  // inline KernelInstantiation instantiate(std::vector<std::string> const&
  // template_args) const {
  inline KernelInstantiation instantiate(
      std::vector<std::string> const& template_args =
          std::vector<std::string>()) const {
    return KernelInstantiation(*this, template_args);
  }
#if __cplusplus >= 201103L
  // Regular template instantiation syntax (note limited flexibility)
  /*! Instantiate the kernel.
   *
   *  \note The template arguments specified on this function are
   *    used to instantiate the kernel. Non-type template arguments must
   *    be wrapped with
   *    \code{.cpp}jitify::reflection::NonType<type,value>\endcode
   *
   *  \note Template type deduction is not possible, so all types must be
   *    explicitly specified.
   */
  template <typename... TemplateArgs>
  inline KernelInstantiation instantiate() const {
    return this->instantiate(
        std::vector<std::string>({reflection::reflect<TemplateArgs>()...}));
  }
  // Template-like instantiation syntax
  //   E.g., instantiate(myvar,Type<MyType>())(grid,block)
  /*! Instantiate the kernel.
   *
   *  \param targs The template arguments for the kernel, represented as
   *    values. Types must be wrapped with
   *    \code{.cpp}jitify::reflection::Type<type>()\endcode or
   *    \code{.cpp}jitify::reflection::type_of(value)\endcode
   *
   *  \note Template type deduction is not possible, so all types must be
   *    explicitly specified.
   */
  template <typename... TemplateArgs>
  inline KernelInstantiation instantiate(TemplateArgs... targs) const {
    return this->instantiate(
        std::vector<std::string>({reflection::reflect(targs)...}));
  }
  template <typename... TemplateArgs>
  inline std::string instantiateLog(TemplateArgs... targs) const {
      auto instans = this->instantiate(
          std::vector<std::string>({ reflection::reflect(targs)... }));
      return instans.getLog();
  }
#endif
};

/*! An object representing a program made up of source code, headers
 *    and options.
 */
class Program {
  friend class Kernel;
  std::unique_ptr<Program_impl const> _impl;

 public:
  Program(JitCache& cache, std::string source,
          jitify::detail::vector<std::string> headers = 0,
          jitify::detail::vector<std::string> options = 0,
          file_callback_type file_callback = 0);

  /*! Select a kernel.
   *
   * \param name The name of the kernel (unmangled and without
   * template arguments).
   * \param options A vector of options to be passed to the NVRTC
   * compiler when compiling this kernel.
   */
  inline Kernel kernel(std::string name,
                       jitify::detail::vector<std::string> options = 0) const {
    return Kernel(*this, name, options);
  }
  /*! Select a kernel.
   *
   *  \see kernel
   */
  inline Kernel operator()(
      std::string name, jitify::detail::vector<std::string> options = 0) const {
    return this->kernel(name, options);
  }
  const std::string getLog() const { return (_impl) ? _impl->getLog() : ""; }
};

/*! An object that manages a cache of JIT-compiled CUDA kernels.
 *
 */
class JitCache {
  friend class Program;
  std::unique_ptr<JitCache_impl> _impl;

 public:
  /*! JitCache constructor.
   *  \param cache_size The number of kernels to hold in the cache
   *    before overwriting the least-recently-used ones.
   */
  enum { DEFAULT_CACHE_SIZE = 128 };
  JitCache(size_t cache_size = DEFAULT_CACHE_SIZE)
      : _impl(new JitCache_impl(cache_size)) {}

  /*! Create a program.
   *
   *  \param source A string containing either the source filename or
   *    the source itself; in the latter case, the first line must be
   *    the name of the program.
   *  \param headers A vector of strings representing the source of
   *    each header file required by the program. Each entry can be
   *    either the header filename or the header source itself; in
   *    the latter case, the first line must be the name of the header
   *    (i.e., the name by which the header is #included).
   *  \param options A vector of options to be passed to the
   *    NVRTC compiler. Include paths specified with \p -I
   *    are added to the search paths used by Jitify. The environment
   *    variable JITIFY_OPTIONS can also be used to define additional
   *    options.
   *  \param file_callback A pointer to a callback function that is
   *    invoked whenever a source file needs to be loaded. Inside this
   *    function, the user can either load/specify the source themselves
   *    or defer to Jitify's file-loading mechanisms.
   *  \note Program or header source files referenced by filename are
   *  looked-up using the following mechanisms (in this order):
   *  \note 1) By calling file_callback.
   *  \note 2) By looking for the file embedded in the executable via the GCC
   * linker.
   *  \note 3) By looking for the file in the filesystem.
   *
   *  \note Jitify recursively scans all source files for \p #include
   *  directives and automatically adds them to the set of headers needed
   *  by the program.
   *  If a \p #include directive references a header that cannot be found,
   *  the directive is automatically removed from the source code to prevent
   *  immediate compilation failure. This may result in compilation errors
   *  if the header was required by the program.
   *
   *  \note Jitify automatically includes NVRTC-safe versions of some
   *  standard library headers.
   */
  inline Program program(std::string source,
                         jitify::detail::vector<std::string> headers = 0,
                         jitify::detail::vector<std::string> options = 0,
                         file_callback_type file_callback = 0) {
    return Program(*this, source, headers, options, file_callback);
  }
};

inline Program::Program(JitCache& cache, std::string source,
                        jitify::detail::vector<std::string> headers,
                        jitify::detail::vector<std::string> options,
                        file_callback_type file_callback)
    : _impl(new Program_impl(*cache._impl, source, headers, options,
                             file_callback)) {}

inline Kernel::Kernel(Program const& program, std::string name,
                      jitify::detail::vector<std::string> options)
    : _impl(new Kernel_impl(*program._impl, name, options)) {}

inline KernelInstantiation::KernelInstantiation(
    Kernel const& kernel, std::vector<std::string> const& template_args)
    : _impl(new KernelInstantiation_impl(*kernel._impl, template_args)) {}

inline KernelLauncher::KernelLauncher(KernelInstantiation const& kernel_inst,
                                      dim3 grid, dim3 block, size_t smem,
                                      cudaStream_t stream)
    : _impl(new KernelLauncher_impl(*kernel_inst._impl, grid, block, smem,
                                    stream)) {}

inline std::ostream& operator<<(std::ostream& stream, dim3 d) {
  if (d.y == 1 && d.z == 1) {
    stream << d.x;
  } else {
    stream << "(" << d.x << "," << d.y << "," << d.z << ")";
  }
  return stream;
}

inline CUresult KernelLauncher_impl::launch(
    jitify::detail::vector<void*> arg_ptrs,
    jitify::detail::vector<std::string> arg_types) const {
#if JITIFY_PRINT_LAUNCH
  Kernel_impl const& kernel = _kernel_inst._kernel;
  std::string arg_types_string =
      (arg_types.empty() ? "..." : reflection::reflect_list(arg_types));
  std::cout << "Launching " << kernel._name << _kernel_inst._template_inst
            << "<<<" << _grid << "," << _block << "," << _smem << "," << _stream
            << ">>>"
            << "(" << arg_types_string << ")" << std::endl;
#endif
  if (!_kernel_inst._cuda_kernel) {
    throw std::runtime_error(
        "Kernel pointer is NULL; you may need to define JITIFY_THREAD_SAFE 1");
  }
  return _kernel_inst._cuda_kernel->launch(_grid, _block, _smem, _stream,
                                           arg_ptrs);
}

inline KernelInstantiation_impl::KernelInstantiation_impl(
    Kernel_impl const& kernel, std::vector<std::string> const& template_args)
    : _kernel(kernel), _options(kernel._options) {
  _template_inst =
      (template_args.empty() ? ""
                             : reflection::reflect_template(template_args));
  using detail::hash_combine;
  using detail::hash_larson64;
  _hash = _kernel._hash;
  _hash = hash_combine(_hash, hash_larson64(_template_inst.c_str()));
  JitCache_impl& cache = _kernel._program._cache;
  uint64_t cache_key = _hash;
#if JITIFY_THREAD_SAFE
  std::lock_guard<std::mutex> lock(cache._kernel_cache_mutex);
#endif
  if (cache._kernel_cache.contains(cache_key)) {
#if 0 && JITIFY_PRINT_INSTANTIATION
    std::cout << "Found ";
    this->print();
#endif
    _cuda_kernel = &cache._kernel_cache.get(cache_key);
  } else {
#if 0 && JITIFY_PRINT_INSTANTIATION
    std::cout << "Building ";
    this->print();
#else
      _compile_log += "Building " + this->print();
#endif
    _cuda_kernel = &cache._kernel_cache.emplace(cache_key);
    this->build_kernel();
  }
}

inline std::string KernelInstantiation_impl::print() const {
  std::string options_string = reflection::reflect_list(_options);
  std::stringstream ss;
  ss << _kernel._name << _template_inst << " [" << options_string << "]"
            << std::endl;
  return ss.str();
}

inline void KernelInstantiation_impl::build_kernel() {
  Program_impl const& program = _kernel._program;

  std::string instantiation = _kernel._name + _template_inst;

  std::vector<std::string> compiler_options;
  std::vector<std::string> linker_files;
  std::vector<std::string> linker_paths;
  detail::split_compiler_and_linker_options(_options, &compiler_options,
                                            &linker_files, &linker_paths);

  std::string log;
  std::string ptx;
  std::string mangled_instantiation;
  nvrtcResult ret = detail::compile_kernel(program.name(), program.sources(),
                                           compiler_options, instantiation,
                                           &log, &ptx, &mangled_instantiation);
#if JITIFY_PRINT_LOG
  if (log.size() > 1) {
    log = log.substr(0, strlen(log.c_str())); // fix '\0' isnerted between the string
    _compile_log += detail::print_compile_log(program.name(), log);
  }
#endif
  if (ret != NVRTC_SUCCESS) {
    throw std::runtime_error(std::string("NVRTC error: ") +
                             nvrtcGetErrorString(ret));
  }

#if JITIFY_PRINT_PTX
  std::stringstream ptx_ss;
  ptx_ss << "---------------------------------------" << std::endl;
  ptx_ss << mangled_instantiation << std::endl;
  ptx_ss << "---------------------------------------" << std::endl;
  ptx_ss << "--- PTX for " << program.name() << " ---" << std::endl;
  ptx_ss << "---------------------------------------" << std::endl;
  ptx_ss << ptx << std::endl;
  ptx_ss << "---------------------------------------" << std::endl;
#endif
  _compile_log += ptx_ss.str();

  _cuda_kernel->set(mangled_instantiation.c_str(), ptx.c_str(), linker_files,
                    linker_paths);
}

Kernel_impl::Kernel_impl(Program_impl const& program, std::string name,
                         jitify::detail::vector<std::string> options)
    : _program(program), _name(name), _options(options) {
  // Merge options from parent
  _options.insert(_options.end(), _program.options().begin(),
                  _program.options().end());
  detail::detect_and_add_cuda_arch(_options);
  std::string options_string = reflection::reflect_list(_options);
  using detail::hash_combine;
  using detail::hash_larson64;
  _hash = _program._hash;
  _hash = hash_combine(_hash, hash_larson64(_name.c_str()));
  _hash = hash_combine(_hash, hash_larson64(options_string.c_str()));
}

Program_impl::Program_impl(JitCache_impl& cache, std::string source,
                           jitify::detail::vector<std::string> headers,
                           jitify::detail::vector<std::string> options,
                           file_callback_type file_callback)
    : _cache(cache) {
  // Compute hash of source, headers and options
  std::string options_string = reflection::reflect_list(options);
  using detail::hash_combine;
  using detail::hash_larson64;
  _hash = hash_combine(hash_larson64(source.c_str()),
                       hash_larson64(options_string.c_str()));
  for (size_t i = 0; i < headers.size(); ++i) {
    _hash = hash_combine(_hash, hash_larson64(headers[i].c_str()));
  }
  _hash = hash_combine(_hash, (uint64_t)file_callback);
  // Add built-in JIT-safe headers
  for (int i = 0; i < detail::jitsafe_headers_count; ++i) {
    const char* hdr_name = detail::jitsafe_header_names[i];
    const char* hdr_source = detail::jitsafe_headers[i];
    headers.push_back(std::string(hdr_name) + "\n" + hdr_source);
  }
  // Merge options from parent
  options.insert(options.end(), _cache._options.begin(), _cache._options.end());
  // Load sources
#if JITIFY_THREAD_SAFE
  std::lock_guard<std::mutex> lock(cache._program_cache_mutex);
#endif
  if (!cache._program_config_cache.contains(_hash)) {
    _config = &cache._program_config_cache.insert(_hash);
    this->load_sources(source, headers, options, file_callback);
  } else {
    _config = &cache._program_config_cache.get(_hash);
  }
}

inline void Program_impl::load_sources(std::string source,
                                       std::vector<std::string> headers,
                                       std::vector<std::string> options,
                                       file_callback_type file_callback) {
  std::vector<std::string>& include_paths = _config->include_paths;
  std::string& name = _config->name;
  ProgramConfig::source_map& sources = _config->sources;

  // Extract include paths from compile options
  std::vector<std::string>::iterator iter = options.begin();
  while (iter != options.end()) {
    std::string const& opt = *iter;
    if (opt.substr(0, 2) == "-I") {
      include_paths.push_back(opt.substr(2));
      options.erase(iter);
    } else {
      ++iter;
    }
  }
  _config->options = options;

  // Load program source
  if (!detail::load_source(source, sources, "", include_paths, file_callback)) {
    throw std::runtime_error("Source not found: " + source);
  }
  name = sources.begin()->first;

  // Load header sources
  for (int i = 0; i < (int)headers.size(); ++i) {
    if (!detail::load_source(headers[i], sources, "", include_paths,
                             file_callback)) {
      // **TODO: Deal with source not found
      throw std::runtime_error("Source not found: " + headers[i]);
    }
  }

  std::stringstream log_ss;
#if JITIFY_PRINT_SOURCE
  std::string& program_source = sources[name];
  log_ss << "---------------------------------------" << std::endl;
  log_ss << "--- Source of " << name << " ---" << std::endl;
  log_ss << "---------------------------------------" << std::endl;
  log_ss << detail::print_with_line_numbers(program_source);
  log_ss << "---------------------------------------" << std::endl;
#endif

  std::vector<std::string> compiler_options;
  std::vector<std::string> linker_files;
  std::vector<std::string> linker_paths;
  detail::split_compiler_and_linker_options(options, &compiler_options,
                                            &linker_files, &linker_paths);

  // If no arch is specified at this point we use whatever the current
  // context is.  This ensures we pick up the correct internal headers
  // for arch-dependent compilation, e.g., some intrinsics are only
  // present for specific architectures
  detail::detect_and_add_cuda_arch(compiler_options);

#if JITIFY_PRINT_LOG
  log_ss << "Compiler options: ";
  for (int i = 0; i < (int)compiler_options.size(); ++i) {
      log_ss << compiler_options[i] << " ";
  }
  log_ss << std::endl;
  _compile_log = log_ss.str();
  log_ss.clear();
#endif

  std::string log;
  nvrtcResult ret;
  while ((ret = detail::compile_kernel(name, sources, compiler_options, "",
                                       &log)) == NVRTC_ERROR_COMPILATION) {
     _compile_log += std::string("\n") + detail::print_compile_log(name, log);

    std::string include_name;
    std::string include_parent;
    int line_num = 0;
    if (!detail::extract_include_info_from_compile_error(
            log, include_name, include_parent, line_num)) {
      // There was a non include-related compilation error
#if JITIFY_PRINT_LOG
      _compile_log += detail::print_compile_log(name, log);
#endif
      // TODO: How to handle error?
      throw std::runtime_error(std::string("Runtime compilation failed\n") + _compile_log);
    }

    // Try to load the new header
    std::string include_path = detail::path_base(include_parent);
    if (!detail::load_source(include_name, sources, include_path, include_paths,
                             file_callback)) {
      // Comment-out the include line and print a warning
      if (!sources.count(include_parent)) {
        // ***TODO: Unless there's another mechanism (e.g., potentially
        //            the parent path vs. filename problem), getting
        //            here means include_parent was found automatically
        //            in a system include path.
        //            We need a WAR to zap it from *its parent*.

        for (ProgramConfig::source_map::const_iterator it = sources.begin();
             it != sources.end(); ++it) {
          std::cout << "  " << it->first << std::endl;
        }
        throw std::out_of_range(include_parent +
                                " not in loaded sources!"
                                " This may be due to a header being loaded by"
                                " NVRTC without Jitify's knowledge.");
      }
      std::string& parent_source = sources[include_parent];
      parent_source = detail::comment_out_code_line(line_num, parent_source);
#if JITIFY_PRINT_LOG
      log_ss << include_parent << "(" << line_num
                << "): warning: " << include_name << ": File not found"
                << std::endl;
      _compile_log += log_ss.str();
#endif
    }
  }
  if (ret != NVRTC_SUCCESS) {
#if JITIFY_PRINT_LOG
    if (ret == NVRTC_ERROR_INVALID_OPTION) {
      log_ss << "Compiler options: ";
      for (int i = 0; i < (int)compiler_options.size(); ++i) {
        log_ss << compiler_options[i] << " ";
      }
      log_ss << std::endl;
    }
    _compile_log += log_ss.str();
#endif
    throw std::runtime_error(std::string("NVRTC error: ") +
                             nvrtcGetErrorString(ret));
  }
}

#if __cplusplus >= 201103L

enum Location { HOST, DEVICE };

/*! Specifies location and parameters for execution of an algorithm.
 *  \param stream        The CUDA stream on which to execute.
 *  \param headers       A vector of headers to include in the code.
 *  \param options       Options to pass to the NVRTC compiler.
 *  \param file_callback See jitify::Program.
 *  \param block_size    The size of the CUDA thread block with which to
 * execute.
 *  \param cache_size    The number of kernels to store in the cache
 * before overwriting the least-recently-used ones.
 */
struct ExecutionPolicy {
  /*! Location (HOST or DEVICE) on which to execute.*/
  Location location;
  /*! List of headers to include when compiling the algorithm.*/
  std::vector<std::string> headers;
  /*! List of compiler options.*/
  std::vector<std::string> options;
  /*! Optional callback for loading source files.*/
  file_callback_type file_callback;
  /*! CUDA stream on which to execute.*/
  cudaStream_t stream;
  /*! CUDA device on which to execute.*/
  int device;
  /*! CUDA block size with which to execute.*/
  int block_size;
  /*! The number of instantiations to store in the cache before overwriting
   *  the least-recently-used ones.*/
  size_t cache_size;
  ExecutionPolicy(Location location_ = DEVICE,
                  jitify::detail::vector<std::string> headers_ = 0,
                  jitify::detail::vector<std::string> options_ = 0,
                  file_callback_type file_callback_ = 0,
                  cudaStream_t stream_ = 0, int device_ = 0,
                  int block_size_ = 256,
                  size_t cache_size_ = JitCache::DEFAULT_CACHE_SIZE)
      : location(location_),
        headers(headers_),
        options(options_),
        file_callback(file_callback_),
        stream(stream_),
        device(device_),
        block_size(block_size_),
        cache_size(cache_size_) {}
};

template <class Func>
class Lambda;

/*! An object that captures a set of variables for use in a parallel_for
 *    expression. See JITIFY_CAPTURE().
 */
class Capture {
 public:
  std::vector<std::string> _arg_decls;
  std::vector<void*> _arg_ptrs;

 public:
  template <typename... Args>
  inline Capture(std::vector<std::string> arg_names, Args const&... args)
      : _arg_ptrs{(void*)&args...} {
    std::vector<std::string> arg_types = {reflection::reflect<Args>()...};
    _arg_decls.resize(arg_names.size());
    for (int i = 0; i < (int)arg_names.size(); ++i) {
      _arg_decls[i] = arg_types[i] + " " + arg_names[i];
    }
  }
};

/*! An object that captures the instantiated Lambda function for use
    in a parallel_for expression and the function string for NVRTC
    compilation
 */
template <class Func>
class Lambda {
 public:
  Capture _capture;
  std::string _func_string;
  Func _func;

 public:
  inline Lambda(Capture const& capture, std::string func_string, Func func)
      : _capture(capture), _func_string(func_string), _func(func) {}
};

template <typename T>
inline Lambda<T> make_Lambda(Capture const& capture, std::string func,
                             T lambda) {
  return Lambda<T>(capture, func, lambda);
}

#define JITIFY_CAPTURE(...)                                            \
  jitify::Capture(jitify::detail::split_string(#__VA_ARGS__, -1, ","), \
                  __VA_ARGS__)

#define JITIFY_MAKE_LAMBDA(capture, x, ...)               \
  jitify::make_Lambda(capture, std::string(#__VA_ARGS__), \
                      [x](int i) { __VA_ARGS__; })

#define JITIFY_ARGS(...) __VA_ARGS__

#define JITIFY_LAMBDA_(x, ...) \
  JITIFY_MAKE_LAMBDA(JITIFY_CAPTURE(x), JITIFY_ARGS(x), __VA_ARGS__)

// macro sequence to strip surrounding brackets
#define JITIFY_STRIP_PARENS(X) X
#define JITIFY_PASS_PARAMETERS(X) JITIFY_STRIP_PARENS(JITIFY_ARGS X)

/*! Creates a Lambda object with captured variables and a function
 *    definition.
 *  \param capture A bracket-enclosed list of variables to capture.
 *  \param ...     The function definition.
 *
 *  \code{.cpp}
 *  float* capture_me;
 *  int    capture_me_too;
 *  auto my_lambda = JITIFY_LAMBDA( (capture_me, capture_me_too),
 *                                  capture_me[i] = i*capture_me_too );
 *  \endcode
 */
#define JITIFY_LAMBDA(capture, ...)                            \
  JITIFY_LAMBDA_(JITIFY_ARGS(JITIFY_PASS_PARAMETERS(capture)), \
                 JITIFY_ARGS(__VA_ARGS__))

// TODO: Try to implement for_each that accepts iterators instead of indices
//       Add compile guard for NOCUDA compilation
/*! Call a function for a range of indices
 *
 *  \param policy Determines the location and device parameters for
 *  execution of the parallel_for.
 *  \param begin  The starting index.
 *  \param end    The ending index.
 *  \param lambda A Lambda object created using the JITIFY_LAMBDA() macro.
 *
 *  \code{.cpp}
 *  char const* in;
 *  float*      out;
 *  parallel_for(0, 100, JITIFY_LAMBDA( (in, out), {char x = in[i]; out[i] =
 * x*x; } ); \endcode
 */
template <typename IndexType, class Func>
CUresult parallel_for(ExecutionPolicy policy, IndexType begin, IndexType end,
                      Lambda<Func> const& lambda) {
  using namespace jitify;

  if (policy.location == HOST) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (IndexType i = begin; i < end; i++) {
      lambda._func(i);
    }
    return CUDA_SUCCESS;  // FIXME - replace with non-CUDA enum type?
  }

  thread_local static JitCache kernel_cache(policy.cache_size);

  std::vector<std::string> arg_decls;
  arg_decls.push_back("I begin, I end");
  arg_decls.insert(arg_decls.end(), lambda._capture._arg_decls.begin(),
                   lambda._capture._arg_decls.end());

  std::stringstream source_ss;
  source_ss << "parallel_for_program\n";
  for (auto const& header : policy.headers) {
    std::string header_name = header.substr(0, header.find("\n"));
    source_ss << "#include <" << header_name << ">\n";
  }
  source_ss << "template<typename I>\n"
               "__global__\n"
               "void parallel_for_kernel("
            << reflection::reflect_list(arg_decls)
            << ") {\n"
               "	I i0 = threadIdx.x + blockDim.x*blockIdx.x;\n"
               "	for( I i=i0+begin; i<end; i+=blockDim.x*gridDim.x ) {\n"
               "	"
            << "\t" << lambda._func_string << ";\n"
            << "	}\n"
               "}\n";

  Program program = kernel_cache.program(source_ss.str(), policy.headers,
                                         policy.options, policy.file_callback);

  std::vector<void*> arg_ptrs;
  arg_ptrs.push_back(&begin);
  arg_ptrs.push_back(&end);
  arg_ptrs.insert(arg_ptrs.end(), lambda._capture._arg_ptrs.begin(),
                  lambda._capture._arg_ptrs.end());

  size_t n = end - begin;
  dim3 block(policy.block_size);
  dim3 grid(std::min((n - 1) / block.x + 1, size_t(65535)));
  cudaSetDevice(policy.device);
  return program.kernel("parallel_for_kernel")
      .instantiate<IndexType>()
      .configure(grid, block, 0, policy.stream)
      .launch(arg_ptrs);
}

#endif  // __cplusplus >= 201103L

}  // namespace jitify

#if defined(_WIN32) || defined(_WIN64)
#pragma pop_macro("strtok_r")
#endif

#ifdef _MSVC_LANG
#pragma pop_macro("__cplusplus")
#endif
