/*
 * Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * ALL NVIDIA DESIGN SPECIFICATIONS, REFERENCE BOARDS, FILES, DRAWINGS,
 * DIAGNOSTICS, LISTS, AND OTHER DOCUMENTS (TOGETHER AND SEPARATELY,
 * ÅgMATERIALSÅh) ARE BEING PROVIDED ÅgAS IS.Åh WITHOUT EXPRESS OR IMPLIED
 * WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD
 * TO THESE LICENSED DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE LICENSE
 * AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT,
 * INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING
 * FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT,
 * NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION
 * WITH THE USE OR PERFORMANCE OF THESE LICENSED DELIVERABLES.
 *
 * Information furnished is believed to be accurate and reliable. However,
 * NVIDIA assumes no responsibility for the consequences of use of such
 * information nor for any infringement of patents or other rights of
 * third parties, which may result from its use.  No License is granted
 * by implication or otherwise under any patent or patent rights of NVIDIA
 * Corporation.  Specifications mentioned in the software are subject to
 * change without notice. This publication supersedes and replaces all
 * other information previously supplied.
 *
 * NVIDIA Corporation products are not authorized for use as critical
 * components in life support devices or systems without express written
 * approval of NVIDIA Corporation.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */


// These are helper functions for the SDK samples (string parsing, timers, etc)
#ifndef STRING_HELPER_H
#define STRING_HELPER_H

#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <string>

#if defined(WIN32) || defined(_WIN32) || defined(WIN64)
#ifndef _CRT_SECURE_NO_DEPRECATE
#define _CRT_SECURE_NO_DEPRECATE
#endif
#ifndef STRCASECMP
#define STRCASECMP  _stricmp
#endif
#ifndef STRNCASECMP
#define STRNCASECMP _strnicmp
#endif
#ifndef STRCPY
#define STRCPY(sFilePath, nLength, sPath) strcpy_s(sFilePath, nLength, sPath)
#endif

#ifndef FOPEN
#define FOPEN(fHandle,filename,mode) fopen_s(&fHandle, filename, mode)
#endif
#ifndef FOPEN_FAIL
#define FOPEN_FAIL(result) (result != 0)
#endif
#ifndef SSCANF
#define SSCANF sscanf_s
#endif
#else
#include <string.h>
#include <strings.h>

#ifndef STRCASECMP
#define STRCASECMP  strcasecmp
#endif
#ifndef STRNCASECMP
#define STRNCASECMP strncasecmp
#endif
#ifndef STRCPY
#define STRCPY(sFilePath, nLength, sPath) strcpy(sFilePath, sPath)
#endif

#ifndef FOPEN
#define FOPEN(fHandle,filename,mode) (fHandle = fopen(filename, mode))
#endif
#ifndef FOPEN_FAIL
#define FOPEN_FAIL(result) (result == NULL)
#endif
#ifndef SSCANF
#define SSCANF sscanf
#endif
#endif

#ifndef EXIT_WAIVED
#define EXIT_WAIVED 2
#endif

// CUDA Utility Helper Functions
inline int stringRemoveDelimiter(char delimiter, const char *string)
{
    int string_start = 0;

    while (string[string_start] == delimiter)
    {
        string_start++;
    }

    if (string_start >= (int)strlen(string)-1)
    {
        return 0;
    }

    return string_start;
}

inline int getFileExtension(char *filename, char **extension)
{
    int string_length = (int)strlen(filename);

    while (filename[string_length--] != '.')
    {
        if (string_length == 0)
            break;
    }

    if (string_length > 0) string_length += 2;

    if (string_length == 0)
        *extension = NULL;
    else
        *extension = &filename[string_length];

    return string_length;
}


inline int checkCmdLineFlag(const int argc, const char **argv, const char *string_ref)
{
    bool bFound = false;

    if (argc >= 1)
    {
        for (int i=1; i < argc; i++)
        {
            int string_start = stringRemoveDelimiter('-', argv[i]);
            const char *string_argv = &argv[i][string_start];

            const char *equal_pos = strchr(string_argv, '=');
            int argv_length = (int)(equal_pos == 0 ? strlen(string_argv) : equal_pos - string_argv);

            int length = (int)strlen(string_ref);

            if (length == argv_length && !STRNCASECMP(string_argv, string_ref, length))
            {

                bFound = true;
                continue;
            }
        }
    }

    return (int)bFound;
}

// This function wraps the CUDA Driver API into a template function
template <class T>
inline bool getCmdLineArgumentValue(const int argc, const char **argv, const char *string_ref, T *value)
{
    bool bFound = false;

    if (argc >= 1)
    {
        for (int i=1; i < argc; i++)
        {
            int string_start = stringRemoveDelimiter('-', argv[i]);
            const char *string_argv = &argv[i][string_start];
            int length = (int)strlen(string_ref);

            if (!STRNCASECMP(string_argv, string_ref, length))
            {
                if (length+1 <= (int)strlen(string_argv))
                {
                    int auto_inc = (string_argv[length] == '=') ? 1 : 0;
                    *value = (T)atoi(&string_argv[length + auto_inc]);
                }

                bFound = true;
                i=argc;
            }
        }
    }

    return bFound;
}

inline int getCmdLineArgumentInt(const int argc, const char **argv, const char *string_ref)
{
    bool bFound = false;
    int value = -1;

    if (argc >= 1)
    {
        for (int i=1; i < argc; i++)
        {
            int string_start = stringRemoveDelimiter('-', argv[i]);
            const char *string_argv = &argv[i][string_start];
            int length = (int)strlen(string_ref);

            if (!STRNCASECMP(string_argv, string_ref, length))
            {
                if (length+1 <= (int)strlen(string_argv))
                {
                    int auto_inc = (string_argv[length] == '=') ? 1 : 0;
                    value = atoi(&string_argv[length + auto_inc]);
                }
                else
                {
                    value = 0;
                }

                bFound = true;
                continue;
            }
        }
    }

    if (bFound)
    {
        return value;
    }
    else
    {
        return 0;
    }
}

inline float getCmdLineArgumentFloat(const int argc, const char **argv, const char *string_ref)
{
    bool bFound = false;
    float value = -1;

    if (argc >= 1)
    {
        for (int i=1; i < argc; i++)
        {
            int string_start = stringRemoveDelimiter('-', argv[i]);
            const char *string_argv = &argv[i][string_start];
            int length = (int)strlen(string_ref);

            if (!STRNCASECMP(string_argv, string_ref, length))
            {
                if (length+1 <= (int)strlen(string_argv))
                {
                    int auto_inc = (string_argv[length] == '=') ? 1 : 0;
                    value = (float)atof(&string_argv[length + auto_inc]);
                }
                else
                {
                    value = 0.f;
                }

                bFound = true;
                continue;
            }
        }
    }

    if (bFound)
    {
        return value;
    }
    else
    {
        return 0;
    }
}

inline bool getCmdLineArgumentString(const int argc, const char **argv,
                                     const char *string_ref, char **string_retval)
{
    bool bFound = false;

    if (argc >= 1)
    {
        for (int i=1; i < argc; i++)
        {
            int string_start = stringRemoveDelimiter('-', argv[i]);
            char *string_argv = (char *)&argv[i][string_start];
            int length = (int)strlen(string_ref);

            if (!STRNCASECMP(string_argv, string_ref, length))
            {
                *string_retval = &string_argv[length+1];
                bFound = true;
                continue;
            }
        }
    }

    if (!bFound)
    {
        *string_retval = NULL;
    }

    return bFound;
}

//////////////////////////////////////////////////////////////////////////////
//! Find the path for a file assuming that
//! files are found in the searchPath.
//!
//! @return the path if succeeded, otherwise 0
//! @param filename         name of the file
//! @param executable_path  optional absolute path of the executable
//////////////////////////////////////////////////////////////////////////////
inline char *sdkFindFilePath(const char *filename, const char *executable_path)
{
    // <executable_name> defines a variable that is replaced with the name of the executable

    // Typical relative search paths to locate needed companion files (e.g. sample input data, or JIT source files)
    // The origin for the relative search may be the .exe file, a .bat file launching an .exe, a browser .exe launching the .exe or .bat, etc
    const char *searchPath[] =
    {
        "./",                                       // same dir
        "./common/",                                // "/common/" subdir
        "./common/data/",                           // "/common/data/" subdir
        "./data/",                                  // "/data/" subdir
        "./src/",                                   // "/src/" subdir
        "./src/<executable_name>/data/",            // "/src/<executable_name>/data/" subdir
        "./inc/",                                   // "/inc/" subdir
        "./0_Simple/",                              // "/0_Simple/" subdir
        "./1_Utilities/",                           // "/1_Utilities/" subdir
        "./2_Graphics/",                            // "/2_Graphics/" subdir
        "./3_Imaging/",                             // "/3_Imaging/" subdir
        "./4_Financial/",                           // "/4_Financial/" subdir
        "./5_Simulations/",                         // "/5_Simulations/" subdir
        "./6_Advanced/",                            // "/6_Advanced/" subdir
        "./7_CUDALibraries/",                       // "/7_CUDALibraries/" subdir

        "../",                                      // up 1 in tree
        "../common/",                               // up 1 in tree, "/common/" subdir
        "../common/data/",                          // up 1 in tree, "/common/data/" subdir
        "../data/",                                 // up 1 in tree, "/data/" subdir
        "../src/",                                  // up 1 in tree, "/src/" subdir
        "../inc/",                                  // up 1 in tree, "/inc/" subdir
        "../C/src/<executable_name>/",              // up 1 in tree, "/C/src/<executable_name>/" subdir
        "../C/src/<executable_name>/data/",         // up 1 in tree, "/C/src/<executable_name>/data/" subdir
        "../C/src/<executable_name>/src/",          // up 1 in tree, "/C/src/<executable_name>/src/" subdir
        "../C/src/<executable_name>/inc/",          // up 1 in tree, "/C/src/<executable_name>/inc/" subdir
        "../C/",                                      // up 1 in tree
        "../C/common/",                               // up 1 in tree, "/common/" subdir
        "../C/common/data/",                          // up 1 in tree, "/common/data/" subdir
        "../C/data/",                                 // up 1 in tree, "/data/" subdir
        "../C/src/",                                  // up 1 in tree, "/src/" subdir
        "../C/inc/",                                  // up 1 in tree, "/inc/" subdir
        "../C/0_Simple/<executable_name>/data/",         // up 1 in tree, "/0_Simple/<executable_name>/" subdir
        "../C/1_Utilities/<executable_name>/data/",      // up 1 in tree, "/1_Utilities/<executable_name>/" subdir
        "../C/2_Graphics/<executable_name>/data/",       // up 1 in tree, "/2_Graphics/<executable_name>/" subdir
        "../C/3_Imaging/<executable_name>/data/",        // up 1 in tree, "/3_Imaging/<executable_name>/" subdir
        "../C/4_Financial/<executable_name>/data/",      // up 1 in tree, "/4_Financial/<executable_name>/" subdir
        "../C/5_Simulations/<executable_name>/data/",    // up 1 in tree, "/5_Simulations/<executable_name>/" subdir
        "../C/6_Advanced/<executable_name>/data/",       // up 1 in tree, "/6_Advanced/<executable_name>/" subdir
        "../C/7_CUDALibraries/<executable_name>/data/",  // up 1 in tree, "/7_CUDALibraries/<executable_name>/" subdir

        "../0_Simple/<executable_name>/data/",           // up 1 in tree, "/0_Simple/<executable_name>/" subdir
        "../1_Utilities/<executable_name>/data/",        // up 1 in tree, "/1_Utilities/<executable_name>/" subdir
        "../2_Graphics/<executable_name>/data/",         // up 1 in tree, "/2_Graphics/<executable_name>/" subdir
        "../3_Imaging/<executable_name>/data/",          // up 1 in tree, "/3_Imaging/<executable_name>/" subdir
        "../4_Financial/<executable_name>/data/",        // up 1 in tree, "/4_Financial/<executable_name>/" subdir
        "../5_Simulations/<executable_name>/data/",      // up 1 in tree, "/5_Simulations/<executable_name>/" subdir
        "../6_Advanced/<executable_name>/data/",         // up 1 in tree, "/6_Advanced/<executable_name>/" subdir
        "../7_CUDALibraries/<executable_name>/data/",    // up 1 in tree, "/7_CUDALibraries/<executable_name>/" subdir
        "../../",                                   // up 2 in tree
        "../../common/",                            // up 2 in tree, "/common/" subdir
        "../../common/data/",                       // up 2 in tree, "/common/data/" subdir
        "../../data/",                              // up 2 in tree, "/data/" subdir
        "../../src/",                               // up 2 in tree, "/src/" subdir
        "../../inc/",                               // up 2 in tree, "/inc/" subdir
        "../../sandbox/<executable_name>/data/",    // up 2 in tree, "/sandbox/<executable_name>/" subdir
        "../../0_Simple/<executable_name>/data/",        // up 2 in tree, "/0_Simple/<executable_name>/" subdir
        "../../1_Utilities/<executable_name>/data/",     // up 2 in tree, "/1_Utilities/<executable_name>/" subdir
        "../../2_Graphics/<executable_name>/data/",      // up 2 in tree, "/2_Graphics/<executable_name>/" subdir
        "../../3_Imaging/<executable_name>/data/",       // up 2 in tree, "/3_Imaging/<executable_name>/" subdir
        "../../4_Financial/<executable_name>/data/",     // up 2 in tree, "/4_Financial/<executable_name>/" subdir
        "../../5_Simulations/<executable_name>/data/",   // up 2 in tree, "/5_Simulations/<executable_name>/" subdir
        "../../6_Advanced/<executable_name>/data/",      // up 2 in tree, "/6_Advanced/<executable_name>/" subdir
        "../../7_CUDALibraries/<executable_name>/data/", // up 2 in tree, "/7_CUDALibraries/<executable_name>/" subdir
        "../../../",                                // up 3 in tree
        "../../../src/<executable_name>/",          // up 3 in tree, "/src/<executable_name>/" subdir
        "../../../src/<executable_name>/data/",     // up 3 in tree, "/src/<executable_name>/data/" subdir
        "../../../src/<executable_name>/src/",      // up 3 in tree, "/src/<executable_name>/src/" subdir
        "../../../src/<executable_name>/inc/",      // up 3 in tree, "/src/<executable_name>/inc/" subdir
        "../../../sandbox/<executable_name>/",      // up 3 in tree, "/sandbox/<executable_name>/" subdir
        "../../../sandbox/<executable_name>/data/", // up 3 in tree, "/sandbox/<executable_name>/data/" subdir
        "../../../sandbox/<executable_name>/src/",  // up 3 in tree, "/sandbox/<executable_name>/src/" subdir
        "../../../sandbox/<executable_name>/inc/",   // up 3 in tree, "/sandbox/<executable_name>/inc/" subdir
        "../../../0_Simple/<executable_name>/data/",     // up 3 in tree, "/0_Simple/<executable_name>/" subdir
        "../../../1_Utilities/<executable_name>/data/",  // up 3 in tree, "/1_Utilities/<executable_name>/" subdir
        "../../../2_Graphics/<executable_name>/data/",   // up 3 in tree, "/2_Graphics/<executable_name>/" subdir
        "../../../3_Imaging/<executable_name>/data/",    // up 3 in tree, "/3_Imaging/<executable_name>/" subdir
        "../../../4_Financial/<executable_name>/data/",  // up 3 in tree, "/4_Financial/<executable_name>/" subdir
        "../../../5_Simulations/<executable_name>/data/",// up 3 in tree, "/5_Simulations/<executable_name>/" subdir
        "../../../6_Advanced/<executable_name>/data/",   // up 3 in tree, "/6_Advanced/<executable_name>/" subdir
        "../../../7_CUDALibraries/<executable_name>/data/", // up 3 in tree, "/7_CUDALibraries/<executable_name>/" subdir
        "../../../common/",                         // up 3 in tree, "../../../common/" subdir
        "../../../common/data/",                    // up 3 in tree, "../../../common/data/" subdir
        "../../../data/",                           // up 3 in tree, "../../../data/" subdir
        "../../../../",                                // up 4 in tree
        "../../../../src/<executable_name>/",          // up 4 in tree, "/src/<executable_name>/" subdir
        "../../../../src/<executable_name>/data/",     // up 4 in tree, "/src/<executable_name>/data/" subdir
        "../../../../src/<executable_name>/src/",      // up 4 in tree, "/src/<executable_name>/src/" subdir
        "../../../../src/<executable_name>/inc/",      // up 4 in tree, "/src/<executable_name>/inc/" subdir
        "../../../../sandbox/<executable_name>/",      // up 4 in tree, "/sandbox/<executable_name>/" subdir
        "../../../../sandbox/<executable_name>/data/", // up 4 in tree, "/sandbox/<executable_name>/data/" subdir
        "../../../../sandbox/<executable_name>/src/",  // up 4 in tree, "/sandbox/<executable_name>/src/" subdir
        "../../../../sandbox/<executable_name>/inc/",   // up 4 in tree, "/sandbox/<executable_name>/inc/" subdir
        "../../../../0_Simple/<executable_name>/data/",     // up 4 in tree, "/0_Simple/<executable_name>/" subdir
        "../../../../1_Utilities/<executable_name>/data/",  // up 4 in tree, "/1_Utilities/<executable_name>/" subdir
        "../../../../2_Graphics/<executable_name>/data/",   // up 4 in tree, "/2_Graphics/<executable_name>/" subdir
        "../../../../3_Imaging/<executable_name>/data/",    // up 4 in tree, "/3_Imaging/<executable_name>/" subdir
        "../../../../4_Financial/<executable_name>/data/",  // up 4 in tree, "/4_Financial/<executable_name>/" subdir
        "../../../../5_Simulations/<executable_name>/data/",// up 4 in tree, "/5_Simulations/<executable_name>/" subdir
        "../../../../6_Advanced/<executable_name>/data/",   // up 4 in tree, "/6_Advanced/<executable_name>/" subdir
        "../../../../7_CUDALibraries/<executable_name>/data/", // up 4 in tree, "/7_CUDALibraries/<executable_name>/" subdir
        "../../../../common/",                         // up 4 in tree, "../../../common/" subdir
        "../../../../common/data/",                    // up 4 in tree, "../../../common/data/" subdir
        "../../../../data/",                           // up 4 in tree, "../../../data/" subdir
    };

    // Extract the executable name
    std::string executable_name;

    if (executable_path != 0)
    {
        executable_name = std::string(executable_path);

#ifdef _WIN32
        // Windows path delimiter
        size_t delimiter_pos = executable_name.find_last_of('\\');
        executable_name.erase(0, delimiter_pos + 1);

        if (executable_name.rfind(".exe") != std::string::npos)
        {
            // we strip .exe, only if the .exe is found
            executable_name.resize(executable_name.size() - 4);
        }

#else
        // Linux & OSX path delimiter
        size_t delimiter_pos = executable_name.find_last_of('/');
        executable_name.erase(0,delimiter_pos+1);
#endif
    }

    // Loop over all search paths and return the first hit
    for (unsigned int i = 0; i < sizeof(searchPath)/sizeof(char *); ++i)
    {
        std::string path(searchPath[i]);
        size_t executable_name_pos = path.find("<executable_name>");

        // If there is executable_name variable in the searchPath
        // replace it with the value
        if (executable_name_pos != std::string::npos)
        {
            if (executable_path != 0)
            {
                path.replace(executable_name_pos, strlen("<executable_name>"), executable_name);
            }
            else
            {
                // Skip this path entry if no executable argument is given
                continue;
            }
        }

#ifdef _DEBUG
        printf("sdkFindFilePath <%s> in %s\n", filename, path.c_str());
#endif

        // Test if the file exists
        path.append(filename);
        FILE *fp;
        FOPEN(fp, path.c_str(), "rb");

        if (fp != NULL)
        {
            fclose(fp);
            // File found
            // returning an allocated array here for backwards compatibility reasons
            char *file_path = (char *) malloc(path.length() + 1);
            STRCPY(file_path, path.length() + 1, path.c_str());
            return file_path;
        }

        if (fp)
        {
            fclose(fp);
        }
    }

    // File not found
    return 0;
}

#endif
