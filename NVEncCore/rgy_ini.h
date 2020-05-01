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
#ifndef __RGY_INI_H__
#define __RGY_INI_H__

#if (defined(_WIN32) || defined(_WIN64))
#define GetPrivateProfileStringCP(Section, Key, Default, buf, nSize, IniFile, codecpage) GetPrivateProfileStringA((Section), (Key), (Default), (buf), (nSize), (IniFile))
#define GetPrivateProfileIntCP(Section, Key, defaultValue, IniFile, codecpage) GetPrivateProfileStringA((Section), (Key), (defaultValue), (IniFile))
#else
uint32_t GetPrivateProfileStringCP(const TCHAR *Section, const TCHAR *Key, const TCHAR *Default, TCHAR *buf, size_t nSize, const TCHAR *IniFile, uint32_t codecpage = CP_THREAD_ACP);
uint32_t GetPrivateProfileIntCP(const TCHAR *Section, const TCHAR *Key, const uint32_t defaultValue, const TCHAR *IniFile, uint32_t codecpage = CP_THREAD_ACP);

#define GetPrivateProfileStringA GetPrivateProfileStringCP
#define GetPrivateProfileIntA GetPrivateProfileIntCP

#endif //#if !(defined(_WIN32) || defined(_WIN64))

#endif //__RGY_INI_H__

