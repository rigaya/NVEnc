#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import sys
import codecs
import os
import subprocess
import traceback

def extract_options_md(filename):
    re1 = re.compile(r'^###\s--([A-Za-z0-9][A-Za-z0-9-_]+)([\s,]+.*)')
    re2 = re.compile(r'^###\s-[A-Za-z0-9],\s--([A-Za-z0-9][A-Za-z0-9-_]+)([,\s]+.*)')
    re3 = re.compile(r'^###\s--\(no-\)([A-Za-z0-9][A-Za-z0-9-_]+)([,\s]+.*)')
    re4 = re.compile(r'^,\s--([A-Za-z0-9][A-Za-z0-9-_]+)([\s,]*.*)')
    
    option_list = set()
    with codecs.open(filename, 'r', 'utf_8') as f:
        for line in f.readlines():
            while True:
                m = re1.match(line)
                if m:
                    option_list.add('--' + m.group(1))
                else:
                    m = re2.match(line)
                    if m:
                        option_list.add('--' + m.group(1))
                    else:
                        m = re3.match(line)
                        if m:
                            option_list.add('--' + m.group(1))
                        else:
                            m = re4.match(line)
                            if m:
                                option_list.add('--' + m.group(1))
                if m:
                    line = m.group(2)
                else:
                    break

    return option_list
    
def get_options_exe(exe_file):
    filename = 'nvencc_option_list.txt'
    cmd = '\"' + exe_file + '\" --option-list'
    option_list = set()
    start_add = False
    try:
        proc = subprocess.Popen(cmd, stdout = subprocess.PIPE, shell=True)
        while True:
            line = proc.stdout.readline().decode('utf-8').strip()
            if start_add and len(line) > 0:
                option_list.add(line)
            if 'Option List:' in line:
                start_add = True

            if not line and proc.poll() is not None:
                break
    except:
        print("failed to run encoder\n");
        print(traceback.format_exc())
        raise

    return option_list

if __name__ == '__main__':
    md_files = []
    md_en_file = 'NVEncC_Options.en.md'
    md_ja_file = 'NVEncC_Options.ja.md'
    md_cn_file = 'NVEncC_Options.zh-cn.md'
    exe_file = r'_build\x64\RelStatic\NVEncC64.exe' if os.name == 'nt' else './nvencc'
    
    iarg = 0
    while iarg < len(sys.argv):
        if sys.argv[iarg] == "-exe":
            iarg=iarg+1
            exe_file = sys.argv[iarg]
        elif sys.argv[iarg] == "-md":
            iarg=iarg+1
            md_files.append(sys.argv[iarg])
        iarg=iarg+1
            
    # デフォルトのファイルの追加
    if os.path.exists(md_en_file):
        md_files.append(md_en_file)
    if os.path.exists(md_ja_file):
        md_files.append(md_ja_file)
    if os.path.exists(md_cn_file):
        md_files.append(md_cn_file)
    
    # mdファイルからのオプションの抽出
    option_list_md = []
    for md in md_files:
        option_list_md.append(extract_options_md(md))

    option_list_exe = get_options_exe(exe_file)
    
    # 全オプションリスト
    option_list_all = option_list_exe
    for optlist in option_list_md:
        option_list_all = option_list_all | optlist

    for imd in range(len(md_files)):
        diff = option_list_all - option_list_md[imd]
        print('Not listed in ' + md_files[imd] + ':' + str(len(diff)))
        for option in sorted(diff):
            print(option)
        print()

    diff = option_list_all - option_list_exe
    print('Not listed in --help:' + str(len(diff)))
    for option in sorted(diff):
        print(option)
