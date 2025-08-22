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

#include <iostream>
#include <fstream>
#include <string.h>
#include <stdio.h>
#include <stddef.h>
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>
#include <shlwapi.h>
#pragma comment(lib, "shlwapi.lib")
#include <json.hpp>
#include "auo_util.h"
#include "auo_conf.h"
#include "auo_options.h"

const int guiEx_config::conf_block_data[CONF_BLOCK_COUNT] = {
    sizeof(CONF_GUIEX::enc),
    sizeof(CONF_GUIEX::vid),
    sizeof(CONF_GUIEX::aud),
    sizeof(CONF_GUIEX::mux),
    sizeof(CONF_GUIEX::oth)
};

const uint32_t guiEx_config::conf_block_pointer[CONF_BLOCK_COUNT] = {
    (uint32_t)offsetof(CONF_GUIEX, enc),
    (uint32_t)offsetof(CONF_GUIEX, vid),
    (uint32_t)offsetof(CONF_GUIEX, aud),
    (uint32_t)offsetof(CONF_GUIEX, mux),
    (uint32_t)offsetof(CONF_GUIEX, oth)
};

guiEx_config::guiEx_config() { }

// 旧形式からJSON文字列に変換
std::string guiEx_config::old_conf_to_json(const CONF_GUIEX_OLD *old_conf) {
    nlohmann::json j;
    
    // ヘッダ情報
    j["conf_name"] = std::string(old_conf->header.conf_name);
    j["version"] = CONF_NAME_JSON;
    
    // エンコーダ設定 (CONF_ENCは構造体なので、バイナリデータとして保存)
#if ENCODER_SVTAV1 || ENCODER_QSV || ENCODER_NVENC || ENCODER_VCEENC
    auto cmd_buffer = char_to_tstring(old_conf->enc.cmd, CP_THREAD_ACP);
#else
    TCHAR cmd_buffer[MAX_CMD_LEN] = { 0 };
#if ENCODER_X264 || ENCODER_X265
    build_cmd_from_conf(cmd_buffer, _countof(cmd_buffer), &old_conf->enc, NULL, FALSE);
#endif    
#endif
    j["enc"] = {
#if ENABLE_AMP || ENCODER_FFMPEG
        {"use_auto_npass", old_conf->enc.use_auto_npass },
        {"auto_npass", old_conf->enc.auto_npass },
#endif
#if ENCODER_SVTAV1
        {"sar_x", old_conf->vid.sar_x},
        {"sar_y", old_conf->vid.sar_y},
#endif
#if ENCODER_FFMPEG
        {"use_highbit_depth", old_conf->enc.use_highbit_depth},
        {"output_csp", old_conf->enc.output_csp},
        {"pass", old_conf->enc.pass},
        {"bitrate", old_conf->enc.bitrate},
        {"interlaced", old_conf->enc.interlaced},
        {"audio_input", old_conf->enc.audio_input},
        {"outext", tchar_to_string(char_to_tstring(old_conf->vid.outext, CP_THREAD_ACP), CP_UTF8)},
        {"incmd", tchar_to_string(char_to_tstring(old_conf->vid.incmd, CP_THREAD_ACP), CP_UTF8)},
#endif
#if ENCODER_QSV || ENCODER_NVENC || ENCODER_VCEENC
        {"codec_rgy", tchar_to_string(get_cx_desc(list_rgy_codec, old_conf->enc.codec_rgy), CP_UTF8)},
        {"resize_enable", old_conf->vid.resize_enable},
        {"resize_width", old_conf->vid.resize_width},
        {"resize_height", old_conf->vid.resize_height},
#endif
        {"cmd", tchar_to_string(cmd_buffer, CP_UTF8) }
    };
    
    // 旧ビデオ設定をJSONに変換（char文字列をUTF-8に変換）
#if ENCODER_QSV || ENCODER_NVENC || ENCODER_VCEENC
    const char *cmdex = old_conf->enc.cmdex;
#else
    const char *cmdex = old_conf->vid.cmdex;
#endif
    j["video"] = {
        {"afs", old_conf->vid.afs},
        {"auo_tcfile_out", old_conf->vid.auo_tcfile_out},
#if !(ENCODER_QSV || ENCODER_NVENC || ENCODER_VCEENC)
        {"afs_bitrate_correction", old_conf->vid.afs_bitrate_correction},
        {"priority", old_conf->vid.priority},
#endif
#if ENCODER_X264 || ENCODER_X265 || ENCODER_SVTAV1
        {"stats", tchar_to_string(char_to_tstring(old_conf->vid.stats, CP_THREAD_ACP), CP_UTF8)},
#endif
#if ENABLE_AMP
        {"amp_check", old_conf->vid.amp_check},
        {"amp_limit_file_size", old_conf->vid.amp_limit_file_size},
        {"amp_limit_bitrate_upper", old_conf->vid.amp_limit_bitrate_upper},
        {"amp_limit_bitrate_lower", old_conf->vid.amp_limit_bitrate_lower},
#endif
#if ENCODER_X264 || ENCODER_X265
        {"check_keyframe", old_conf->vid.check_keyframe},
        {"input_as_lw48", old_conf->vid.input_as_lw48},
        {"tcfile_in", tchar_to_string(char_to_tstring(old_conf->vid.tcfile_in, CP_THREAD_ACP), CP_UTF8)},
        {"cqmfile", tchar_to_string(char_to_tstring(old_conf->vid.cqmfile, CP_THREAD_ACP), CP_UTF8)},
#endif
#if ENCODER_X265
        {"analysis_file", tchar_to_string(char_to_tstring(old_conf->vid.analysis_file, CP_THREAD_ACP), CP_UTF8)},
        {"parallel_div_info", tchar_to_string(char_to_tstring(old_conf->vid.parallel_div_info, CP_THREAD_ACP), CP_UTF8)},
        {"sync_process_affinity", old_conf->vid.sync_process_affinity},
#endif
        {"cmdex", tchar_to_string(char_to_tstring(cmdex, CP_THREAD_ACP), CP_UTF8)}
    };
    
    // オーディオ設定（変更なし）
    CONF_AUDIO_BASE aud_ = { 0 };
#if ENCODER_X264 || ENCODER_X265 || ENCODER_SVTAV1
    const CONF_AUDIO_BASE *aud_ext = &old_conf->aud;
    const CONF_AUDIO_BASE *aud_in = &aud_;
    const int aud_use_internal = 0;
#else
    const CONF_AUDIO_BASE *aud_ext = &old_conf->aud.ext;
    const CONF_AUDIO_BASE *aud_in = &old_conf->aud.in;
    const int aud_use_internal = old_conf->aud.use_internal;
#endif
    j["audio"] = {
        {"ext", {
            {"encoder", aud_ext->encoder},
            {"enc_mode", aud_ext->enc_mode},
            {"bitrate", aud_ext->bitrate},
            {"use_2pass", aud_ext->use_2pass},
            {"use_wav", aud_ext->use_wav},
            {"faw_check", aud_ext->faw_check},
            {"priority", aud_ext->priority},
            {"minimized", aud_ext->minimized},
            {"aud_temp_dir", aud_ext->aud_temp_dir},
            {"audio_encode_timing", aud_ext->audio_encode_timing},
            {"delay_cut", aud_ext->delay_cut}
        }},
        {"in", {
            {"encoder", aud_in->encoder},
            {"enc_mode", aud_in->enc_mode},
            {"bitrate", aud_in->bitrate},
            {"use_2pass", aud_in->use_2pass},
            {"use_wav", aud_in->use_wav},
            {"faw_check", aud_in->faw_check},
            {"priority", aud_in->priority},
            {"minimized", aud_in->minimized},
            {"aud_temp_dir", aud_in->aud_temp_dir},
            {"audio_encode_timing", aud_in->audio_encode_timing},
            {"delay_cut", aud_in->delay_cut}
        }},
        {"use_internal", aud_use_internal}
    };
    
    // Mux設定（変更なし）
#if ENCODER_X264 || ENCODER_X265 || ENCODER_SVTAV1
    const int mux_use_internal = 0;
#else
    const int mux_use_internal = old_conf->mux.use_internal;
#endif
    j["mux"] = {
        {"disable_mp4ext", old_conf->mux.disable_mp4ext},
        {"disable_mkvext", old_conf->mux.disable_mkvext},
        {"mp4_mode", old_conf->mux.mp4_mode},
        {"mkv_mode", old_conf->mux.mkv_mode},
        {"minimized", old_conf->mux.minimized},
        {"priority", old_conf->mux.priority},
        {"mp4_temp_dir", old_conf->mux.mp4_temp_dir},
        {"apple_mode", old_conf->mux.apple_mode},
        {"use_internal", mux_use_internal},
        {"internal_mode", old_conf->mux.internal_mode}
    };

    j["other"] = {
#if !(ENCODER_QSV || ENCODER_NVENC || ENCODER_VCEENC)
        {"disable_guicmd", old_conf->oth.disable_guicmd},
#endif
        {"temp_dir", old_conf->oth.temp_dir},
        {"out_audio_only", old_conf->oth.out_audio_only},
        {"notes", tchar_to_string(char_to_tstring(old_conf->oth.notes, CP_THREAD_ACP), CP_UTF8)},
        {"run_bat", old_conf->oth.run_bat},
        {"dont_wait_bat_fin", old_conf->oth.dont_wait_bat_fin},
        {"batfiles", {
            {"before_process", tchar_to_string(char_to_tstring(old_conf->oth.batfile.before_process, CP_THREAD_ACP), CP_UTF8)},
            {"after_process", tchar_to_string(char_to_tstring(old_conf->oth.batfile.after_process, CP_THREAD_ACP), CP_UTF8)},
            {"before_audio", tchar_to_string(char_to_tstring(old_conf->oth.batfile.before_audio, CP_THREAD_ACP), CP_UTF8)},
            {"after_audio", tchar_to_string(char_to_tstring(old_conf->oth.batfile.after_audio, CP_THREAD_ACP), CP_UTF8)}
        }}
    };
    
    return j.dump(4);
}

// ブロック別JSON変換関数の実装

// ビデオ設定をJSONに変換
void guiEx_config::video_to_json(nlohmann::json& j, const CONF_VIDEO& vid) {
    j["video"] = {
        {"afs", vid.afs},
        {"afs_bitrate_correction", vid.afs_bitrate_correction},
        {"auo_tcfile_out", vid.auo_tcfile_out},
        {"check_keyframe", vid.check_keyframe},
        {"priority", vid.priority},
        {"stats", tchar_to_string(vid.stats, CP_UTF8)},
        {"tcfile_in", tchar_to_string(vid.tcfile_in, CP_UTF8)},
        {"cqmfile", tchar_to_string(vid.cqmfile, CP_UTF8)},
        {"cmdex", tchar_to_string(vid.cmdex, CP_UTF8)},
        {"amp_check", vid.amp_check},
        {"amp_limit_file_size", vid.amp_limit_file_size},
        {"amp_limit_bitrate_upper", vid.amp_limit_bitrate_upper},
        {"input_as_lw48", vid.input_as_lw48},
        {"amp_limit_bitrate_lower", vid.amp_limit_bitrate_lower},
        {"analysis_file", tchar_to_string(vid.analysis_file, CP_UTF8)},
        {"parallel_div_info", tchar_to_string(vid.parallel_div_info, CP_UTF8)},
        {"sync_process_affinity", vid.sync_process_affinity}
    };
}

// オーディオ設定をJSONに変換
void guiEx_config::audio_to_json(nlohmann::json& j, const CONF_AUDIO& aud) {
    j["audio"] = {
        {"ext", {
            {"encoder", aud.ext.encoder},
            {"enc_mode", aud.ext.enc_mode},
            {"bitrate", aud.ext.bitrate},
            {"use_2pass", aud.ext.use_2pass},
            {"use_wav", aud.ext.use_wav},
            {"faw_check", aud.ext.faw_check},
            {"priority", aud.ext.priority},
            {"minimized", aud.ext.minimized},
            {"aud_temp_dir", aud.ext.aud_temp_dir},
            {"audio_encode_timing", aud.ext.audio_encode_timing},
            {"delay_cut", aud.ext.delay_cut}
        }},
        {"in", {
            {"encoder", aud.in.encoder},
            {"enc_mode", aud.in.enc_mode},
            {"bitrate", aud.in.bitrate},
            {"use_2pass", aud.in.use_2pass},
            {"use_wav", aud.in.use_wav},
            {"faw_check", aud.in.faw_check},
            {"priority", aud.in.priority},
            {"minimized", aud.in.minimized},
            {"aud_temp_dir", aud.in.aud_temp_dir},
            {"audio_encode_timing", aud.in.audio_encode_timing},
            {"delay_cut", aud.in.delay_cut}
        }},
        {"use_internal", aud.use_internal}
    };
}

// Mux設定をJSONに変換
void guiEx_config::mux_to_json(nlohmann::json& j, const CONF_MUX& mux) {
    j["mux"] = {
        {"disable_mp4ext", mux.disable_mp4ext},
        {"disable_mkvext", mux.disable_mkvext},
        {"mp4_mode", mux.mp4_mode},
        {"mkv_mode", mux.mkv_mode},
        {"minimized", mux.minimized},
        {"priority", mux.priority},
        {"mp4_temp_dir", mux.mp4_temp_dir},
        {"apple_mode", mux.apple_mode},
        {"use_internal", mux.use_internal},
        {"internal_mode", mux.internal_mode}
    };
}

// その他設定をJSONに変換
void guiEx_config::other_to_json(nlohmann::json& j, const CONF_OTHER& oth) {
    j["other"] = {
        {"disable_guicmd", oth.disable_guicmd},
        {"temp_dir", oth.temp_dir},
        {"out_audio_only", oth.out_audio_only},
        {"notes", tchar_to_string(oth.notes, CP_UTF8)},
        {"run_bat", oth.run_bat},
        {"dont_wait_bat_fin", oth.dont_wait_bat_fin},
        {"batfiles", {
            {"before_process", tchar_to_string(oth.batfile.before_process, CP_UTF8)},
            {"after_process", tchar_to_string(oth.batfile.after_process, CP_UTF8)},
            {"before_audio", tchar_to_string(oth.batfile.before_audio, CP_UTF8)},
            {"after_audio", tchar_to_string(oth.batfile.after_audio, CP_UTF8)}
        }}
    };
}

// JSONからビデオ設定を復元
void guiEx_config::json_to_video(const nlohmann::json& j, CONF_VIDEO& vid) {
    if (j.contains("video")) {
        auto& v = j["video"];
        vid.afs = v.value("afs", 0);
        vid.afs_bitrate_correction = v.value("afs_bitrate_correction", 0);
        vid.auo_tcfile_out = v.value("auo_tcfile_out", 0);
        vid.check_keyframe = v.value("check_keyframe", 0);
        vid.priority = v.value("priority", 0);
        vid.amp_check = v.value("amp_check", 0);
        vid.amp_limit_file_size = v.value("amp_limit_file_size", 0.0);
        vid.amp_limit_bitrate_upper = v.value("amp_limit_bitrate_upper", 0.0);
        vid.input_as_lw48 = v.value("input_as_lw48", 0);
        vid.amp_limit_bitrate_lower = v.value("amp_limit_bitrate_lower", 0.0);
        vid.sync_process_affinity = v.value("sync_process_affinity", 0);
        
        // 文字列の復元
        auto stats_tstr = char_to_tstring(v.value("stats", ""), CP_UTF8);
        _tcscpy_s(vid.stats, stats_tstr.c_str());
        
        auto tcfile_in_tstr = char_to_tstring(v.value("tcfile_in", ""), CP_UTF8);
        _tcscpy_s(vid.tcfile_in, tcfile_in_tstr.c_str());
        
        auto cqmfile_tstr = char_to_tstring(v.value("cqmfile", ""), CP_UTF8);
        _tcscpy_s(vid.cqmfile, cqmfile_tstr.c_str());
        
        auto cmdex_tstr = char_to_tstring(v.value("cmdex", ""), CP_UTF8);
        _tcscpy_s(vid.cmdex, cmdex_tstr.c_str());
        
        auto analysis_file_tstr = char_to_tstring(v.value("analysis_file", ""), CP_UTF8);
        _tcscpy_s(vid.analysis_file, analysis_file_tstr.c_str());
        
        auto parallel_div_info_tstr = char_to_tstring(v.value("parallel_div_info", ""), CP_UTF8);
        _tcscpy_s(vid.parallel_div_info, parallel_div_info_tstr.c_str());
    }
}

// JSONからオーディオ設定を復元
void guiEx_config::json_to_audio(const nlohmann::json& j, CONF_AUDIO& aud) {
    if (j.contains("audio")) {
        auto& a = j["audio"];
        
        // ext設定の復元
        if (a.contains("ext")) {
            auto& ext = a["ext"];
            aud.ext.encoder = ext.value("encoder", 0);
            aud.ext.enc_mode = ext.value("enc_mode", 0);
            aud.ext.bitrate = ext.value("bitrate", 0);
            aud.ext.use_2pass = ext.value("use_2pass", 0);
            aud.ext.use_wav = ext.value("use_wav", 0);
            aud.ext.faw_check = ext.value("faw_check", 0);
            aud.ext.priority = ext.value("priority", 0);
            aud.ext.minimized = ext.value("minimized", 0);
            aud.ext.aud_temp_dir = ext.value("aud_temp_dir", 0);
            aud.ext.audio_encode_timing = ext.value("audio_encode_timing", 0);
            aud.ext.delay_cut = ext.value("delay_cut", 0);
        }
        
        // in設定の復元
        if (a.contains("in")) {
            auto& in = a["in"];
            aud.in.encoder = in.value("encoder", 0);
            aud.in.enc_mode = in.value("enc_mode", 0);
            aud.in.bitrate = in.value("bitrate", 0);
            aud.in.use_2pass = in.value("use_2pass", 0);
            aud.in.use_wav = in.value("use_wav", 0);
            aud.in.faw_check = in.value("faw_check", 0);
            aud.in.priority = in.value("priority", 0);
            aud.in.minimized = in.value("minimized", 0);
            aud.in.aud_temp_dir = in.value("aud_temp_dir", 0);
            aud.in.audio_encode_timing = in.value("audio_encode_timing", 0);
            aud.in.delay_cut = in.value("delay_cut", 0);
        }
        
        aud.use_internal = a.value("use_internal", 0);
    }
}

// JSONからMux設定を復元
void guiEx_config::json_to_mux(const nlohmann::json& j, CONF_MUX& mux) {
    if (j.contains("mux")) {
        auto& m = j["mux"];
        mux.disable_mp4ext = m.value("disable_mp4ext", 0);
        mux.disable_mkvext = m.value("disable_mkvext", 0);
        mux.mp4_mode = m.value("mp4_mode", 0);
        mux.mkv_mode = m.value("mkv_mode", 0);
        mux.minimized = m.value("minimized", 0);
        mux.priority = m.value("priority", 0);
        mux.mp4_temp_dir = m.value("mp4_temp_dir", 0);
        mux.apple_mode = m.value("apple_mode", 0);
        mux.use_internal = m.value("use_internal", 0);
        mux.internal_mode = m.value("internal_mode", 0);
    }
}

// JSONからその他設定を復元
void guiEx_config::json_to_other(const nlohmann::json& j, CONF_OTHER& oth) {
    if (j.contains("other")) {
        auto& o = j["other"];
        oth.disable_guicmd = o.value("disable_guicmd", 0);
        oth.temp_dir = o.value("temp_dir", 0);
        oth.out_audio_only = o.value("out_audio_only", 0);
        oth.run_bat = o.value("run_bat", 0);
        oth.dont_wait_bat_fin = o.value("dont_wait_bat_fin", 0);
        
        auto notes_tstr = char_to_tstring(o.value("notes", ""), CP_UTF8);
        _tcscpy_s(oth.notes, notes_tstr.c_str());
        
        if (o.contains("batfiles") && o["batfiles"].is_object()) {
            auto& batfiles = o["batfiles"];
            
            auto before_process_tstr = char_to_tstring(batfiles.value("before_process", ""), CP_UTF8);
            _tcscpy_s(oth.batfile.before_process, before_process_tstr.c_str());
            
            auto after_process_tstr = char_to_tstring(batfiles.value("after_process", ""), CP_UTF8);
            _tcscpy_s(oth.batfile.after_process, after_process_tstr.c_str());
            
            auto before_audio_tstr = char_to_tstring(batfiles.value("before_audio", ""), CP_UTF8);
            _tcscpy_s(oth.batfile.before_audio, before_audio_tstr.c_str());
            
            auto after_audio_tstr = char_to_tstring(batfiles.value("after_audio", ""), CP_UTF8);
            _tcscpy_s(oth.batfile.after_audio, after_audio_tstr.c_str());
        }
    }
}

// 設定をJSON文字列に変換
std::string guiEx_config::conf_to_json(const CONF_GUIEX *conf, int indent) {
    nlohmann::json j;
    
    // ヘッダ情報
    j["conf_name"] = std::string(conf->header.conf_name);
    j["version"] = CONF_NAME_JSON;
    
    // エンコーダ設定
#if ENCODER_X264 || ENCODER_X265
    std::vector<TCHAR> cmd_buffer(MAX_CMD_LEN, 0);
    build_cmd_from_conf(cmd_buffer.data(), cmd_buffer.size(), &conf->enc, &conf->vid, FALSE);
    auto cmd_enc = tchar_to_string(cmd_buffer.data(), CP_UTF8);
#elif !ENCODER_FFMPEG
    auto cmd_enc = tchar_to_string(conf->enc.cmd, CP_UTF8);
#else
    auto cmd_enc = std::string();
#endif
    j["enc"] = {
#if ENABLE_AMP || ENCODER_FFMPEG
        {"use_auto_npass", conf->enc.use_auto_npass },
        {"auto_npass", conf->enc.auto_npass },
#endif
#if ENCODER_SVTAV1
        {"sar_x", conf->enc.sar_x},
        {"sar_y", conf->enc.sar_y},
#endif
#if ENCODER_FFMPEG
        {"use_highbit_depth", conf->enc.use_highbit_depth},
        {"output_csp", conf->enc.output_csp},
        {"pass", conf->enc.pass},
        {"bitrate", conf->enc.bitrate},
        {"interlaced", conf->enc.interlaced},
        {"audio_input", conf->enc.audio_input},
        {"outext", tchar_to_string(conf->enc.outext, CP_UTF8)},
        {"incmd", tchar_to_string(conf->enc.incmd, CP_UTF8)},
#endif
#if ENCODER_QSV || ENCODER_NVENC || ENCODER_VCEENC
        {"codec_rgy", tchar_to_string(CodecToStr(conf->enc.codec_rgy), CP_UTF8)},
        {"resize_enable", conf->enc.resize_enable},
        {"resize_width", conf->enc.resize_width},
        {"resize_height", conf->enc.resize_height},
#endif
        {"cmd", cmd_enc}
    };
    
    // 各ブロックを個別の関数で変換
    video_to_json(j, conf->vid);
    audio_to_json(j, conf->aud);
    mux_to_json(j, conf->mux);
    other_to_json(j, conf->oth);
    
    return j.dump(indent);
}

// JSON文字列から設定を復元
bool guiEx_config::json_to_conf(CONF_GUIEX *conf, const std::string &json_str) {
    try {
        auto j = nlohmann::json::parse(json_str);
        
        // 初期化
        ZeroMemory(conf, sizeof(CONF_GUIEX));
        write_conf_header(&conf->header);
        
        // ヘッダ情報
        if (j.contains("conf_name")) {
            std::string conf_name = j["conf_name"];
            strncpy_s(conf->header.conf_name, conf_name.c_str(), _countof(conf->header.conf_name) - 1);
        }

        // 各ブロックを個別の関数で復元
        json_to_video(j, conf->vid);
        json_to_audio(j, conf->aud);
        json_to_mux(j, conf->mux);
        json_to_other(j, conf->oth);
        
        // エンコーダ設定の復元
        if (j.contains("enc")) {
            auto& enc = j["enc"];
            auto cmd_str = char_to_tstring(enc.value("cmd", ""), CP_UTF8);
#if ENCODER_X264 || ENCODER_X265
            set_cmd_to_conf_full(cmd_str.c_str(), &conf->enc);
#elif ENCODER_SVTAV1
            _tcscpy_s(conf->enc.cmd, cmd_str.c_str());
            conf->enc.sar_x = enc.value("sar_x", 0);
            conf->enc.sar_y = enc.value("sar_y", 0);
#endif
#if ENABLE_AMP || ENCODER_FFMPEG
            conf->enc.use_auto_npass = enc.value("use_auto_npass", 0);
            conf->enc.auto_npass = enc.value("auto_npass", 0);
#endif
#if ENCODER_FFMPEG
            conf->enc.use_highbit_depth = enc.value("use_highbit_depth", 0);
            conf->enc.output_csp = enc.value("output_csp", 0);
            conf->enc.pass = enc.value("pass", 0);
            conf->enc.bitrate = enc.value("bitrate", 0);
            conf->enc.interlaced = enc.value("interlaced", 0);
            conf->enc.audio_input = enc.value("audio_input", 0);
            auto outext_tstr = char_to_tstring(enc.value("outext", ""), CP_UTF8);
            _tcscpy_s(conf->enc.outext, outext_tstr.c_str());
            auto incmd_tstr = char_to_tstring(enc.value("incmd", ""), CP_UTF8);
            _tcscpy_s(conf->enc.incmd, incmd_tstr.c_str());
#endif
#if ENCODER_QSV || ENCODER_NVENC || ENCODER_VCEENC
            conf->enc.codec_rgy = (RGY_CODEC)get_cx_value(list_rgy_codec, char_to_tstring(enc.value("codec_rgy", ""), CP_UTF8).c_str());
            conf->enc.resize_enable = enc.value("resize_enable", 0);
            conf->enc.resize_width = enc.value("resize_width", 0);
            conf->enc.resize_height = enc.value("resize_height", 0);
#endif
        }
        
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

void guiEx_config::write_conf_header(CONF_GUIEX_HEADER *save_conf) {
    sprintf_s(save_conf->conf_name, CONF_NAME);
    save_conf->size_all = sizeof(CONF_GUIEX);
    save_conf->head_size = CONF_HEAD_SIZE;
    save_conf->block_count = CONF_BLOCK_COUNT;
    for (int i = 0; i < CONF_BLOCK_COUNT; ++i) {
        save_conf->block_size[i] = conf_block_data[i];
        save_conf->block_head_p[i] = conf_block_pointer[i];
    }
}

// 新しい統合ロード関数 (JSON & バイナリ対応)
int guiEx_config::load_guiEx_conf(CONF_GUIEX *conf, const TCHAR *stg_file) {
    FILE *fp = NULL;
    if (_tfopen_s(&fp, stg_file, _T("rb")) != NULL)
        return CONF_ERROR_FILE_OPEN;
    
    // ファイルの最初の数バイトをチェックしてJSONかバイナリかを判定
    char header[32] = {0};
    fread(header, 1, sizeof(header) - 1, fp);
    fclose(fp);
    
    // JSONファイルかどうかチェック（'{' で始まるかどうか）
    if (header[0] == '{') {
        // JSONファイルとして読み込み
        std::ifstream file(stg_file, std::ios::binary);
        if (!file.is_open()) {
            return CONF_ERROR_FILE_OPEN;
        }
        
        std::string json_content((std::istreambuf_iterator<char>(file)),
                                std::istreambuf_iterator<char>());
        file.close();
        
        init_CONF_GUIEX(conf, FALSE);
        if (!json_to_conf(conf, json_content)) {
            return CONF_ERROR_FILE_OPEN;
        }
        
        write_conf_header(&conf->header);
        conf->header.size_all = CONF_INITIALIZED;
        return 0;
    } else {
        // 従来のバイナリファイルとして読み込み
        return load_guiEx_conf_legacy(conf, stg_file);
    }
}

// 旧形式のstgファイルから読み込み
int guiEx_config::load_guiEx_conf_legacy(CONF_GUIEX *conf, const TCHAR *stg_file) {
    size_t conf_size = 0;
    BYTE *dst, *filedat;
    //初期化
    ZeroMemory(conf, sizeof(CONF_GUIEX));
    //ファイルからロード
    FILE *fp = NULL;
    if (_tfopen_s(&fp, stg_file, _T("rb")) != NULL)
        return CONF_ERROR_FILE_OPEN;
    //設定ファイルチェック
    char conf_name[CONF_NAME_BLOCK_LEN + 32];
    fread(&conf_name, sizeof(char), CONF_NAME_BLOCK_LEN, fp);
    if (   strcmp(CONF_NAME,       conf_name)
        && strcmp(CONF_NAME_OLD_1, conf_name)
        && strcmp(CONF_NAME_OLD_2, conf_name)
        && strcmp(CONF_NAME_OLD_3, conf_name)
        && strcmp(CONF_NAME_OLD_4, conf_name)) {
        fclose(fp);
        return CONF_ERROR_FILE_OPEN;
    }
    fread(&conf_size, sizeof(int), 1, fp);
    BYTE *dat = (BYTE*)calloc(conf_size, 1);
    fseek(fp, 0, SEEK_SET);
    fread(dat, conf_size, 1, fp);
    fclose(fp);


    CONF_GUIEX_OLD old_conf_conv = { 0 };
    write_conf_header(&old_conf_conv.header);

    //旧設定ファイルから変換
    if (strcmp(conf_name, CONF_NAME_OLD_1) == 0) {
        convert_nvencstg_to_nvencstgv4(&old_conf_conv, dat);
        dat = (BYTE *)& old_conf_conv;
    } else if (strcmp(conf_name, CONF_NAME_OLD_2) == 0) {
        convert_nvencstgv2_to_nvencstgv4(&old_conf_conv, dat);
        dat = (BYTE *)&old_conf_conv;
    } else if (strcmp(conf_name, CONF_NAME_OLD_3) == 0) {
        convert_nvencstgv3_to_nvencstgv4(&old_conf_conv, dat);
        dat = (BYTE *)&old_conf_conv;
    }
    //ブロックサイズチェック
    if (((CONF_GUIEX_OLD *)dat)->header.block_count > CONF_BLOCK_COUNT)
        return CONF_ERROR_BLOCK_SIZE;

    CONF_GUIEX_OLD old_conf = { 0 };
    write_conf_header(&old_conf.header);
    dst = (BYTE *)&old_conf;
    //filedat = (BYTE *)data;
    //memcpy(dst, filedat, data->head_size);
    dst += CONF_HEAD_SIZE;


    const size_t conf_block_pointer_old[CONF_BLOCK_COUNT] = {
        offsetof(CONF_GUIEX_OLD, enc),
        offsetof(CONF_GUIEX_OLD, vid),
        offsetof(CONF_GUIEX_OLD, aud),
        offsetof(CONF_GUIEX_OLD, mux),
        offsetof(CONF_GUIEX_OLD, oth)
    };
    //ブロック部分のコピー
    for (int i = 0; i < ((CONF_GUIEX_OLD *)dat)->header.block_count; i++) {
        filedat = dat + ((CONF_GUIEX_OLD *)dat)->header.block_head_p[i];
        dst = (BYTE *)&old_conf + conf_block_pointer_old[i];
        memcpy(dst, filedat, std::min(((CONF_GUIEX_OLD *)dat)->header.block_size[i], conf_block_data[i]));
    }


    // 旧構造体からJSON文字列を作成
    std::string json_str = old_conf_to_json(&old_conf);
    
    //初期化
    ZeroMemory(conf, sizeof(CONF_GUIEX));
    init_CONF_GUIEX(conf, FALSE);
    write_conf_header(&conf->header);
    
    // JSON文字列から新構造体に変換
    if (!json_to_conf(conf, json_str)) {
        free(dat);
        return CONF_ERROR_FILE_OPEN;
    }

    //初期化するかどうかで使うので。
    conf->header.size_all = CONF_INITIALIZED;
    free(dat);
    return 0;
}

// 新しいJSON保存関数
int guiEx_config::save_guiEx_conf(const CONF_GUIEX *conf, const TCHAR *stg_file) {
    try {
        // 設定をJSONに変換
        std::string json_content = conf_to_json(conf, 2);
        
        // UTF-8でファイルに保存
        std::ofstream file(stg_file, std::ios::binary);
        if (!file.is_open()) {
            return CONF_ERROR_FILE_OPEN;
        }
        
        file.write(json_content.c_str(), json_content.length());
        file.close();
        
        return 0;
    } catch (const std::exception&) {
        return CONF_ERROR_FILE_OPEN;
    }
}
