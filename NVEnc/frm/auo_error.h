//  -----------------------------------------------------------------------------------------
//    NVEnc by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  -----------------------------------------------------------------------------------------

#ifndef _AUO_ERROR_H_
#define _AUO_ERROR_H_

void warning_failed_getting_temp_path();
void warning_no_temp_root(const char *dir);
void warning_no_aud_temp_root(const char *dir);
void error_filename_too_long();
void error_nothing_to_output();
void info_afs_audio_delay_confliction();
void error_invalid_resolution(BOOL width, int mul, int w, int h);
void error_log_line_cache();
void error_no_exe_file(const char *name, const char *path);
void warning_auto_afs_disable();
void error_afs_setup(BOOL afs, BOOL auto_afs_disable);
void error_run_process(const char *exe_name, int rp_ret);
void warning_auto_qpfile_failed();
void warning_auo_tcfile_failed();
void error_open_wavfile();
void warning_audio_length();

void error_malloc_pixel_data();
void error_malloc_tc();
void error_malloc_8bit();
void error_afs_interlace_stg();
void error_x264_dead();
void error_x264_version();
void error_afs_get_frame();
void error_audenc_failed(const char *name, const char *args);
void error_no_wavefile();
void error_mux_failed(const char *name, const char *args);

void warning_no_mux_tmp_root(const char *dir);
void warning_failed_mux_tmp_drive_space();
void error_failed_muxer_drive_space();
void error_failed_out_drive_space();
void warning_failed_get_aud_size();
void warning_failed_get_vid_size();
void error_no_vid_file();
void error_no_aud_file();
void error_vid_file_zero_byte();
void error_aud_file_zero_byte();
void warning_mux_tmp_not_enough_space();
void error_muxer_drive_not_enough_space();
void error_out_drive_not_enough_space();
void error_check_muxout_exist();
void error_check_muxout_too_small(int expected_filesize_KB, int muxout_filesize_KB);
void warning_failed_check_muxout_filesize();

void warning_no_auto_save_log_dir();

void info_encoding_aborted();

void warning_mux_no_chapter_file();
void warning_mux_chapter(int sts);

void warning_no_batfile(const char *batfile);
void warning_malloc_batfile_tmp();
void warning_failed_open_bat_orig();
void warning_failed_open_bat_new();

#endif //_AUO_ERROR_H_