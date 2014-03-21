//----------------------------------------------------------------------------------
//	出力プラグイン ヘッダーファイル for AviUtl version 0.99g4 以降
//	By ＫＥＮくん
//----------------------------------------------------------------------------------
//AviUtlPluginSDK License
//-----------------------
//
//The MIT License
//
//Copyright (c) 1999-2011 Kenkun
//
//Permission is hereby granted, free of charge, to any person obtaining a copy
//of this software and associated documentation files (the "Software"), to deal
//in the Software without restriction, including without limitation the rights
//to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//copies of the Software, and to permit persons to whom the Software is
//furnished to do so, subject to the following conditions:
//
//The above copyright notice and this permission notice shall be included in
//all copies or substantial portions of the Software.
//
//THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
//THE SOFTWARE.
//
//----------------------------------------------------------------------------------

#ifndef _AUO_OUTPUT_H_ 
#define _AUO_OUTPUT_H_

//	出力情報構造体
typedef struct {
	int		flag;			//	フラグ
							//	OUTPUT_INFO_FLAG_VIDEO	: 画像データあり
							//	OUTPUT_INFO_FLAG_AUDIO	: 音声データあり
							//	OUTPUT_INFO_FLAG_BATCH	: バッチ出力中
	int		w,h;			//	縦横サイズ
	int		rate,scale;		//	フレームレート
	int		n;				//	フレーム数
	int		size;			//	１フレームのバイト数
	int		audio_rate;		//	音声サンプリングレート
	int		audio_ch;		//	音声チャンネル数
	int		audio_n;		//	音声サンプリング数
	int		audio_size;		//	音声１サンプルのバイト数
	LPSTR	savefile;		//	セーブファイル名へのポインタ
	void	*(*func_get_video)( int frame );
							//	DIB形式(RGB24bit)の画像データへのポインタを取得します。
							//	frame	: フレーム番号
							//	戻り値	: データへのポインタ
							//			  画像データポインタの内容は次に外部関数を使うかメインに処理を戻すまで有効
	void	*(*func_get_audio)( int start,int length,int *readed );
							//	16bitPCM形式の音声データへのポインタを取得します。
							//	start	: 開始サンプル番号
							//	length	: 読み込むサンプル数
							//	readed	: 読み込まれたサンプル数
							//	戻り値	: データへのポインタ
							//			  音声データポインタの内容は次に外部関数を使うかメインに処理を戻すまで有効
	BOOL	(*func_is_abort)( void );
							//	中断するか調べます。
							//	戻り値	: TRUEなら中断
	BOOL	(*func_rest_time_disp)( int now,int total );
							//	残り時間を表示させます。
							//	now		: 処理しているフレーム番号
							//	total	: 処理する総フレーム数
							//	戻り値	: TRUEなら成功
	int		(*func_get_flag)( int frame );
							//	フラグを取得します。
							//	frame	: フレーム番号
							//	戻り値	: フラグ
							//  OUTPUT_INFO_FRAME_FLAG_KEYFRAME		: キーフレーム推奨
							//  OUTPUT_INFO_FRAME_FLAG_COPYFRAME	: コピーフレーム推奨
	BOOL	(*func_update_preview)( void );
							//	プレビュー画面を更新します。
							//	最後にfunc_get_videoで読み込まれたフレームが表示されます。
							//	戻り値	: TRUEなら成功
	void	*(*func_get_video_ex)( int frame,DWORD format );
							//	DIB形式の画像データを取得します。
							//	frame	: フレーム番号
							//	format	: 画像フォーマット( NULL = RGB24bit / 'Y''U''Y''2' = YUY2 / 'Y''C''4''8' = PIXEL_YC )
							//			  ※PIXEL_YC形式 は YUY2フィルタモードでは使用出来ません。
							//	戻り値	: データへのポインタ
							//			  画像データポインタの内容は次に外部関数を使うかメインに処理を戻すまで有効
} OUTPUT_INFO;
#define	OUTPUT_INFO_FLAG_VIDEO	1
#define	OUTPUT_INFO_FLAG_AUDIO	2
#define	OUTPUT_INFO_FLAG_BATCH	4
#define	OUTPUT_INFO_FRAME_FLAG_KEYFRAME		1
#define	OUTPUT_INFO_FRAME_FLAG_COPYFRAME	2

//	出力プラグイン構造体
typedef	struct {
	int		flag;			//	フラグ
	LPSTR	name;			//	プラグインの名前
	LPSTR	filefilter;		//	ファイルのフィルタ
	LPSTR	information;	//	プラグインの情報
	BOOL	(*func_init)( void );
							//	DLL開始時に呼ばれる関数へのポインタ (NULLなら呼ばれません)
	BOOL	(*func_exit)( void );
							//	DLL終了時に呼ばれる関数へのポインタ (NULLなら呼ばれません)
	BOOL	(*func_output)( OUTPUT_INFO *oip );
							//	出力時に呼ばれる関数へのポインタ
	BOOL	(*func_config)( HWND hwnd,HINSTANCE dll_hinst );
							//	出力設定のダイアログを要求された時に呼ばれる関数へのポインタ (NULLなら呼ばれません)
	int		(*func_config_get)( void *data,int size );
							//	出力設定データを取得する時に呼ばれる関数へのポインタ (NULLなら呼ばれません)
							//	data	: 設定データを書き込むバッファへのポインタ (NULLなら設定データサイズを返すだけ)
							//	size	: 設定データを書き込むバッファのサイズ
							//	戻り値	: 設定データのサイズ
	int		(*func_config_set)( void *data,int size );
							//	出力設定データを設定する時に呼ばれる関数へのポインタ (NULLなら呼ばれません)
							//	data	: 設定データへのポインタ
							//	size	: 設定データのサイズ
							//	戻り値	: 使用した設定データのサイズ
	int		reserve[16];	//	拡張用に予約されてます
} OUTPUT_PLUGIN_TABLE;

BOOL func_init( void );
BOOL func_exit( void );
BOOL func_output( OUTPUT_INFO *oip );
BOOL func_config( HWND hwnd,HINSTANCE hinst );
int func_config_get( void *data,int size );
int func_config_set( void *data,int size );


#endif //_AUO_OUTPUT_H_