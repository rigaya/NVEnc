#pragma once
//----------------------------------------------------------------------------------
//	出力プラグイン ヘッダーファイル for AviUtl ExEdit2
//	By ＫＥＮくん
//----------------------------------------------------------------------------------

//	出力プラグインは下記の関数を外部公開すると呼び出されます
//
//	出力プラグイン構造体のポインタを渡す関数 (必須)
//		OUTPUT_PLUGIN_TABLE* GetOutputPluginTable(void)
//
//	必要とする本体バージョン番号取得関数 (任意)
//		DWORD RequiredVersion() ※必要な本体のバージョン番号を返却します
//
//	プラグインDLL初期化関数 (任意)
//		bool InitializePlugin(DWORD version) ※versionは本体のバージョン番号
// 
//	プラグインDLL終了関数 (任意)
//		void UninitializePlugin()
// 
//	ログ出力機能初期化関数 (任意) ※logger2.h
//		void InitializeLogger(LOG_HANDLE* logger)
// 
//	設定関連機能初期化関数 (任意) ※config2.h
//		void InitializeConfig(CONFIG_HANDLE* config)
//
//	キャッシュ関連機能初期化関数 ※cache2.h
//		void InitializeCache(CACHE_HANDLE* cache)

//----------------------------------------------------------------------------------

// plugin2.hに定義されています
struct PROJECT_FILE;

// 出力情報構造体
struct OUTPUT_INFO {
	int flag;			//	フラグ
	static constexpr int FLAG_VIDEO = 1; // 画像データあり
	static constexpr int FLAG_AUDIO = 2; // 画像データあり
	int w, h;			//	縦横サイズ
	int rate, scale;	//	フレームレート、スケール
	int n;				//	フレーム数
	int audio_rate;		//	音声サンプリングレート
	int audio_ch;		//	音声チャンネル数
	int audio_n;		//	音声サンプリング数
	LPCWSTR savefile;	//	セーブファイル名へのポインタ

	// DIB形式の画像データを取得します
	// frame	: フレーム番号
	// format	: 画像フォーマット
	//			  0(BI_RGB) = RGB24bit / 'P''A''6''4' = PA64 / 'H''F''6''4' = HF64 / 'Y''U''Y''2' = YUY2 / 'Y''C''4''8' = YC48
	// ※PA64はDXGI_FORMAT_R16G16B16A16_UNORM(乗算済みα)です
	// ※HF64はDXGI_FORMAT_R16G16B16A16_FLOAT(乗算済みα)です(内部フォーマット)
	// ※YC48は互換対応のフォーマットです
	// 戻り値	: データへのポインタ
	//			  画像データポインタの内容は次に外部関数を使うかメインに処理を戻すまで有効
	void* (*func_get_video)(int frame, DWORD format);

	// PCM形式の音声データへのポインタを取得します
	// start	: 開始サンプル番号
	// length	: 読み込むサンプル数
	// readed	: 読み込まれたサンプル数
	// format	: 音声フォーマット
	//			  1(WAVE_FORMAT_PCM) = PCM16bit / 3(WAVE_FORMAT_IEEE_FLOAT) = PCM(float)32bit
	// 戻り値	: データへのポインタ
	//			  音声データポインタの内容は次に外部関数を使うかメインに処理を戻すまで有効
	void* (*func_get_audio)(int start, int length, int* readed, DWORD format);

	// 中断するか調べます
	// 戻り値	: trueなら中断
	bool (*func_is_abort)();

	// 残り時間を表示させます
	// now		: 処理しているフレーム番号
	// total	: 処理する総フレーム数
	// 戻り値	: trueなら成功
	void (*func_rest_time_disp)(int now, int total);

	// データ取得のバッファ数(フレーム数)を設定します ※標準は4になります
	// バッファ数の半分のデータを先読みリクエストするようになります
	// video	: 画像データのバッファ数
	// audio	: 音声データのバッファ数
	void (*func_set_buffer_size)(int video_size, int audio_size);
};

// 出力プラグイン構造体
struct OUTPUT_PLUGIN_TABLE {
	int flag;				// フラグ
	static constexpr int FLAG_VIDEO = 1; // 画像をサポートする
	static constexpr int FLAG_AUDIO = 2; // 音声をサポートする
	static constexpr int FLAG_IMAGE = 4; // 静止画出力のみサポートする (OUTPUT_INFOが1フレーム出力になります)
										 // ※静止画出力では出力完了時の通知やサウンド再生をしません
	static constexpr int FLAG_PROJECT_CONFIG = 8; // プロジェクトファイルの設定保持をサポートする
												  // ※プロジェクトファイル側に出力設定を保持する場合に指定します
	LPCWSTR name;			// プラグインの名前
	LPCWSTR filefilter;		// ファイルのフィルタ
	LPCWSTR information;	// プラグインの情報

	// 出力時に呼ばれる関数へのポインタ
	// 戻り値	: 成功時はtrueを返却
	bool (*func_output)(OUTPUT_INFO* oip);

	// 出力設定のダイアログを要求された時に呼ばれる関数へのポインタ (nullptrなら呼ばれません)
	// 戻り値	: 成功時はtrueを返却
	bool (*func_config)(HWND hwnd, HINSTANCE dll_hinst);

	// 出力設定のテキスト情報を取得する時に呼ばれる関数へのポインタ (nullptrなら呼ばれません)
	// 戻り値	: 出力設定のテキスト情報へのポインタ (次に関数が呼ばれるまで内容を有効にしておく)
	LPCWSTR (*func_get_config_text)();

	// プロジェクトファイル側から出力設定の読み込み要求時に呼ばれる関数へのポインタ (FLAG_PROJECT_CONFIGが有効の時のみ呼ばれます)
	// project	: プロジェクトファイル構造体へのポインタ
	// 戻り値	: 成功時はtrueを返却
	bool (*func_load_project_config)(PROJECT_FILE* project);

	// プロジェクトファイル側への出力設定の書き込み要求時に呼ばれる関数へのポインタ (FLAG_PROJECT_CONFIGが有効の時のみ呼ばれます)
	// project	: プロジェクトファイル構造体へのポインタ
	// 戻り値	: 成功時はtrueを返却
	bool (*func_save_project_config)(PROJECT_FILE* project);

};
