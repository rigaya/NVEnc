//----------------------------------------------------------------------------------
//	ログ出力機能 ヘッダーファイル for AviUtl ExEdit2
//	By ＫＥＮくん
//----------------------------------------------------------------------------------

//	各種プラグインで下記の関数を外部公開すると呼び出されます
// 
//	ログ出力機能初期化関数
//		void InitializeLogger(LOG_HANDLE* logger)
//		※InitializePlugin()より先に呼ばれます

//----------------------------------------------------------------------------------

// ログ出力ハンドル
struct LOG_HANDLE {
	// プラグイン用のログを出力します
	// handle	: ログ出力ハンドル
	// message	: ログメッセージ
	void (*log)(LOG_HANDLE* handle, LPCWSTR message);

	// infoレベルのログを出力します
	// handle	: ログ出力ハンドル
	// message	: ログメッセージ
	void (*info)(LOG_HANDLE* handle, LPCWSTR message);

	// warnレベルのログを出力します
	// handle	: ログ出力ハンドル
	// message	: ログメッセージ
	void (*warn)(LOG_HANDLE* handle, LPCWSTR message);

	// errorレベルのログを出力します
	// handle	: ログ出力ハンドル
	// message	: ログメッセージ
	void (*error)(LOG_HANDLE* handle, LPCWSTR message);

	// verboseレベルのログを出力します
	// handle	: ログ出力ハンドル
	// message	: ログメッセージ
	void (*verbose)(LOG_HANDLE* handle, LPCWSTR message);

};
