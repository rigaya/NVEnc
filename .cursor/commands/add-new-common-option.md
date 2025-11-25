# add-new-common-option

新たに共通オプションを作成するタスクです。
別途与えられた指定に従って、新しいオプションを作成してください。

- NVEncCore/rgy_prm.hに以下を追加してください。

  RGYParamXXX構造体に指定の変数を追加してください。
  どの構造体かはこちらで別途してします。指定がない場合は、確認を行ってください。

- NVEncCore/rgy_prm.cppに以下を追加してください。

  NVEncCore/rgy_prm.hに追加した変数の初期化子を追加してください。

- NVEncCore/rgy_cmd.cppに以下を追加してください。

  - parse_one_xxx_optionに読み取り部分を追加。
  - gen_cmdにコマンド生成部分を追加。
  - gen_cmd_help_xxxにヘルプ生成部分を追加。

- NVEncC_Options.ja.md にヘルプを記載してください。

  追記位置は、関連オプションの近くに配置できるよう検討しましょう。

- NVEncC_Options.en.md に NVEncC_Options.ja.md に記載した内容を英語に翻訳してヘルプを記載してください。