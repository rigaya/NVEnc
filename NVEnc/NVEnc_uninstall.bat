@echo off
setlocal
rem バッチファイル名: %TARGET%_uninstall.bat を前提に TARGET を抽出
set "TARGET=%~n0"
if /i "%TARGET:~-10%"=="_uninstall" (
  set "TARGET=%TARGET:~0,-10%"
) else (
  echo.
  echo [警告] バッチファイル名が "%TARGET%_uninstall.bat" の形式ではありません: "%~nx0"
  echo        このまま "%TARGET%" を TARGET として続行します。
)

rem 実行場所をこのbatのあるフォルダに固定
cd /d "%~dp0" >nul 2>&1

rem 目印ファイルの存在確認
set TARGET_AUO_VER=2
if not exist "%TARGET%.auo2" (
  if not exist "%TARGET%.auo" (
    echo.
    echo [%TARGET%] のアンインストールを開始できません。
    echo "%TARGET%.auo2" および "%TARGET%.auo" が見つかりません。
    echo このbatは %TARGET% の設定ファイルがあるフォルダで実行してください。
    echo.
    pause
    endlocal
    exit /b 1
  )
  set TARGET_AUO_VER=1
)

echo.
echo [%TARGET%] をアンインストールします。
echo 設定ファイル等を削除します。よろしいですか?
echo.
choice /c yn /m "削除を実行しますか (Yes/No)"
if errorlevel 2 (
  echo.
  echo アンインストールを中止しました。
  echo.
  pause
  endlocal
  exit /b 0
)

echo.
echo 削除中...
del /q "%TARGET%.auo2"  >nul 2>&1
del /q "%TARGET%.auo"   >nul 2>&1
del /q "%TARGET%.conf"  >nul 2>&1
del /q "%TARGET%.ini"   >nul 2>&1
del /q "%TARGET%.*.ini" >nul 2>&1
if exist "%TARGET%_stg" rmdir /s /q "%TARGET%_stg" >nul 2>&1

set EXE_DIR="exe_files"
if %TARGET_AUO_VER%==1 if exist "..\exe_files" set EXE_DIR="..\exe_files"

rem %TARGET% から後ろのguiExを削除
set "TARGET_EXE=%TARGET:~0,-5%"
if exist "%EXE_DIR%\%TARGET_EXE%*.exe" del /q "%EXE_DIR%\%TARGET_EXE%*.exe" >nul 2>&1

echo.
echo アンインストール処理が完了しました。
echo なにかキーを押すと終了します。
echo.
pause > nul 2>&1
rem 正常終了時のみ、このバッチ自身を削除（別プロセスで遅延削除）
start "" /b cmd /c "timeout /t 1 /nobreak >nul & del /f /q ""%~f0"" >nul 2>&1"
endlocal
