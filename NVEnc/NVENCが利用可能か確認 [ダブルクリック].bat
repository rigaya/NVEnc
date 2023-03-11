@echo off
set NVENCC_PATH=exe_files\NVEncC\x64\NVEncC64.exe
if "%PROCESSOR_ARCHITECTURE%" == "x86" (
    set NVENCC_PATH=exe_files\NVEncC\x86\NVEncC.exe
)

%NVENCC_PATH% --check-hw --log-level debug
if %errorlevel% == 0 (
    echo NVENCは利用可能です。
) else (
    echo 環境変数CUDA_VISIBLE_DEVICESを指定して再試行します…
    set CUDA_VISIBLE_DEVICES=0
    %NVENCC_PATH% --check-hw --log-level debug

    if %errorlevel% == 0 (
        echo NVENCは利用可能です。
    ) else (
        echo NVENCが利用できません。
        nvidia-smi
    )
)
pause