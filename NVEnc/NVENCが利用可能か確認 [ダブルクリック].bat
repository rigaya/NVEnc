@echo off
exe_files\NVEncC\x86\NVEncC.exe --check-hw
if %errorlevel% == 0 (
    echo NVENCは利用可能です。
) else (
    echo NVENCが利用できません。
)
pause