cd /d "%~dp0"
call "%VS140COMNTOOLS%VsMSBuildCmd.bat"

set CUDA_PATH=%CUDA_PATH_V8_0%
msbuild /m /p:Configuration=Release /p:Platform=Win32 /t:Clean ..\NVEnc.sln
msbuild /m /p:Configuration=RelStatic /p:Platform=x64 /t:Clean ..\NVEnc.sln
pause