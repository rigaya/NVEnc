cd /d "%~dp0"
call "%VS140COMNTOOLS%VsMSBuildCmd.bat"

set CUDA_PATH=%CUDA_PATH_V8_0%
msbuild /p:Configuration=RelStatic /p:Platform=Win32 /t:Clean ..\NVEnc.sln
msbuild /p:Configuration=RelFilters /p:Platform=Win32 /t:Clean ..\NVEnc.sln
pause