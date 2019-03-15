cd /d "%~dp0"
call "%VS140COMNTOOLS%VsMSBuildCmd.bat"

set CUDA_PATH=%CUDA_PATH_V8_0%
msbuild release.build.proj /m:2

set CUDA_PATH=%CUDA_PATH_V10_1%
msbuild /t:Rebuild /p:Configuration=RelStatic;Platform=x64 ..\NVEnc.sln
