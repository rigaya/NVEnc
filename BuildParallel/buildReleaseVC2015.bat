cd /d "%~dp0"
call "%VS140COMNTOOLS%VsMSBuildCmd.bat"

set CUDA_PATH=%CUDA_PATH_V8_0%
msbuild release.build.vc2015.proj /m:2
pause