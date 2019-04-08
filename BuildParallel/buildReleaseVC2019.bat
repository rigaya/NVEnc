cd /d "%~dp0"
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat" x64
set CUDA_PATH=%CUDA_PATH_V10_1%
msbuild release.build.vc2019.proj /m:2
pause
