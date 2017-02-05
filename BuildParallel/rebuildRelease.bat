cd /d "%~dp0"
call "%VS140COMNTOOLS%VsMSBuildCmd.bat"
msbuild release.rebuild.proj /m
