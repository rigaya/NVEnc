cd /d "%~dp0"
call "%VS140COMNTOOLS%VsMSBuildCmd.bat"
msbuild build.all.proj /m