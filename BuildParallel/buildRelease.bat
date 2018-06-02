cd /d "%~dp0"
call "%VS140COMNTOOLS%VsMSBuildCmd.bat"
msbuild release.build.proj /m:3
