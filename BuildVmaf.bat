set WIN_TARGET_PLATFORM=10.0
set PLATFORM_TOOLSET=v142
MSBuild "..\vmaf-1.3.15\vmaf.sln" /property:WindowsTargetPlatformVersion=%WIN_TARGET_PLATFORM%;PlatformToolset=%PLATFORM_TOOLSET%;Configuration="Release";Platform=%1;WholeProgramOptimization=true;ConfigurationType=StaticLibrary;ForceImportBeforeCppTargets="BuildVmaf.props" /p:BuildProjectReferences=true /p:SpectreMitigation=false