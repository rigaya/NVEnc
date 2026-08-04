#Requires -Version 5.1
<#
.SYNOPSIS
  NVEnc をビルドし、AviUtl2 Plugin へ配置して起動する。

.EXAMPLE
  .\scripts\dev_deploy.ps1
  .\scripts\dev_deploy.ps1 -NoLaunch
  .\scripts\dev_deploy.ps1 -SkipBuild
#>
[CmdletBinding()]
param(
    [ValidateSet("Release", "Debug")]
    [string]$Configuration = "Release",

    [ValidateSet("x64", "Win32")]
    [string]$Platform = "x64",

    [string]$AviUtl2Exe = "C:\ProgramEx\aviutl2\aviutl2.exe",

    [string]$PluginDir = "C:\ProgramEx\aviutl2\data\Plugin",

    [string]$Aup2 = "F:\temp\test3.aup2",

    [string]$MsBuild = "",

    [switch]$NoLaunch,

    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$Sln = Join-Path $RepoRoot "NVEnc.sln"

function Find-MSBuild {
    param([string]$Explicit)
    if ($Explicit -and (Test-Path -LiteralPath $Explicit)) {
        return (Resolve-Path -LiteralPath $Explicit).Path
    }
    $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path -LiteralPath $vswhere) {
        $found = & $vswhere -latest -products * -requires Microsoft.Component.MSBuild -find "MSBuild\**\Bin\MSBuild.exe" 2>$null
        if ($found) {
            return ($found | Select-Object -First 1)
        }
    }
    foreach ($cand in @(
            "${env:ProgramFiles}\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\MSBuild.exe",
            "${env:ProgramFiles}\Microsoft Visual Studio\18\Community\MSBuild\Current\Bin\MSBuild.exe"
        )) {
        if (Test-Path -LiteralPath $cand) { return $cand }
    }
    throw "MSBuild.exe が見つかりません。-MsBuild でパスを指定してください。"
}

Write-Host "==> repo: $RepoRoot"

if (-not $SkipBuild) {
    $msbuild = Find-MSBuild -Explicit $MsBuild
    Write-Host ("==> build: {0}/{1}" -f $Configuration, $Platform)
    Write-Host "    $msbuild"
    & $msbuild $Sln /t:Build "/p:Configuration=$Configuration" "/p:Platform=$Platform" /m /nologo /v:m
    if ($LASTEXITCODE -ne 0) {
        throw "MSBuild が失敗しました (exit=$LASTEXITCODE)"
    }
}

# NVEnc OutDir: _build\<Platform>\<Configuration>\
$outDir = Join-Path $RepoRoot ("_build\{0}\{1}" -f $Platform, $Configuration)
if ($Platform -eq "x64") {
    $srcAuo2 = Join-Path $outDir "NVEnc.auo2"
} else {
    $srcAuo2 = Join-Path $outDir "NVEnc.auo"
}

if (-not (Test-Path -LiteralPath $srcAuo2)) {
    throw "ビルド成果物が見つかりません: $srcAuo2"
}

if (-not (Test-Path -LiteralPath $PluginDir)) {
    throw "Plugin ディレクトリがありません: $PluginDir"
}

Write-Host "==> stop AviUtl2 (DLL unlock)"
Get-Process -Name "aviutl2" -ErrorAction SilentlyContinue | ForEach-Object {
    Write-Host ("    stop pid={0}" -f $_.Id)
    Stop-Process -Id $_.Id -Force
}
Start-Sleep -Milliseconds 400

$dstAuo2 = Join-Path $PluginDir "NVEnc.auo2"
Write-Host "==> copy"
Write-Host "    $srcAuo2"
Write-Host " -> $dstAuo2"
Copy-Item -LiteralPath $srcAuo2 -Destination $dstAuo2 -Force

# ini はソースツリーから配置 (OutDir にはコピーされない)
$srcIniDir = Join-Path $RepoRoot "NVEnc"
foreach ($name in @("NVEnc.ini", "NVEnc.en.ini", "NVEnc.zh.ini")) {
    $srcIni = Join-Path $srcIniDir $name
    if (Test-Path -LiteralPath $srcIni) {
        Copy-Item -LiteralPath $srcIni -Destination (Join-Path $PluginDir $name) -Force
    }
}

if ($NoLaunch) {
    Write-Host "==> done (no launch)"
    exit 0
}

if (-not (Test-Path -LiteralPath $AviUtl2Exe)) {
    throw "AviUtl2 が見つかりません: $AviUtl2Exe"
}
if (-not (Test-Path -LiteralPath $Aup2)) {
    throw "aup2 が見つかりません: $Aup2"
}

Write-Host "==> launch AviUtl2"
Write-Host "    $AviUtl2Exe"
Write-Host "    $Aup2"
Start-Process -FilePath $AviUtl2Exe -ArgumentList "`"$Aup2`""
Write-Host "==> done"
