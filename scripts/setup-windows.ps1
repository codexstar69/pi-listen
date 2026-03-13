#Requires -Version 5.1
<#
.SYNOPSIS
  pi-listen Windows bootstrap — Deepgram streaming voice input for Pi

.DESCRIPTION
  Installs SoX, the pi-listen extension, and writes voice settings.
  You should not need to run `/voice setup` on the happy path.

.PARAMETER Scope
  Where to save settings: "global" or "project" (default: global)

.PARAMETER ProjectDir
  Project directory for -Scope project (default: current dir)

.PARAMETER DeepgramKey
  Deepgram API key. Defaults to $env:DEEPGRAM_API_KEY.

.PARAMETER SkipPiInstall
  Skip `pi install npm:@codexstar/pi-listen`

.EXAMPLE
  .\scripts\setup-windows.ps1 -DeepgramKey "your-key-here"
#>

param(
    [ValidateSet("global", "project")]
    [string]$Scope = "global",

    [string]$ProjectDir = (Get-Location).Path,

    [string]$DeepgramKey = $env:DEEPGRAM_API_KEY,

    [switch]$SkipPiInstall
)

$ErrorActionPreference = 'Stop'

function Write-Step($msg) { Write-Host "  ✓ $msg" -ForegroundColor Green }
function Write-Warn($msg) { Write-Host "  ! $msg" -ForegroundColor Yellow }
function Write-Fail($msg) { Write-Host "  ✗ $msg" -ForegroundColor Red; exit 1 }

# ── Step 1: Install SoX ──────────────────────────────────────────────────

if (Get-Command rec -ErrorAction SilentlyContinue) {
    Write-Step "SoX already installed"
} else {
    Write-Step "Installing SoX via winget…"
    winget install --id ChrisBagwell.SoX --accept-source-agreements --accept-package-agreements
    Write-Step "SoX installed"
}

# ── Step 2: Install pi-listen extension ───────────────────────────────────

if (-not $SkipPiInstall) {
    Write-Step "Installing pi-listen extension…"
    pi install npm:@codexstar/pi-listen
    Write-Step "pi-listen installed"
} else {
    Write-Step "Skipping pi install"
}

# ── Step 3: Deepgram key ─────────────────────────────────────────────────

if (-not $DeepgramKey) {
    Write-Warn "DEEPGRAM_API_KEY not set."
    Write-Warn "Get a free key → https://dpgr.am/pi-voice (`$200 credits)"
    Write-Warn "Then: `$env:DEEPGRAM_API_KEY = 'your_key_here'"
}

# ── Step 4: Write voice settings ──────────────────────────────────────────

if ($Scope -eq "project") {
    $settingsDir = Join-Path $ProjectDir ".pi"
} else {
    $settingsDir = Join-Path $env:USERPROFILE ".pi\agent"
}

$settingsFile = Join-Path $settingsDir "settings.json"
New-Item -ItemType Directory -Path $settingsDir -Force | Out-Null

$voiceConfig = @{
    version = 2
    enabled = $true
    language = "en"
    scope = $Scope
    onboarding = @{
        completed = $true
        schemaVersion = 2
        completedAt = (Get-Date -Format "o")
        source = "setup-script"
    }
}

if ($DeepgramKey) {
    $voiceConfig.deepgramApiKey = $DeepgramKey
}

if (Test-Path $settingsFile) {
    $settings = Get-Content $settingsFile -Raw | ConvertFrom-Json
    $settings | Add-Member -NotePropertyName "voice" -NotePropertyValue $voiceConfig -Force
} else {
    $settings = @{ voice = $voiceConfig }
}

$settings | ConvertTo-Json -Depth 10 | Set-Content $settingsFile -Encoding UTF8
Write-Step "Settings written to $settingsFile"

# ── Step 5: Done ──────────────────────────────────────────────────────────

Write-Host ""
Write-Host "  Setup complete!"
Write-Host ""
Write-Host "  Next: open Pi and hold SPACE to talk."
if (-not $DeepgramKey) {
    Write-Host ""
    Write-Warn "Don't forget: set DEEPGRAM_API_KEY"
}
Write-Host ""
