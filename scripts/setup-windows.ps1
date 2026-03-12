param(
    [ValidateSet('local', 'api')]
    [string]$Mode = 'local',

    [ValidateSet('faster-whisper', 'moonshine', 'parakeet')]
    [string]$Backend = 'faster-whisper',

    [ValidateSet('global', 'project')]
    [string]$Scope = 'global',

    [string]$ProjectDir = (Get-Location).Path,

    [string]$DeepgramKey = $env:DEEPGRAM_API_KEY,

    [switch]$PersistDeepgramKey,

    [switch]$SkipPiInstall
)

$ErrorActionPreference = 'Stop'
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$PackageRoot = Split-Path -Parent $ScriptDir

function Write-Step {
    param([string]$Message)
    Write-Host "`n[pi-listen] $Message"
}

function Write-Warn {
    param([string]$Message)
    Write-Warning $Message
}

function Test-Command {
    param([string]$Name)
    return [bool](Get-Command $Name -ErrorAction SilentlyContinue)
}

function Install-WingetPackage {
    param([string]$Id)
    if (-not (Test-Command winget)) {
        throw "winget is required for automatic Windows setup. Install App Installer / winget first."
    }

    & winget install -e --id $Id --accept-package-agreements --accept-source-agreements
}

function Get-SelectedBackend {
    if ($Mode -eq 'api') { return 'deepgram' }
    return $Backend
}

function Get-SelectedModel {
    switch (Get-SelectedBackend) {
        'faster-whisper' { return 'small' }
        'moonshine' { return 'moonshine/base' }
        'parakeet' { return 'nvidia/parakeet-tdt-0.6b-v2' }
        'deepgram' { return 'nova-3' }
        default { throw "No default model for backend $(Get-SelectedBackend)" }
    }
}

function Get-SettingsPath {
    if ($Scope -eq 'project') {
        return Join-Path $ProjectDir '.pi\settings.json'
    }
    return Join-Path $HOME '.pi\agent\settings.json'
}

function Invoke-Python {
    param([string[]]$Arguments)
    & $script:PythonCmd @script:PythonArgs @Arguments
}

function Install-PiPackage {
    if ($SkipPiInstall) { return }

    if (-not (Test-Command pi)) {
        Write-Warn "The 'pi' command was not found, so package installation was skipped. Install Pi first, then run: pi install npm:pi-listen"
        return
    }

    if ($Scope -eq 'project') {
        Write-Step 'Installing pi-listen into Pi project settings'
        Push-Location $ProjectDir
        try {
            & pi install -l npm:pi-listen
        }
        finally {
            Pop-Location
        }
    }
    else {
        Write-Step 'Installing pi-listen into global Pi settings'
        & pi install npm:pi-listen
    }
}

function Write-VoiceConfig {
    $settingsPath = Get-SettingsPath
    $settingsDir = Split-Path -Parent $settingsPath
    $null = New-Item -ItemType Directory -Force -Path $settingsDir

    $data = @{}
    if (Test-Path $settingsPath) {
        try {
            $data = Get-Content $settingsPath -Raw | ConvertFrom-Json -AsHashtable
        }
        catch {
            $data = @{}
        }
    }

    $timestamp = (Get-Date).ToUniversalTime().ToString('o').Replace('+00:00', 'Z')
    $data['voice'] = @{
        version = 2
        enabled = $true
        language = 'en'
        mode = $Mode
        backend = (Get-SelectedBackend)
        model = (Get-SelectedModel)
        scope = $Scope
        btwEnabled = $true
        onboarding = @{
            completed = $true
            schemaVersion = 2
            completedAt = $timestamp
            lastValidatedAt = $timestamp
            source = 'setup-command'
        }
    }

    $data | ConvertTo-Json -Depth 8 | Set-Content -Path $settingsPath -Encoding UTF8
    Write-Step "Wrote ready-to-use voice config to $settingsPath"
}

function Smoke-Test {
    $selectedBackend = Get-SelectedBackend
    $selectedModel = Get-SelectedModel
    $transcribePath = Join-Path $PackageRoot 'transcribe.py'
    $daemonPath = Join-Path $PackageRoot 'daemon.py'

    Write-Step 'Checking backend availability via transcribe.py'
    $backends = Invoke-Python @($transcribePath, '--list-backends') | ConvertFrom-Json
    $selected = $backends | Where-Object { $_.name -eq $selectedBackend } | Select-Object -First 1
    if (-not $selected) {
        throw "Backend $selectedBackend was not returned by transcribe.py --list-backends"
    }
    if (-not $selected.available) {
        throw "Backend $selectedBackend is still not available after setup"
    }

    Write-Step "Starting daemon smoke test for backend=$selectedBackend model=$selectedModel"
    $socket = Join-Path $env:TEMP ("pi-listen-bootstrap-" + [guid]::NewGuid().ToString() + '.sock')
    $stdoutLog = Join-Path $env:TEMP ("pi-listen-daemon-stdout-" + [guid]::NewGuid().ToString() + '.log')
    $stderrLog = Join-Path $env:TEMP ("pi-listen-daemon-stderr-" + [guid]::NewGuid().ToString() + '.log')
    $pythonExe = (Get-Command $PythonCmd).Source
    $daemonArgs = @($PythonArgs + @($daemonPath, 'start', '--socket', $socket, '--backend', $selectedBackend, '--model', $selectedModel))
    $daemon = Start-Process -FilePath $pythonExe -ArgumentList $daemonArgs -PassThru -WindowStyle Hidden -RedirectStandardOutput $stdoutLog -RedirectStandardError $stderrLog

    try {
        $ready = $false
        for ($i = 0; $i -lt 120; $i++) {
            Invoke-Python @($daemonPath, 'ping', '--socket', $socket) *> $null
            if ($LASTEXITCODE -eq 0) {
                $ready = $true
                break
            }
            Start-Sleep -Seconds 1
        }

        if (-not $ready) {
            $stderrTail = if (Test-Path $stderrLog) { Get-Content $stderrLog -Tail 40 | Out-String } else { '' }
            throw "Daemon failed to start cleanly.`n$stderrTail"
        }

        Invoke-Python @($daemonPath, 'status', '--socket', $socket) | Out-Null
        Write-Step 'Daemon smoke test passed'
    }
    finally {
        Invoke-Python @($daemonPath, 'stop', '--socket', $socket) *> $null
        if (-not $daemon.HasExited) {
            Wait-Process -Id $daemon.Id -Timeout 20 -ErrorAction SilentlyContinue
        }
        Remove-Item -Force -ErrorAction SilentlyContinue $socket, "$socket.pid", $stdoutLog, $stderrLog
    }
}

if (-not (Test-Command python) -and -not (Test-Command py)) {
    Write-Step 'Installing Python 3.12'
    Install-WingetPackage 'Python.Python.3.12'
}

if (-not (Test-Command rec)) {
    Write-Step 'Installing SoX'
    Install-WingetPackage 'ChrisBagwell.SoX'
} else {
    Write-Step 'SoX already available'
}

$ProjectDir = (Resolve-Path $ProjectDir).Path
$script:PythonCmd = if (Test-Command py) { 'py' } else { 'python' }
$script:PythonArgs = if ($script:PythonCmd -eq 'py') { @('-3') } else { @() }

Write-Step 'Ensuring pip tooling is ready'
Invoke-Python @('-m', 'pip', 'install', '--upgrade', 'pip') | Out-Null

if ($Mode -eq 'local') {
    switch ($Backend) {
        'faster-whisper' {
            Write-Step 'Installing faster-whisper'
            Invoke-Python @('-m', 'pip', 'install', 'faster-whisper')
        }
        'moonshine' {
            Write-Step 'Installing moonshine'
            Invoke-Python @('-m', 'pip', 'install', 'useful-moonshine[onnx]')
        }
        'parakeet' {
            Write-Step 'Installing parakeet'
            Invoke-Python @('-m', 'pip', 'install', 'nemo_toolkit[asr]')
        }
    }
}
else {
    if (-not $DeepgramKey) {
        throw 'API mode requires DEEPGRAM_API_KEY. Pass -DeepgramKey or set DEEPGRAM_API_KEY first.'
    }

    $env:DEEPGRAM_API_KEY = $DeepgramKey
    Write-Step 'Loaded DEEPGRAM_API_KEY into the current PowerShell session'
    if ($PersistDeepgramKey) {
        [Environment]::SetEnvironmentVariable('DEEPGRAM_API_KEY', $DeepgramKey, 'User')
        Write-Step 'Persisted DEEPGRAM_API_KEY to the current user environment'
    }
}

Install-PiPackage
Smoke-Test
Write-VoiceConfig

@"

Done.

Happy path result:
  - dependencies installed
  - selected backend validated
  - daemon smoke test passed
  - Pi voice settings written automatically

Selected configuration:
  mode: $Mode
  backend: $(Get-SelectedBackend)
  model: $(Get-SelectedModel)
  scope: $Scope

Next steps:
  1. Grant microphone permission once:
       Settings -> Privacy & security -> Microphone
       Allow microphone access for your terminal app / shell host

  2. Start Pi:
       $(if ($Scope -eq 'project') { "Set-Location '$ProjectDir'; pi" } else { 'pi' })

  3. Try hold-to-talk in an empty editor.

You should not need to run `/voice setup` on the happy path.
If you want a quick verification inside Pi, run:
  /voice test
  /voice doctor
"@ | Write-Host
