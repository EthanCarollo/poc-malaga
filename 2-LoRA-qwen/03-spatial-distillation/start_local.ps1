param(
    [string]$Distro = 'Ubuntu',
    [string]$Profile = "$PSScriptRoot/configs/8gb.json",
    [int]$WaitSetupPid = 0,
    [int]$WaitDownloadPid = 0,
    [switch]$UsePreparedRuntime,
    [switch]$WaitForDataset,
    [switch]$Pokeapi,
    [string]$SpritesPath,
    [string]$AssetsPath
)
$ErrorActionPreference = 'Stop'
$taskProject = (Resolve-Path "$PSScriptRoot/../..").Path
$taskArtifacts = Join-Path $taskProject '2-LoRA-qwen/artifacts/spatial'
New-Item -ItemType Directory -Path $taskArtifacts -Force | Out-Null
$taskStatus = Join-Path $taskArtifacts 'bootstrap-status.json'

function Set-TaskStatus([string]$Stage, [string]$State = 'running', [string]$Message = '') {
    $value = @{ stage=$Stage; state=$State; pid=$PID; updated_at=[DateTime]::UtcNow.ToString('o'); message=$Message }
    $temporary = "$taskStatus.tmp"
    [IO.File]::WriteAllText($temporary, ($value | ConvertTo-Json), [Text.UTF8Encoding]::new($false))
    Move-Item -LiteralPath $temporary -Destination $taskStatus -Force
}

function Invoke-TaskWsl([string[]]$Arguments, [switch]$AsRoot) {
    if ($AsRoot) { & wsl -d $Distro --user root --exec @Arguments }
    else { & wsl -d $Distro --exec @Arguments }
    if ($LASTEXITCODE -ne 0) { throw "La commande WSL a échoué (code $LASTEXITCODE). Voir le journal bootstrap." }
}

try {
    Set-TaskStatus 'initializing'
    $taskLinuxRoot = (& wsl -d $Distro --exec wslpath -a $taskProject).Trim()
    $taskLinuxProfile = (& wsl -d $Distro --exec wslpath -a (Resolve-Path $Profile).Path).Trim()
    $taskPython = "$taskLinuxRoot/1-local-generation/.venv-qwen/bin/python"
    Set-TaskStatus 'waiting_existing_install'
    if ($WaitSetupPid -gt 0) {
        while ($true) {
            & wsl -d $Distro --exec test -d "/proc/$WaitSetupPid"
            if ($LASTEXITCODE -ne 0) { break }
            Start-Sleep -Seconds 5
        }
    }
    if (-not $UsePreparedRuntime) {
        Set-TaskStatus 'install_runtime'
        & wsl -d $Distro --exec /bin/sh -c 'command -v cc >/dev/null'
        if ($LASTEXITCODE -ne 0) {
            Invoke-TaskWsl -AsRoot -Arguments @('env', 'DEBIAN_FRONTEND=noninteractive', 'apt-get', 'update')
            Invoke-TaskWsl -AsRoot -Arguments @('env', 'DEBIAN_FRONTEND=noninteractive', 'apt-get', 'install', '-y', '--no-install-recommends', 'build-essential')
        }
        Invoke-TaskWsl -Arguments @('env', 'PIP_NO_CACHE_DIR=1', 'bash', "$taskLinuxRoot/2-LoRA-qwen/setup_sharm.sh")
        if ($WaitDownloadPid -gt 0) {
            Set-TaskStatus 'waiting_existing_download'
            $taskDownloadInfo = Get-CimInstance Win32_Process -Filter "ProcessId=$WaitDownloadPid"
            if ($taskDownloadInfo.CommandLine -like '*prepare_models.py*') {
                $taskDownload = Get-Process -Id $WaitDownloadPid -ErrorAction SilentlyContinue
                if ($taskDownload) { $taskDownload.WaitForExit() }
            }
        }
        Set-TaskStatus 'prepare_models'
        Invoke-TaskWsl -Arguments @('env', 'HF_XET_CHUNK_CACHE_SIZE_BYTES=0', $taskPython, "$taskLinuxRoot/2-LoRA-qwen/01-encoder-distillation/prepare_models.py")
    }
    if ($WaitForDataset) {
        Set-TaskStatus 'waiting_dataset' 'running' 'Attente du manifeste du corpus choisi.'
        while (-not (Test-Path -LiteralPath (Join-Path $taskArtifacts 'dataset/manifest.json'))) { Start-Sleep -Seconds 5 }
    }
    $taskArguments = @($taskPython, "$taskLinuxRoot/2-LoRA-qwen/03-spatial-distillation/run.py", '--profile', $taskLinuxProfile)
    if ($Pokeapi) { $taskArguments += '--pokeapi' }
    if ($SpritesPath) { $taskArguments += @('--sprites', (& wsl -d $Distro --exec wslpath -a $SpritesPath).Trim()) }
    if ($AssetsPath) { $taskArguments += @('--assets', (& wsl -d $Distro --exec wslpath -a $AssetsPath).Trim()) }
    Set-TaskStatus 'training_chain'
    # setup_sharm.sh and prepare_models.py have already verified/downloaded
    # the kernel and model files. Avoid Hub network calls in the long run.
    Invoke-TaskWsl -Arguments (@('env', 'HF_HUB_OFFLINE=1') + $taskArguments)
    Set-TaskStatus 'finished' 'complete' 'Candidat local terminé ; revue spatiale requise.'
}
catch {
    Set-TaskStatus 'failed' 'failed' $_.Exception.Message
    throw
}
