$ErrorActionPreference = 'Stop'
$LinuxPath = (wsl.exe wslpath -a "$PSScriptRoot").Trim()
if (-not $LinuxPath) { throw 'Impossible de résoudre le chemin WSL du projet.' }
wsl.exe --cd "$LinuxPath" python3 server.py
