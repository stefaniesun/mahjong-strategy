param([string]$Config = "configs/s5_console.local.yaml")
$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "../../..")
Set-Location $Root
python -m console.run_agent_a --config $Config
