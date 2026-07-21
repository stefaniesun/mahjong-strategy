param([string]$TaskName = "S5ConsoleAgent", [string]$Config = "configs/s5_console.local.yaml")
$ErrorActionPreference = "Stop"
$Script = Resolve-Path (Join-Path $PSScriptRoot "start-agent.ps1")
$Action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$Script`" -Config `"$Config`""
$Trigger = New-ScheduledTaskTrigger -AtStartup
$Principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType S4U -RunLevel Highest
Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Principal $Principal -Description "S5 local training Agent A (service only; never auto-starts training)" -Force
