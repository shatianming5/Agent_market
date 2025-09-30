Param(
  [switch]$DryRun,
  [switch]$KeepDirs,
  [Parameter(ValueFromRemainingArguments = $true)] [string[]]$Targets
)

function Require-Python {
  $py = Get-Command python -ErrorAction SilentlyContinue
  if (-not $py) { Write-Error '未找到 python，请先安装 Python 并加入 PATH'; exit 1 }
}

Require-Python

$argsList = @('scripts/clean_workspace.py')
if ($DryRun)   { $argsList += '--dry-run' }
if ($KeepDirs) { $argsList += '--keep-dirs' }
if ($Targets)  { $argsList += $Targets }

Write-Host "python $($argsList -join ' ')"
& python @argsList
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

