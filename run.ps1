param(
    [int]$Samples = 21,
    [string[]]$Modes,
    [string]$Device,
    [switch]$List,
    [switch]$NonInteractive,
    [switch]$Gui,
    [ValidateSet('auto','reference','c8w1','c16w1','c32w1','c8w2','c16w2','c32w2')][string]$Fp32Kernel = 'auto',
    [ValidateSet('auto','dp4a','tensor')][string]$Int8Path = 'auto',
    [ValidateSet(16,32)][int]$AmdInt4K = 32
)
$ErrorActionPreference = 'Stop'
$bundledPython = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
if (Test-Path -LiteralPath $bundledPython) {
    $benchmarkPython = $bundledPython
} else {
    $candidate = Get-Command python.exe -ErrorAction SilentlyContinue
    if (-not $candidate -or $candidate.Source -like '*WindowsApps*') {
        throw 'Python 3.10+ de 64 bits necessario. Instale Python ou execute bench.py com seu interpretador.'
    }
    $benchmarkPython = $candidate.Source
}
if ($Gui) {
    $windowPython = Join-Path (Split-Path $benchmarkPython) 'pythonw.exe'
    if (-not (Test-Path -LiteralPath $windowPython)) { $windowPython = $benchmarkPython }
    Start-Process -FilePath $windowPython -ArgumentList ('"' + (Join-Path $PSScriptRoot 'gui.py') + '"') -WorkingDirectory $PSScriptRoot -WindowStyle Hidden
    exit 0
}
$benchmarkArgs = @((Join-Path $PSScriptRoot 'gpu_bench.py'), '--samples', $Samples, '--int8-path', $Int8Path, '--amd-int4-k', $AmdInt4K, '--fp32-kernel', $Fp32Kernel)
if ($Modes) { $benchmarkArgs += '--modes'; $benchmarkArgs += $Modes }
if ($Device) { $benchmarkArgs += @('--device', $Device) }
if ($List) { $benchmarkArgs += '--list' }
elseif (-not $NonInteractive) { $benchmarkArgs += '--interactive' }
& $benchmarkPython @benchmarkArgs
exit $LASTEXITCODE
