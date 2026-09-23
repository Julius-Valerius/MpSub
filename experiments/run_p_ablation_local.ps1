$ErrorActionPreference = 'Stop'

$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPackages = 'C:\Users\31203\Desktop\Haoyu-Pengcheng\2DMoSub\2DMoSub\venv311\Lib\site-packages'
$pythonExe = 'C:\Users\31203\.local\bin\python3.11.exe'

$env:PYTHONPATH = $pythonPackages
$env:MPSUB_PYTHON = $pythonExe
$env:MPSUB_HF_HOME = Join-Path $projectRoot 'results\hf_runtime_cache'
$env:CUDA_VISIBLE_DEVICES = '0'
$env:MPSUB_CUDA_MEMORY_FRACTION = '0.48'
$env:FORWARD_BUDGET = '8400'
$env:P_VALUES = '5 10 15 25 30'
$env:SEEDS = '0 1 2'
$env:WORKERS = '1'

$driver = Join-Path $PSScriptRoot 'run_p_ablation_equal_budget.py'
$driverLog = Join-Path $projectRoot 'results\p_ablation_equal_budget\driver_local.log'

Set-Location -LiteralPath $projectRoot
& $pythonExe -u $driver 2>&1 | Tee-Object -FilePath $driverLog -Append
exit $LASTEXITCODE
