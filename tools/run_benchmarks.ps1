param(
    [string]$Case = "benchmarks/cases/san_miguel_first_scene.json",
    [ValidateSet("all", "opengl", "cuda")]
    [string]$Backend = "all",
    [string]$OutputRoot = "benchmarks/results",
    [string]$Baseline = ""
)

$ErrorActionPreference = "Stop"
$sourceRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
function Resolve-FromSource([string]$Path) {
    if ([System.IO.Path]::IsPathRooted($Path)) {
        return [System.IO.Path]::GetFullPath($Path)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $sourceRoot $Path))
}

$timestamp = (Get-Date).ToUniversalTime().ToString("yyyyMMdd-HHmmssZ")
$commit = (git -C $sourceRoot rev-parse --short=12 HEAD).Trim()
$dirty = -not [string]::IsNullOrWhiteSpace(
    (git -C $sourceRoot status --porcelain --untracked-files=no | Out-String))
$runDirectory = Join-Path (Resolve-FromSource $OutputRoot) "$timestamp-$commit"
$timingDirectory = Join-Path $runDirectory "timing"
$diagnosticDirectory = Join-Path $runDirectory "diagnostics"

Push-Location $sourceRoot
try {
cmake --preset cuda -DRENDERER_BUILD_BENCHMARKS=ON
if ($LASTEXITCODE -ne 0) { throw "benchmark configure failed" }

cmake --build --preset cuda-release --target `
    viewer_benchmark viewer_benchmark_diagnostics benchmark_tests `
    benchmark_diagnostics_tests
if ($LASTEXITCODE -ne 0) { throw "benchmark build failed" }

& (Join-Path $sourceRoot "build/cuda/bin/benchmark_tests.exe")
if ($LASTEXITCODE -ne 0) { throw "benchmark tests failed" }
& (Join-Path $sourceRoot "build/cuda/bin/benchmark_diagnostics_tests.exe")
if ($LASTEXITCODE -ne 0) { throw "benchmark diagnostics tests failed" }

$commonArguments = @(
    "--case", (Resolve-FromSource $Case),
    "--git-commit", $commit
)
if ($dirty) { $commonArguments += "--git-dirty" }

$timingArguments = $commonArguments + @(
    "--backend", $Backend,
    "--output-dir", $timingDirectory
)
if (-not [string]::IsNullOrWhiteSpace($Baseline)) {
    $timingArguments += @("--baseline", (Resolve-FromSource $Baseline))
}
& (Join-Path $sourceRoot "build/cuda/bin/viewer_benchmark.exe") @timingArguments
if ($LASTEXITCODE -ne 0) { throw "timing benchmark failed" }

if ($Backend -ne "opengl") {
    $diagnosticArguments = $commonArguments + @(
        "--backend", "cuda",
        "--output-dir", $diagnosticDirectory
    )
    & (Join-Path $sourceRoot "build/cuda/bin/viewer_benchmark_diagnostics.exe") `
        @diagnosticArguments
    if ($LASTEXITCODE -ne 0) { throw "diagnostics benchmark failed" }
}

$combined = [ordered]@{
    schema_version = 1
    generated_at_utc = (Get-Date).ToUniversalTime().ToString("o")
    timing = Get-Content (Join-Path $timingDirectory "raw.json") -Raw | ConvertFrom-Json
}
$summary = "# Combined benchmark report`n`n## Timing`n`n" +
    (Get-Content (Join-Path $timingDirectory "summary.md") -Raw)
if (Test-Path (Join-Path $diagnosticDirectory "raw.json")) {
    $combined["diagnostics"] = Get-Content `
        (Join-Path $diagnosticDirectory "raw.json") -Raw | ConvertFrom-Json
    $summary += "`n`n## Diagnostics`n`n" +
        (Get-Content (Join-Path $diagnosticDirectory "summary.md") -Raw)
}
$combined | ConvertTo-Json -Depth 100 | Set-Content `
    (Join-Path $runDirectory "raw.json") -Encoding utf8
$summary | Set-Content (Join-Path $runDirectory "summary.md") -Encoding utf8

Write-Host "Benchmark report: $runDirectory"
}
finally {
    Pop-Location
}
