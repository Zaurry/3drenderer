param([string]$Destination = "$PSScriptRoot/../build/dxr-sdk")
$ErrorActionPreference = 'Stop'
$root = [IO.Path]::GetFullPath($Destination)
New-Item -ItemType Directory -Force -Path $root | Out-Null
$packages = @(
    @{name='agility'; sha256='08f0489281401aa430fc37322d6c3fc98a8025175aacd714c10d562f4963f1e9'; url='https://api.nuget.org/v3-flatcontainer/microsoft.direct3d.d3d12/1.619.6/microsoft.direct3d.d3d12.1.619.6.nupkg'},
    @{name='dxc'; sha256='ad31b1fc8443175d204f77a611fdb3ef2ec42759bdc2f1167368de24a4a7e7f1'; url='https://github.com/microsoft/DirectXShaderCompiler/releases/download/v1.9.2609/dxc_2026_09_29.zip'},
    @{name='rtxdi'; sha256='e4abcadfaf4e44942bcd8f4345c2b85823515b9779e28c636a879d03307675a5'; url='https://github.com/NVIDIA-RTX/RTXDI/archive/refs/tags/v3.1.0.zip'},
    @{name='rtxdi-library'; sha256='f91de92d7c27f9824915ee5fcf62b0da664a6eb294edb659056ff3c462823f75'; url='https://github.com/NVIDIA-RTX/RTXDI-Library/archive/f12037fa8e97ebc08e9e3edfd2de528ed1772a4b.zip'},
    @{name='nrd'; sha256='c3a71eb0c3577f664f6f8bf3be7c585a39911fbf2895c144af10c58b861ad6ba'; url='https://github.com/NVIDIA-RTX/NRD/archive/refs/tags/v4.17.3.zip'},
    @{name='streamline'; sha256='92c4d954631a1710da86ca3fa8d5034f2b9503838c95fc4ae977ae149319781b'; url='https://github.com/NVIDIA-RTX/Streamline/releases/download/v2.14.1/streamline-sdk-v2.14.1.zip'},
    @{name='shadermake'; sha256='35de2547e28cf10f18a0e2782cf154a3d1610212d00d829fa8e73a18210623b6'; url='https://github.com/NVIDIA-RTX/ShaderMake/archive/18f5a344e7ca8fa65daaf079d07bc8ce38453e05.zip'},
    @{name='mathlib'; sha256='4a0772103fd7ef832f8c2fb3ec7fb6bf5b3bebbd41ecafc476671d11eca28a06'; url='https://github.com/NVIDIA-RTX/MathLib/archive/refs/tags/v11.zip'}
)
$manifest = @()
foreach ($package in $packages) {
    $archive = Join-Path $root ($package.name + '.zip')
    if (!(Test-Path -LiteralPath $archive)) {
        Write-Output ('Downloading ' + $package.name)
        Invoke-WebRequest -Uri $package.url -OutFile $archive
    }
    $hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $archive).Hash.ToLowerInvariant()
    if ($hash -ne $package.sha256) { throw ('Pinned SHA256 mismatch for ' + $package.name + ': ' + $hash) }
    $directory = Join-Path $root $package.name
    if (!(Test-Path -LiteralPath $directory)) {
        Expand-Archive -LiteralPath $archive -DestinationPath $directory
    }
    $manifest += @{ name=$package.name; url=$package.url; sha256=$hash }
    Write-Output ($package.name + ' SHA256=' + $hash)
}
$manifest | ConvertTo-Json -Depth 3 | Set-Content -LiteralPath (Join-Path $root 'manifest.json') -Encoding utf8
