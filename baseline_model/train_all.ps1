param(
    [int]$Epochs = 80,
    [int]$BatchSize = 4,
    [int]$Accumulate = 1,
    [string]$DataRoot = "D:\landcover\data\loveda\extracted",
    [string]$Output = "D:\landcover\baseline_model\runs"
)

$ErrorActionPreference = "Stop"
$Models = @(
    "pspnet", "deeplabv3plus", "danet", "semantic_fpn", "farseg",
    "ocrnet", "lanet", "isnet", "segmenter", "swin_upernet", "manet",
    "flanet", "convnext", "poolformer", "biformer", "efficientvit",
    "ddp", "logcanpp", "scsm"
)

Push-Location $PSScriptRoot
try {
    foreach ($Model in $Models) {
        Write-Host "=== Training $Model on LoveDA Train; evaluating on LoveDA Val ==="
        python train.py `
            --model $Model `
            --data-root $DataRoot `
            --output $Output `
            --epochs $Epochs `
            --batch-size $BatchSize `
            --accumulate $Accumulate `
            --crop-size 512 `
            --amp
        if ($LASTEXITCODE -ne 0) {
            throw "Training failed for $Model with exit code $LASTEXITCODE"
        }
    }
}
finally {
    Pop-Location
}
