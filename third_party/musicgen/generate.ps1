# MusicGen 快速生成脚本
param(
    [Parameter(Mandatory=$true)]
    [string]$Prompt,
    [int]$Duration = 30,
    [string]$Output = "generated_music.wav",
    [string]$Model = "small"
)
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
. "$scriptDir\venv\Scripts\Activate.ps1"
Set-Location $scriptDir

Write-Host "生成AI音乐..." -ForegroundColor Green
Write-Host "描述: $Prompt" -ForegroundColor Yellow

python -c "
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

model = MusicGen.get_pretrained('$Model')
model.set_generation_params(duration=$Duration)

print('Generating...')
wav = model.generate(['$Prompt'])

audio_write(
    '$Output'.replace('.wav', ''),
    wav[0].cpu(),
    model.sample_rate,
    strategy='loudness',
    loudness_compressor=True
)
print('Done: $Output')
"
