# Установка окружения с нуля.
#
#   .\install.ps1                  создать окружение mmaction и всё поставить
#   .\install.ps1 -EnvName myenv   другое имя окружения
#
# Порядок шагов важен: torch и mmcv ставятся с отдельных индексов, а numpy надо
# закрепить на 1.x до и после установки OpenMMLab — иначе он подтянется как 2.x
# и стек упадёт при импорте.

param(
    [string]$EnvName = "mmaction",
    [string]$PythonVersion = "3.10"
)

$ErrorActionPreference = "Stop"

Write-Host "=== 1/7 создание окружения $EnvName (Python $PythonVersion) ===" -ForegroundColor Cyan
conda create -n $EnvName python=$PythonVersion -y

$envPath = (conda run -n $EnvName python -c "import sys; print(sys.executable)").Trim()
if (-not $envPath) { throw "не удалось определить путь к python в окружении $EnvName" }
Write-Host "интерпретатор: $envPath"

Write-Host "`n=== 2/7 PyTorch 2.1.0 + CUDA 12.1 ===" -ForegroundColor Cyan
& $envPath -m pip install torch==2.1.0 torchvision==0.16.0 `
    --index-url https://download.pytorch.org/whl/cu121

Write-Host "`n=== 3/7 numpy 1.x (до OpenMMLab) ===" -ForegroundColor Cyan
& $envPath -m pip install "numpy<2"

Write-Host "`n=== 4/7 chumpy (git-форк; версия с PyPI не собирается) ===" -ForegroundColor Cyan
& $envPath -m pip install --no-build-isolation "chumpy @ git+https://github.com/mattloper/chumpy"

Write-Host "`n=== 5/7 OpenMMLab: mmengine, mmcv, mmdet, mmpose ===" -ForegroundColor Cyan
& $envPath -m pip install -U openmim
& $envPath -m mim install mmengine==0.10.7
# не 2.2.0 — mmdet 3.3.0 требует mmcv<2.2.0 и падает на импорте
& $envPath -m mim install "mmcv==2.1.0"
& $envPath -m mim install "mmdet==3.3.0"
& $envPath -m mim install "mmpose==1.3.2"

Write-Host "`n=== 6/7 mmaction2 из исходников ===" -ForegroundColor Cyan
if (-not (Test-Path "mmaction2")) {
    git clone --depth 1 https://github.com/open-mmlab/mmaction2.git
}
& $envPath -m pip install -e mmaction2

Write-Host "`n=== 7/7 закрепление версий, конфликтующих с numpy 2 ===" -ForegroundColor Cyan
# opencv 5.x требует numpy>=2; mim мог подтянуть numpy 2 — возвращаем 1.x
& $envPath -m pip install "opencv-python==4.9.0.80" "opencv-contrib-python==4.9.0.80"
& $envPath -m pip install "numpy<2"

Write-Host "`n=== проверка ===" -ForegroundColor Cyan
& $envPath -c @"
import torch, numpy, cv2, mmengine, mmcv, mmdet, mmpose, mmaction
print('numpy   ', numpy.__version__)
print('opencv  ', cv2.__version__)
print('torch   ', torch.__version__, '| CUDA:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU     ', torch.cuda.get_device_name(0))
print('mmengine', mmengine.__version__)
print('mmcv    ', mmcv.__version__)
print('mmdet   ', mmdet.__version__)
print('mmpose  ', mmpose.__version__)
print('mmaction', mmaction.__version__)
"@

Write-Host "`nГотово. Запуск:" -ForegroundColor Green
Write-Host "  $envPath demo_pipeline.py"
Write-Host "Веса моделей (~140 МБ) скачаются автоматически при первом запуске в .\models\"
