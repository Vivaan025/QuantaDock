# Local Python Setup (Alternative to Docker)
# Run this if Docker is having issues

Write-Host "ML Model Optimization - Local Python Setup" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan

# Check Python installation
Write-Host "Checking Python installation..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Found: $pythonVersion" -ForegroundColor Green
}
catch {
    Write-Host "Python not found. Please install Python 3.10+ from python.org" -ForegroundColor Red
    exit 1
}

# Check if CUDA is available
Write-Host "Checking NVIDIA GPU..." -ForegroundColor Yellow
try {
    nvidia-smi | Out-Null
    Write-Host "NVIDIA GPU detected" -ForegroundColor Green
}
catch {
    Write-Host "No NVIDIA GPU found. CPU-only mode will be used." -ForegroundColor Yellow
}

# Create virtual environment
Write-Host "Creating Python virtual environment..." -ForegroundColor Yellow
python -m venv ml_optimization_env

# Activate environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& ".\ml_optimization_env\Scripts\Activate.ps1"

# Upgrade pip
Write-Host "Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# Install PyTorch with CUDA support
Write-Host "Installing PyTorch with CUDA support..." -ForegroundColor Yellow
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install ML libraries
Write-Host "Installing ML optimization libraries..." -ForegroundColor Yellow
pip install -r requirements.txt

# Test installation
Write-Host "Testing installation..." -ForegroundColor Yellow
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU Count: {torch.cuda.device_count()}')"

# Create activation script
Write-Host "Creating activation script..." -ForegroundColor Yellow
$activationScript = @'
# Activate ML Optimization Environment
& ".\ml_optimization_env\Scripts\Activate.ps1"
Write-Host "ML Optimization Environment Activated!" -ForegroundColor Green
Write-Host "Run: python scripts/system_info.py" -ForegroundColor Cyan
'@

Set-Content -Path "activate_local.ps1" -Value $activationScript

Write-Host ""
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host "================" -ForegroundColor Green
Write-Host ""
Write-Host "To use the local environment:" -ForegroundColor Cyan
Write-Host "1. Run: .\activate_local.ps1" -ForegroundColor White
Write-Host "2. Run: python scripts/system_info.py" -ForegroundColor White
Write-Host "3. Run: python scripts/optimize_llm.py" -ForegroundColor White
Write-Host ""
Write-Host "Available scripts:" -ForegroundColor Yellow
Write-Host "  scripts/system_info.py              - Hardware analysis" -ForegroundColor White
Write-Host "  scripts/optimize_llm.py             - LLM optimization" -ForegroundColor White
Write-Host "  scripts/optimize_stable_diffusion.py - Image model optimization" -ForegroundColor White
Write-Host "  scripts/optimize_text_encoder.py    - Text encoder optimization" -ForegroundColor White
Write-Host "  scripts/optimize_sound_to_vec.py    - Audio model optimization" -ForegroundColor White
Write-Host "  scripts/master_optimizer.py         - Run all optimizations" -ForegroundColor White