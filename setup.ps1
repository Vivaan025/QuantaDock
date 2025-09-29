# ML Model Optimization Docker Setup Script
# PowerShell script to build and run the ML optimization environment

Write-Host "ML Model Optimization Docker Setup" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan

# Check if Docker is running
Write-Host "Checking Docker status..." -ForegroundColor Yellow
try {
    docker version | Out-Null
    Write-Host "Docker is running" -ForegroundColor Green
}
catch {
    Write-Host "Docker is not running. Please start Docker Desktop." -ForegroundColor Red
    exit 1
}

# Check for NVIDIA Docker support
Write-Host "Checking NVIDIA Docker support..." -ForegroundColor Yellow
try {
    docker run --rm --gpus all nvidia/cuda:12.2-base nvidia-smi
    Write-Host "NVIDIA Docker support is available" -ForegroundColor Green
}
catch {
    Write-Host "NVIDIA Docker support not detected. GPU acceleration may not work." -ForegroundColor Yellow
    Write-Host "Please install NVIDIA Container Toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html" -ForegroundColor Yellow
}

# Build the Docker image
Write-Host "Building Docker image..." -ForegroundColor Yellow
docker-compose build

if ($LASTEXITCODE -eq 0) {
    Write-Host "Docker image built successfully" -ForegroundColor Green
}
else {
    Write-Host "Docker build failed" -ForegroundColor Red
    exit 1
}

# Start the container
Write-Host "Starting ML optimization container..." -ForegroundColor Yellow
docker-compose up -d

if ($LASTEXITCODE -eq 0) {
    Write-Host "Container started successfully" -ForegroundColor Green
}
else {
    Write-Host "Container start failed" -ForegroundColor Red
    exit 1
}

# Wait a moment for services to start
Start-Sleep -Seconds 5

# Check container status
Write-Host "Container status:" -ForegroundColor Yellow
docker-compose ps

# Display access information
Write-Host ""
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host "================" -ForegroundColor Green
Write-Host ""
Write-Host "Jupyter Notebook: http://localhost:8888" -ForegroundColor Cyan
Write-Host "TensorBoard: http://localhost:6006" -ForegroundColor Cyan
Write-Host ""
Write-Host "Useful Commands:" -ForegroundColor Yellow
Write-Host "  View logs:           docker-compose logs -f" -ForegroundColor White
Write-Host "  Stop container:      docker-compose down" -ForegroundColor White
Write-Host "  Access shell:        docker-compose exec ml-optimizer bash" -ForegroundColor White
Write-Host "  Run optimization:    docker-compose exec ml-optimizer python /workspace/scripts/master_optimizer.py" -ForegroundColor White
Write-Host ""
Write-Host "Directories:" -ForegroundColor Yellow
Write-Host "  Scripts:   ./scripts/" -ForegroundColor White
Write-Host "  Config:    ./configs/" -ForegroundColor White
Write-Host "  Workspace: ./workspace/" -ForegroundColor White
Write-Host "  Results:   ./results/" -ForegroundColor White
Write-Host "  Models:    ./models/" -ForegroundColor White

# Show system info
Write-Host ""
Write-Host "Getting system specifications..." -ForegroundColor Yellow
docker-compose exec ml-optimizer python /workspace/scripts/system_info.py

Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Cyan
Write-Host "1. Open Jupyter Notebook at http://localhost:8888" -ForegroundColor White
Write-Host "2. Run complete optimization suite:" -ForegroundColor White
Write-Host "   docker-compose exec ml-optimizer python /workspace/scripts/master_optimizer.py" -ForegroundColor White
Write-Host "3. Check results in ./results/ directory" -ForegroundColor White
