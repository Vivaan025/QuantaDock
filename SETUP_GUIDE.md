# ML Model Optimization Docker - Setup Guide

## Prerequisites

### 1. Install Docker Desktop
1. Download Docker Desktop from: https://www.docker.com/products/docker-desktop/
2. Install and start Docker Desktop
3. Ensure Docker Desktop is running (check system tray)

### 2. Install NVIDIA Container Toolkit (for GPU support)
1. Download from: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
2. Follow Windows installation guide
3. Restart Docker Desktop after installation

### 3. Verify GPU Support
```powershell
# Test NVIDIA Docker support
docker run --rm --gpus all nvidia/cuda:12.2-base nvidia-smi
```

## Quick Start

### Option 1: Automated Setup (Recommended)
```powershell
# Navigate to project directory
cd c:\Users\vivaa\docker-gpu

# Run the automated setup script
.\setup.ps1
```

### Option 2: Manual Setup
```powershell
# Navigate to project directory
cd c:\Users\vivaa\docker-gpu

# Build the Docker image (this takes 20-30 minutes first time)
docker-compose build

# Start the container
docker-compose up -d

# Check container status
docker-compose ps
```

## Running the ML Optimization Suite

### 1. Access Jupyter Notebook
- Open browser to: http://localhost:8888
- Navigate to `workspace/ML_Optimization_Demo.ipynb`
- Run cells interactively

### 2. Run Complete Optimization Suite
```powershell
# Run all optimizations (LLM, Stable Diffusion, Text Encoder, Audio)
docker-compose exec ml-optimizer python /workspace/scripts/master_optimizer.py
```

### 3. Run Individual Optimizations
```powershell
# System information and hardware detection
docker-compose exec ml-optimizer python /workspace/scripts/system_info.py

# Optimize Large Language Models
docker-compose exec ml-optimizer python /workspace/scripts/optimize_llm.py

# Optimize Stable Diffusion models
docker-compose exec ml-optimizer python /workspace/scripts/optimize_stable_diffusion.py

# Optimize text encoders/tokenizers
docker-compose exec ml-optimizer python /workspace/scripts/optimize_text_encoder.py

# Optimize audio/speech models
docker-compose exec ml-optimizer python /workspace/scripts/optimize_sound_to_vec.py
```

## Access Points

| Service | URL | Description |
|---------|-----|-------------|
| **Jupyter Notebook** | http://localhost:8888 | Interactive development environment |
| **TensorBoard** | http://localhost:6006 | Model training visualization |

## Directory Structure

```
docker-gpu/
├── scripts/           # Python optimization scripts
├── configs/           # Configuration files
├── workspace/         # Jupyter notebooks and development
├── results/           # Optimization results and benchmarks
├── models/            # Downloaded and optimized models
└── data/              # Training and test data
```

## Useful Commands

### Container Management
```powershell
# Start container
docker-compose up -d

# Stop container
docker-compose down

# View logs
docker-compose logs -f

# Access container shell
docker-compose exec ml-optimizer bash

# Restart container
docker-compose restart
```

### Development Commands
```powershell
# Install additional Python packages
docker-compose exec ml-optimizer pip install <package-name>

# Copy files to container
docker cp <local-file> ml-optimizer:/workspace/

# Copy files from container
docker cp ml-optimizer:/workspace/<file> <local-destination>
```

## Configuration

### Hardware Configuration
Edit `configs/optimization_config.py` to customize:
- Memory limits and batch sizes
- Quantization settings (INT8, INT4, FP16)
- Model-specific optimization parameters

### Environment Variables
Edit `docker-compose.yml` to modify:
- Port mappings
- GPU allocation
- Volume mounts
- Memory limits

## Troubleshooting

### Common Issues

#### 1. Docker Build Fails
```powershell
# Clean rebuild
docker-compose build --no-cache
```

#### 2. GPU Not Detected
```powershell
# Verify NVIDIA Docker support
docker run --rm --gpus all nvidia/cuda:12.2-base nvidia-smi

# Check Docker Desktop GPU settings
# Settings > Resources > Advanced > GPU
```

#### 3. Container Won't Start
```powershell
# Check detailed logs
docker-compose logs ml-optimizer

# Verify port availability
netstat -an | findstr 8888
```

#### 4. Out of Memory Errors
```powershell
# Increase Docker memory limit
# Docker Desktop > Settings > Resources > Advanced > Memory
```

### Performance Issues
- **First build takes 20-30 minutes** - This is normal for CUDA image download
- **Model downloads are slow** - Models are cached after first download
- **Low GPU utilization** - Check `nvidia-smi` and adjust batch sizes

## Expected Results

After running optimizations, you'll find results in `./results/`:

### Performance Improvements
- **LLM Models**: 3-8x inference speedup with quantization
- **Stable Diffusion**: 5-10x generation speedup with FP16
- **Text Encoders**: 2-5x encoding speedup
- **Audio Models**: 4-7x processing speedup

### Output Files
- `system_specs.json` - Hardware analysis
- `optimization_results.json` - Benchmark comparisons  
- `model_metrics.csv` - Performance metrics
- `tensorboard_logs/` - Training visualizations

## Advanced Usage

### Custom Model Integration
```python
# Add your own models to optimize_custom.py
from scripts.optimize_llm import LLMOptimizer

optimizer = LLMOptimizer("your-model-name")
results = optimizer.run_optimization()
```

### Batch Processing
```powershell
# Process multiple models
docker-compose exec ml-optimizer python scripts/batch_optimize.py --models model1,model2,model3
```

### Production Deployment
```powershell
# Build production image
docker build -f Dockerfile.prod -t ml-optimizer:prod .

# Deploy with resource limits
docker run --gpus all --memory 16g --cpus 8 ml-optimizer:prod
```

## Support

### Hardware Requirements
- **Minimum**: 8GB RAM, 4GB VRAM
- **Recommended**: 16GB RAM, 8GB+ VRAM
- **Optimal**: 32GB RAM, 12GB+ VRAM

### Software Requirements
- Windows 10/11 with WSL2
- Docker Desktop 4.0+
- NVIDIA drivers 520.61.05+
- CUDA-compatible GPU

### Getting Help
1. Check logs: `docker-compose logs -f`
2. Verify hardware: Run `system_info.py`
3. Test individual components before full suite
4. Monitor resource usage during optimization

---

**Quick Start Summary:**
1. `.\setup.ps1` - Run automated setup
2. Open http://localhost:8888 - Access Jupyter
3. Run optimization scripts or use interactive notebook
4. Check `./results/` for performance improvements