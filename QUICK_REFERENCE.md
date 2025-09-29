# Quick Reference - ML Optimization Docker

##  One-Line Setup
```powershell
cd c:\Users\vivaa\docker-gpu; .\setup.ps1
```

##  Essential Commands

### Start/Stop
```powershell
docker-compose up -d        # Start container
docker-compose down         # Stop container
docker-compose restart      # Restart container
```

### Run Optimizations
```powershell
# Complete optimization suite
docker-compose exec ml-optimizer python /workspace/scripts/master_optimizer.py

# Individual optimizations
docker-compose exec ml-optimizer python /workspace/scripts/system_info.py
docker-compose exec ml-optimizer python /workspace/scripts/optimize_llm.py
docker-compose exec ml-optimizer python /workspace/scripts/optimize_stable_diffusion.py
```

### Access Points
- **Jupyter**: http://localhost:8888
- **TensorBoard**: http://localhost:6006
- **Shell Access**: `docker-compose exec ml-optimizer bash`

### Check Status
```powershell
docker-compose ps           # Container status
docker-compose logs -f      # View logs
nvidia-smi                  # GPU usage
```

##  Troubleshooting

### Build Issues
```powershell
docker-compose build --no-cache    # Clean rebuild
docker system prune -a             # Clean Docker cache
```

### Performance Check
```powershell
docker-compose exec ml-optimizer python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
```

### Container Shell
```powershell
docker-compose exec ml-optimizer bash
cd /workspace && python scripts/system_info.py
```

##  File Locations
- **Scripts**: `./scripts/`
- **Results**: `./results/`  
- **Notebooks**: `./workspace/`
- **Config**: `./configs/`