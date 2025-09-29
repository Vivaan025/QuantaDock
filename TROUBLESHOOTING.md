# Docker Troubleshooting Guide

## Current Issue: Docker Storage Corruption

You're experiencing Docker storage corruption with I/O errors. Here are solutions:

### Solution 1: Reset Docker Desktop (Recommended)

1. **Open Docker Desktop**
2. **Go to Settings > Advanced**
3. **Click "Clean / Purge data"**
4. **Restart Docker Desktop**

### Solution 2: Command Line Reset
```powershell
# Stop all Docker processes
Get-Process "*docker*" | Stop-Process -Force

# Clear Docker data (WARNING: Removes all images/containers)
Remove-Item -Path "$env:USERPROFILE\AppData\Local\Docker" -Recurse -Force -ErrorAction SilentlyContinue

# Restart Docker Desktop
Start-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe"
```

### Solution 3: Alternative Container Runtime
If Docker continues to fail, use Windows containers or WSL2 backend:

```powershell
# Switch to WSL2 backend in Docker Desktop settings
# Or use Windows containers temporarily
```

### Solution 4: Lightweight Setup (Emergency Backup)
If Docker remains problematic, run locally:

```powershell
# Install Python environment locally
pip install torch torchvision transformers diffusers accelerate
pip install bitsandbytes optimum auto-gptq
pip install jupyter tensorboard

# Run scripts directly
cd scripts
python system_info.py
python optimize_llm.py
```

## Quick Fixes to Try First

### Fix 1: Clean Docker System
```powershell
docker system prune -a --volumes -f
docker builder prune -a -f
```

### Fix 2: Restart Docker Service
```powershell
Restart-Service -Name "com.docker.service" -Force
```

### Fix 3: Check Disk Space
```powershell
Get-WmiObject -Class Win32_LogicalDisk | Select-Object DeviceID, @{Name="Size(GB)";Expression={[math]::Round($_.Size/1GB,2)}}, @{Name="FreeSpace(GB)";Expression={[math]::Round($_.FreeSpace/1GB,2)}}
```

## Prevention

1. **Regular Cleanup**: Run `docker system prune` weekly
2. **Disk Space**: Keep 20+ GB free space
3. **Docker Updates**: Keep Docker Desktop updated
4. **WSL2**: Use WSL2 backend for better performance

## Alternative: Local Python Setup

If Docker remains problematic, you can run the optimization scripts locally:

1. **Install Anaconda/Miniconda**
2. **Create environment**:
   ```bash
   conda create -n ml-opt python=3.10
   conda activate ml-opt
   conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
   pip install -r requirements.txt
   ```
3. **Run scripts directly**:
   ```bash
   python scripts/system_info.py
   python scripts/optimize_llm.py
   ```

## Support Contacts

- **Docker Issues**: https://docs.docker.com/desktop/troubleshoot/
- **NVIDIA Docker**: https://github.com/NVIDIA/nvidia-docker/issues
- **Project Issues**: Check logs in `./results/error_logs.txt`