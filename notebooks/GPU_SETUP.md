# Setting Up GPU Support for AI Learning Notebooks

This guide will help you set up PyTorch with GPU support for faster model training and inference.

## Prerequisites

1. **NVIDIA GPU** - You need an NVIDIA GPU with CUDA support
2. **Updated GPU Drivers** - Make sure your GPU drivers are up to date
3. **CUDA Toolkit** - Install the appropriate CUDA Toolkit for your system

## Installation Steps

### 1. Check Your GPU

First, make sure your system has an NVIDIA GPU:

**Windows:**
```
nvidia-smi
```

**Linux:**
```
nvidia-smi
```

**Mac:**
Note: Modern Macs with Apple Silicon (M1/M2/M3) do not support CUDA. For these Macs, PyTorch uses MPS (Metal Performance Shaders) instead.

### 2. Install CUDA Toolkit

Download and install the appropriate CUDA Toolkit for your system:
https://developer.nvidia.com/cuda-downloads

The current recommended CUDA version for PyTorch is CUDA 11.8 or CUDA 12.1.

### 3. Install PyTorch with CUDA Support

#### Option 1: Using pip (Recommended)

Install PyTorch with CUDA support using the command appropriate for your CUDA version:

**For CUDA 11.8:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**For CUDA 12.1:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### Option 2: Using conda

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

### 4. Verify CUDA Support

To verify that PyTorch can access your GPU:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"Number of GPU devices: {torch.cuda.device_count()}")
print(f"GPU name: {torch.cuda.get_device_name(0)}")
```

### 5. Configure CUDA_VISIBLE_DEVICES

If you have multiple GPUs and want to use a specific one, you can set the environment variable `CUDA_VISIBLE_DEVICES`:

**Windows PowerShell:**
```
$env:CUDA_VISIBLE_DEVICES=0
```

**Linux/macOS:**
```
export CUDA_VISIBLE_DEVICES=0
```

### 6. GPU Memory Management

If you encounter "CUDA out of memory" errors:

1. Reduce batch sizes
2. Use gradient accumulation
3. Enable mixed precision training:

```python
# Enable mixed precision training
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

# Training loop
for batch in dataloader:
    optimizer.zero_grad()
    
    # Forward pass with autocast
    with autocast():
        outputs = model(inputs)
        loss = criterion(outputs, targets)
    
    # Backward pass with scaled gradients
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

## Troubleshooting

1. **CUDA version mismatch**: Make sure the CUDA version you installed matches the PyTorch build
2. **Driver too old**: Update your NVIDIA drivers to support your CUDA version
3. **Wrong PyTorch installation**: Reinstall PyTorch with the correct CUDA version

## For Apple Silicon (M1/M2/M3) Macs

For Macs with Apple Silicon:

```bash
pip install torch torchvision torchaudio
```

To use Metal Performance Shaders (MPS) instead of CUDA:

```python
import torch

# Check if MPS is available
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS device")
else:
    device = torch.device("cpu")
    print("MPS device not found, using CPU")

# Move model and tensors to device
model = model.to(device)
inputs = inputs.to(device)
``` 