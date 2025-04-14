# Module 1.2: Development Environment Setup

## Overview

Setting up the right development environment is crucial for AI and machine learning work. A properly configured environment will save you time, prevent compatibility issues, and allow you to leverage your hardware effectively. This section will guide you through setting up a professional AI development environment optimized for your RTX 3080 GPU.

## Python Environment Configuration

Python is the primary language for AI development due to its simplicity, readability, and the vast ecosystem of libraries available for data science and machine learning.

### Installing Python

While you already have some Python experience, let's ensure you're using a version that's compatible with modern AI libraries:

1. **Recommended Version**: Python 3.8-3.10 (many deep learning libraries have compatibility issues with newer versions)

2. **Installation Options**:
   - **Windows**: Download from [python.org](https://www.python.org/downloads/) or use [Anaconda](https://www.anaconda.com/products/distribution)
   - **macOS**: Use Homebrew (`brew install python`) or Anaconda
   - **Linux**: Most distributions come with Python pre-installed, but you can update using your package manager

3. **Verification**: After installation, verify your Python version:
   ```bash
   python --version
   # or
   python3 --version
   ```

### Virtual Environments

Virtual environments are isolated Python environments that allow you to work on different projects with different dependencies without conflicts.

#### Option 1: venv (Python's built-in solution)

```bash
# Create a virtual environment
python -m venv ai_env

# Activate the environment
# On Windows
ai_env\Scripts\activate
# On macOS/Linux
source ai_env/bin/activate

# Deactivate when done
deactivate
```

#### Option 2: Conda (Recommended for AI work)

Conda is a package, dependency, and environment management system that's particularly useful for data science and AI:

```bash
# Install Miniconda (a minimal version of Anaconda)
# Download from: https://docs.conda.io/en/latest/miniconda.html

# Create a new environment
conda create -n ai_env python=3.10

# Activate the environment
conda activate ai_env

# Deactivate when done
conda deactivate
```

## GPU Setup for Deep Learning (RTX 3080)

Your RTX 3080 is a powerful GPU that can significantly accelerate deep learning tasks. Here's how to set it up:

### NVIDIA Driver Installation

1. **Windows**:
   - Download the latest drivers from [NVIDIA's website](https://www.nvidia.com/Download/index.aspx)
   - Run the installer and follow the prompts

2. **Linux**:
   ```bash
   # Ubuntu example
   sudo apt update
   sudo apt install nvidia-driver-535  # Use the latest version available
   sudo reboot
   ```

3. **Verify Installation**:
   ```bash
   # Windows
   nvidia-smi
   
   # Linux
   nvidia-smi
   ```
   
   This command should display information about your GPU, including the driver version and CUDA version.

### CUDA Installation

CUDA is NVIDIA's parallel computing platform that enables GPU-accelerated computing:

1. **Download CUDA Toolkit**:
   - Visit [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads)
   - Select your operating system and follow the installation instructions
   - Recommended version: CUDA 11.8 (compatible with most current deep learning frameworks)

2. **Verify Installation**:
   ```bash
   nvcc --version
   ```

### cuDNN Installation

cuDNN is a GPU-accelerated library for deep neural networks:

1. **Download cuDNN**:
   - Register for an NVIDIA Developer account
   - Download cuDNN from [NVIDIA's website](https://developer.nvidia.com/cudnn)
   - Select the version compatible with your CUDA installation

2. **Installation**:
   - **Windows**: Extract and copy files to your CUDA installation directory
   - **Linux**: Follow the instructions in the cuDNN Installation Guide

### RTX 3080 Optimization Tips

Your RTX 3080 has 10GB of VRAM, which is excellent for most deep learning tasks. Here are some optimization tips:

1. **Power Management**:
   - Set your GPU to "Prefer maximum performance" in NVIDIA Control Panel
   - Ensure proper cooling as deep learning workloads can be intensive

2. **Memory Management**:
   - Monitor VRAM usage with tools like `nvidia-smi`
   - Adjust batch sizes based on available memory
   - Use mixed precision training (FP16) to reduce memory usage

3. **Multi-GPU Considerations**:
   - If you add more GPUs in the future, ensure proper spacing for thermal management
   - Consider using NVLink for multi-GPU setups

## Essential Python Libraries for AI

Let's install the core libraries you'll need for AI development:

```bash
# Activate your environment first
conda activate ai_env  # or your environment name

# Scientific computing and data manipulation
pip install numpy pandas scipy matplotlib seaborn

# Machine learning
pip install scikit-learn

# Deep learning - choose one or both based on your preference
# PyTorch (recommended for research and flexibility)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# TensorFlow (recommended for production deployment)
pip install tensorflow

# Jupyter for interactive development
pip install jupyter notebook jupyterlab

# Additional useful libraries
pip install opencv-python pillow tqdm
```

## IDE Setup

A good Integrated Development Environment (IDE) enhances productivity:

### VSCode Setup (Recommended)

Visual Studio Code is a lightweight, powerful editor with excellent Python and Jupyter support:

1. **Installation**:
   - Download from [code.visualstudio.com](https://code.visualstudio.com/)

2. **Essential Extensions**:
   - Python extension by Microsoft
   - Jupyter extension
   - Pylance for improved code intelligence
   - indent-rainbow for better code readability
   - GitLens for Git integration

3. **Configuration**:
   - Set your Python interpreter to your virtual environment
   - Configure linting and formatting (PEP 8)
   - Set up keyboard shortcuts for common tasks

### Jupyter Setup

Jupyter notebooks are ideal for experimentation and visualization:

1. **Start Jupyter**:
   ```bash
   # From your environment
   jupyter notebook
   # or
   jupyter lab  # for the newer interface
   ```

2. **Configuration**:
   - Set up dark mode if preferred
   - Install extensions like Table of Contents, Variable Inspector
   - Configure autosave

3. **Best Practices**:
   - Use markdown cells for documentation
   - Keep code cells focused on single tasks
   - Restart and run all cells periodically to verify workflow

## Version Control with Git

Version control is essential for tracking changes and collaborating:

1. **Installation**:
   - Download from [git-scm.com](https://git-scm.com/)

2. **Basic Configuration**:
   ```bash
   git config --global user.name "Your Name"
   git config --global user.email "your.email@example.com"
   ```

3. **GitHub/GitLab Integration**:
   - Create an account if you don't have one
   - Set up SSH keys for secure access

4. **Essential Commands**:
   ```bash
   # Initialize a repository
   git init
   
   # Add files
   git add .
   
   # Commit changes
   git commit -m "Descriptive message"
   
   # Create and switch to a new branch
   git checkout -b feature-name
   
   # Push to remote repository
   git push origin branch-name
   ```

## Project Structure Best Practices

Organizing your AI projects well makes them easier to maintain and share:

```
project_name/
├── data/               # Raw and processed data
├── notebooks/          # Jupyter notebooks for exploration
├── src/                # Source code
│   ├── __init__.py
│   ├── data/           # Data processing scripts
│   ├── models/         # Model definitions
│   ├── training/       # Training scripts
│   └── utils/          # Utility functions
├── tests/              # Unit tests
├── configs/            # Configuration files
├── requirements.txt    # Dependencies
├── README.md           # Project documentation
└── .gitignore          # Files to ignore in version control
```

## Monitoring Tools

These tools will help you track your model training and system performance:

1. **TensorBoard/PyTorch Lightning**:
   - Visualize training metrics
   - Compare different model runs
   - Inspect model graphs

2. **NVIDIA-SMI and GPU Monitoring**:
   ```bash
   # Basic monitoring
   watch -n 1 nvidia-smi
   
   # More detailed monitoring with nvidia-smi
   nvidia-smi --query-gpu=timestamp,name,pci.bus_id,driver_version,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv -l 1
   ```

3. **System Monitoring**:
   - Windows: Task Manager, GPU-Z
   - Linux: htop, glances

## Practical Exercise: Environment Setup

Now, let's set up your development environment:

1. Install Python and create a virtual environment
2. Install the NVIDIA drivers, CUDA, and cuDNN
3. Install the essential Python libraries
4. Set up VSCode or your preferred IDE
5. Create a simple project structure
6. Verify GPU acceleration with this test script:

```python
# Save as gpu_test.py
import torch
import tensorflow as tf

# PyTorch GPU check
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device:", torch.cuda.get_device_name(0))
    
# TensorFlow GPU check
print("\nTensorFlow version:", tf.__version__)
print("GPU available:", len(tf.config.list_physical_devices('GPU')) > 0)
if len(tf.config.list_physical_devices('GPU')) > 0:
    print("GPU devices:", tf.config.list_physical_devices('GPU'))

# Simple GPU test
if torch.cuda.is_available():
    # Create random tensors
    x = torch.rand(5000, 5000).cuda()
    y = torch.rand(5000, 5000).cuda()
    
    # Measure time for matrix multiplication
    import time
    start = time.time()
    z = torch.matmul(x, y)
    torch.cuda.synchronize()  # Wait for GPU operation to complete
    end = time.time()
    
    print(f"\nGPU matrix multiplication time: {end - start:.4f} seconds")
    
    # Compare with CPU
    x_cpu = x.cpu()
    y_cpu = y.cpu()
    
    start = time.time()
    z_cpu = torch.matmul(x_cpu, y_cpu)
    end = time.time()
    
    print(f"CPU matrix multiplication time: {end - start:.4f} seconds")
    print(f"GPU speedup: {(end - start) / (z.device.type == 'cuda' and (end - start) or 1):.2f}x")
```

Run this script to verify your GPU setup:
```bash
python gpu_test.py
```

## Troubleshooting Common Issues

### CUDA Installation Problems
- Ensure your NVIDIA driver is compatible with your CUDA version
- Check system PATH variables include CUDA directories
- Verify installation with `nvcc --version` and `nvidia-smi`

### Library Compatibility Issues
- Use compatible versions of libraries (e.g., PyTorch/TensorFlow versions that support your CUDA version)
- Check error messages for specific version requirements
- Consider using conda environments which handle dependencies better

### GPU Not Detected
- Verify driver installation with `nvidia-smi`
- Check if CUDA paths are correctly set
- Ensure your GPU is supported by the installed CUDA version

### Out of Memory Errors
- Reduce batch size
- Use gradient accumulation for large models
- Implement mixed precision training
- Close other GPU-intensive applications

## Resources for Further Learning

- [NVIDIA CUDA Documentation](https://docs.nvidia.com/cuda/)
- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
- [TensorFlow GPU Guide](https://www.tensorflow.org/guide/gpu)
- [VSCode Python Tutorial](https://code.visualstudio.com/docs/python/python-tutorial)
- [Git and GitHub Learning Resources](https://docs.github.com/en/get-started/quickstart)

In the next section, we'll explore the essential Python skills specifically needed for AI development, building on your existing Python knowledge.
