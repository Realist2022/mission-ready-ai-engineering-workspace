import torch

print(f"PyTorch version: {torch.__version__}")
cuda_available = torch.cuda.is_available()
print(f"CUDA available: {cuda_available}")

if cuda_available:
    print(f"GPU detected: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version PyTorch was built with: {torch.version.cuda}")
else:
    print("No GPU detected. PyTorch will use the CPU.")