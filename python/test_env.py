import os
import sys

site_torch_lib = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv", "Lib", "site-packages", "torch", "lib")
if os.path.exists(site_torch_lib) and hasattr(os, "add_dll_directory"):
    try:
        os.add_dll_directory(site_torch_lib)
    except Exception:
        pass

import torch
import transformers
import peft
import bitsandbytes

print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)} GB")
print(f"Transformers: {transformers.__version__}")
print(f"PEFT: {peft.__version__}")
print(f"BitsAndBytes: {bitsandbytes.__version__}")
print("STATUS: ENVIRONMENT_READY")
