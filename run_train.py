"""
Wrapper to run train_wandb.py with Windows DLL compatibility fixes.

Usage:
    python run_train.py --epochs 50 --batch-size 16 --num-assets 20
    python run_train.py --no-wandb --epochs 3

All arguments are forwarded to train_wandb.py.
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import importlib.util

# Pre-load torch by adding its lib directory to DLL search path
spec = importlib.util.find_spec("torch")
if spec and spec.origin:
    torch_lib = os.path.join(os.path.dirname(spec.origin), "lib")
    if os.path.isdir(torch_lib):
        try:
            os.add_dll_directory(torch_lib)
        except (OSError, AttributeError):
            pass
        os.environ["PATH"] = torch_lib + os.pathsep + os.environ.get("PATH", "")

import runpy
sys.argv[0] = "train_wandb.py"
runpy.run_path("train_wandb.py", run_name="__main__")
