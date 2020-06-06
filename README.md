# BFITT: Bit Flip Injection Into PyTorch Tensors
This repo provides efficient CUDA kernel based Bit Flip Injection into PyTorch Tensors for all PyTorch data types and for any number of dimensions.

Using BFITT with e.g. 8 bit:
- install with "python setup.py install" in folder src (with option --user if preferred)
- use "import bfitt"
- use "bfitt.bfi_8bit(input, p01, p10)" to call the bfi function that injects bit flips, where p01 is the probability in decimal for 0->1 and p10 for 1->0

There are functions available for 8, 16, 32, and 64 bits:
- bfi_8bit
- bfi_16bit
- bfi_32bit
- bfi_64bit

---
This software contains source code provided by NVIDIA Corporation.
