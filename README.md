# BFITT: Bit Flip Injection Into PyTorch Tensors
This repo provides efficient CUDA kernel based Bit Flip Injection into PyTorch Tensors for all PyTorch data types and for any number of dimensions.

Using BFITT:
- ```cd src/``` and install with ```python setup.py install``` in folder src
- ```import torch``` and ```import bfitt```
- use ```bfitt.bfi_8bit(input, p01, p10)``` to call the bfi function that injects bit flips into all 8 bit data types, where ```p01``` is the probability in decimal for flipping 0->1 and ```p10``` for 1->0
- in interactive mode, try ```bfitt.bfi_8bit(torch.ones(2,2,2,2).type(torch.uint8).cuda(), 0.03, 0)```, to test and see the effect of bit flips

There are functions available for 8, 16, 32, and 64 bits:
- bfi_8bit
- bfi_16bit
- bfi_32bit
- bfi_64bit

For configuring threads per block before compilation, see ```src/bfitt_kernels.cu``` and set the macros ```TPB_X```, ```TPB_Y```, and ```TPB_Z```.
---
This software contains source code provided by NVIDIA Corporation.
