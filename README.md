![logo](https://i.imgur.com/DmiHwzW.png) 

Bit Flip Injection into PyTorch Tensors
---
This repository provides efficient CUDA kernel-based bit flip injection into PyTorch tensors for any number of dimensions and all data types except torch.bool and torch.float16.

Install requirements: CUDA toolkit, pytorch, pybind11

Using BFITT:
- install with ```python setup.py install --user```
- ```import torch``` and ```import bfitt```
- in interactive mode, try ```bfitt.bfi_8bit(torch.ones(2,2,2,2).type(torch.uint8).cuda(), 0.03, 0)```, to test the installation and see the effect of bit flips
- we use ```bfitt.bfi_8bit(input, p01, p10)``` to call the bfi function that injects bit flips into all 8 bit data types, where ```input``` is a PyTorch tensor and ```p01``` is the probability in decimal for flipping 0->1 and ```p10``` for 1->0

There are functions available for datatypes with 8, 16, 32, and 64 bits:
- bfitt.bfi_8bit(input, p01, p10)
- bfitt.bfi_16bit(input, p01, p10)
- bfitt.bfi_32bit(input, p01, p10)
- bfitt.bfi_64bit(input, p01, p10)

For configuring threads per block before compilation, see ```bfitt/bfitt_kernels.cu``` and set the macros ```TPB_X```, ```TPB_Y```, and ```TPB_Z```.

---
This software contains source code provided by NVIDIA Corporation.
