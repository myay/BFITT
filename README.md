# BFITT: Bit Flip Injection Into PyTorch Tensors
This repo provides efficient CUDA kernel based Bit Flip Injection into PyTorch Tensors for all PyTorch data types and for any number of dimensions.

Using BFITT with e.g. 8 bit:
- install by executing "python setup.py install --user" in folder src
- use "import bfitt"
- use "bfitt.bfi_8bit(input, p01, p10)" to call the bfi function that injects bit flips, where p01 is the probability in decimal for 0->1 and p10 for 1->0

There are functions available for 8, 16, 32, and 64 bits. In the following command, just replace X with the number of bits you have: "bfitt.bfi_Xbit(input, p01, p10)". 

---
This software contains source code provided by NVIDIA Corporation.
