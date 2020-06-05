# BFITT
This repo provides efficient CUDA kernel based Bit Flip Injection into PyTorch Tensors up to 4d. (Can easily be extended for higher dimensions.)

Using BFITT with e.g. 8 bit:
- sys.path.append("../") (to where bfitt_lib.py resides)
- use "from bfitt_lib import bfi_8bit_cuda" to use the function in your Python file
- use "bfi_8bit_cuda(input, p01, p10)" to call the bfi function that injects bit flips, where p01 is the probability in decimal for 0->1 and p10 for 1->0
