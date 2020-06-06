#/usr/bin/environment python3

import sys
import gc
import argparse
import torch

sys.path.append("../")
import bfitt

# wrapper for profiling functions
def cuda_profiler(profile_function, input, p01, p10):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    output = profile_function(input, p01, p10)
    end.record()
    torch.cuda.synchronize()
    print(start.elapsed_time(end))
    return output

def main():
    p01 = 0.01
    p10 = 0
    test_range = 7
    dim_cases = [[2], [2, 2], [2, 2, 2], [2 ,2, 2, 2]]
    rest_cases = [[1,1,2,2], [2,2,1,1], [4,2,2,2], [2,4,2,2], [2,2,4,2], [2,2,2,4]]
    for rest_case in rest_cases:
        dim_cases.append(rest_case)

    for dim_case in dim_cases:
        for batch_mult in range(1,test_range):
            test_dims = [d*(pow(2,batch_mult-1)) for d in dim_case]

            input = torch.FloatTensor(*test_dims).uniform_(-1, 1).cuda()
            print("Shape: ", input.shape)

            print("\n8 bit")
            input = torch.ones_like(input)
            X_cuda_uint8 = input.type(torch.uint8)
            X_cuda_uint8 = cuda_profiler(bfitt.bfi_8bit, X_cuda_uint8, p01, p10)
            #print(X_cuda_uint8)

            print("\n16 bit")
            input = torch.ones_like(input)
            X_cuda_int16 = input.type(torch.int16)
            X_cuda_int16 = cuda_profiler(bfitt.bfi_16bit, X_cuda_int16, p01, p10)
            #print(X_cuda_int16)

            print("\n32 bit")
            input = torch.ones_like(input)
            X_cuda_int32 = input.type(torch.int32)
            X_cuda_int32 = cuda_profiler(bfitt.bfi_32bit, X_cuda_int32, p01, p10)
            #print(X_cuda_int32)

            print("\n64 bit")
            input = torch.ones_like(input)
            X_cuda_int64 = input.type(torch.int64)
            X_cuda_int64 = cuda_profiler(bfitt.bfi_64bit, X_cuda_int64, p01, p10)
            #print(X_cuda_int64)

            print("\n\n---\n\n")

    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(type(obj), obj.size())
        except:
            pass

if __name__=="__main__":
    main()
