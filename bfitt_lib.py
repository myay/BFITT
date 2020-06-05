import torch
import bfitt

def view_as_3d(input, input_shape):
    input_shape_len = len(input_shape)
    if input_shape_len == 4:
        input = input.view(input_shape[0], input_shape[1], -1)
    if input_shape_len == 3:
        input = input.view(input_shape[0], input_shape[1], input_shape[2])
    if input_shape_len == 2:
        input = input.view(input_shape[0], input_shape[1], 1)
    if input_shape_len == 1:
        input = input.view(input_shape[0], 1, 1)
    return input

def bfi_8bit_cuda(input, p01, p10):
    input_shape = list(input.shape)
    input = view_as_3d(input, input_shape)
    inputL = bfitt.bfi_8bit(input, p01, p10)
    input = inputL[0].view(input_shape)
    return input

def bfi_16bit_cuda(input, p01, p10):
    input_shape = list(input.shape)
    input = view_as_3d(input, input_shape)
    inputL = bfitt.bfi_16bit(input, p01, p10)
    input = inputL[0].view(input_shape)
    return input

def bfi_32bit_cuda(input, p01, p10):
    input_shape = list(input.shape)
    input = view_as_3d(input, input_shape)
    inputL = bfitt.bfi_32bit(input, p01, p10)
    input = inputL[0].view(input_shape)
    return input

def bfi_64bit_cuda(input, p01, p10):
    input_shape = list(input.shape)
    input = view_as_3d(input, input_shape)
    inputL = bfitt.bfi_64bit(input, p01, p10)
    input = inputL[0].view(input_shape)
    return input
