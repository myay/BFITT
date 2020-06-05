from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='bfitt',
    ext_modules=[
        CUDAExtension('bfitt', [
            'bfitt.cpp',
            'bfitt_kernels.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
