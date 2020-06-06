from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='bfitt',
    author='Mikail Yayla',
    author_email='mikail.yayla@tu-dortmund.de',
    description='Bit Flip Injection Into PyTorch Tensors',
    ext_modules=[
        CUDAExtension('bfitt', [
            'bfitt.cpp',
            'bfitt_kernels.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
