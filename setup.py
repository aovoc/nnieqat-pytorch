from setuptools import setup, find_packages
import pathlib
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

from build_helper import check_cuda_version
assert(check_cuda_version())

import os
os.system('make -j%d' % os.cpu_count())

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / 'README.md').read_text(encoding='utf-8')

setup(
    name='nnieqat',
    version='0.1.0',
    description='A nnie quantization aware training tool on pytorch.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/aovoc/nnieqat-pytorch',
    author='Minqin Chen',
    author_email='minqinchen@deepglint.com',
    license='MIT',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        "Intended Audience :: Science/Research",
        'Intended Audience :: Developers',
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3 :: Only',
    ],
    keywords=[
        "quantization aware training",
        "deep learning",
        "neural network",
        "CNN",
        "machine learning",
    ],
    packages=find_packages(),
    package_data={
        "nnieqat": ["gpu/lib/*gfpq*"],
    },
    python_requires='>=3.5, <4',
    install_requires=[
        "torch>=1.5",
        "numba>=0.42.0",
        "numpy>=1.18.1"
    ],
    extras_require={
        'test': ["torchvision>=0.4",
                 "nose",
                 "ddt"
                 ],
        'docs': [
            'sphinx==2.4.4',
            'sphinx_rtd_theme'
        ]
    },
    ext_modules=[
        CUDAExtension(
            name="quant_impl",
            sources=[
                "./src/fake_quantize.cpp",
            ],
            libraries=['quant_impl'],
            library_dirs=['obj'],
        )
    ],
    cmdclass={'build_ext': BuildExtension},
    test_suite="nnieqat.test.test_cifar10",
)
