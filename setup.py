#! /usr/bin/env python
#
# Copyright (C) 2018 Raphael Vallat

DESCRIPTION = "TorchSpectra: compute spectrograms of signals using PyTorch on a GPU, enabling efficient analysis of frequency content over time."
DISTNAME = "TorchSpectra"
MAINTAINER = "Amir Hossein Daraie"
MAINTAINER_EMAIL = "daraieamirh@gmail.com"
URL = "https://www.adaraie.com/"
LICENSE = "BSD (3-clause)"
DOWNLOAD_URL = "https://github.com/amirhdre/TorchSpectra"
VERSION = "0.1.0"
# PACKAGE_DATA = {"antropy.data.icons": ["*.ico"]}

try:
    from setuptools import setup

    _has_setuptools = True
except ImportError:
    from distutils.core import setup


def check_dependencies():
    install_requires = []

    try:
        import numpy
    except ImportError:
        install_requires.append("numpy")
    try:
        import scipy
    except ImportError:
        install_requires.append("scipy")

    try:
        import torch
    except ImportError:
        install_requires.append("torch")

    return install_requires


if __name__ == "__main__":
    install_requires = check_dependencies()

    setup(
        name=DISTNAME,
        author=MAINTAINER,
        author_email=MAINTAINER_EMAIL,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        description=DESCRIPTION,
        license=LICENSE,
        url=URL,
        version=VERSION,
        download_url=DOWNLOAD_URL,
        install_requires=install_requires,
        include_package_data=True,
        packages=["TorchSpectra"],
        # package_data=PACKAGE_DATA,
        classifiers=[
            "Intended Audience :: Science/Research",
            "Programming Language :: Python",
            "License :: OSI Approved :: BSD License",
            "Topic :: Scientific/Engineering :: Mathematics",
            "Operating System :: POSIX",
            "Operating System :: Unix",
            "Operating System :: MacOS",
        ],
    )