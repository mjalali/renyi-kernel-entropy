from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="rke-score",
    version="0.1",
    author="",
    autor_email="mjalali0079@gmail.com",
    description="Compute Renyi Kernel Entropy scores (RKE-MC and RRKE) for two sets of vectors.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mjalali/renyi-kernel-entropy-score",
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'numpy',
    ],
)
