from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [
        line.strip() for line in fh if line.strip() and not line.startswith("#")
    ]

setup(
    name="torch_drl",
    version="0.1.0",
    author="lee",
    author_email="jacklee13520@gmail.com",
    description="A PyTorch implementation of various distributional reinforcement learning algorithms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bajfi/Distributional-RL.git",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "torch-drl=torch_drl.main:app",
        ],
    },
)
