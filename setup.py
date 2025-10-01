"""Setup script for llmperf."""

from setuptools import setup, find_packages
import os

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
def read_requirements(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

# Get core requirements
core_requirements = read_requirements("requirements-core.txt")

setup(
    name="llmperf",
    version="0.1.0",
    author="llmperf contributors",
    author_email="",
    description="A lightweight CLI toolkit for benchmarking and profiling LLM inference",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/llmperf",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=core_requirements,
    extras_require={
        "dev": read_requirements("requirements-dev.txt"),
        "macos": read_requirements("requirements-macos.txt"),
    },
    entry_points={
        "console_scripts": [
            "llmperf=llmperf.cli:cli",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
