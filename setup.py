from setuptools import find_packages, setup

setup(
    name="blink-aero",
    version="1.0.0",
    description="Project BLINK - Satellite Temporal Nowcasting & Neural Frame Interpolation Engine",
    author="BLINK Research Team",
    packages=find_packages(),
    py_modules=["blink"],
    install_requires=[
        "torch>=2.0.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.23.0",
        "scipy>=1.10.0",
        "pillow>=10.0.0",
        "requests>=2.28.0",
        "httpx>=0.24.0",
        "h5py>=3.8.0",
        "s3fs>=2023.6.0",
    ],
    entry_points={
        "console_scripts": [
            "blink=blink:main",
        ],
    },
    python_requires=">=3.10",
)
