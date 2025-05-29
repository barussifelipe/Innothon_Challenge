from setuptools import setup, find_packages

setup(
    name="anomaly_transformer",
    version="0.1",
    packages=find_packages(where= "src"),  # Auto-discovers Python packages
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.21.0",
        "torch>=1.9.0",
        "scikit-learn>=0.24.0",
        "pandas>=1.2.0",
        "matplotlib>=3.3.0",
        "tqdm>=4.60.0"
    ],
)