from setuptools import find_packages, setup

setup(
    name="trilemma",
    version="1.0.0",
    description=(
        "Accuracy–Fairness–Efficiency Trilemma: "
        "Pareto Benchmark for Mobile Image Classification"
    ),
    author="Tran Van Thanh, Nguyen Van An, Nguyen Van Anh",
    author_email="tvthanh@lhu.edu.vn",
    packages=find_packages(where="."),
    python_requires=">=3.9",
    install_requires=[
        "torch>=1.13.0",
        "torchvision>=0.14.0",
        "numpy>=1.23.0",
        "pandas>=1.5.0",
        "scikit-learn>=1.2.0",
        "Pillow>=9.4.0",
        "PyYAML>=6.0",
        "tqdm>=4.64.0",
    ],
    extras_require={
        "3d_aug": ["deca-pytorch>=0.1.0", "mediapipe>=0.9.0"],
        "mobile":  ["tensorflow>=2.11.0", "onnx>=1.13.0"],
        "dev":     ["pytest>=7.2.0", "pytest-cov>=4.0.0", "black", "flake8"],
    },
    entry_points={
        "console_scripts": [
            "trilemma-train=scripts.train:main",
            "trilemma-eval=scripts.evaluate:main",
            "trilemma-pareto=scripts.pareto_analysis:main",
        ]
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
