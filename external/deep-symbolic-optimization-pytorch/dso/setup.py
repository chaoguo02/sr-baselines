from setuptools import setup, find_packages
import os


required = [
    "pytest",
    "black",
    "flake8",
    "cython",
    "numpy>=1.24",
    "torch",
    "sympy",
    "pandas",
    "scikit-learn",
    "click",
    "deap",
    "pathos",
    "seaborn",
    "progress",
    "tqdm",
    "commentjson",
    "PyYAML",
    "prettytable",
]

extras = {
    "control": [
        "mpi4py",
        "gym[box2d]==0.15.4",
        "pybullet",
        "stable-baselines[mpi]==2.10.0",
    ],
    "regression": [],
}
extras["all"] = list({item for group in extras.values() for item in group})

# Lazy import numpy for include_dirs — avoids import error during build env setup
try:
    import numpy
    include_dirs = [numpy.get_include()]
except ImportError:
    include_dirs = []

setup(
    name="dso",
    version="1.0dev",
    description="Deep symbolic optimization.",
    author="LLNL",
    packages=find_packages(),
    setup_requires=["numpy", "Cython"],
    ext_modules=[],
    include_dirs=include_dirs,
    install_requires=required,
    extras_require=extras,
)
