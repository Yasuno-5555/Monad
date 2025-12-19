import sys
import os

try:
    import eigen
    print(f"Eigen found at: {eigen.__file__}")
    base = os.path.dirname(eigen.__file__)
    print(f"Contents: {os.listdir(base)}")
except ImportError:
    print("Eigen not importable")

try:
    import pybind11
    print(f"Pybind11 include: {pybind11.get_include()}")
except:
    pass
