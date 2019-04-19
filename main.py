import pycuda.autoinit
import pycuda.driver as drv
import numpy
from pycuda.compiler import SourceModule


filename = input("File to execute: ")
with open(filename, "r") as f:
    code = f.readlines()

for line in code:
    print(line, end="")
