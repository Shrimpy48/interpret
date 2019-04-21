import pycuda.autoinit
import pycuda.gpuarray as gpuarr
import pycuda.driver as drv
import numpy
from pycuda.compiler import SourceModule
from copy import deepcopy

from core import Function, value


class RangeGPU(Function):
    def __init__(self, name=None):
        if name is not None:
            super().__init__(name)
        else:
            super().__init__("rangeGPU")
        self.add_def([value("start", {}), value("stop", {}), value("step", {})], "")

    def __repr__(self):
        return "RangeGPU(" + self.name + ")"

    def call(self, args, which, namespace):
        if self.defs[which] != "":
            return super().call(args, which, namespace)
        taken_num = len(self.args[which])
        local_namespace = deepcopy(namespace)
        for i in range(taken_num):
            if isinstance(self.args[which][i], Function):
                local_namespace[self.args[which][i].name] = args[i]
        start = local_namespace["start"]
        stop = local_namespace["stop"]
        step = local_namespace["step"]
        remaining_args = args[taken_num:]
        if isinstance(start, Function) or isinstance(stop, Function) or isinstance(step, Function):
            result = type(self)(self.name)
            result.given_args = [start, stop, step]
            if len(remaining_args) > 0:
                result.evaluate(remaining_args, namespace)
            return result
        result = gpuarr.arange(start, stop, step)
        if len(remaining_args) > 0:
            raise RuntimeError("Too many arguments for " + self.name)
        return result


class ToGPU(Function):
    def __init__(self, name=None):
        if name is not None:
            super().__init__(name)
        else:
            super().__init__("toGPU")
        self.add_def([value("array", {})], "")

    def __repr__(self):
        return "ToGPU(" + self.name + ")"

    def call(self, args, which, namespace):
        if self.defs[which] != "":
            return super().call(args, which, namespace)
        taken_num = len(self.args[which])
        local_namespace = deepcopy(namespace)
        for i in range(taken_num):
            if isinstance(self.args[which][i], Function):
                local_namespace[self.args[which][i].name] = args[i]
        arr = local_namespace["array"]
        remaining_args = args[taken_num:]
        if isinstance(arr, Function):
            result = type(self)(self.name)
            result.given_args = [arr]
            if len(remaining_args) > 0:
                result.evaluate(remaining_args, namespace)
            return result
        if isinstance(arr, numpy.ndarray):
            result = gpuarr.to_gpu(arr)
        else:
            raise RuntimeError("Invalid argument for " + self.name)
        if len(remaining_args) > 0:
            raise RuntimeError("Too many arguments for " + self.name)
        return result


class FromGPU(Function):
    def __init__(self, name=None):
        if name is not None:
            super().__init__(name)
        else:
            super().__init__("fromGPU")
        self.add_def([value("array", {})], "")

    def __repr__(self):
        return "FromGPU(" + self.name + ")"

    def call(self, args, which, namespace):
        if self.defs[which] != "":
            return super().call(args, which, namespace)
        taken_num = len(self.args[which])
        local_namespace = deepcopy(namespace)
        for i in range(taken_num):
            if isinstance(self.args[which][i], Function):
                local_namespace[self.args[which][i].name] = args[i]
        arr = local_namespace["array"]
        remaining_args = args[taken_num:]
        if isinstance(arr, Function):
            result = type(self)(self.name)
            result.given_args = [arr]
            if len(remaining_args) > 0:
                result.evaluate(remaining_args, namespace)
            return result
        if isinstance(arr, gpuarr.GPUArray):
            result = arr.get()
        else:
            raise RuntimeError("Invalid argument for " + self.name)
        if len(remaining_args) > 0:
            raise RuntimeError("Too many arguments for " + self.name)
        return result


func_dict_CUDA = {"rangeGPU": RangeGPU(), "toGPU": ToGPU(), "fromGPU": FromGPU()}
