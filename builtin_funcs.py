import numpy
from copy import deepcopy

from core import Function, value


class Add(Function):
    def __init__(self, name=None):
        if name is not None:
            super().__init__(name)
        else:
            super().__init__("add")
        self.add_def([value("a", {}), value("b", {})], "")

    def __repr__(self):
        return "Add(" + self.name + ")"

    def call(self, args, which, namespace):
        if self.defs[which] != "":
            return super().call(args, which, namespace)
        taken_num = len(self.args[which])
        local_namespace = deepcopy(namespace)
        for i in range(taken_num):
            if isinstance(self.args[which][i], Function):
                local_namespace[self.args[which][i].name] = args[i]
        a = local_namespace["a"]
        b = local_namespace["b"]
        remaining_args = args[taken_num:]
        if isinstance(a, Function) or isinstance(b, Function):
            result = type(self)(self.name)
            result.given_args = [a, b]
            if len(remaining_args) > 0:
                result.evaluate(remaining_args, namespace)
            return result
        result = a + b
        if len(remaining_args) > 0:
            raise RuntimeError("Too many arguments for " + self.name)
        return result


class Mult(Function):
    def __init__(self, name=None):
        if name is not None:
            super().__init__(name)
        else:
            super().__init__("mult")
        self.add_def([value("a", {}), value("b", {})], "")

    def __repr__(self):
        return "Mult(" + self.name + ")"

    def call(self, args, which, namespace):
        if self.defs[which] != "":
            return super().call(args, which, namespace)
        taken_num = len(self.args[which])
        local_namespace = deepcopy(namespace)
        for i in range(taken_num):
            if isinstance(self.args[which][i], Function):
                local_namespace[self.args[which][i].name] = args[i]
        a = local_namespace["a"]
        b = local_namespace["b"]
        remaining_args = args[taken_num:]
        if isinstance(a, Function) or isinstance(b, Function):
            result = type(self)(self.name)
            result.given_args = [a, b]
            if len(remaining_args) > 0:
                result.evaluate(remaining_args, namespace)
            return result
        result = a * b
        if len(remaining_args) > 0:
            raise RuntimeError("Too many arguments for " + self.name)
        return result


class Sub(Function):
    def __init__(self, name=None):
        if name is not None:
            super().__init__(name)
        else:
            super().__init__("sub")
        self.add_def([value("a", {}), value("b", {})], "")

    def __repr__(self):
        return "Sub(" + self.name + ")"

    def call(self, args, which, namespace):
        if self.defs[which] != "":
            return super().call(args, which, namespace)
        taken_num = len(self.args[which])
        local_namespace = deepcopy(namespace)
        for i in range(taken_num):
            if isinstance(self.args[which][i], Function):
                local_namespace[self.args[which][i].name] = args[i]
        a = local_namespace["a"]
        b = local_namespace["b"]
        remaining_args = args[taken_num:]
        if isinstance(a, Function) or isinstance(b, Function):
            result = type(self)(self.name)
            result.given_args = [a, b]
            if len(remaining_args) > 0:
                result.evaluate(remaining_args, namespace)
            return result
        result = a - b
        if len(remaining_args) > 0:
            raise RuntimeError("Too many arguments for " + self.name)
        return result


class Div(Function):
    def __init__(self, name=None):
        if name is not None:
            super().__init__(name)
        else:
            super().__init__("div")
        self.add_def([value("a", {}), value("b", {})], "")

    def __repr__(self):
        return "Div(" + self.name + ")"

    def call(self, args, which, namespace):
        if self.defs[which] != "":
            return super().call(args, which, namespace)
        taken_num = len(self.args[which])
        local_namespace = deepcopy(namespace)
        for i in range(taken_num):
            if isinstance(self.args[which][i], Function):
                local_namespace[self.args[which][i].name] = args[i]
        a = local_namespace["a"]
        b = local_namespace["b"]
        remaining_args = args[taken_num:]
        if isinstance(a, Function) or isinstance(b, Function):
            result = type(self)(self.name)
            result.given_args = [a, b]
            if len(remaining_args) > 0:
                result.evaluate(remaining_args, namespace)
            return result
        result = a / b
        if len(remaining_args) > 0:
            raise RuntimeError("Too many arguments for " + self.name)
        return result


class Map(Function):
    def __init__(self, name=None):
        if name is not None:
            super().__init__(name)
        else:
            super().__init__("map")
        self.add_def([value("a", {}), value("b", {})], "")

    def __repr__(self):
        return "Map(" + self.name + ")"

    def call(self, args, which, namespace):
        if self.defs[which] != "":
            return super().call(args, which, namespace)
        taken_num = len(self.args[which])
        local_namespace = deepcopy(namespace)
        for i in range(taken_num):
            if isinstance(self.args[which][i], Function):
                local_namespace[self.args[which][i].name] = args[i]
        a = local_namespace["a"]
        b = local_namespace["b"]
        remaining_args = args[taken_num:]
        if isinstance(b, Function):
            result = type(self)(self.name)
            result.given_args = [a, b]
            if len(remaining_args) > 0:
                result.evaluate(remaining_args, namespace)
            return result
        if not isinstance(a, Function):
            raise RuntimeError("Invalid argument for " + self.name)
        result = numpy.empty_like(b)
        for i in range(len(b)):
            result[i] = a.evaluate([b[i]], namespace)
        if len(remaining_args) > 0:
            raise RuntimeError("Too many arguments for " + self.name)
        return result


class Range(Function):
    def __init__(self, name=None):
        if name is not None:
            super().__init__(name)
        else:
            super().__init__("range")
        self.add_def([value("start", {}), value("stop", {}), value("step", {})], "")

    def __repr__(self):
        return "Range(" + self.name + ")"

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
        result = numpy.arange(start, stop, step)
        if len(remaining_args) > 0:
            raise RuntimeError("Too many arguments for " + self.name)
        return result


func_dict = {"add": Add(), "mult": Mult(), "sub": Sub(), "div": Div(), "map": Map(), "range": Range()}
