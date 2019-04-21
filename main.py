import pycuda.autoinit
import pycuda.driver as drv
import numpy
from pycuda.compiler import SourceModule
from copy import deepcopy


class Function:
    def __init__(self, name):
        self.name = name
        self.args = []
        self.defs = []
        self.given_args = []

    def __str__(self):
        string = self.name
        for arg in self.given_args:
            string += " (" + str(arg) + ")"
        return string

    def __repr__(self):
        return "Function(" + self.name + ")"

    def add_def(self, args, body):
        self.args.append(args)
        self.defs.append(body)

    def evaluate(self, new_args, namespace):
        args = self.given_args + new_args
        string = self.name
        for arg in args:
            string += " (" + str(arg) + ")"
        arg_matches = deepcopy(self.args)
        for i in range(len(self.given_args), len(args)):
            for j in range(len(self.args)):
                if i < len(self.args[j]) and not isinstance(self.args[j][i], Function):
                    if args[i] != self.args[j][i]:
                        arg_matches[j] = None
        for i in range(len(arg_matches)):
            if arg_matches[i] is not None and len(arg_matches[i]) <= len(args):
                result = self.call(args, i, namespace)
                return result
        matches = [arg for arg in arg_matches if arg is not None]
        defs = [self.defs[i] for i in range(len(arg_matches)) if arg_matches[i] is not None]
        result = type(self)(self.name)
        for i in range(len(matches)):
            result.add_def(matches[i], defs[i])
        result.given_args = args
        return result

    def call(self, args, which, namespace):
        taken_num = len(self.args[which])
        local_namespace = namespace
        for i in range(taken_num):
            if isinstance(self.args[which][i], Function):
                local_namespace[self.args[which][i].name] = args[i]
                if isinstance(args[i], Function):
                    local_namespace[self.args[which][i].name].name = self.args[which][i].name
        body_parts = [value(part, local_namespace) for part in separate(self.defs[which])]
        if not isinstance(body_parts[0], Function):
            return body_parts[0]
        func = body_parts[0]
        func_args = body_parts[1:]
        result = func.evaluate(func_args, local_namespace)
        remaining_args = args[taken_num:]
        if not isinstance(result, Function):
            if len(remaining_args) > 0:
                raise RuntimeError("Too many arguments for " + self.name)
            return result
        return result.evaluate(remaining_args, namespace)


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
        local_namespace = namespace
        for i in range(taken_num):
            if isinstance(self.args[which][i], Function):
                local_namespace[self.args[which][i].name] = args[i]
                if isinstance(args[i], Function):
                    local_namespace[self.args[which][i].name].name = self.args[which][i].name
        result = local_namespace["a"] + local_namespace["b"]
        remaining_args = args[taken_num:]
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
        local_namespace = namespace
        for i in range(taken_num):
            if isinstance(self.args[which][i], Function):
                local_namespace[self.args[which][i].name] = args[i]
                if isinstance(args[i], Function):
                    local_namespace[self.args[which][i].name].name = self.args[which][i].name
        result = local_namespace["a"] * local_namespace["b"]
        remaining_args = args[taken_num:]
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
        local_namespace = namespace
        for i in range(taken_num):
            if isinstance(self.args[which][i], Function):
                local_namespace[self.args[which][i].name] = args[i]
                if isinstance(args[i], Function):
                    local_namespace[self.args[which][i].name].name = self.args[which][i].name
        result = local_namespace["a"] - local_namespace["b"]
        remaining_args = args[taken_num:]
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
        local_namespace = namespace
        for i in range(taken_num):
            if isinstance(self.args[which][i], Function):
                local_namespace[self.args[which][i].name] = args[i]
                if isinstance(args[i], Function):
                    local_namespace[self.args[which][i].name].name = self.args[which][i].name
        result = local_namespace["a"] / local_namespace["b"]
        remaining_args = args[taken_num:]
        if len(remaining_args) > 0:
            raise RuntimeError("Too many arguments for " + self.name)
        return result


class Program:
    def __init__(self):
        self.functions = {"add": Add(), "mult": Mult(), "sub": Sub(), "div": Div()}

    def read_line(self, line):
        if line.isspace():
            return True
        parts = line.split("=")
        if len(parts) > 2:
            raise RuntimeError("Expected func arg1 arg2 .. argN = def, got " + line)
        elif len(parts) == 2:
            declaration = parts[0].strip()
            definition = parts[1].strip()
            dec_parts = separate(declaration)
            name = dec_parts[0]
            args = dec_parts[1:]
            arg_vals = [value(arg, {}) for arg in args]
            if name not in self.functions:
                self.functions[name] = Function(name)
            self.functions[name].add_def(arg_vals, definition)
            return True
        else:
            action_parts = parts[0].split(":")
            if len(action_parts) != 2:
                raise RuntimeError("Expected action : data, got " + line)
            action, data = action_parts
            return self.run(action.strip(), data.strip())

    def run(self, action, data):
        if action == "output":
            result = value(data, self.functions)
            print(data, "=", result)
            return True
        elif action == "input":
            val = input(data + ": ")
            self.functions[data] = value(val, self.functions)
            return True
        elif action == "run":
            with open(data, "r") as file:
                code = file.readlines()
            for line in code:
                self.read_line(line)
            return True
        elif action == "quit":
            return False
        else:
            raise RuntimeError("Unknown action")


def value(string, namespace):
    stripped = string.strip()
    while stripped[0] == "(" and stripped[-1] == ")":
        stripped = stripped[1:-1]
    if stripped[0] == stripped[-1] == '"':
        return stripped[1:-1]
    if stripped[0] == "[" and stripped[-1] == "]":
        parts = separate(stripped[1:-1])
        val = []
        for part in parts:
            val.append(value(part, namespace))
        return numpy.array(val)
    try:
        val = numpy.int(stripped)
    except ValueError:
        pass
    else:
        return val
    try:
        val = numpy.float32(stripped)
    except ValueError:
        pass
    else:
        return val
    parts = separate(stripped)
    name = parts[0]
    args = [value(arg, namespace) for arg in parts[1:]]
    if name in namespace:
        if not isinstance(namespace[name], Function):
            return namespace[name]
        return namespace[name].evaluate(args, namespace)
    return Function(name).evaluate(args, namespace)


def separate(string):
    parts = []
    paren_depth = 0
    in_str = False
    in_space = True
    part_start = 0
    for i in range(len(string)):
        char = string[i]
        if not in_space and char == " ":
            in_space = True
            if paren_depth == 0 and not in_str:
                parts.append(string[part_start:i])
        elif in_space and char != " ":
            in_space = False
            if paren_depth == 0 and not in_str:
                part_start = i
        if char in "[(":
            paren_depth += 1
        elif char in "])":
            paren_depth -= 1
        elif char == '"':
            in_str = not in_str
    if not in_space:
        parts.append(string[part_start:])
    return parts


if __name__ == "__main__":
    program = Program()
    line_in = input("> ")
    while program.read_line(line_in):
        line_in = input("> ")
