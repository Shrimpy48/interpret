import pycuda.autoinit
import pycuda.driver as drv
import numpy
from pycuda.compiler import SourceModule


class Function:
    def __init__(self, args, body=None):
        self.args = args
        self.body = body


class Program:
    def __init__(self, code_lines):
        self.objects = {}
        self.actions = []
        for line in code_lines:
            stripped = line.strip()
            if stripped != "":
                self.parse_dec(stripped)

        print(self.objects)

    def parse_dec(self, line):
        parts = line.split("=")
        if len(parts) > 2:
            raise SyntaxError
        elif len(parts) == 2:
            declaration = parts[0].strip()
            definition = parts[1].strip()
            dec_parts = separate(declaration)
            name = dec_parts[0]
            args = dec_parts[1:]
            arg_vals = []
            arg_vars = []
            for arg in args:
                val = self.literal(arg)
                if val is None:
                    arg_vars.append(arg)
                arg_vals.append(val)
            if name not in self.objects:
                self.objects[name] = []
            self.objects[name].append((arg_vals, Function(arg_vars, body=definition)))
        else:
            action_parts = parts[0].split("<-")
            if len(action_parts) != 2:
                raise SyntaxError
            action, data = action_parts
            self.actions.append((action.strip(), data.strip()))

    def run(self):
        for action, data in self.actions:
            if action == "output":
                value = self.evaluate(data, {})
                print(value)
            else:
                raise RuntimeError("Unknown action", action)

    def evaluate(self, exp, local_vars):
        print("Evaluating", exp)
        while exp[0] == "(" and exp[-1] == ")":
            exp = exp[1:-1]
        parts = separate(exp)
        val = self.literal(parts[0])
        if val is not None:
            return val
        args = parts[1:]
        arg_vals = []
        for arg in args:
            val = self.evaluate(arg, local_vars)
            arg_vals.append(val)
        func_name = parts[0]
        if func_name in local_vars:
            funcs = local_vars[func_name]
        elif func_name in self.objects:
            funcs = self.objects[func_name]
        else:
            raise RuntimeError("Could not find function", func_name)
        func = None
        func_args = None
        for func_opt_args, func_opt in funcs:
            match = True
            if len(arg_vals) != len(func_opt_args):
                match = False
            for i in range(len(func_opt_args)):
                if func_opt_args[i] is not None and func_opt_args[i] != arg_vals[i]:
                    match = False
            if match:
                func = func_opt
                func_args = func_opt_args
                break
        if func is None:
            raise RuntimeError("Could not match arguments")
        variables = {}
        count = 0
        for i in range(len(func_args)):
            if func_args[i] is None:
                variables[func.args[count]] = arg_vals[i]
                count += 1
        return self.evaluate(func.body, variables)

    @classmethod
    def literal(cls, string):
        stripped = string.strip()
        if stripped[0] == stripped[-1] == '"':
            return stripped[1:-1]
        if stripped[0] == "[" and stripped[-1] == "]":
            parts = separate(stripped[1:-1])
            val = []
            for part in parts:
                val.append(cls.literal(part))
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
        return None


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


filename = "testprog.shr"  # input("File to execute: ")
with open(filename, "r") as f:
    code = f.readlines()

program = Program(code)
program.run()
