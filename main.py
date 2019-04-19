import pycuda.autoinit
import pycuda.driver as drv
import numpy
from pycuda.compiler import SourceModule


class Function:
    def __init__(self, args, body):
        self.objects = {}
        for arg in args:
            val = self.value(arg)
            if val is None:
                self.objects[arg] = val
        self.args = args
        self.body = body

    @classmethod
    def value(cls, string):
        stripped = string.strip()
        if stripped[0] == stripped[-1] == '"':
            return stripped[1:-1]
        if stripped[0] == "[" and stripped[-1] == "]":
            parts = stripped[1:-1].split(",")
            val = []
            for part in parts:
                val.append(cls.value(part))
            return val
        try:
            val = int(stripped)
        except ValueError:
            pass
        else:
            return val
        try:
            val = float(stripped)
        except ValueError:
            pass
        else:
            return val
        return None


class Program:
    def __init__(self, code_lines):
        self.objects = {}
        for line in code_lines:
            self.parse(line)

    def parse(self, line):
        parts = line.split("=")
        if len(parts) > 2:
            raise SyntaxError
        elif len(parts) == 2:
            declaration = parts[0].strip()
            definition = parts[1].strip()
            dec_parts = separate(declaration)
            name = dec_parts[0]
            args = dec_parts[1:]
            self.objects[name] = Function(args, definition)
        else:
            raise NotImplementedError


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
