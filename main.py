from core import Function, value, separate, enable_log, log_file
from builtin_funcs import Add, Mult, Sub, Div


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
            if len(action_parts) == 2:
                action, data = action_parts
            elif len(action_parts) == 1:
                data = action_parts[0]
                action = "output"
            else:
                raise RuntimeError("Expected action : data, got " + line)
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
                if not self.read_line(line):
                    return False
            return True
        elif action == "quit":
            return False
        else:
            raise RuntimeError("Unknown action")


if __name__ == "__main__":
    if enable_log:
        with open(log_file, "w") as file:
            file.write("")
    program = Program()
    line_in = input("> ")
    while program.read_line(line_in):
        line_in = input("> ")
