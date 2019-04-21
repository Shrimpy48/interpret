import numpy
from copy import deepcopy


enable_log = False
log_file = "log.txt"
log_indent = 0


class Function:
    def __init__(self, name):
        self.name = name
        self.args = []
        self.defs = []
        self.given_args = []

    def __str__(self):
        string = self.name
        for arg in self.given_args:
            if separate(str(arg))[0] != str(arg):
                string += " (" + str(arg) + ")"
            else:
                string += " " + str(arg)
        return string

    def __repr__(self):
        return "Function(" + self.name + ")"

    def add_def(self, args, body):
        self.args.append(args)
        self.defs.append(body)

    def evaluate(self, new_args, namespace):
        args = self.given_args + new_args
        if enable_log:
            string = self.name
            for arg in args:
                if separate(str(arg))[0] != str(arg):
                    string += " (" + str(arg) + ")"
                else:
                    string += " " + str(arg)
            log("(start)   evaluating " + string + " " + str(namespace), 1)
        arg_matches = deepcopy(self.args)
        for i in range(len(self.given_args), len(args)):
            for j in range(len(self.args)):
                if i < len(self.args[j]) and not isinstance(self.args[j][i], Function):
                    if args[i] != self.args[j][i]:
                        arg_matches[j] = None
        for i in range(len(arg_matches)):
            if arg_matches[i] is not None and len(arg_matches[i]) <= len(args):
                result = self.call(args, i, namespace)
                if enable_log:
                    log("(full)    evaluating " + string + " got " + str(result), -1)
                return result
        matches = [arg for arg in arg_matches if arg is not None]
        defs = [self.defs[i] for i in range(len(arg_matches)) if arg_matches[i] is not None]
        result = type(self)(self.name)
        for i in range(len(matches)):
            result.add_def(matches[i], defs[i])
        result.given_args = args
        if enable_log:
            log("(partial) evaluating " + string + " got " + str(result), -1)
        return result

    def call(self, args, which, namespace):
        taken_num = len(self.args[which])
        local_namespace = deepcopy(namespace)
        for i in range(taken_num):
            if isinstance(self.args[which][i], Function):
                local_namespace[self.args[which][i].name] = args[i]
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


def value(string, namespace):
    if enable_log:
        log("(start)   finding value of " + string + " " + str(namespace), 1)
    stripped = string.strip()
    while stripped[0] == "(" and stripped[-1] == ")":
        stripped = stripped[1:-1]
    if stripped[0] == stripped[-1] == '"':
        result = stripped[1:-1]
        if enable_log:
            log("(value)   value of " + string + " is " + str(result), -1)
        return result
    if stripped[0] == "[" and stripped[-1] == "]":
        parts = separate(stripped[1:-1])
        val = []
        for part in parts:
            val.append(value(part, namespace))
        result = numpy.array(val)
        if enable_log:
            log("(value)   value of " + string + " is " + str(result), -1)
        return result
    try:
        val = numpy.int(stripped)
    except ValueError:
        pass
    else:
        result = val
        if enable_log:
            log("(value)   value of " + string + " is " + str(result), -1)
        return result
    try:
        val = numpy.float32(stripped)
    except ValueError:
        pass
    else:
        result = val
        if enable_log:
            log("(value)   value of " + string + " is " + str(result), -1)
        return result
    parts = separate(stripped)
    name = parts[0]
    args = [value(arg, namespace) for arg in parts[1:]]
    if name in namespace:
        if not isinstance(namespace[name], Function):
            result = namespace[name]
            if enable_log:
                log("(name)    value of " + string + " is " + str(result), -1)
            return result
        result = namespace[name].evaluate(args, namespace)
        if enable_log:
            log("(namefun) value of " + string + " is " + str(result), -1)
        return result
    result = Function(name).evaluate(args, namespace)
    if enable_log:
        log("(func)    value of " + string + " is " + str(result), -1)
    return result


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


def log(string, ind_inc):
    global log_indent
    with open(log_file, "a") as file:
        file.write("  " * log_indent + string + "\n")
    log_indent += ind_inc
