'''APL : the Artificial Programmer Language'''
import sys
import math


def _tokenize(line):
    line = line.lower()
    for sep in ["(", ")", "-", "+", "*", "/", "#"]:
        line = line.replace(sep, " " + sep + " ")
    for sep in [",", "'", ""]:
        line = line.replace(sep, "")
    return [token for token in line.split(" ") if token != '']


class Parser:
    ''''''
    def __init__(self):
        self.end_of_program = ""

    def _first_token(self):
        if len(self.tokens) > 0:
            self.i = 0
            self.token = self.tokens[self.i]
            self.peek_token = self.tokens[self.i + 1] if self.i + 1 < len(self.tokens) else self.end_of_program
        else:
            self.token = self.end_of_program
            self.peek_token = self.end_of_program

    def _next_token(self):
        if len(self.tokens) > self.i + 1:
            self.i += 1
            self.token = self.tokens[self.i]
            self.peek_token = self.tokens[self.i + 1] if self.i + 1 < len(self.tokens) else self.end_of_program
        else:
            self.token = self.end_of_program
            self.peek_token = self.end_of_program

    def _token_to_str(self, token):
        return token if token != self.end_of_program else "end-of-program"

    def _expect_token(self, expected):
        if self.token != expected:
            raise RuntimeError(f"'{self._token_to_str(expected)}' expected instead of '{self._token_to_str(self.token)}'")
        self._next_token()

    def _parse_element(self):
        # print("_parse_element start", self.token)
        if self.token.isnumeric(): # positive integral number
            result = int(self.token)
            self._next_token()
        elif self.token == "-" and self.peek_token.isnumeric(): # negative integral number
            self._next_token()
            result = -int(self.token)
            self._next_token()
        elif self.token.isidentifier(): # identifier, but also "setq", "for1",
            result = self.token
            self._next_token()
        elif self.token in ["+", "*"]: # operator
            result = self.token
            self._next_token()
        elif self.token == "(": # list
            self._next_token()
            result = []
            while self.token != ")":
                result.append(self._parse_element())
            self._next_token()
        else:
            raise RuntimeError(f"atom or list expected instead of '{self._token_to_str(self.token)}'")
        # print("_parse_element result", result, "next token", self.token)
        return result

    def compile(self, program_str):
        print("compile: program_str", program_str)
        self.tokens = _tokenize(program_str)
        self._first_token()
        print("compile: tokens", self.tokens)
        program = self._parse_element()
        self._expect_token(self.end_of_program)
        return program

    def _parse_deap_element(self):
        # print("_parse_deap_element start", self.token)
        if self.token.isnumeric(): # positive integral number
            result = int(self.token)
            self._next_token()
        elif self.token == "-" and self.peek_token.isnumeric(): # negative integral number
            self._next_token()
            result = -int(self.token)
            self._next_token()
        elif self.token in ["_integer2integer", "_identifier2integer", "_identifier2identifier", "_empty_list", "_identifier2element", "_integer2element"]: # type-casts can be removed
            self._next_token()
            self._expect_token("(")
            result = self._parse_deap_element() if self.token != ")" else []
            self._expect_token(")")
        elif self.token in ["for1", "setq", "_print", "mul", "cons"]:
            # formula(arg1, ...argn)
            result = [self.token]
            self._next_token()
            self._expect_token("(")
            while self.token != ")":
                result.append(self._parse_deap_element())
            self._next_token()
        elif self.token.isidentifier(): # identifier
            result = self.token
            self._next_token()
        else:
            raise RuntimeError(f"number, function or identifyer expected instead of '{self._token_to_str(self.token)}'")
        # print("_parse_deap_element result", result, "next token", self.token)
        return result

    def compile_deap(self, program_str):
        # print("compile_deap: program_str", program_str)
        self.tokens = _tokenize(program_str)
        self._first_token()
        # print("compile_deap: tokens", self.tokens)
        program = self._parse_deap_element()
        self._expect_token(self.end_of_program)
        # print("compile_deap: result", program)
        return program

_parser = Parser()
_memory = dict()
_output = []


def _run(program):
    global _memory, _output
    if type(program) == type([]):
        if len(program) == 0:
            return 0
        if type(program[0]) == type([]): # compound statement
            for statement in program:
                result = _run(statement)
            return result
        if program[0] == "setq": # example (setq x 5)
            if len(program) != 3:
                print(f"setq: name value expected {program[1:]}")
                exit(1)
                return 0
            variable_name = program[1]
            if type(variable_name) != type(""):
                print(f"setq: name expected instead of {variable_name}")
                exit(1)
                return 0
            variable_value = _run(program[2])
            _memory[variable_name] = variable_value
            return variable_value
        if program[0] == "for1": # example (for1 i n (_print i))
            if len(program) != 4:
                print(f"for1: loop_variable upper_value statement expected instead of {program[1:]}")
                exit(1)
                return 0
            loop_variable = program[1]
            if type(loop_variable) != type(""):
                print(f"for1: loop_variable expected instead of '{loop_variable}'")
                exit(1)
                return 0
            upper_bound = _run(program[2])
            if type(upper_bound) != type(5):
                print(program[2], "-->", upper_bound)
                print(f"for1: integer upper bound expected instead of '{upper_bound}'")
                exit(1)
                return 0
            result = 0
            for i in range(1, upper_bound+1):
                _memory[loop_variable] = i
                result = _run(program[3])
            return result
        if program[0] == "_print": # example(_print i)
            if len(program) != 2:
                print(f"_print: argument expected instead of {program[1:]}")
                exit(1)
                return 0
            value = _run(program[1])
            _output.append(value)
            return value
        if program[0] == "mul": # example(_print i)
            if len(program) != 3:
                print(f"mul: 2 operands expected instead of {program[1:]}")
                exit(1)
                return 0
            value1 = _run(program[1])
            value2 = _run(program[2])
            assert type(value1) == type(1) and type(value2) == type(1)
            return value1 * value2
        if program[0] == "cons": # example(cons a b)
            if len(program) != 3:
                print(f"cons: 2 operands expected instead of {program[1:]}")
                exit(1)
                return 0
            value1 = _run(program[1])
            value2 = _run(program[2])
            if type(value1) != type([]):
                value1 = [value1]
            if type(value2) != type([]):
                value2 = [value2]
            return value1 + value2
        print (f"apl: onbekende functie '{program[0]}'")
        return 0
    elif type(program) == type(""):
        variable_name = program[0]
        return _memory[variable_name] if variable_name in _memory else 0
    else:
        if type(program) != type(1):
            print(f"list, identifyer or int expected instead of '{program}'")
            exit(1)
            return 0
        return program


# ============================================== INTERFACE ====================


def compile(program_str):
    '''Compiles "Lisp" program_str into an APL program'''
    return _parser.compile(program_str)


def compile_deap(program_str):
    '''Compiles DEAP program_str into an APL program'''
    program = _parser.compile_deap(program_str)
    return program


def run(program, memory):
    '''runs compiled program in given input and return the output'''
    # print("run", str(program), "memory", memory)
    global _output, _memory
    _output = []
    _memory = dict(memory) # makes a copy
    _run(program)
    return _output


def evaluate_output(model_output, expected_output, debug=False):
    '''compute and return error on this output'''
    if debug:
        print("evaluate_output")
        print("model_output", model_output)
        print("expected_output", expected_output)
    k = len(expected_output)
    if k > 0:
        error = 0.33 * min(1.0, abs(len(model_output) - len(expected_output))/5.0)
        for i in range(len(expected_output)):
            if i >= len(model_output) or (model_output[i] != expected_output[i]):
                error += 0.33 / k
        # print("model_output", model_output)
        try:
            model_output = set(model_output)
        except:
            model_output = set()
        expected_output = set(expected_output)
        k = len(expected_output)
        for output in expected_output:
            if output not in model_output:
                error += 0.33 / k
    else:
        error = 0.0 if len(model_output) == 0 else min(1.0, len(model_output)/100.0)
    return error


def convert_accumulated_evaluation_into_error(accumulated_evaluation):
    '''compute one error value from all thse evaluations.'''
    return sum(accumulated_evaluation) / len(accumulated_evaluation)


def get_examples(example_file):
    '''Examples are in a tab delimited file, with the columns '''
    examples = []
    with open(example_file, "r") as f:
        solution = f.readline().rstrip()
        hdr = f.readline().rstrip().lower().split("\t")
        input_labels = [label for label in hdr if label not in ["", "output",]]
        input_dimension = len(input_labels)
        for line in f:
            values = [int(s) for s in line.rstrip().lower().split("\t")]
            examples.append((values[:input_dimension], values[input_dimension:]))
    return examples, input_labels, len(solution)


def bind_example(labels, values):
    '''Return dict with each label having the corresponding value'''
    return {label: value for label, value in zip(labels, values)}


if __name__ == "__main__":
    # a few simple tests
    for program_str in [ \
            "(for1 i n (_print i))", \
            "((setq x 1) (for1 i n (setq x (mul x i))) (_print x))", \
            "(for1 i n ((setq x 1) (for1 j i (setq x (mul x j))) (_print x)))", ]:
        program = compile(program_str)
        print(program)
        memory = {"n": 5}
        print("run with", memory)
        output = run(program, memory)
        print(output)
    examples, labels = get_examples("easy.txt")
    print("examples")
    for input, output in examples:
        print(input, output)
    print("labels", labels)
    print("bind", bind_example(labels, examples[1]))
    errors = []
    for model_output, expected_output in [ \
            [[1, 3], [1, 3]], [[], []], [[1], []], \
            [[1, 2], [1, 3]], [[2, 1], [1, 3]], [[3, 1], [1, 3]], [[3,], [1, 3]], \
            [[1, 2, 3], [1, 3]], [[2, 1, 3], [1, 3]], [[3, 1, 2], [1, 3]], [[1, 3, 2], [1, 3]], ]:
        errors.append(evaluate_output(model_output, expected_output))
        print("evaluate_output", model_output, expected_output, errors[-1])
    print("accumulated error", convert_accumulated_evaluation_into_error(errors))