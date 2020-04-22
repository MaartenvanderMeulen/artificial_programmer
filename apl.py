'''APL : the Artificial Programmer Language'''
import sys
import math
import numpy as np


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
        # print("compile: program_str", program_str)
        self.tokens = _tokenize(program_str)
        self._first_token()
        # print("compile: tokens", self.tokens)
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

global _memory, _output
_parser = Parser()
_memory = dict()
_output = []


def _run(program, debug, indent):
    global _memory, _output
    if debug:
        print(indent, "_run start", "program", program, "_memory", _memory, "_output", _output)
    if type(program) == type([]):
        result = None
        if len(program) == 0:
            result = 0
        elif type(program[0]) == type([]): # non-empty compound statement
            for statement in program:
                result = _run(statement, debug, indent+" ")
        elif program[0] == "setq": # example (setq x 5)
            if len(program) != 3:
                raise RuntimeError(f"setq: name value expected {program[1:]}")
            variable_name = program[1]
            if type(variable_name) != type(""):
                raise RuntimeError(f"setq: name expected instead of {variable_name}")
            result = _run(program[2], debug, indent+" ")
            _memory[variable_name] = result
        elif program[0] == "for1": # example (for1 i n (_print i))
            if len(program) != 4:
                raise RuntimeError(f"for1: loop_variable upper_value statement expected instead of {program[1:]}")
            loop_variable = program[1]
            if type(loop_variable) != type(""):
                raise RuntimeError(f"for1: loop_variable expected instead of '{loop_variable}'")
            upper_bound = _run(program[2], debug, indent+" ")
            if type(upper_bound) != type(5):
                raise RuntimeError(f"for1: integer upper bound expected instead of '{upper_bound}'")
            result = 0
            for i in range(1, upper_bound+1):
                _memory[loop_variable] = i
                result = _run(program[3], debug, indent+" ")
        elif program[0] == "_print": # example(_print i)
            if len(program) != 2:
                raise RuntimeError(f"_print: argument expected instead of {program[1:]}")
            result = _run(program[1], debug, indent+" ")
            _output.append(result)
        elif program[0] == "mul": # example(_print i)
            if len(program) != 3:
                raise RuntimeError(f"mul: 2 operands expected instead of {program[1:]}")
            value1 = _run(program[1], debug, indent+" ")
            value2 = _run(program[2], debug, indent+" ")
            assert type(value1) == type(1) and type(value2) == type(1)
            result = value1 * value2
        elif program[0] == "cons": # example(cons a b)
            if len(program) != 3:
                raise RuntimeError(f"cons: 2 operands expected instead of {program[1:]}")
            value1 = _run(program[1], debug, indent+" ")
            value2 = _run(program[2], debug, indent+" ")
            result = value2
        else:
            raise RuntimeError(f"apl: onbekende functie '{program[0]}'")
        assert result is not None
    elif type(program) == type(""):
        variable_name = program[0]
        result = _memory[variable_name] if variable_name in _memory else 0
    else:
        if type(program) != type(1):
            raise RuntimeError(f"list, identifyer or int expected instead of '{program}'")
        result = program
    if debug:
        print(indent, "_run   end", "program", program, "_memory", _memory, "_output", _output, "result", result)
    return result


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
    value = _run(program, False, "")
    assert type(value) == type(1)
    if len(_output) == 0:
        _output.append(value)
    return _output


dynamic_weight_iteration = 1
sum_weighted_errors = np.zeros((5))
weights = np.array([0.0433, 0.0002, 0.0024, 0.0002, 0.9539])


def distance_with_closest_values(x, values):
    '''distance of x with nearest'''
    if len(values) > 0:
        result = abs(x - values[0])
        for value in values[1:]:
            if result > abs(x - value):
               result = abs(x - value)
    else:
        result = abs(x - 0)
    return result


def evaluate_output(model_output, expected_output, debug=False):
    '''compute and return error on this output'''
    if debug:
        print("evaluate_output")
        print("model_output", model_output)
        print("expected_output", expected_output)
    k = len(expected_output)
    if k == 0:
        raise RuntimeError("TODO: handle case were expected output is empty")
    errors = np.zeros((4))
    # aantal outputs
    errors[0] = (len(model_output) - len(expected_output)) ** 2
    # hoever zitten de expected getallen van de model getallen af
    for output in expected_output:
        errors[1] += distance_with_closest_values(output, model_output) ** 1.5
    # hoever zitten de model getallen van de expected getallen af
    for output in model_output:
        errors[2] += distance_with_closest_values(output, expected_output) ** 1.5
    # absolute verschil van de outputs met de gewenste output
    for i in range(len(expected_output)):
        if i < len(model_output):
            errors[3] += abs(model_output[i] - expected_output[i]) ** 1.5
        else:
            errors[3] += abs(expected_output[i]) ** 1.5
    global sum_weighted_errors, weights
    weighted_errors = errors[:4] * weights[:4]
    sum_weighted_errors[:4] += weighted_errors
    return np.sum(weighted_errors)


def evaluate_program(program_str, hints):
    missed_hints = 0
    for hint in hints:
        if program_str.find(hint) == -1:
            missed_hints += 1
    errors = missed_hints ** 2
    global sum_weighted_errors, weights
    weighted_errors = errors * weights[4]
    sum_weighted_errors[4] += weighted_errors
    # print("progrsam_str", program_str, "hints", hints, "missed_hints", missed_hints)
    return weighted_errors

def dynamic_error_weight_adjustment():
    global dynamic_weight_iteration, sum_weighted_errors, weights
    dynamic_weight_iteration += 1
    if dynamic_weight_iteration >= 1000:
        print("weights before", weights)
        n = len(weights)
        weights *= (np.sum(sum_weighted_errors) / n) / sum_weighted_errors
        weights /= np.sum(weights)
        sum_weighted_errors = np.zeros((5))
        dynamic_weight_iteration = 1
        print("weights after", weights)


def convert_accumulated_evaluation_into_error(accumulated_evaluation):
    '''compute one error value from all thse evaluations.'''
    # aantal getallen
    return sum(accumulated_evaluation) / len(accumulated_evaluation)


def get_examples(example_file):
    '''Examples are in a tab delimited file, with the columns '''
    examples = []
    with open(example_file, "r") as f:
        solution = f.readline().rstrip()
        hints = f.readline().rstrip().lower().split("\t")
        hdr = f.readline().rstrip().lower().split("\t")
        input_labels = [label for label in hdr if label not in ["", "output",]]
        input_dimension = len(input_labels)
        for line in f:
            values = [int(s) for s in line.rstrip().lower().split("\t")]
            examples.append((values[:input_dimension], values[input_dimension:]))
    return examples, input_labels, len(solution), hints


def bind_example(labels, values):
    '''Return dict with each label having the corresponding value'''
    return {label: value for label, value in zip(labels, values)}


if __name__ == "__main__":
    # a few simple tests
    if False:
        for program_str in [ \
                "(for1 i n (_print i))", \
                "((setq x 1) (for1 i n (setq x (mul x i))) (_print x))", \
                "(for1 i n ((setq x 1) (for1 j i (setq x (mul x j))) (_print x)))", \
                ]:
            program = compile(program_str)
            print("apl.__main__: program", program)
            memory = {"n": 5}
            print("apl.__main__: memory", memory)
            output = run(program, memory)
            print("apl.__main__: output of run", output)
    if False:
        for program_str in [ \
                # "for1(_identifier2identifier(_identifier2identifier('j')), _identifier2identifier(_identifier2identifier('i')), setq(_identifier2identifier('x'), mul(1, ARG0)))", \
                # "mul(for1(_identifier2identifier('x'), _identifier2identifier('i'), cons(setq('n', ARG0), _print(1))), _identifier2integer(_identifier2identifier('j')))", \
                "cons(_print(for1('i', 'i', 1)), mul(setq('x', 1), setq('i', 1)))", \
                ]:
            program = compile_deap(program_str)
            print("apl.__main__: program deap", program)
            memory = {"n": 5}
            print("apl.__main__: memory", memory)
            output = run(program, memory)
            print("apl.__main__: output of run", output)
    if False:
        for file_name in ["easy.txt", "medium.txt", ]:
            examples, labels, len_solution, hints = get_examples(file_name)
            print("examples", file_name)
            for input, output in examples:
                print(input, output)
            print("labels", labels)
            print("len_solution", len_solution)
            print("hints", hints)
            print("bind", bind_example(labels, examples[1]))
    if False: # test of evaluation function
        weighted_errors = 0.0
        for model_output, expected_output in [ \
                [[1, 3], [1, 3]], \
                [[], [1, 3]], \
                [[1, ], [1, 3]], [[2, ], [1, 3]], [[3, ], [1, 3]], \
                [[1, 2], [1, 3]], [[2, 1], [1, 3]], [[3, 1], [1, 3]], [[1, 1,], [1, 3]], \
                [[1, 2, 3], [1, 3]], [[2, 1, 3], [1, 3]], [[3, 1, 2], [1, 3]], [[1, 3, 2], [1, 3]], ]:
            weighted_error = evaluate_output(model_output, expected_output)
            weighted_errors += weighted_error
            print("evaluate_output", model_output, expected_output, weighted_error)
        print("accumulated error", weighted_errors)
    if True: # test of evaluation function
        weighted_errors = 0.0
        for model_output, expected_output in [ \
                [[1,], [1,]], \
                [[2,], [2]], \
                [[3,], [6,]], \
                [[4,], [24,]], \
                [[5,], [120,]], ]:
            weighted_error = evaluate_output(model_output, expected_output)
            weighted_errors += weighted_error
            print("evaluate_output", model_output, expected_output, weighted_error)
        print("accumulated error", weighted_errors)
        print("dynamic weight adjustment", sum_weighted_errors)
    if True: # test of evaluation function
        weighted_errors = 0.0
        for model_output, expected_output in [ \
                [[0,], [1,]], \
                [[0,], [2]], \
                [[0,], [6,]], \
                [[0,], [24,]], \
                [[0,], [120,]], ]:
            weighted_error = evaluate_output(model_output, expected_output)
            weighted_errors += weighted_error
            print("evaluate_output", model_output, expected_output, weighted_error)
        print("accumulated error", weighted_errors)
        print("dynamic weight adjustment", sum_weighted_errors)
