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

    def raise_error(self, msg2):
        msg1 = ""
        for i in range(len(self.tokens)):
            if i == self.i:
                msg1 += ">>" + self.tokens[i] + "<< "
            else:
                msg1 += self.tokens[i] + " "
        raise RuntimeError(msg1 + "\n              " + msg2)
            
    
    def _expect_token(self, expected):
        if self.token != expected:
            self.raise_error(f"'{self._token_to_str(expected)}' expected")
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

    def _parse_deap_element(self, quoted=False):
        # print("_parse_deap_element start", self.token)
        if self.token.isnumeric(): # positive integral number
            result = int(self.token)
            self._next_token()
        elif self.token == "-" and self.peek_token.isnumeric(): # negative integral number
            self._next_token()
            result = -int(self.token)
            self._next_token()
        elif self.token in ["_empty_list",]: # type-casts can be removed
            self._next_token()
            result = []
        elif self.token in ["_function2function"]: # type-casts can be removed, but here is has to be removed in a special way
            self._next_token()
            self._expect_token("(")
            result = self._parse_deap_element(True) if self.token != ")" else []
            self._expect_token(")")
        elif self.token in ["_int2int", "_str2int", "_str2str", "_str2element", \
                "_str2element", "_int2element", "_int2element", "_element2element", ]: # type-casts can be removed
            self._next_token()
            self._expect_token("(")
            result = self._parse_deap_element() if self.token != ")" else []
            self._expect_token(")")
        elif self.token in ["apply"]:
            # apply(function, args_list)
            self._next_token()
            self._expect_token("(")
            if self.token != "mul":
                function_name = self._parse_deap_element() # can be "_function2function(mul)"
            else:
                self._expect_token("mul")
                function_name = "mul"
            args_list = self._parse_deap_element()
            self._expect_token(")")
            result = ["apply", function_name, args_list]
        elif self.token in ["for1", "setq", "_print", "mul", "cons"]:
            # formula(arg1, ...argn)
            result = [self.token]
            self._next_token()
            if not quoted:
                self._expect_token("(")
                while self.token != ")":
                    result.append(self._parse_deap_element())
                self._next_token()
        elif self.token.isidentifier(): # identifier
            result = self.token
            self._next_token()
        else:
            self.raise_error(f"number, function or identifyer expected")
        # print("_parse_deap_element result", result, "next token", self.token)
        return result

    def compile_deap(self, program_str, debug):
        if debug:
            print("compile_deap: 1/3 program_str", program_str)
        self.tokens = _tokenize(program_str)
        self._first_token()
        if debug:
            print("compile_deap: 2/3 tokens", self.tokens)
        program = self._parse_deap_element()
        self._expect_token(self.end_of_program)
        if debug:
            print("compile_deap: 3/3 result", program)
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
            result = []
        elif type(program[0]) == type([]): # non-empty compound statement
            for statement in program:
                result = _run(statement, debug, indent+" ")
        elif program[0] == "setq": # example (setq x 5)
            if len(program) != 3:
                if debug:
                    print(f"setq: name value expected {program[1:]}")
                return 0
            variable_name = program[1]
            if type(variable_name) != type(""):
                if debug:
                    print(f"setq: name expected instead of {variable_name}")
                return 0
            result = _run(program[2], debug, indent+" ")
            _memory[variable_name] = result
        elif program[0] == "for1": # example (for1 i n i))
            if len(program) < 4:
                if debug:
                    print(f"for1: loop_variable upper_value statement expected instead of {program[1:]}")
                return []
            loop_variable = program[1]
            if type(loop_variable) != type(""):
                if debug:
                    print(f"for1: loop_variable expected instead of '{loop_variable}'")
                return []
            upper_bound = _run(program[2], debug, indent+" ")
            if type(upper_bound) != type(1):
                if debug:
                    print(f"for1: integer upper bound expected instead of '{upper_bound}'")
                return []
            result = []
            for i in range(1, upper_bound+1):
                _memory[loop_variable] = i
                result.append(_run(program[3], debug, indent+" "))
        elif program[0] == "_print": # example (_print i)
            result = 0
            for p in program[1:]:
                result = _run(program[1], debug, indent+" ")
                _output.append(result)
        elif program[0] == "mul": # example (mul 1 2 3)
            result = 1
            for p in program[1:]:
                value = _run(p, debug, indent+" ")
                if type(value) == type(1):
                    result *= value
                else:
                    result = 0
        elif program[0] == "cons": # example (cons a b)
            result = 0
            for p in program[1:]:
                result = _run(p, debug, indent+" ")
        elif program[0] == "apply": # example (apply mul (1 2 3))
            if len(program) < 3:
                if debug:
                    print(f"apply: function list expected instead of {program[1:]}")
                return 0
            function_name = _run(program[1], debug, indent+" ")
            if function_name not in ["mul", "apply", "for1", "_print", "cons", "setq"]:
                if debug:
                    print(f"apply: function expected instead of {program[1:]}")
                return 0
            arguments = _run(program[2], debug, indent+" ")
            if type(arguments) != type([]):
                if debug:
                    print(f"apply: function arguments list expected instead of {program[2:]}")
                return 0
            result = _run([function_name] + arguments, debug, indent+" ")
        else:
            result = _run(program[0], debug, indent+" ")
        assert result is not None
    elif type(program) == type(""):
        identifyer = program
        if identifyer == "_empty_list":
            result = []
        elif identifyer in ["mul", "apply", "for1", "_print", "cons", "setq"]:
            result = identifyer
        else:
            result = _memory[identifyer] if identifyer in _memory else 0
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
    program = _parser.compile_deap(program_str, False)
    return program


def convert_to_list_of_ints(data):
    result = []
    for x in data:
        if type(x) == type(1):
            if x < -1000000:
                x = -1000000
            if x > 1000000:
                x = 1000000
            result.append(x)
        else:
            assert type(x) == type([])
            result.extend(convert_to_list_of_ints(x))
    return result


def run(program, memory):
    '''runs compiled program in given input and return the output'''
    # print("run", str(program), "memory", memory)
    global _output, _memory
    _output = []
    _memory = dict(memory) # makes a copy
    value = _run(program, False, "")
    if len(_output) == 0:
        if type(value) == type(1):
            _output = [value]
        else:
            _output = value
    return convert_to_list_of_ints(_output)


dynamic_weight_iteration = 1
sum_weighted_errors = np.zeros((6))
# weights = np.array([4.99324622e-01, 3.72453567e-05, 3.72453567e-05, 3.72453567e-05, 5.00487193e-01, 7.64482815e-05])
weights = np.array([3.75871763e-02, 3.51454465e-04, 1.78800674e-08, 3.02494615e-08, 7.69075095e-01, 1.92986226e-01])

def distance_with_closest_values(x, values):
    '''distance of x with nearest'''
    if x > 1000000:
        x = 1000000
    if len(values) > 0:
        result = 1000000
        for value in values:
            if value > 1000000:
                value = 1000000
            if result > abs(x - value):
               result = abs(x - value)
    else:
        result = abs(x - 0)
    assert result <= 1000000
    #if result > 1000:
    #    print(result, x, values)
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
            # print(model_output[i], expected_output[i])
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


def evaluate_postcondition(model_output, prev_model_output, input):
    if len(model_output) != 1 or prev_model_output is None or len(prev_model_output) != 1 or len(input) != 1:
        return 0.0
    model_increase = model_output[0] - prev_model_output[0]
    expected_increase = input[0]
    errors = abs(model_increase - expected_increase) ** 1.5
    global sum_weighted_errors, weights
    weighted_errors = errors * weights[5]
    sum_weighted_errors[5] += weighted_errors
    return weighted_errors
    

def dynamic_error_weight_adjustment():
    global dynamic_weight_iteration, sum_weighted_errors, weights
    dynamic_weight_iteration += 1
    if dynamic_weight_iteration >= 1000:
        debug = True
        n = len(weights)
        if debug:
            print("weights before", weights, sum_weighted_errors, n)
        average_weighted_error = np.sum(sum_weighted_errors) / n
        for i in range(6):
            if sum_weighted_errors[i] > 0:
                weights[i] *= average_weighted_error / sum_weighted_errors[i]
            else:
                weights[i] = 1.0 / n
        weights /= np.sum(weights)
        sum_weighted_errors = np.zeros((len(sum_weighted_errors)))
        dynamic_weight_iteration = 1
        if debug:
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
                # "(_print (for1 i n i))", \
                # "((setq x 1) (for1 i n (setq x (mul x i))) (_print x))", \
                # "((setq x 1) (for1 i n (setq x (mul x i))) x)", \
                "(apply mul (for1 i n i))", \
                # "(for1 i n ((setq x 1) (for1 j i (setq x (mul x j))) (_print x)))", \
                "(for1 i n (_print (apply mul (for1 j i j))))", \
                ]:
            program = compile(program_str)
            print("apl.__main__: program", program)
            memory = {"n": 5}
            print("apl.__main__: memory", memory)
            output = run(program, memory)
            print("apl.__main__: output of run", output)
    if True:
        easy = "for1('i', 'n', _print(_str2element('i')))"
        medium = "apply('mul', for1('i', 'n', _str2element('i')))"
        hard = "for1('i', 'n', _print(apply('mul', for1('j', 'i', _str2element('j')))))"
        for program_str in [ easy, medium, hard ]:
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
    if False: # test of evaluation function
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
    if False: # test of evaluation function
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
