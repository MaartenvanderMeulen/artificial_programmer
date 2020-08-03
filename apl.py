'''APL : the Artificial Programmer Language'''
import sys


class Parser:
    ''''''
    _end_of_program = ""
    _i = None
    _peek_token = None
    _token = None
    _tokens = None

    def _tokenize(line):
        line = line.lower()
        for sep in ["(", ")", "-", "+", "*", "/", "#"]:
            line = line.replace(sep, " " + sep + " ")
        for sep in [",", "'", '"']:
            line = line.replace(sep, "")
        return [token for token in line.split(" ") if token != '']

    def _first_token():
        if len(Parser._tokens) > 0:
            Parser._i = 0
            Parser._token = Parser._tokens[Parser._i]
            Parser._peek_token = Parser._tokens[Parser._i + 1] if Parser._i + 1 < len(Parser._tokens) else Parser._end_of_program
        else:
            Parser._token = Parser._end_of_program
            Parser._peek_token = Parser._end_of_program

    def _next_token():
        if len(Parser._tokens) > Parser._i + 1:
            Parser._i += 1
            Parser._token = Parser._tokens[Parser._i]
            Parser._peek_token = Parser._tokens[Parser._i + 1] if Parser._i + 1 < len(Parser._tokens) else Parser._end_of_program
        else:
            Parser._token = Parser._end_of_program
            Parser._peek_token = Parser._end_of_program

    def _token_to_str(token):
        return token if token != Parser._end_of_program else "end-of-program"

    def _raise_error(msg2):
        msg1 = ""
        for i in range(len(Parser._tokens)):
            if i == Parser._i:
                msg1 += ">>" + Parser._tokens[i] + "<< "
            else:
                msg1 += Parser._tokens[i] + " "
        raise RuntimeError(msg1 + "\n              " + msg2)

    def _expect_token(expected):
        if Parser._token != expected:
            Parser._raise_error(f"'{Parser._token_to_str(expected)}' expected")
        Parser._next_token()

    def _parse_element():
        if Parser._token.isnumeric(): # positive integral number
            result = int(Parser._token)
            Parser._next_token()
        elif Parser._token == "-" and Parser._peek_token.isnumeric(): # negative integral number
            Parser._next_token()
            result = -int(Parser._token)
            Parser._next_token()
        elif Parser._token.isidentifier(): # identifier, but also "setq", "for1",
            result = Parser._token
            Parser._next_token()
        elif Parser._token == "(": # list
            Parser._next_token()
            result = []
            while Parser._token != ")":
                result.append(Parser._parse_element())
            Parser._next_token()
        else:
            raise RuntimeError(f"atom or list expected instead of '{Parser._token_to_str(Parser._token)}'")
        return result

    def compile(program_str):
        Parser._tokens = Parser._tokenize(program_str)
        Parser._first_token()
        program = Parser._parse_element()
        Parser._expect_token(Parser._end_of_program)
        return program


# ====================== interpreter ======================================


def _run(program, memory, debug=False, indent=""):
    if debug:
        print(indent, "_run start", "program", program, "memory", memory)
    if type(program) == type([]):
        result = None
        if len(program) == 0:
            result = []
        elif program[0] == "add": # example (add 3 2)
            result = 0
            for p in program[1:]:
                value = _run(p, memory, debug, indent+" ")
                if type(value) == type(1):
                    result += value
                else:
                    result = 0
        elif program[0] == "sub": # example (sub 3 2)
            result = 0
            for i in range(1, len(program)):
                value = _run(program[i], memory, debug, indent+" ")
                if type(value) == type(1):
                    if i == 1:
                        result = value
                    else:
                        result -= value
                else:
                    result = 0
        elif program[0] == "mul": # example (mul 1 2 3)
            result = 1
            for p in program[1:]:
                value = _run(p, memory, debug, indent+" ")
                if type(value) == type(1):
                    result *= value
                else:
                    result = 0
        elif program[0] == "div": # example (div 3 2)
            a = _run(program[1], memory, debug, indent+" ") if len(program) > 1 else 0
            if type(a) != type(1):
                a = 0
            b = _run(program[2], memory, debug, indent+" ") if len(program) > 2 else 0
            if type(b) != type(1):
                b = 0
            result = a // b if b != 0 else 0
        elif program[0] in ["lt", "le", "eq", "ne", "ge", "gt", "and", "or"]:
            a = _run(program[1], memory, debug, indent+" ") if len(program) > 1 else 0
            b = _run(program[2], memory, debug, indent+" ") if len(program) > 2 else 0
            if type(a) != type(b):
                result = 0
            elif program[0] == "lt": # example (le 3 2)
                result = int(a < b)
            elif program[0] == "le": # example (le 3 2)
                result = int(a <= b)
            elif program[0] == "eq": # example (eq 3 2)
                result = int(a == b)
            elif program[0] == "ne": # example (ne 3 2)
                result = int(a != b)
            elif program[0] == "ge": # example (ge 3 2)
                result = int(a >= b)
            elif program[0] == "gt": # example (gt 3 2)
                result = int(a > b)
            elif program[0] == "and": # example (and 1 1)
                result = int(a and b)
            elif program[0] == "or": # example (or 1 1)
                result = int(a or b)
            else:
                raise RuntimeError("internal error at line 156 of the APL interpreter")
        elif program[0] == "len": # example (len x)
            result = 0
            if len(program) > 1:
                x = _run(program[1], memory, debug, indent+" ")
                if type(x) == type([]):
                    result = len(x)
        elif program[0] == "at": # example (at x i j)
            result = 0
            if len(program) > 2:
                x = _run(program[1], memory, debug, indent+" ")
                if type(x) == type([]):
                    for dim in range(len(program) - 2):
                        index = _run(program[2 + dim], memory, debug, indent+" ")
                        if type(x) != type([]) or index > len(x):
                            result = 0
                            break
                        x = x[index]
                        result = x
        elif program[0] == "last": # example (last (assign n 3) (for0 i n i)) --> (0 1 2)
            result = 0
            for p in program[1:]:
                result = _run(p, memory, debug, indent+" ")
        elif program[0] == "if": # example (if cond x y))
            print("if : not implemented yet")
            result = 0
        elif program[0] == "call": # example (call get_row ((1 2 3) (4 5 6) (7 8 9)) 1)
            result = 0
            if len(program) > 1:
                function_name = program[1]
                if function_name in memory:
                    formal_params, code = memory[function_name]
                    actual_params = [_run(param, memory, debug, indent+" ") for param in program[2:]]
                    while len(actual_params) < len(formal_params):
                        actual_params.append(0)
                    actual_params = actual_params[:len(formal_params)]
                    old_values = [(name, memory[name]) for name in formal_params if name in memory]
                    for name, value in zip(formal_params, actual_params):
                        memory[name] = value
                    result = _run(code, memory, debug, indent+" ")
                    for name, value in old_values:
                        memory[name] = value
        elif program[0] == "assign": # example (assign x 5)
            if len(program) < 2:
                return 0
            variable_name = program[1]
            if type(variable_name) != type(""):
                return 0
            result = _run(program[2], memory, debug, indent+" ") if len(program) >= 3 else 0
            memory[variable_name] = result
        elif program[0] == "function":
            # example (function fac n (last (assign result 1) (for1 i n (assign result (mul n i)))))
            result = 0
            if len(program) >= 4:
                function_name = program[1]
                if type(function_name) == type(""):
                    params = program[2]
                    code = program[3]
                    result = (params, code)
                    memory[function_name] = result
        elif program[0] == "for1": # example (for1 i n i))
            if len(program) < 4:
                return []
            loop_variable = program[1]
            if type(loop_variable) != type(""):
                return []
            upper_bound = _run(program[2], memory, debug, indent+" ")
            if type(upper_bound) != type(1):
                return []
            result = []
            for i in range(1, upper_bound+1):
                memory[loop_variable] = i
                result.append(_run(program[3], memory, debug, indent+" "))
        elif program[0] == "for0": # example (for0 i n i))
            if len(program) < 4:
                return []
            loop_variable = program[1]
            if type(loop_variable) != type(""):
                return []
            upper_bound = _run(program[2], memory, debug, indent+" ")
            if type(upper_bound) != type(1):
                return []
            result = []
            for i in range(0, upper_bound):
                memory[loop_variable] = i
                result.append(_run(program[3], memory, debug, indent+" "))
        elif program[0] == "assert": # example (assert i))
            result = 0
            if len(program) > 1:
                value = _run(program[1], memory, debug, indent+" ")
                if not value:
                    raise RuntimeError(f"assertion failed : {str(program)}")
                result = 1
        else:
            result = []
            for p in program:
                value = _run(p, memory, debug, indent+" ")
                result.append(_run(p, memory, debug, indent+" "))
        assert result is not None
    elif type(program) == type(""):
        identifyer = program
        if identifyer == "_empty_list":
            result = []
        elif identifyer in ["mul", "apply", "for1", "_print", "cons", "setq", "sub", "add", "div"]:
            result = identifyer
        else:
            result = memory[identifyer] if identifyer in memory else 0
    elif type(program) == type(1):
        result = program
    else:
        raise RuntimeError(f"list, identifyer or int expected instead of '{program}'")
    if debug:
        print(indent, "_run   end", "program", program, "memory", memory, "result", result)
    return result


# ============================================== INTERFACE ====================


def load(file_name):
    '''Returns file content as string: skips lines that start with #; all is converted to lower case.'''
    with open(file_name, "r") as f:
        program_str = ""
        for line in f:
            line = line.strip().lower()
            if len(line) > 0 and line[0] != '#':
                program_str += line
    return program_str
    
    
def compile(program_str):
    '''Compiles program_str'''
    return Parser.compile(program_str)


def run(program, memory, debug=False):
    '''Runs compiled program'''
    return _run(program, memory, debug, "")


if __name__ == "__main__":
    file_name = sys.argv[1] if len(sys.argv) >= 2 else "test_apl.txt" 
    print(run(compile(load(file_name)), {}))
    
    if False:
        for program_str, memory, expected in [ \
                #("(add n 1)", {"n": 5}, 6), \
                #("(sub n 1)", {"n": 5}, 4), \
                #("(mul n 2)", {"n": 5}, 10), \
                #("(div n 2)", {"n": 12}, 6), \
                #("(for1 i n i)", {"n": 5}, [1, 2, 3, 4, 5]), \
                #("(for1 i n (i))", {"n": 5}, [1, 2, 3, 4, 5]), \
                #("(1 2 3 4 5 6 7 8 9 0)", {}, [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]), \
                #("(assign x (1 2 3 4 5 6 7 8 9 0))", {}, [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]), \
                #("((assign x (1 2 3 4 5 6 7 8 9 0)) x)", {}, [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]), \
                # get_row:
                ("((1 2 3) (4 5 6) (7 8 9))", {}, [[1, 2, 3], [4, 5, 6], [7, 8, 9]]), \
                ("(assign board ((1 2 3) (4 5 6) (7 8 9)))", {}, [[1, 2, 3], [4, 5, 6], [7, 8, 9]]), \
                ("(at ((1 2 3) (4 5 6) (7 8 9)) 1 0)", {}, 4), \
                ("((assign board ((1 2 3) (4 5 6) (7 8 9))) (at board 1 0))", {}, [[[1, 2, 3], [4, 5, 6], [7, 8, 9]], 4]), \
                ("(last (assign board ((1 2 3) (4 5 6) (7 8 9))) (at board 1 0))", {}, 4), \
                ("(last (assign board ((1 2 3) (4 5 6) (7 8 9))) (assign n (len board)) (for0 col n (at board row col)))", {"row":1}, [4, 5, 6]), \
                ("(call get_row ((1 2 3) (4 5 6) (7 8 9)) 1)", {
                    "get_row":[["board", "row"],
                        ["last",
                            ["assign", "n",  ["len", "board"]],
                            ["for0", "col", "n", ["at", "board", "row", "col"]]]]}, [4, 5, 6]), \
                ]:
            actual = run(compile(program_str), memory, False)
            if actual != expected:
                print("apl.__main__: program_str", program_str)
                print("apl.__main__: final memory", memory)
                print("apl.__main__: actual output of run", actual)
                print("apl.__main__: expected output of run", expected)
