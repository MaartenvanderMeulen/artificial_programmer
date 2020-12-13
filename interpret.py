'''Interpreter for LISP like programming language'''
import sys
import copy # mandatory
import time


class Parser:
    ''''''
    _end_of_program = ""
    _i = None
    _peek_token = None
    _token = None
    _tokens = None

    def _tokenize(line):
        line = line.lower()
        for deap_stuff in [",", '"', "'"]:
            line = line.replace(deap_stuff, " ")
        for sep in ["(", ")", ","]:
            line = line.replace(sep, " " + sep + " ")
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

    def _parse_deap_element(functions):
        # print("DEBUG 84 : _parse_deap_element start", Parser._token)
        if Parser._token.isnumeric(): # positive integral number
            result = int(Parser._token)
            Parser._next_token()
        elif Parser._token == "-" and Parser._peek_token.isnumeric(): # negative integral number
            Parser._next_token()
            result = -int(Parser._token)
            Parser._next_token()
        elif Parser._token in get_build_in_functions() or Parser._token in functions:
            # print("DEBU 95: buildin", Parser._token)
            # formula(arg1, ...argn)
            result = [Parser._token]
            Parser._next_token()
            Parser._expect_token("(")
            while Parser._token != ")":
                result.append(Parser._parse_deap_element(functions))
            Parser._next_token()
        elif Parser._token.isidentifier(): # identifier
            # print("DEBU 104: identifier", Parser._token)
            result = Parser._token
            Parser._next_token()
        else:
            Parser._raise_error(f"number, function or identifyer expected")
        return result

    def compile_deap(program_str, functions):
        try:
            Parser._tokens = Parser._tokenize(program_str)
            Parser._first_token()
            program = Parser._parse_deap_element(functions)
            Parser._expect_token(Parser._end_of_program)
            return program
        except RuntimeError as e:
            print("Runtimeerror", str(e))
            print("compile_deap(", program_str, ")")
            raise e



# ====================== interpreter ======================================


# TODO: put this in a Run object
global count_runs_calls
count_runs_calls = 0
longest_run = 0

def check_depth(data, current_depth):
    max_depth = 5
    if current_depth > max_depth:
        raise RuntimeError("data depth exceeded")
    count_items = 1
    if type(data) == type([]):
        count_items += 1
        if len(data) > 100:
            raise RuntimeError("data list length exceeded")
        for item in data:
            count_items += check_depth(item, current_depth+1)
    if count_items > 1000:
        raise RuntimeError("data size exceeded")
    return count_items


def _run(program, variables, functions, debug, depth):
    if depth > 100:
        #print("code depth exceeded")
        raise RuntimeError("code depth exceeded")
    global count_runs_calls
    count_runs_calls += 1
    if count_runs_calls > 10000:
        #print("code run calls exceeded")
        raise RuntimeError("code run calls exceeded")
    if debug:
        print("    "*depth, "_run start", "program", program, "variables", variables, "functions", functions)
    if type(program) == type([]):
        result = None
        if len(program) == 0:
            result = []
        elif program[0] in ["lt", "le", "ge", "gt", "add", "sub", "mul", "div"]: # operands must be NUMERIC
            a = _run(program[1], variables, functions, debug, depth+1) if len(program) > 1 else 0
            b = _run(program[2], variables, functions, debug, depth+1) if len(program) > 2 else 0
            if type(a) != type(1) or type(b) != type(1):
                result = 0
            else:
                if program[0] == "lt": # example (le 3 2)
                    result = 1 if a < b else 0
                elif program[0] == "le": # example (le 3 2)
                    result = 1 if a <= b else 0
                elif program[0] == "ge": # example (ge 3 2)
                    result = 1 if a >= b else 0
                elif program[0] == "gt": # example (gt 3 2)
                    result = 1 if a > b else 0
                elif program[0] == "add":
                    result = a + b
                elif program[0] == "sub":
                    result = a - b
                elif program[0] == "mul":
                    result = a * b
                elif program[0] == "div":
                    result = a // b if b != 0 else 0
                else:
                    raise RuntimeError("internal error at line 156 of the APL interpreter")
        elif program[0] in ["eq", "ne", ]:
            a = _run(program[1], variables, functions, debug, depth+1) if len(program) > 1 else 0
            b = _run(program[2], variables, functions, debug, depth+1) if len(program) > 2 else 0
            if program[0] == "eq":
                result = 1 if a == b else 0
            elif program[0] == "ne":
                result = 1 if a != b else 0
            else:
                raise RuntimeError("internal error at line 156 of the APL interpreter")
        elif program[0] in ["and",]: # with lazy evaluation of operands : only evaluate when needed
            result = 0 # at least one operand must be True
            for i in range(1, len(program)):
                result = 1 if _run(program[i], variables, functions, debug, depth+1) else 0
                if not result: # found an operand that is False.  result of AND is False.  Skip rest.
                    break
        elif program[0] in ["or"]: # with lazy evaluation of operands : only evaluate when needed
            result = 0 # at least one operand must be True
            for i in range(1, len(program)):
                result = 1 if _run(program[i], variables, functions, debug, depth+1) else 0
                if result: # found an operand that is nonzero.  result of OR is THIS NON-ZERO value.  skip rest
                    break
        elif program[0] in ["not"]:
            result = 0
            if len(program) > 1:
                x = _run(program[1], variables, functions, debug, depth+1)
                result = 0 if x else 1
        elif program[0] in ["first"]:
            result = 0
            if len(program) > 1:
                x = _run(program[1], variables, functions, debug, depth+1)
                result = x[0] if type(x) == type([]) and len(x) > 0 else 0
        elif program[0] in ["rest"]:
            result = 0
            if len(program) > 1:
                x = _run(program[1], variables, functions, debug, depth+1)
                result = x[1:] if type(x) == type([]) else 0
        elif program[0] == "extend": # example (extend (1 2) (3 4))
            result = 0
            for i in range(1, len(program)):
                value = _run(program[i], variables, functions, debug, depth+1)
                if type(value) != type([]):
                    result = 0
                    break
                if i == 1:
                    result = value
                else:
                    result += value
        elif program[0] == "append":
            result = 0
            for i in range(1, len(program)):
                value = _run(program[i], variables, functions, debug, depth+1)
                if i == 1:
                    if type(value) != type([]):
                        result = 0
                        break
                    result = value
                else:
                    result.append(value)
        elif program[0] == "cons":
            result = 0
            for i in range(1, len(program)):
                value = _run(program[i], variables, functions, debug, depth+1)
                if i == 1:
                    result = [value]
                else:
                    if type(value) != type([]):
                        result = 0
                        break
                    result.extend(value)
        elif program[0] == "len": # example (len x)
            result = 0
            if len(program) > 1:
                x = _run(program[1], variables, functions, debug, depth+1)
                if type(x) == type([]):
                    result = len(x)
        elif program[0] in ["at", "at1", "at2", "at3"]: # example (at x i j)
            result = 0
            if len(program) > 2:
                x = _run(program[1], variables, functions, debug, depth+1)
                for dim in range(len(program) - 2):
                    if type(x) != type([]): # x gets reassigned below "x = x[index]", but check the result type here
                        result = 0
                        break
                    index = _run(program[2 + dim], variables, functions, debug, depth+1)
                    if type(index) != type(1):
                        result = 0
                        break
                    if index >= len(x) or index < 0:
                        result = 0
                        break
                    x = x[index]
                    result = x
        elif program[0] in ["list", "list1", "list2", "list3"]: # example (list (at board 0 0) (at board 1 1)) --> (x y)
            result = []
            for p in program[1:]:
                result.append(_run(p, variables, functions, debug, depth+1))
        elif program[0] in ["last", "last1", "last2", "last3"]: # example (last (for i 1 i) (for i 3 i)) --> (0 1 2)
            result = 0
            for p in program[1:]:
                result = _run(p, variables, functions, debug, depth+1)
        elif program[0] == "var": # syntax : "var" local_variable expression code_using_local_variable
            # [for id [value] id] --> [value]
            # [var id value id] --> value
            if len(program) < 4:
                return 0
            local_variable = program[1]
            if type(local_variable) == type(""):
                expr = _run(program[2], variables, functions, debug, depth+1)
                if local_variable in variables:
                    old_value = variables[local_variable]
                else:
                    old_value = None
                variables[local_variable] = expr
                result = _run(program[3], variables, functions, debug, depth+1)
                if old_value is not None:
                    variables[local_variable] = old_value
                else:
                    del variables[local_variable]
            else:
                if False: # set to True if expr may have side effects
                    expr = _run(program[2], variables, functions, debug, depth+1)
                result = _run(program[3], variables, functions, debug, depth+1)
        elif program[0] == "assign":
            result = 0
            if len(program) == 3:
                local_variable = program[1]
                if type(local_variable) == type(""):
                    expr = _run(program[2], variables, functions, debug, depth+1)
                    check_depth(expr, 0)
                    variables[local_variable] = copy.deepcopy(expr)
                    result = expr
        elif program[0] == "function":
            result = 0
            if len(program) >= 4:
                function_name = program[1]
                if type(function_name) == type(""):
                    params = program[2]
                    code = program[3]
                    result = (params, code)
                    functions[function_name] = result
        elif type(program[0]) == type("") and program[0] in functions:
            result = call_function(program, variables, functions, debug, depth)
        elif program[0] in ["if", "if_then_else"]: # example (if cond x)), with lazy evaluation
            result = 0
            if len(program) >= 3:
                condition = _run(program[1], variables, functions, debug, depth+1)
                if condition:
                    result = _run(program[2], variables, functions, debug, depth+1)
                elif len(program) >= 4:
                    result = _run(program[3], variables, functions, debug, depth+1)
                else:
                    result = 0
        elif program[0] == "for": # example (for i n i))
            if len(program) < 4:
                return []
            loop_variable = program[1]
            steps = _run(program[2], variables, functions, debug, depth+1)
            if type(steps) == type(1):
                if steps > 1000:
                    raise RuntimeError("for loop max iterations exceeded")
                steps = [i for i in range(steps)]
            result = []
            if type(loop_variable) == type("") and loop_variable in variables:
                old_value = variables[loop_variable]
                variables[loop_variable] = 0 # make sure the old value cannot be accessed anymore
            else:
                old_value = None
            for i in steps:
                if type(loop_variable) == type(""):
                    variables[loop_variable] = i
                result.append(_run(program[3], variables, functions, debug, depth+1))
            if type(loop_variable) == type(""):
                if old_value is not None:
                    variables[loop_variable] = old_value
                else:
                    variables[loop_variable] = 0 # identical effect to value of unknown variable
        elif program[0] == "print":
            result = 0
            for p in program[1:]:
                result = _run(p, variables, functions, debug, depth+1)
                print(str(p), "=", str(result))
        elif program[0] == "assert": # example (assert 1))
            result = 0
            if len(program) > 1:
                value = _run(program[1], variables, functions, debug, depth+1)
                if not value:
                    raise RuntimeError(f"assertion failed : {str(program)}")
                result = 1
        elif program[0] == "exit":
            exit()
        elif program[0] in ["sum"]: # example (sum (1 2 3)) --> 6
            result = 0
            if len(program) > 1:
                values = _run(program[1], variables, functions, debug, depth+1)
                if type(values) == type([]):
                    for v in values:
                        if type(v) != type(1):
                            result = 0
                            break
                        result += v
        else:
            if type(program[0]) == type("") and program[0] not in variables and program[0] not in functions:
                print(f"Warning: list starts with non-function {str(program[0])}")
            result = []
            for p in program:
                result.append(_run(p, variables, functions, debug, depth+1))
        if result is None:
            print("WARNING: program", program, "has result None, which is unexpected")
            result = 0
    elif type(program) == type(""):
        identifyer = program
        # NOTE : the copy.deepcopy is mandatory here to prevent self-referential loops with
        # (var n (1) (extend n (n))

        result = copy.deepcopy(variables[identifyer]) if identifyer in variables else 0
    elif type(program) == type(1):
        result = program
    else:
        raise RuntimeError(f"list, identifyer or int expected instead of '{program}'")
    if debug:
        print("    "*depth, "_run   end", "program", program, "variables", variables, "functions", functions, "result", result)
    check_depth(result, 0)
    return result


# ============================================== INTERFACE ====================


def load(file_name):
    '''Returns file content as string: skips lines that start with #; all is converted to lower case.'''
    with open(file_name, "r") as f:
        program_str = ""
        for line in f:
            line_stripped = line.strip().lower().split(" ")
            for part in line_stripped:
                if len(part) > 0:
                    if part[0] == '#':
                        break
                    if len(program_str) > 0:
                        program_str += " "
                    program_str += part
    return program_str


def compile(program_str):
    '''Compiles program_str'''
    return Parser.compile(program_str)


def run(program, variables, functions, debug=False):
    '''Runs compiled program'''
    #print("run start")
    global count_runs_calls, longest_run
    count_runs_calls = 0
    try:
        t0 = time.time()
        result = _run(program, variables, functions, debug, 0)
        t1 = time.time()
        if longest_run < t1 - t0:
            longest_run = t1 - t0
            #print(convert_code_to_str(program))
            #print(longest_run, "seconds")
    except RuntimeError as e:
        if str(e) not in ["code run calls exceeded", "code depth exceeded", "for loop max iterations exceeded",
                "data depth exceeded", "data list length exceeded", "data size exceeded"]:
            print(convert_code_to_str(program))
            print("RuntimeError", str(e))
        return 0
    except MemoryError as e:
        print(convert_code_to_str(program))
        print("MemoryError", str(e))
        return 0
    #print("run end")
    return result


def get_functions(file_name):
    variables = dict()
    functions = dict()
    run(compile(load(file_name)), variables, functions)
    return functions


def bind_params(formal_params, actual_params):
    while len(actual_params) < len(formal_params):
        actual_params = actual_params + [0] # don't use actual_params.append(0) : that alters actual_params
    actual_params = actual_params[:len(formal_params)]
    return {name:value for name, value in zip(formal_params, actual_params)}


def call_function(function_call, variables, functions, debug, depth):
    function_name = function_call[0]
    formal_params, code = functions[function_name]
    actual_params = [_run(param, variables, functions, debug, depth+1) for param in function_call[1:]]
    new_scope = bind_params(formal_params, actual_params)
    result = _run(code, new_scope, functions, debug, depth+1)
    return result


def get_build_in_functions():
    return [
        "len", "sum", "not", "first", "rest", # arity 1
        "eq", "ne", # arity 2
        "add", "sub", "mul", "div", "lt", "le", "ge", "gt", # arity 2, numeric
        "and", "or", # arity 2
        "extend", "append", "cons",
        "if", # arity 2
        "if_then_else", # arity 3
        "for", # artity 3, but 1st operand must be a variable name
        "var", # artity 3, but 1st operand must be a variable name

        "list1", # arity 1
        "list2", "last2", "at2", # arity 2
        # "list3", "last3", "at3", # arity 3
        ]


def get_build_in_function_param_types(fname):
    # return a list with type-indication of the params of the build-in function.
    # type-indications : 1=numeric; "*"=zero or more numeric; "?"=0 or 1 numeric; "v"=variable; []=list
    arity_dict = {
        "len":(1,), "sum":([],), "not":(1,), "first":([],), "rest":(1,), # arity 1
        "eq":(1,1), "ne":(1,1), # arity 2
        "div":(1,1), "lt":(1,1), "le":(1,1), "ge":(1,1), "gt":(1,1), "sub":(1,1), "mul":(1,1), "add":(1,1), # arity 2, numeric
        "and":(1,1), "or":(1,1), # arity 2
        "extend":([],[]),
        "append":([],1),
        "cons":(1,[]),
        "if":(1,1), # arity 2
        "if_then_else":(1,1,1), # arity 3
        "for":("v",1,1), # artity 3, but 1st operand must be a variable name
        "var":("v",1,1), # artity 3, but 1st operand must be a variable name

        "list1":(1,), # arity 1
        "list2":(1,1,), "last2":(1,1,), "at2":(1,1,), # arity 2
        # "list3":(1,1,1,), "last3":(1,1,1,), "at3":(1,1,1,), # arity 3
        }
    return arity_dict[fname]


def is_pure_numeric(fname):
    return fname in ["add", "sub", "mul", "div", "lt", "le", "ge", "gt", ]


def convert_code_to_str(code):
    if type(code) == type([]):
        result = "(" + " ".join([convert_code_to_str(item) for item in code]) + ")"
    else:
        result = str(code)
        if result in ["list1", "list2", "list3"]:
            result = "list"
        if result in ["last1", "last2", "last3"]:
            result = "last"
        if result in ["at2", "at3"]:
            result = "at"
    return result


def convert_code_to_deap_str(code, toolbox):
    if type(code) == type([]):
        fname = convert_code_to_deap_str(code[0], toolbox)
        if fname in ["list", "last", "at"]:
            arity = len(code) - 1
            fname += str(arity)
        result = fname + "(" + ", ".join([convert_code_to_deap_str(item, toolbox) for item in code[1:]]) + ")"
    else:
        result = str(code)
    return result


def add_function(function, functions, write_functions_to_file=None, mode="a"):
    keyword, fname, params, code = function
    if keyword != "function":
        raise RuntimeError(f"interpret.add_function : keyword 'function' expected")
    if type(fname) != type(""):
        raise RuntimeError(f"interpret.add_function : fname expected")
    functions[fname] = [params, code]
    if write_functions_to_file is not None:
        with open(write_functions_to_file, mode) as f:
            params = convert_code_to_str(params)
            code = convert_code_to_str(code)
            f.write(f"#    (function {fname} {params} {code})\n")


def write_functions(functions, mode="w"):
    with open(write_functions_to_file, mode) as f:
        for function in functions:
            keyword, fname, params, code = function
            params = convert_code_to_str(params)
            code = convert_code_to_str(code)
            f.write(f"#    (function {fname} {params} {code})\n")


def compile_deap(program_str, functions):
    '''Compiles DEAP program_str into an LISP program'''
    program = Parser.compile_deap(program_str, functions)
    return program


default_test_file = "experimenten/test_interpret.txt"
def self_test(self_test_file=default_test_file):
    result = run(compile(load(self_test_file)), variables={}, functions={}, debug=False)
    result = "OK" if result == 1 else "FAILED"
    print("interpreter self_test", result)


if __name__ == "__main__":
    file_name = sys.argv[1] if len(sys.argv) >= 2 else default_test_file
    self_test(file_name)
