'''Interpreter for LISP like programming language'''
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
        Parser._tokens = Parser._tokenize(program_str)
        Parser._first_token()
        program = Parser._parse_deap_element(functions)
        Parser._expect_token(Parser._end_of_program)
        return program
    


# ====================== interpreter ======================================


# TODO: put this in a Run object
global count_runs_calls
count_runs_calls = 0
longest_run = 0
    
def _run(program, variables, functions, debug, depth):
    if depth > 100:
        print("code depth exceeded")
        raise RuntimeError("code depth exceeded")
    global count_runs_calls
    count_runs_calls += 1
    if count_runs_calls > 10000:
        print("code run calls exceeded")
        raise RuntimeError("code run calls exceeded")    
    if debug:
        print("    "*depth, "_run start", "program", program, "variables", variables, "functions", functions)
    if type(program) == type([]):
        result = None
        if len(program) == 0:
            result = []
        elif program[0] == "add": # example (add 3 2)
            result = 0
            for i in range(1, len(program)):
                value = _run(program[i], variables, functions, debug, depth+1)
                if i == 1:
                    result = value
                else:
                    if type(result) == type(value):
                        result += value
                    else:
                        result = 0
        elif program[0] == "sub": # example (sub 3 2)
            result = 0
            for i in range(1, len(program)):
                value = _run(program[i], variables, functions, debug, depth+1)
                if i == 1:
                    result = value
                else:
                    if type(result) == type(1):
                        if type(value) == type(1):
                            result -= value
                        # else ignore this value
                    else:
                        result = 0
        elif program[0] == "mul": # example (mul 1 2 3)
            result = 0
            for i in range(1, len(program)):
                value = _run(program[i], variables, functions, debug, depth+1)
                if i == 1:
                    result = value
                else:
                    if type(result) == type(1):
                        result *= value
                    else:
                        result = 0
        elif program[0] == "div": # example (div 3 2)
            a = _run(program[1], variables, functions, debug, depth+1) if len(program) > 1 else 0
            if type(a) != type(1):
                a = 0
            b = _run(program[2], variables, functions, debug, depth+1) if len(program) > 2 else 0
            if type(b) != type(1):
                b = 0
            result = a // b if b != 0 else 0
        elif program[0] in ["eq", "ne", "lt", "le", "ge", "gt", "and", "or"]:
            a = _run(program[1], variables, functions, debug, depth+1) if len(program) > 1 else 0
            b = _run(program[2], variables, functions, debug, depth+1) if len(program) > 2 else 0
            if type(a) != type(b):
                result = 0
            elif program[0] == "eq": # example (eq 3 2)
                result = 1 if a == b else 0
            elif program[0] == "ne": # example (ne 3 2)
                result = 1 if a != b else 0
            elif type(a) != type(1):
                assert type(b) != type(1)
                result = 0
            else:
                assert type(b) == type(1)
                if program[0] == "lt": # example (le 3 2)
                    result = 1 if a < b else 0
                elif program[0] == "le": # example (le 3 2)
                    result = 1 if a <= b else 0
                elif program[0] == "ge": # example (ge 3 2)
                    result = 1 if a >= b else 0
                elif program[0] == "gt": # example (gt 3 2)
                    result = 1 if a > b else 0
                elif program[0] == "and": # example (and 1 1)
                    result = 1 if a and b else 0
                elif program[0] == "or": # example (or 1 1)
                    result = 1 if a or b else 0
                else:
                    raise RuntimeError("internal error at line 156 of the APL interpreter")
        elif program[0] == "len": # example (len x)
            result = 0
            if len(program) > 1:
                x = _run(program[1], variables, functions, debug, depth+1)
                if type(x) == type([]):
                    result = len(x)
        elif program[0] == "at": # example (at x i j)
            result = 0
            if len(program) > 2:
                x = _run(program[1], variables, functions, debug, depth+1)
                if type(x) == type([]):
                    for dim in range(len(program) - 2):
                        index = _run(program[2 + dim], variables, functions, debug, depth+1)
                        if type(index) != type(1):
                            index = 0
                        if type(x) != type([]) or index >= len(x) or index < 0:
                            result = 0
                            break
                        assert index < len(x)
                        x = x[index]
                        result = x
        elif program[0] == "last": # example (last (assign n 3) (for i n i)) --> (0 1 2)
            result = 0
            for p in program[1:]:
                result = _run(p, variables, functions, debug, depth+1)
        elif program[0] == "assign": # example (assign x 5)
            if len(program) < 2:
                return 0
            variable_name = program[1]
            if type(variable_name) != type(""):
                return 0
            result = _run(program[2], variables, functions, debug, depth+1) if len(program) >= 3 else 0
            variables[variable_name] = result
        elif program[0] == "function":
            # example (function fac n (last (assign result 1) (for1 i n (assign result (mul n i)))))
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
        elif program[0] == "if": # example (if cond x y))
            result = 0
            if len(program) >= 2:
                result = _run(program[1], variables, functions, debug, depth+1)
                if len(program) >= 3:
                    condition = result
                    if condition:
                        result = _run(program[2], variables, functions, debug, depth+1)
                    elif len(program) >= 4:
                        result = _run(program[3], variables, functions, debug, depth+1)
        elif program[0] == "for": # example (for i n i))
            if len(program) < 4:
                return []
            loop_variable = program[1]
            steps = _run(program[2], variables, functions, debug, depth+1)
            if type(steps) == type(1):
                steps = [i for i in range(steps)]
            result = []
            for i in steps:
                if type(loop_variable) == type(""):
                    variables[loop_variable] = i
                result.append(_run(program[3], variables, functions, debug, depth+1))
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
        else:
            result = []
            for p in program:
                result.append(_run(p, variables, functions, debug, depth+1))
        if result is None:
            print("program", program)
            print("result", result)
        assert result is not None
    elif type(program) == type(""):
        identifyer = program
        # NOTE : the copy.deepcopy is mandatory here to prevent self-referential loops with
        # (last (assign n (1)) (add n (n)))
        result = copy.deepcopy(variables[identifyer]) if identifyer in variables else 0
    elif type(program) == type(1):
        result = program
    else:
        raise RuntimeError(f"list, identifyer or int expected instead of '{program}'")
    if debug:
        print("    "*depth, "_run   end", "program", program, "variables", variables, "functions", functions, "result", result)
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
        print(convert_code_to_str(program))
        print(str(e))
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
        "len", # arity 1
        "last", # arity 1+
        "div", "lt", "le", "eq", "ne", "ge", "gt", # arity 2
        "assign", # artity 2, but 1st operand must be a variable name
        "add", "sub", "mul", "and", "or", # arity 2+
        "at", # arity 2*
        "if", # arity 2 or 3
        "for", # artity 3, but 1st operand must be a variable name
        ]


def get_build_in_function_param_types(fname):
    # return a list with type-indication of the params of the build-in function.
    # type-indications : 1=numeric; "*"=zero or more numeric; "?"=0 or 1 numeric; "v"=variable
    arity_dict = {        
        "len":(1,), # arity 1
        "last":(1,"*"), # arity 1*
        "div":(1,1), "lt":(1,1), "le":(1,1), "eq":(1,1), "ne":(1,1), "ge":(1,1), "gt":(1,1), # arity 2
        "assign":("v",1), # artity 2, but 1st operand must be a variable name
        "add":(1,1,"*"), "sub":(1,1,"*"), "mul":(1,1,"*"), "and":(1,1,"*"), "or":(1,1,"*"), # arity 2*
        "at":(1,1,"*"), # arity 2*
        "if":(1,1,"?"), # arity 2 or 3
        "for":("v",1,1), # artity 3, but 1st operand must be a variable name
        }
    return arity_dict[fname]


def convert_code_to_str(code):
    if type(code) == type([]):
        result = "(" + " ".join([convert_code_to_str(item) for item in code]) + ")"
    else:
        result = str(code)
    return result

    
def add_function(function, functions, functions_file_name):
    # print("DEBUG 405", function)
    keyword, fname, params, code = function
    assert type(code) == type([])
    assert keyword == "function" # keyword of the function statement
    functions[fname] = [params, code]
    with open(functions_file_name, "a") as f:
        params = convert_code_to_str(params)
        code = convert_code_to_str(code)
        f.write(f"#    (function {fname} {params} {code})\n")


def compile_deap(program_str, functions):
    '''Compiles DEAP program_str into an LISP program'''
    program = Parser.compile_deap(program_str, functions)
    return program


if __name__ == "__main__":
    file_name = sys.argv[1] if len(sys.argv) >= 2 else "test_interpret.txt" 
    print(run(compile(load(file_name)), {}, {}, False))
