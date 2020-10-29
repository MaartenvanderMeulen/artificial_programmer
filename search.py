'''Search for new functions.  Usage:
python search.py [functions.txt [problems.txt]]
'''
import sys
import time
import interpret
import evaluate
import random
import numpy as np
import ga_search_deap
import evaluate


def is_solved_by_function(example_inputs, evaluation_functions, fname, functions):
    for input in example_inputs:
        code = [fname] + input
        variables = dict()
        actual_output = interpret.run(code, variables, functions)
        if evaluate.evaluate(input, actual_output, evaluation_functions, False) > 0.0:
            return False
    return True


def solve_by_existing_function(problem, functions):
    problem_label, params, example_inputs, evaluation_functions, hints, layer = problem
    build_in_functions = interpret.get_build_in_functions()
    for fname in build_in_functions:
        layer0_no_functions = dict()
        if is_solved_by_function(example_inputs, evaluation_functions, fname, layer0_no_functions):
            return fname
    for fname, (params, code) in functions.items():
        if is_solved_by_function(example_inputs, evaluation_functions, fname, functions):
            return fname
    return None


def find_new_functions(problems, functions, layer, append_functions_to_file=None):
    '''If append_functions_to_file is a string, the new functions will be appended to that file'''
    print("Solving problems, layer", layer, "...")
    new_functions = []
    for problem in problems:
        problem_label = problem[0]
        problem_layer = problem[-1]
        if problem_layer <= layer:
            function_str = solve_by_existing_function(problem, functions)
            if function_str:
                pass
                # print("problem", problem_label, "is solved by existing function", function_str)
            else:
                print("problem", problem_label, "...")
                function_code = ga_search_deap.solve_by_new_function(problem, functions)
                if function_code:
                    function_str = interpret.convert_code_to_str(function_code)
                    #print("problem", problem_label, "is be solved by new function", function_str)
                    new_functions.append(function_code)
                else:
                    print("problem", problem_label, "cannot be solved in this layer")
        else:
            pass
            # print("problem", problem_label, "will be tried at layer", problem_layer)        
    for function_code in new_functions:
        interpret.add_function(function_code, functions, append_functions_to_file)
    return len(new_functions) > 0


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    functions_file_name = sys.argv[1] if len(sys.argv) >= 2 else "functions.txt"
    problems_file_name = sys.argv[2] if len(sys.argv) >= 3 else "problems.txt"
    functions = interpret.get_functions(functions_file_name)
    problems = interpret.compile(interpret.load(problems_file_name))
    t0 = time.time()
    max_layer = max([problem[-1] for problem in problems])
    for layer in range(1, max_layer+1):
        find_new_functions(problems, functions, layer, append_functions_to_file=None)
    t1 = time.time()
    print("total execution time", int(t1 - t0), "seconds", "total evaluations", ga_search_deap.total_eval_count)
