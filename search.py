'''Search for new functions.  Usage:
python search.py [functions.txt [problems.txt]]
'''
import os
import sys
import time
import interpret
import evaluate
import random
import numpy as np
import json
import ga_search_deap
import evaluate


def is_solved_by_function(example_inputs, evaluation_function, fname, functions):
    actual_outputs = []
    for input in example_inputs:
        code = [fname] + input
        variables = dict()
        actual_output = interpret.run(code, variables, functions)
        actual_outputs.append(actual_output)
    error, _ = evaluate.evaluate_all(example_inputs, actual_outputs, evaluation_function, None, 0)
    return error <= 0.0


def solve_by_existing_function(problem, functions):
    problem_label, params, example_inputs, evaluation_function, hints, layer = problem
    build_in_functions = interpret.get_build_in_functions()
    for fname in build_in_functions:
        layer0_no_functions = dict()
        if is_solved_by_function(example_inputs, evaluation_function, fname, layer0_no_functions):
            return fname
    for fname, (params, code) in functions.items():
        if is_solved_by_function(example_inputs, evaluation_function, fname, functions):
            return fname
    return None


def find_new_functions(problems, functions, layer, f, params, append_functions_to_file=None):
    '''If append_functions_to_file is a string, the new functions will be appended to that file'''
    verbose = params["verbose"]
    if verbose > 0:
        f.write(f"Solving problems, layer {layer} ...\n")
    new_functions = []
    for problem in problems:
        problem_label = problem[0]
        problem_layer = problem[-1]
        if problem_layer <= layer:
            function_str = solve_by_existing_function(problem, functions)
            if function_str:
                pass
            else:
                if verbose > 0:
                    f.write(f"problem  {problem_label} ...\n")
                function_code = ga_search_deap.solve_by_new_function(problem, functions, f, params)
                if function_code:
                    function_str = interpret.convert_code_to_str(function_code)
                    new_functions.append(function_code)
        else:
            pass
    for function_code in new_functions:
        interpret.add_function(function_code, functions, append_functions_to_file)
    return len(new_functions) > 0


def main(seed, param_file):
    with open(param_file, "r") as f:
        params = json.load(f)
    output_folder = params["output_folder"]
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    with open(f"{output_folder}/params.txt", "w") as f:
        # write a copy to the output folder
        json.dump(params, f, sort_keys=True, indent=4)
    seed += params["seed_prefix"]
    random.seed(seed)
    np.random.seed(seed)
    with open(f"{output_folder}/log_{seed}.txt", "w") as f:
        functions_file_name = params["functions_file"]
        problems_file_name = params["problems_file"]
        functions = interpret.get_functions(functions_file_name)
        problems = interpret.compile(interpret.load(problems_file_name))
        t0 = time.time()
        max_layer = max([problem[-1] for problem in problems])
        for layer in range(1, max_layer+1):
            find_new_functions(problems, functions, layer, f, params, append_functions_to_file=None)
        t1 = time.time()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        exit(f"Usage: python search.py seed paramsfile")
    seed = int(sys.argv[1])
    param_file = sys.argv[2]
    main(seed, param_file)
