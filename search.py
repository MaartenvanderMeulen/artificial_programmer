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


def is_solved_by_function(example_inputs, evaluation_function, fname, functions, log_file, verbose):
    actual_outputs = []
    for input in example_inputs:
        # print("fname", fname, type(fname), "input", input, type(input))
        code = [fname] + input
        variables = dict()
        actual_output = interpret.run(code, variables, functions)
        actual_outputs.append(actual_output)
    error, _ = evaluate.evaluate_all(example_inputs, actual_outputs, evaluation_function, log_file, verbose)
    if verbose >= 2:
        log_file.write(f"is_solved_by_function({fname}), actual_outputs {actual_outputs}, error {error}\n")
    return error <= 0.0


def solve_by_existing_function(problem, functions, log_file, verbose):
    problem_label, _, example_inputs, evaluation_function, _, _ = problem
    if verbose >= 2:
        log_file.write(f"solve_by_existing_function {problem_label}\n")
    build_in_functions = interpret.get_build_in_functions()
    for fname in build_in_functions:
        layer0_no_functions = dict()
        if is_solved_by_function(example_inputs, evaluation_function, fname, layer0_no_functions, log_file, verbose):
            return fname
    for fname, (_, _) in functions.items():
        if is_solved_by_function(example_inputs, evaluation_function, fname, functions, log_file, verbose):
            return fname
    if verbose >= 2:
        log_file.write(f"solve_by_existing_function {problem_label} fails\n")
    return None


def solve_problems(problems, functions, log_file, params, append_functions_to_file=None):
    '''If append_functions_to_file is a string, the new functions will be appended to that file'''
    verbose = params["verbose"]
    new_functions = []
    current_layer = -1
    for problem in problems:
        problem_label = problem[0]
        problem_layer = problem[-1]
        if problem_layer > current_layer:
            for function_code in new_functions:
                interpret.add_function(function_code, functions, append_functions_to_file)
            new_functions = []
            current_layer = problem_layer
            if verbose >= 1:
                log_file.write(f"Solving problems, layer {current_layer} ...\n")
        elif problem_layer < current_layer:
            raise RuntimeError("Problems must be specified with nondecreasing layer number")
        function_str = solve_by_existing_function(problem, functions, log_file, verbose)
        if function_str:
            if verbose >= 1:
                log_file.write(f"problem  {problem_label} is solved by existing function {function_str}\n")
            pass
        else:
            if verbose >= 1:
                log_file.write(f"problem  {problem_label} ...\n")
            function_code = ga_search_deap.solve_by_new_function(problem, functions, log_file, params)
            if function_code:
                function_str = interpret.convert_code_to_str(function_code)
                new_functions.append(function_code)
            else:
                return False
    return True


def main(seed, param_file):
    if param_file[:len("experimenten/params_")] != "experimenten/params_" or param_file[-len(".txt"):] != ".txt":
        exit("param file must have format 'experimenten/params_id.txt'")
    id = param_file[len("experimenten/params_"):-len(".txt")]    
    output_folder = f"tmp/{id}"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    with open(param_file, "r") as f:
        params = json.load(f)
    seed += params["seed_prefix"]
    log_file = f"{output_folder}/log_{seed}.txt" 
    if params.get("do_not_overwrite_logfile", False):
        if os.path.exists(log_file):
            exit(0)

    with open(f"{output_folder}/params.txt", "w") as f:
        # write a copy to the output folder
        json.dump(params, f, sort_keys=True, indent=4)

    random.seed(seed)
    np.random.seed(seed)
    with open(f"{output_folder}/log_{seed}.txt", "w") as log_file:
        log_file.reconfigure(line_buffering=True)
        functions_file_name = params["functions_file"]
        problems_file_name = params["problems_file"]
        functions = interpret.get_functions(functions_file_name)
        problems = interpret.compile(interpret.load(problems_file_name))
        solved_all = solve_problems(problems, functions, log_file, params, append_functions_to_file=None)
        return 0 if solved_all else 1


if __name__ == "__main__":
    if len(sys.argv) != 3:
        exit(f"Usage: python search.py seed paramsfile")
    seed = int(sys.argv[1])
    param_file = sys.argv[2]
    exit(main(seed, param_file))
