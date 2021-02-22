import os
import sys
import time
import random
import json
import numpy as np

import interpret
import evaluate
import find_new_function


def is_solved_by_function(example_inputs, error_function, fname, functions, log_file, verbose):
    actual_outputs = []
    if fname in functions:         
        formal_params = functions[fname]
        arity = len(formal_params)
    else:
        arity = len(interpret.get_build_in_function_param_types(fname))    
    used_example_inputs = []
    for input in example_inputs:
        if len(input) == arity:
            used_example_inputs.append(input)
            code = [fname] + input
            variables = dict()
            actual_output = interpret.run(code, variables, functions)
            actual_outputs.append(actual_output)
    if len(used_example_inputs) == 0:
        return False
    raw_error_matrix = evaluate.compute_raw_error_matrix(used_example_inputs, actual_outputs, error_function, log_file, verbose, False)
    return np.sum(raw_error_matrix) <= 0.0


def solve_by_existing_function(problem, functions, log_file, verbose):
    problem_label, _, example_inputs, error_function, _, _ = problem
    if verbose >= 3:
        log_file.write(f"solve_by_existing_function {problem_label}\n")
    build_in_functions = interpret.get_build_in_functions()
    for fname in build_in_functions:
        layer0_no_functions = dict()
        if is_solved_by_function(example_inputs, error_function, fname, layer0_no_functions, log_file, verbose):
            return fname
    for fname, (_, _) in functions.items():
        if is_solved_by_function(example_inputs, error_function, fname, functions, log_file, verbose):
            return fname
    if verbose >= 3:
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
            if verbose >= 3:
                log_file.write(f"Solving problems, layer {current_layer} ...\n")
        elif problem_layer < current_layer:
            raise RuntimeError("Problems must be specified with nondecreasing layer number")
        function_str = solve_by_existing_function(problem, functions, log_file, verbose)
        if function_str:
            if verbose >= 3:
                log_file.write(f"problem  {problem_label} is solved by existing function {function_str}\n")
            pass
        else:
            if verbose >= 3:
                log_file.write(f"problem  {problem_label} ...\n")
            function_code = find_new_function.solve_by_new_function(problem, functions, log_file, params)
            if function_code:
                function_str = interpret.convert_code_to_str(function_code)
                new_functions.append(function_code)
            else:
                return False
    return True


def main(seed, id):
    param_file = f"experimenten/params_{id}.txt" 
    if not os.path.exists(param_file):
        exit(f"param file {param_file} does not exist")
    output_folder = f"tmp/{id}"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    with open(param_file, "r") as f:
        params = json.load(f)
    seed += params["seed_prefix"]
    skip_seeds = params["skip_seeds"]
    if seed in skip_seeds:
        exit()
    os.system(f"rm -f {output_folder}/end_{seed}.txt")
    log_file = f"{output_folder}/log_{seed}.txt" 
    if params.get("do_not_overwrite_logfile", False):
        if os.path.exists(log_file):
            exit(0)
    params["param_file"] = param_file
    params["id"] = id
    params["output_folder"] = output_folder
    params["seed"] = seed

    if False:
        with open(f"{output_folder}/params.txt", "w") as f:
            # write a copy to the output folder
            json.dump(params, f, sort_keys=True, indent=4)

    if params["use_one_random_seed"]:
        random.seed(seed)
    else:
        del params["seed"]
        params["seed2"] = seed
        params["random_seed"] = params["seed_prefix"]
        params["id_seed"] = seed
        random.seed(params["seed_prefix"])
    with open(f"{output_folder}/log_{seed}.txt", "w") as log_file:
        if hasattr(log_file, "reconfigure"):
            log_file.reconfigure(line_buffering=True)
        functions_file_name = params["functions_file"]
        problems_file_name = params["problems_file"]
        functions = interpret.get_functions(functions_file_name)
        problems = interpret.compile(interpret.load(problems_file_name))        
        solved_all = solve_problems(problems, functions, log_file, params, append_functions_to_file=None)
        log_file.write("done\n")
        if params["touch_at_end"]:
            os.system(f"touch {output_folder}/end_{seed}.txt")
        return 0 if solved_all else 1


if __name__ == "__main__":
    if len(sys.argv) != 3:
        exit(f"Usage: python search.py seed param_id")
    seed = int(sys.argv[1])
    id = sys.argv[2]
    exit(main(seed, id))
