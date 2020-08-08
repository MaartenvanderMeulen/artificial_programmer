'''Search for new functions, like deap, but coded by Maarten'''
import interpret
import evaluate
import random


def create_random_function_name(functions):
    i = 0
    while f"f{i}" in functions:
        i += 1
    return f"f{i}"


def create_random_terminal(variables, params, locals):
    if random.random() < 0.3: # in 30% of the cases, return a number
        # most numbers are 1
        if random.random() < 0.9:
            return 1
        if random.random() < 0.2:
            return 0
        return random.randint(-9, 9)
    if random.random() < 0.2: # return somtimes an empty list
        return []
    # in the remaining cases, return a random variable
    var = random.choice(variables)
    if var not in locals:
        params.add(var)
    return var


def create_random_code(variables, depth, min_depth, max_depth, params, locals):
    if depth >= max_depth or (depth >= min_depth and random.random() < 0.4):
        return create_random_terminal(variables, params, locals)
    if depth == 0:
        fname = "last"
        statements = random.randint(1, 5)
        operand_types = (1 for i in range(statements))
    else:
        fname = random.choice(interpret.get_build_in_functions())
        operand_types = interpret.get_build_in_function_param_types(fname)
    operands = []
    #print("DEBUG 60:", fname, operand_types)
    for operand_type in operand_types:
        if operand_type == 1:
            operands.append(create_random_code(variables, depth+1, min_depth, max_depth, params, locals))
        elif operand_type == "v":
            var = random.choice(variables)
            operands.append(var)
            if var not in params:
                locals.add(var)
        elif operand_type == "*":
            while random.random() < 0.2:
                operands.append(create_random_code(variables, depth+1, min_depth, max_depth, params, locals))
        elif operand_type == "?":
            if random.random() < 0.2:
                operands.append(create_random_code(variables, depth+1, min_depth, max_depth, params, locals))
    return [fname] + operands


def create_random_function(functions):
    stub = False
    if stub:
        return ["function", "add_ab1", ["a", "b"], ["add", "a", "b", 1]]
    fname = create_random_function_name(functions)
    params, locals = set(), set()
    code = create_random_code(["n", "m", "k", "p", "s", "t", "v", "w", "i", "j"], 0, 2, 5, params, locals)
    params = list(params)
    params.sort() # to get it in a reproducable order
    return ["function", fname, params, code]


def compute_error(function, problem, functions, debug):
    result = 0.0
    _, fname, formal_params, code = function
    problem_label, examples = problem
    for actual_params, expected_output in examples:
        if debug:
            print("DEBUG 100:", formal_params, actual_params, expected_output)
        variables = interpret.bind_params(formal_params, actual_params)
        if debug:
            print("DEBUG 103:", variables)
        actual_output = interpret.run(code, variables, functions)
        result += evaluate.evaluate(actual_output, expected_output, False)
        if debug:
            print("DEBUG 105:", formal_params, actual_params, expected_output, actual_output)
    if debug:
        print("DEBUG 107:", result)
    return result


def room_for_improvement(error):
    return error[-1] > 0 and (len(error) == 1 or error[-2] > error[-1])


def local_search(function, problem, functions, initial_error, debug, hop):
    # TODO : do the local search
    improved_function = function

    final_error = compute_error(improved_function, problem, functions, debug)
    if final_error > initial_error:
        print("DEBUG 132", final_error, initial_error)
    assert final_error <= initial_error
    return final_error


def solve_by_new_function(problem, functions):
    best_error = 1e9
    for hop in range(100000):
        function = create_random_function(functions)
        debug = False # hop in [7]
        if debug:
            print("DEBUG 136:", interpret.convert_code_to_str(function))
        hop_error = [compute_error(function, problem, functions, debug)]
        while room_for_improvement(hop_error):
            hop_error.append(local_search(function, problem, functions, hop_error[-1], debug, hop))
        if debug:
            print(problem[0], "hop", hop, "error", hop_error[-1])        
        if best_error > hop_error[-1]:
            best_error = hop_error[-1]
            print(problem[0], "hop", hop, "best_error", best_error)
        if hop_error[-1] == 0:
            return function
    return None
