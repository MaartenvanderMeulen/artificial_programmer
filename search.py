'''Search for new functions'''
import sys
import interpret
import evaluate


def is_solved_by_function(examples, fname, functions):
    for actual_params, expected_output in examples:
        code = [fname] + actual_params
        variables = dict()
        actual_output = interpret.run(code, variables, functions)
        if actual_output != expected_output:
            return False
    return True


def solve_by_existing_function(problem, functions):
    problem_label, examples = problem
    build_in_functions = interpret.get_build_in_functions()
    for fname in build_in_functions:
        if is_solved_by_function(examples, fname, dict()):
            return fname
    for fname, (params, code) in functions.items():
        if is_solved_by_function(examples, fname, functions):
            return fname
    return None


def create_random_function_name(functions):
    i = 0
    while f"f{i}" in functions:
        i += 1
    return f"f{i}"
    
    
def create_random_code(functions):
    # TODO
    return params, code


def create_random_function(functions):
    stub = True
    if stub:
        return ["add_ab1", ["a", "b"], ["add", "a", "b", 1]]
    fname = create_random_function_name(functions)
    params, code = create_random_code(functions)
    return [fname, params, code]
    
    
def compute_error(function, problem, functions):
    result = 0.0
    fname, formal_params, code = function
    problem_label, examples = problem
    for actual_params, expected_output in examples:
        variables = interpret.bind_params(formal_params, actual_params)
        actual_output = interpret.run(code, variables, functions)
        result += evaluate.evaluate(actual_output, expected_output, False)
    return result


def room_for_improvement(error):
    return error[-1] > 0 and (len(error) == 1 or error[-2] > error[-1])
    
    
def local_search(function, problem, functions):
    initial_error = compute_error(function, problem, functions)
    
    # TODO : do the local search
    improved_function = function
    
    final_error = compute_error(improved_function, problem, functions)
    assert final_error <= initial_error
    return final_error


def solve_by_new_function(problem, functions):
    for hop in range(10):
        function = create_random_function(functions)
        error = [compute_error(function, problem, functions)]
        while room_for_improvement(error):
            error.append(local_search(function, problem, functions))
        if error[-1] == 0:
            return function
        print(problem[0], "hop", hop, "error", error[-1])
    return None


def find_new_functions(problems, functions, functions_file_name, iteration):
    print("Find new functions, iteration", iteration, "...")
    count_new_functions = 0
    for problem in problems:
        problem_label = problem[0]
        function = solve_by_existing_function(problem, functions)
        if function:
            print(problem_label, "is solved by", function)
        else:
            function = solve_by_new_function(problem, functions)
            if function:
                print(problem_label, "can be solved by new function", function)
                interpret.add_function(function, functions, functions_file_name)
                count_new_functions += 1
            else:
                print(problem_label, "cannot be solved")
    return count_new_functions > 0


if __name__ == "__main__":
    functions_file_name = sys.argv[1] if len(sys.argv) >= 2 else "functions.txt"
    problems_file_name = sys.argv[2] if len(sys.argv) >= 3 else "problems.txt"
    functions = interpret.get_functions(functions_file_name)
    problems = interpret.compile(interpret.load(problems_file_name))
    iteration = 1
    while find_new_functions(problems, functions, functions_file_name, iteration):
        iteration += 1
