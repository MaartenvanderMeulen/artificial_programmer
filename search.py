'''Search for new functions'''
import sys
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
    problem_label, params, example_inputs, evaluation_functions, hints = problem
    build_in_functions = interpret.get_build_in_functions()
    for fname in build_in_functions:
        layer0_no_functions = dict()
        if is_solved_by_function(example_inputs, evaluation_functions, fname, layer0_no_functions):
            return fname
    for fname, (params, code) in functions.items():
        if is_solved_by_function(example_inputs, evaluation_functions, fname, functions):
            return fname
    return None


def find_new_functions(problems, functions, functions_file_name, iteration):
    print("Find new functions, iteration", iteration, "...")
    count_new_functions = 0
    for problem in problems:
        problem_label = problem[0]
        print(problem_label, "...")
        function = solve_by_existing_function(problem, functions)
        if function:
            print(problem_label, "is solved by", function)
        else:
            function = ga_search_deap.solve_by_new_function(problem, functions)
            if function:
                print(problem_label, "can be solved by new function", function)
                interpret.add_function(function, functions, functions_file_name)
                count_new_functions += 1
            else:
                print(problem_label, "cannot be solved")
    return count_new_functions > 0


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    functions_file_name = sys.argv[1] if len(sys.argv) >= 2 else "functions.txt"
    problems_file_name = sys.argv[2] if len(sys.argv) >= 3 else "problems.txt"
    functions = interpret.get_functions(functions_file_name)
    problems = interpret.compile(interpret.load(problems_file_name))
    print("DEBUG SEARCH 61 : problems ", problems)
    iteration = 1
    while find_new_functions(problems, functions, functions_file_name, iteration):
        iteration += 1
