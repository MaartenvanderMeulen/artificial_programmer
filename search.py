'''Search for new functions'''
import sys
import interpret


def is_solved_by_function(examples, fname, functions):
    for actual_params, expected_output in examples:
        fcall = [fname] + actual_params
        actual_output = interpret.call_function(fcall, dict(), functions, False, "")
        if actual_output != expected_output:
            return False        
    return True


def solve_by_existing_function(problem, functions):
    problem_label, examples = problem
    for fname, (params, code) in functions.items():
        if is_solved_by_function(examples, fname, functions):
            return fname
    return None
   

def solve_by_new_function(problem, functions):
    # TODO
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
