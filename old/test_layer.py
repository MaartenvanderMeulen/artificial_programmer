import sys
import interpret
import solve_problems


def compute_solved_all(input_chunks, functions):
    for example_inputs, evaluation_functions in input_chunks:
        for fname, (_, _) in functions.items():
            for evaluation_function in evaluation_functions:
                if solve_problems.is_solved_by_function(example_inputs, evaluation_function, fname, functions, None, 0):                    
                    print(f"    {fname} is evaluated OK by {evaluation_function}")


def main(functions_file_name, inputs_file_name):
    functions = interpret.get_functions(functions_file_name)
    input_chunks = interpret.compile(interpret.load(inputs_file_name))
    compute_solved_all(input_chunks, functions)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        exit(f"Usage: python {sys.argv[0]} functions_file_name inputs_file_name")
    main(sys.argv[1], sys.argv[2])
