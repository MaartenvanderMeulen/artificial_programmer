'''Evaluate the error of the actual data with respect to the expected data'''
import sys
import numpy as np
import interpret
import time


# used in dynamic weight adjustment
global sum_errors, weights
weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2]) # [0.80, 0.17, 0.01, 0.01, 0.01])
sum_errors = np.zeros_like(weights)


def extract_numbers(values):
    if type(values) == type([]):
        result = set()
        for item in values:
            item_set = extract_numbers(item)
            result.update(item_set)
    else:
        assert type(values) == type(1)
        result = set([values])
    return result

    
def _distance_with_closest_numbers(x, values):
    '''distance of x with nearest'''
    assert type(x) == type(1)
    if x > 1000000:
        x = 1000000
    if len(values) > 0:
        result = 1000000
        for value in values:
            assert type(value) == type(1)
            if value > 1000000:
                value = 1000000
            if result > abs(x - value):
               result = abs(x - value)
    else:
        result = abs(x - 0)
    assert result <= 1000000
    #if result > 1000:
    #    print(result, x, values)
    return result
    
    
def evaluate_list_of_ints(actual, expect, debug=False):
    '''compute and return error on this output'''
    errors = []
    assert type(expect) == type([])
    for item in expect:
        assert type(item) == type(1)
    
    # error : type difference
    error = 0.0
    if type(actual) != type([]):
        error = 1.0
    errors.append(error)
    if type(actual) == type(1):
        actual = [actual]
    k = len(expect)
    if k == 0:
        raise RuntimeError("TODO: handle case were expect output is empty")
    # error : aantal outputs
    n = 2 if len(actual) < len(expect) else 1.2
    errors.append(abs(len(actual) - len(expect)) ** n)
    # error : hoever zitten de expect getallen van de model getallen af
    actual_numbers = extract_numbers(actual)
    expected_numbers = extract_numbers(expect)
    error = 0.0
    for expected_number in expected_numbers:
        error += _distance_with_closest_numbers(expected_number, actual_numbers) ** 1.5
    errors.append(error)
    # error : hoever zitten de model getallen van de expect getallen af
    error = 0.0
    for actual_number in actual_numbers:
        error += _distance_with_closest_numbers(actual_number, expected_numbers) ** 1.5
    errors.append(error)
    # error : absolute verschil van de outputs met de gewenste output
    error = 0.0
    for i in range(len(expect)):
        assert type(expect[i]) == type(1)
        if i < len(actual) and type(actual[i]) == type(1):
            # print(actual[i], expect[i])            
            error += abs(actual[i] - expect[i]) ** 1.5
        else:
            error += abs(expect[i]) ** 1.5
    errors.append(error)
    return errors


def eval_board_col_diag_common(input, actual, extra_function_params, expect):
    board = input[0]
    n = len(board)
    
    # error : type difference
    if type(actual) != type(expect):
        error_type = 1.0
    else:
        error_type = 0.0
    if type(actual) == type(1):
        actual = [actual]

    # error : aantal outputs
    error_len = 0.0
    if len(actual) < len(expect):
        error_len = (len(expect) - len(actual)) ** 2.0
    elif len(actual) > len(expect):
        error_len = 0.1 * (len(actual) - len(expect)) ** 1.2
    else:
        error_len = 0.0

    # error : zit er uit elke rij wat in
    error_fromrows = 0.0
    actual_set = set([elem for elem in actual if type(elem) == type(1)])
    for row in range(n):
        row_set = set(board[row])
        if len(actual_set.intersection(row_set)) == 0:
            error_fromrows += 1
    error_fromrows = error_fromrows ** 2
    
    # error : is actual[row] een element van board[row]
    error_fromrows_ordered = 0.0
    for row in range(n):
        if row >= len(actual):
            error_fromrows_ordered += 3.0
        elif actual[row] not in board[row]:
            error_fromrows_ordered += 1.0
    error_fromrows_ordered = error_fromrows_ordered ** 2
    
    # error :zijn het de juiste elementen uit de rows
    error_correct_elements_ordered = 0.0
    for row in range(n):
        if row >= len(actual):
            error_correct_elements_ordered += 3.0
        elif actual[row] != expect[row]:
            error_correct_elements_ordered += 1.0
    error_correct_elements_ordered = error_correct_elements_ordered ** 2
            
    return [error_type, error_len, error_fromrows, error_fromrows_ordered, error_correct_elements_ordered]
    
    
def eval_board_col(input, actual, extra_function_params):
    board, col = input
    n = len(board)
    expect = [row[col] for row in board]    
    return eval_board_col_diag_common(input, actual, extra_function_params, expect)
    
    
def eval_board_diag1(input, actual, extra_function_params):
    board = input[0]
    n = len(board)
    expect = [row[i] for i, row in enumerate(board)]
    return eval_board_col_diag_common(input, actual, extra_function_params, expect)
    
    
def eval_board_diag2(input, actual, extra_function_params):
    board = input[0]
    n = len(board)
    expect = [row[n-1 - i] for i, row in enumerate(board)]    
    return eval_board_col_diag_common(input, actual, extra_function_params, expect)


def eval_sums_rows_cols_diags(input, actual, extra_function_params):
    board = input[0]
    n = len(board)
    expect = [sum(row) for row in board]    
    expect += [sum([row[col] for row in board]) for col in range(n)]    
    expect.append(sum([board[i][i] for i in range(n)]))
    expect.append(sum([board[i][(n-1) - i] for i in range(n)]))
    return evaluate_list_of_ints(actual, expect, False)
    
    
# ============================================== INTERFACE ====================


def evaluate(input, actual_output, evaluation_functions, debug):
    errors = []
    for function_name, extra_function_params in evaluation_functions:
        f = eval(function_name)
        errors.extend(f(input, actual_output, extra_function_params))
    errors = np.array(errors)
    global sum_errors, weights
    if len(weights) != len(errors):
        weights = np.ones((len(errors))) / len(errors)
        print("DEBUG 121 : initializing weights")
    weighted_errors = errors * weights
    if len(sum_errors) != len(errors):
        sum_errors = np.zeros_like(errors)
        print("DEBUG 125 : initializing sum weights")
    sum_errors += errors
    if debug:
        print("evaluate")
        print("    input", input)
        print("    actual_output", actual_output)
        print("    individual errors", errors)
        print("    sum weighted error this sample", np.sum(weighted_errors))
        print("    sums errors this hop", sum_errors)
        print("    sums weighted errors this hop", sum_errors * weights)
    return np.sum(weighted_errors)    
    

def dynamic_error_weight_adjustment(debug=True):
    global sum_errors, weights
    n = len(weights)
    if debug:
        print("weights before", weights, sum_errors * weights)
    average_weighted_error = np.sum(sum_errors) / n
    for i in range(n):
        if sum_errors[i] > 0:
            weights[i] = average_weighted_error / sum_errors[i]
        else:
            weights[i] = 1.0 / n
    weights /= np.sum(weights)
    if debug:
        print("weights after", weights, sum_errors * weights)
    sum_errors.fill(0)


if __name__ == "__main__":
    file_name = sys.argv[1] if len(sys.argv) >= 2 else "test_evaluate.txt"
    tests = interpret.compile(interpret.load(file_name))
    for actual, expect in tests:
        print(evaluate(actual, expect, debug=True))
    print(sum_errors)
