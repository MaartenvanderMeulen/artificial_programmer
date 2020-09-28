'''Evaluate the error of the actual data with respect to the expected data'''
import sys
import numpy as np
import interpret
import time


# used in dynamic weight adjustment
global sum_errors, weights
weights = np.ones((6)) / 6
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
    
    
def evaluate_int(actual, expect, debug=False):
    '''compute and return error on this output'''
    errors = []
    assert type(expect) == type(1)
    
    # error : type difference
    error = 0.0
    if type(actual) != type(1):
        error = 1.0
    errors.append(error)
    # error : length
    error = 0.0
    if type(actual) != type(1):
        actual_numbers = extract_numbers(actual)  
        if len(actual_numbers) == 0:
            error += 1.0
            actual = expect + 1000
        else:
            actual = list(actual_numbers)[0]
    errors.append(error)
    # error : hoever zitten de expect getallen van de model getallen af
    error = (actual - expect) ** 2
    errors.append(error)
    return errors


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


def find_col(board, v):
    for row in board:
        for col, v_col in enumerate(row):
            if v_col == v:
                return col
    return None
    
    
def eval_board_col_diag_common(input, actual, extra_function_params, expect, expect_cols):
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
    
    # error : aantal kolommen
    col_values = np.zeros((n), dtype="int")
    for actual_value in actual:
        col = find_col(board, actual_value)
        if col is not None:
            col_values[col] = 1
    actual_cols = np.sum(col_values)
    error_columns = (actual_cols - expect_cols) ** 2
    
    # error :zijn het de juiste elementen uit de rows
    error_correct_elements_ordered = 0.0
    for row in range(n):
        if row >= len(actual):
            error_correct_elements_ordered += 3.0
        elif actual[row] != expect[row]:
            error_correct_elements_ordered += 1.0
    error_correct_elements_ordered = error_correct_elements_ordered ** 2
            
    return [error_type, error_len, error_fromrows, error_fromrows_ordered, error_columns,
            error_correct_elements_ordered]
    
    
def eval_board_col(input, actual, extra_function_params):
    board, col = input
    n = len(board)
    expect = [row[col] for row in board]    
    return eval_board_col_diag_common(input, actual, extra_function_params, expect, 1)
    
    
def eval_board_diag1(input, actual, extra_function_params):
    board = input[0]
    n = len(board)
    expect = [row[i] for i, row in enumerate(board)]
    return eval_board_col_diag_common(input, actual, extra_function_params, expect, n)
    
    
def eval_board_diag2(input, actual, extra_function_params):
    board = input[0]
    n = len(board)
    expect = [row[n-1 - i] for i, row in enumerate(board)]    
    return eval_board_col_diag_common(input, actual, extra_function_params, expect, n)


def eval_magic_square_sums(input, actual, extra_function_params):
    board = input[0]
    n = len(board)
    expect = []
    expect += [sum(row) for row in board]    
    expect += [sum([row[col] for row in board]) for col in range(n)]    
    expect.append(sum([board[i][i] for i in range(n)]))
    expect.append(sum([board[i][(n-1) - i] for i in range(n)]))
    return evaluate_list_of_ints(actual, expect, False)
    
    
def eval_get_row_sums(input, actual, extra_function_params):
    board = input[0]
    n = len(board)
    expect = []
    expect += [sum(row) for row in board]    
    return evaluate_list_of_ints(actual, expect, False)
    
    
def eval_get_col_sums(input, actual, extra_function_params):
    board = input[0]
    n = len(board)
    expect = []
    expect += [sum([row[col] for row in board]) for col in range(n)]    
    return evaluate_list_of_ints(actual, expect, False)
    
    
def eval_get_diag_sums(input, actual, extra_function_params):
    board = input[0]
    n = len(board)
    expect = []
    expect.append(sum([board[i][i] for i in range(n)]))
    expect.append(sum([board[i][(n-1) - i] for i in range(n)]))
    return evaluate_list_of_ints(actual, expect, False)
    
    
def eval_get_magic_number_n(input, actual, extra_function_params):
    n = input[0]
    assert type(n) == type(1)
    expect = (n * (n * n + 1)) // 2
    if 0 <= n <= 5:
        assert expect == [0, 1, 5, 15, 34, 65][n]
    return evaluate_int(actual, expect, False)
    
    
def eval_get_magic_number(input, actual, extra_function_params):
    board = input[0]
    assert type(board) == type([])
    n = len(board)
    expect = (n * (n * n + 1)) // 2
    if 0 <= n <= 5:
        assert expect == [0, 1, 5, 15, 34, 65][n]
    return evaluate_int(actual, expect, False)
    
    
def eval_are_all_equal_to(input, actual, extra_function_params):
    values = input[0]
    x = input[1]
    expect = sum([1 if value == x else 0 for value in values]) == len(values)
    return evaluate_int(actual, int(expect), False)

    
def eval_is_magic_square(input, actual, extra_function_params):
    board = input[0]
    n = len(board)
    magic_number = (n * (n * n + 1)) // 2
    sums = []
    for row in board:
        sums.append(sum(row))
    for i in range(n):
        sums.append(sum([row[i] for row in board]))
    sums.append(sum([board[i][i] for i in range(len(board))]))
    sums.append(sum([board[i][(n-1) - i] for i in range(len(board))]))
    expect = sum([1 if value == magic_number else 0 for value in sums]) == len(sums)
    return evaluate_int(actual, int(expect), False)

    
# ============================================== INTERFACE ====================


def evaluate_code(actual_code_str, expected_code_str):
    n_eq = 0.0
    for i in range(min(len(actual_code_str), len(expected_code_str))):
        if actual_code_str[i] != expected_code_str[i]:
            break
        n_eq += 1.0
    error = len(expected_code_str) - n_eq
    if error == 0 and (len(actual_code_str) > len(expected_code_str)):
        error += (len(actual_code_str) - len(expected_code_str)) / 10000.0
    # print("evaluate_code", actual_code_str, expected_code_str, error)
    return error


def evaluate(input, actual_output, evaluation_functions, debug):
    errors = []
    for function_name, extra_function_params in evaluation_functions:
        f = eval(function_name)
        errors.extend(f(input, actual_output, extra_function_params))
    errors = np.array(errors)
    global sum_errors, weights
    if len(weights) != len(errors):
        weights = np.ones((len(errors))) / len(errors)
    weighted_errors = errors * weights
    if len(sum_errors) != len(errors):
        sum_errors = np.zeros_like(errors)
    sum_errors += errors
    if debug:
        print("    evaluate")
        print("      input", input)
        print("      actual_output", actual_output)
        print("      individual errors", errors)
        print("      evaluation", np.sum(weighted_errors))
        #print("      sums errors this hop", sum_errors)
        #print("      sums weighted errors this hop", sum_errors * weights)
    return np.sum(weighted_errors)    
    
    
def init_dynamic_error_weight_adjustment():
    global sum_errors, weights
    n = 6
    sum_errors = np.zeros((n))
    weights = np.ones((n)) / n


def dynamic_error_weight_adjustment(debug=True):
    return

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
    for input, actual_output, evaluation_function, debug in tests:
        evaluate(input, actual_output, evaluation_function, debug)
