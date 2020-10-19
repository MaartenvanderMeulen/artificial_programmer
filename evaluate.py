'''Evaluate the error of the actual data with respect to the expected data'''
import sys
import numpy as np
import interpret
# import time
import math


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
    error = (abs(float(actual) - float(expect))) ** 1.2
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
            error += (abs(actual[i] - expect[i])) ** 1.5
        else:
            error += (abs(expect[i])) ** 1.5
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
        error_len = (abs(len(expect) - len(actual))) ** 2.0
    elif len(actual) > len(expect):
        error_len = 0.1 * (abs(len(actual) - len(expect))) ** 1.2
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
    error_columns = (abs(actual_cols - expect_cols)) ** 2
    
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
    result = eval_board_col_diag_common(input, actual, extra_function_params, expect, n)
    return result


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
    assert n in [1, 2, 3, 4, 5, 6]
    expect = (n * (n * n + 1)) // 2
    if 1 <= n <= 6:
        assert expect == [0, 1, 5, 15, 34, 65, 111][n]
    result = evaluate_int(actual, expect, False)
    return result
    
    
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

    
def eval_is_magic(inputs, actual, extra_function_params):
    error = 0.0
    if type(actual) != type(1):
        error += 2.0
    if type(actual) == type(1):
        if actual not in [0, 1]:
            error += 0.1
    model_says_its_magic = bool(actual)
    board = inputs[0]
    assert len(board) == len(board[0]) and type(board[0][0]) == type(1)
    n = len(board)
    magic_number = (n * (n * n + 1)) // 2
    count_magic_rows, count_magic_cols, count_magic_diags, sum_board, sum_diag1, sum_diag2 = 0, 0, 0, 0, 0, 0
    for row in board:
        sum_board += sum(row)
    for row in board:
        if sum(row) == magic_number:
            count_magic_rows += 1
    for i in range(n):
        col = [row[i] for row in board]
        if sum(col) == magic_number:
            count_magic_cols += 1
    for i, row in enumerate(board):
        sum_diag1 += row[i]
        sum_diag2 += row[(n-1) - i]
    if sum_diag1 == magic_number:
        count_magic_diags += 1
    if sum_diag2 == magic_number:
        count_magic_diags += 1
    check_rows = bool(extra_function_params[0])
    check_cols = bool(extra_function_params[1])
    check_diags = bool(extra_function_params[2])
    if model_says_its_magic:
        if sum_board != n * magic_number:
            error += 0.1
        if check_rows:
            error += 0.425 * (n - count_magic_rows) / n
        if check_cols:
            error += 0.325 * (n - count_magic_cols) / n
        if check_diags:
            error += 0.25 * (2 - count_magic_diags) / 2
    else:
        is_magic = True
        if check_rows and count_magic_rows < n:
            is_magic = False
        if check_cols and count_magic_cols < n:
            is_magic = False
        if check_diags and count_magic_diags < 2:
            is_magic = False
        if is_magic:
            if check_rows:
                error += 0.45
            if check_cols:
                error += 0.35
            if check_diags:
                error += 0.2
    return error,

    
def eval_is_sorted(input, actual, extra_function_params):
    expect = 1
    for i in range(len(input) - 1):
        if input[i] > input[i+1]:
            expect = 0
    return evaluate_int(actual, expect, False)


def count_equal_prefix_length(str1, str2):
    n_eq = 0
    for i in range(min(len(str1), len(str2))):
        if str1[i] != str2[i]:
            break
        n_eq += 1
    return n_eq
    
    
def count_equal_and_unequal_chars(str1, str2):
    count_eq, count_ne = 0, 0
    i, j = 0, 0
    while i < len(str1) and j < len(str2):
        if str1[i] == str2[j]:
            count_eq += 1
            i += 1
            j += 1
        else:
            while j < len(str2) and str1[i] != str2[j]:
                j += 1
                count_ne += 1
    count_ne += (len(str1) - i) + (len(str2) - j)/10
    return count_eq, count_ne 
    
    
# ============================================== INTERFACE ====================


def evaluate_code(actual_code_str, expected_code_str):
    # count_eq, count_ne = count_equal_and_unequal_chars(actual_code_str, expected_code_str)
    # return count_ne + count_ne/(count_eq + 1)


    error = 0
    error += len(expected_code_str) - count_equal_prefix_length(actual_code_str, expected_code_str)
    return error
    
    actual_code_str = actual_code_str[::-1]
    expected_code_str = expected_code_str[::-1]
    error += (len(expected_code_str) - count_equal_prefix_length(actual_code_str, expected_code_str)) / 10

    actual_code_str = sorted(actual_code_str)
    expected_code_str = sorted(expected_code_str)
    error += (len(expected_code_str) - count_equal_prefix_length(actual_code_str, expected_code_str)) / 100
    
    actual_code_str = actual_code_str[::-1]
    expected_code_str = expected_code_str[::-1]
    error += (len(expected_code_str) - count_equal_prefix_length(actual_code_str, expected_code_str)) / 1000

    error += abs(len(expected_code_str) - len(actual_code_str)) / 10000.0
    # print("evaluate_code", actual_code_str, expected_code_str, error)
    return error


def evaluate(input, actual_output, evaluation_functions, debug):
    errors = []
    for function_name, extra_function_params in evaluation_functions:
        f = eval(function_name)
        y = f(input, actual_output, extra_function_params)
        errors.extend(y)
    assert len(errors) > 0
    for e in errors:
        e = float(e) # fails on complex numbers that are a result of (a - b) ** 1.5 with a-b negative.
        assert math.isfinite(e)
    errors = np.array(errors).astype(float)    
    global sum_errors, weights
    if weights.shape[0] != errors.shape[0]:
        weights = np.ones((errors.shape[0])) / errors.shape[0]
    weighted_errors = errors * weights
    if sum_errors.shape[0] != errors.shape[0]:
        sum_errors = np.zeros_like(errors)
    sum_errors += errors
    result = float(np.sum(weighted_errors))
    assert type(result) == type(1.0)
    if debug >= 2 or (debug >= 1 and result > 0.0):
        print("    evaluate")
        print("      input", input)
        print("      actual_output", actual_output)
        print("      individual errors", errors)
        print("      evaluation", np.sum(weighted_errors))
        #print("      sums errors this hop", sum_errors)
        #print("      sums weighted errors this hop", sum_errors * weights)
    return result
    
    
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
