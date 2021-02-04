'''Evaluate the error of the actual data with respect to the expected data'''
import sys
import numpy as np
import copy
import math
# import time
import interpret


# used in error normalisation
global avg_raw_error_vector, inv_avg_raw_error_vector
avg_raw_error_vector = np.ones((8))
inv_avg_raw_error_vector = np.ones((8))

global g_recursive_tuple_max_depth
g_recursive_tuple_max_depth = 0

def recursive_tuple(value, depth=0):
    global g_recursive_tuple_max_depth
    if g_recursive_tuple_max_depth < depth:
        g_recursive_tuple_max_depth = depth
    if type(value) == type(1) or type(value) == type(""):
        return value
    result = []
    for v in value:
        if type(v) == type(1):
            result.append(v)
        else:
            result.append(recursive_tuple(v, depth+1))
    result = tuple(result)
    return result


def extract_numbers_list(values):
    if type(values) == type([]):
        if len(values) == 0:
            result = [0]
        else:
            result = []
            for item in values:
                if type(item) == type(1):
                    result.append(item)
                else:
                    result.extend(extract_numbers_list(item))
    else:
        assert type(values) == type(1)
        result = [values]
    return result


def extract_numbers_set(values):
    return set(extract_numbers_list(values))


def count_empty_sublists(actual):
    if type(actual) == type(1):
        return 0
    if len(actual) == 0:
        return 1
    return sum([count_empty_sublists(v) for v in actual])


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
               if result == 0:
                   break
    else:
        result = abs(x - 0)
    assert result <= 1000000
    return result


def compute_error_int(actual, expect, debug=False):
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
        actual_numbers = extract_numbers_set(actual)
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

global g_w1, g_w2a, g_w2b, g_w3, g_w4, g_w5, g_w6, g_w7, g_w8
g_w1, g_w2a, g_w2b, g_w3, g_w4, g_w5, g_w6, g_w7, g_w8 = 1.0, 2.0, 1.1, 1.5, 1.5, 1.5, 1.0, 1.0, 1.0
def compute_error_list_of_ints(actual, expect, debug=False):
    '''compute and return error on this output'''
    errors = []
    assert type(expect) == type([])
    for item in expect:
        assert type(item) == type(1)

    # error1 : type difference
    error = 0.0
    if False:
        if type(actual) != type([]):
            error = 1.0
            assert type(actual) == type(1)
            actual = [actual]
        else:
            for i in range(min(len(expect), len(actual))):
                if type(actual[i]) != type(1):
                    error += 1/len(expect)
            if len(actual) == 0:
                actual = [0]
                error = 1.0
    else:
        if type(actual) != type([]):
            error = 1.0 + len(expect)
            assert type(actual) == type(1)
            actual = [actual]
        else:
            for i in range(len(expect)):
                if i < len(actual):
                    if type(actual[i]) != type(1):
                        error += 1
                else:
                    error += 1
    if error > 0:
        error = error ** g_w1
    errors.append(error)
    k = len(expect)
    if k == 0:
        raise RuntimeError("TODO: handle case were expect output is empty")
    # error2 : aantal outputs
    actual_list = extract_numbers_list(actual)
    n = g_w2a if len(actual_list) < len(expect) else g_w2b
    errors.append(abs(len(actual_list) - len(expect)) ** n)
    # error3 : hoever zitten de expect getallen van de model getallen af
    actual_set = set(actual_list)
    if len(actual_set) == 0:
        actual_set = set([0])
    error = 0.0
    for expected_number in expect:
        error += _distance_with_closest_numbers(expected_number, actual_set) ** g_w3
    errors.append(error)
    # error4 : hoever zitten de model getallen van de expect getallen af
    error = 0.0
    for actual_number in actual_set:
        error += _distance_with_closest_numbers(actual_number, expect) ** g_w4
    errors.append(error)
    # error5 : absolute verschil van de outputs met de gewenste output
    error = 0.0
    for i in range(len(expect)):
        assert type(expect[i]) == type(1)
        if i < len(actual):
            if type(actual[i]) == type(1):
                error += (abs(actual[i] - expect[i])) ** g_w5
            else:
                error += (abs(expect[i])) ** g_w5
    errors.append(error)
    # error6 : hoeveel staan er in volgorde?
    error = 0.0
    j = 0 
    for i in range(len(expect)):
        while j < len(actual_list) and expect[i] != actual_list[j]:
            j += 1
        if j >= len(actual_list):
            error += 1 # /len(expect)
    if error > 0:
        error = error ** g_w6
    errors.append(error)
    # error7 : hoeveel staan er in volgorde (kijkend van achter naar voren)?
    error = 0.0
    j = len(actual_list)-1
    i = len(expect)-1
    while i >= 0:
        while j >= 0 and expect[i] != actual_list[j]:
            j -= 1
        if j < 0:
            error += 1 # /len(expect)
        i -= 1
    if error > 0:
        error = error ** g_w7
    errors.append(error)
    # error 8: # empty sublists
    error = count_empty_sublists(actual)
    if error > 0:
        error = error ** g_w8
    errors.append(error)

    if False:
        # geeft hele slechte resultaten!
        avg_errors = [0.060, 24.240, 47.012, 284.554, 260.724, 0.378, 0.666, 1.309]
        errors = [e / u for e, u in zip(errors, avg_errors)] 
    return errors


def find_col(board, v):
    for row in board:
        for col, v_col in enumerate(row):
            if v_col == v:
                return col
    return None


def compute_error_board_col_diag_common(input, actual, extra_function_params, expect, expect_cols):
    board = input[0]
    n = len(board)

    # error : type difference
    if type(actual) != type(expect):
        error_type = 1.0
    else:
        error_type = 0.0
    if type(actual) == type(1):
        actual = [actual]
    assert type(expect) == type([])
    if type(actual) != type([]):
        actual = [int(actual)]

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

    w = [0.61, 0.06, 0.03, 0.09, 0.1, 0.11] # default weights

    return [error_type*w[0], error_len*w[1], error_fromrows*w[2], error_fromrows_ordered*w[3],
         error_columns*w[4], error_correct_elements_ordered*w[5]]


def compute_error_board_col(input, actual, extra_function_params, log_file, verbose):
    board, col = input
    expect = [row[col] for row in board]
    error = compute_error_board_col_diag_common(input, actual, extra_function_params, expect, 1)
    return error


def compute_error_board_diag1(input, actual, extra_function_params, log_file, verbose):
    board = input[0]
    n = len(board)
    expect = [row[i] for i, row in enumerate(board)]
    return compute_error_board_col_diag_common(input, actual, extra_function_params, expect, n)


def compute_error_board_diag2(input, actual, extra_function_params, log_file, verbose):
    board = input[0]
    n = len(board)
    expect = [row[n-1 - i] for i, row in enumerate(board)]
    result = compute_error_board_col_diag_common(input, actual, extra_function_params, expect, n)
    return result


def compute_error_get_row_sums(input, actual, extra_function_params, log_file, verbose):
    board = input[0]
    expect = []
    expect += [sum(row) for row in board]
    return compute_error_list_of_ints(actual, expect, False)


def compute_error_get_col_sums(input, actual, extra_function_params, log_file, verbose):
    board = input[0]
    n = len(board)
    expect = []
    expect += [sum([row[col] for row in board]) for col in range(n)]
    return compute_error_list_of_ints(actual, expect, False)


def compute_error_get_diag_sums(input, actual, extra_function_params, log_file, verbose):
    board = input[0]
    n = len(board)
    expect = []
    expect.append(sum([board[i][i] for i in range(n)]))
    expect.append(sum([board[i][(n-1) - i] for i in range(n)]))
    return compute_error_list_of_ints(actual, expect, False)


def compute_error_get_magic_number_n(input, actual, extra_function_params, log_file, verbose):
    n = input[0]
    assert type(n) == type(1)
    assert n in [1, 2, 3, 4, 5, 6]
    expect = (n * (n * n + 1)) // 2
    if 1 <= n <= 6:
        assert expect == [0, 1, 5, 15, 34, 65, 111][n]
    result = compute_error_int(actual, expect, False)
    return result


def compute_error_get_magic_number(input, actual, extra_function_params, log_file, verbose):
    board = input[0]
    assert type(board) == type([])
    n = len(board)
    expect = (n * (n * n + 1)) // 2
    if 0 <= n <= 5:
        assert expect == [0, 1, 5, 15, 34, 65][n]
    return compute_error_int(actual, expect, False)


def compute_error_are_all_equal(input, actual, extra_function_params, log_file, verbose):
    values = input[0]
    expect = sum([1 if value == values[0] else 0 for value in values]) == len(values)
    return compute_error_int(actual, int(expect), False)


def compute_error_is_magic(inputs, actual, extra_function_params, log_file, verbose):
    error = 0.0
    if type(actual) != type(1):
        error += 0.64
    else:
        if actual not in [0, 1]:
            error += 0.1
    model_says_its_magic = bool(actual)
    board = inputs[0]
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
    if model_says_its_magic:
        error += 0.425 * (n - count_magic_rows) / n
        error += 0.325 * (n - count_magic_cols) / n
        error += 0.25 * (2 - count_magic_diags) / 2
    else:
        is_magic = count_magic_rows == n and count_magic_cols == n and count_magic_diags == 2
        if is_magic:
            error += 1.0
    return error,


def compute_error_is_sorted(input, actual, extra_function_params, log_file, verbose):
    error = 0.0
    if type(actual) != type(1):
        error += 0.8
    else:
        if actual not in [0, 1]:
            error += 0.2
    model_says_its_sorted = bool(actual)

    data = input[0]
    count_out_of_order = 0
    for i in range(len(data) - 1):
        if data[i] > data[i+1]:
            count_out_of_order += 1

    if model_says_its_sorted:
        if len(data) >= 2:        
            error += count_out_of_order / (len(data) - 1)
    else:
        is_sorted = count_out_of_order == 0
        if is_sorted:
            error += 0.5
    return error,


def compute_error_merge_elem(input, actual, extra_function_params, log_file, verbose):
    elem = input[0]
    data = input[1]
    for i in range(1, len(data)):
        assert data[i-1] <= data[i]
    expect = data + [elem] 
    expect.sort()
    return compute_error_list_of_ints(actual, expect)


def compute_error_sort(input, actual, extra_function_params, log_file, verbose):
    expect = copy.deepcopy(input[0])
    expect.sort()
    return compute_error_list_of_ints(actual, expect)


# ================================== EXACT errors voor testen van laagjes ==================


def compute_error_exact_inc(input, actual, extra_function_params, log_file, verbose):
    expect = input[0] + 1
    error = 0 if expect == actual else 1
    return error,


def compute_error_exact_inc2(input, actual, extra_function_params, log_file, verbose):
    expect = input[0] + 2
    error = 0 if expect == actual else 1
    return error,


def compute_error_exact_inc3(input, actual, extra_function_params, log_file, verbose):
    expect = input[0] + 3
    error = 0 if expect == actual else 1
    return error,


def compute_error_exact_inc4(input, actual, extra_function_params, log_file, verbose):
    expect = input[0] + 4
    error = 0 if expect == actual else 1
    return error,


def compute_error_exact_inc5(input, actual, extra_function_params, log_file, verbose):
    expect = input[0] + 5
    error = 0 if expect == actual else 1
    return error,


def compute_error_exact_add(input, actual, extra_function_params, log_file, verbose):
    expect = input[0] + input[1]
    error = 0 if expect == actual else 1
    return error,


def compute_error_exact_add_and_inc(input, actual, extra_function_params, log_file, verbose):
    expect = (input[0] + input[1]) + 1
    error = 0 if expect == actual else 1
    return error,


def compute_error_exact_inc_and_add(input, actual, extra_function_params, log_file, verbose):
    expect = (input[0] + 1) + (input[1] + 1)
    error = 0 if expect == actual else 1
    return error,


def compute_error_exact_add3(input, actual, extra_function_params, log_file, verbose):
    expect = input[0] + input[1] + input[2]
    error = 0 if expect == actual else 1
    return error,


def compute_error_get_diag1_cell(input, actual, extra_function_params, log_file, verbose):
    board, i = input
    expect = board[i][i]
    error = 0 if expect == actual else 1
    return error,


def compute_error_get_diag2_cell(input, actual, extra_function_params, log_file, verbose):
    board, i = input
    expect = board[i][len(board)-1-i]
    error = 0 if expect == actual else 1
    return error,


# ============================================== INTERFACE ====================


def find_worst_raw_error_vector(raw_error_matrix):
    worst_raw_error_vector = raw_error_matrix[0]
    for raw_error_vector in raw_error_matrix[1:]:
        if np.sum(worst_raw_error_vector) < np.sum(raw_error_vector):
            worst_raw_error_vector = raw_error_vector
    return worst_raw_error_vector


def compute_raw_error_matrix(example_inputs, actual_outputs, raw_error_function, log_file, verbose, penalise_non_reacting_models=False):
    raw_error_matrix = []
    if type(raw_error_function) == type(""):
        function_name, extra_function_params = raw_error_function, []
    else:
        function_name, extra_function_params = raw_error_function
    raw_error_function = eval(function_name)
    if verbose >= 4:
        log_file.write(f"compute_error_matrix({function_name})\n")
    assert len(example_inputs) == len (actual_outputs)
    domain_output_set = set()
    for i, (example_input, actual_output) in enumerate(zip(example_inputs, actual_outputs)):
        domain_output_set.add(recursive_tuple(actual_output))
        raw_error_vector = raw_error_function(example_input, actual_output, extra_function_params, log_file, verbose)
        raw_error_vector = np.array(raw_error_vector).astype(float)
        if verbose >= 4:
            log_file.write(f"    {raw_error_vector} = error(input={example_input}, output={actual_output})\n")
        if i == 0:
            raw_error_matrix = np.empty((len(example_inputs), raw_error_vector.shape[0]))
        raw_error_matrix[i, :] = raw_error_vector
    if penalise_non_reacting_models:
        if len(domain_output_set) == 1:
            worst_raw_error_vector = find_worst_raw_error_vector(raw_error_matrix)
            raw_error_matrix[:] = worst_raw_error_vector
    simplify = False
    if simplify:
        raw_error_matrix = np.sum(raw_error_matrix, axis=0).reshape((1, raw_error_matrix.shape[1]))
    return raw_error_matrix


def compute_raw_error(raw_error_matrix):
    return float(np.sum(raw_error_matrix))

