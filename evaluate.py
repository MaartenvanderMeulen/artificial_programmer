'''Evaluate the error of the actual data with respect to the expected data'''
import sys
import numpy as np
import copy
import math
# import time
import interpret


# used in dynamic weight adjustment
global sum_errors, weights
weights = np.ones((6)) / 6
sum_errors = np.zeros_like(weights)


def recursive_tuple(value):
    if type(value) == type(1) or type(value) == type(""):
        return value
    return tuple([recursive_tuple(v) for v in value])


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
        assert type(actual) == type(1)
        actual = [actual]
    else:
        for v in actual:
            if type(v) != type(1):
                error += 1/(1+len(actual))
        if len(actual) == 0:
            actual = [0]
            error = 1.0
    errors.append(error)
    k = len(expect)
    if k == 0:
        raise RuntimeError("TODO: handle case were expect output is empty")
    # error : aantal outputs
    n = 2 if len(actual) < len(expect) else 1.2
    errors.append(abs(len(actual) - len(expect)) ** n)
    # error : hoever zitten de expect getallen van de model getallen af
    actual_numbers = extract_numbers(actual)
    if len(actual_numbers) == 0:
        actual_numbers = set([0])
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


def eval_board_col(input, actual, extra_function_params, log_file, verbose):
    board, col = input
    expect = [row[col] for row in board]
    error = eval_board_col_diag_common(input, actual, extra_function_params, expect, 1)
    return error


def eval_board_diag1(input, actual, extra_function_params, log_file, verbose):
    board = input[0]
    n = len(board)
    expect = [row[i] for i, row in enumerate(board)]
    return eval_board_col_diag_common(input, actual, extra_function_params, expect, n)


def eval_board_diag2(input, actual, extra_function_params, log_file, verbose):
    board = input[0]
    n = len(board)
    expect = [row[n-1 - i] for i, row in enumerate(board)]
    result = eval_board_col_diag_common(input, actual, extra_function_params, expect, n)
    return result


def eval_get_row_sums(input, actual, extra_function_params, log_file, verbose):
    board = input[0]
    expect = []
    expect += [sum(row) for row in board]
    return evaluate_list_of_ints(actual, expect, False)


def eval_get_col_sums(input, actual, extra_function_params, log_file, verbose):
    board = input[0]
    n = len(board)
    expect = []
    expect += [sum([row[col] for row in board]) for col in range(n)]
    return evaluate_list_of_ints(actual, expect, False)


def eval_get_diag_sums(input, actual, extra_function_params, log_file, verbose):
    board = input[0]
    n = len(board)
    expect = []
    expect.append(sum([board[i][i] for i in range(n)]))
    expect.append(sum([board[i][(n-1) - i] for i in range(n)]))
    return evaluate_list_of_ints(actual, expect, False)


def eval_get_magic_number_n(input, actual, extra_function_params, log_file, verbose):
    n = input[0]
    assert type(n) == type(1)
    assert n in [1, 2, 3, 4, 5, 6]
    expect = (n * (n * n + 1)) // 2
    if 1 <= n <= 6:
        assert expect == [0, 1, 5, 15, 34, 65, 111][n]
    result = evaluate_int(actual, expect, False)
    return result


def eval_get_magic_number(input, actual, extra_function_params, log_file, verbose):
    board = input[0]
    assert type(board) == type([])
    n = len(board)
    expect = (n * (n * n + 1)) // 2
    if 0 <= n <= 5:
        assert expect == [0, 1, 5, 15, 34, 65][n]
    return evaluate_int(actual, expect, False)


def eval_are_all_equal(input, actual, extra_function_params, log_file, verbose):
    values = input[0]
    expect = sum([1 if value == values[0] else 0 for value in values]) == len(values)
    return evaluate_int(actual, int(expect), False)


def eval_is_magic(inputs, actual, extra_function_params, log_file, verbose):
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


def eval_is_sorted(input, actual, extra_function_params, log_file, verbose):
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


def eval_is_sorted_self_test():
    assert math.isclose(eval_is_sorted([[1, 2, 3, 4, 5]], 1, None, None, None)[0], 0.0)
    assert math.isclose(eval_is_sorted([[1, 2, 3, 5, 4]], 1, None, None, None)[0], 0.25)
    assert math.isclose(eval_is_sorted([[1, 2, 5, 4, 3]], 1, None, None, None)[0], 0.5)
    assert math.isclose(eval_is_sorted([[1, 5, 4, 3, 2]], 1, None, None, None)[0], 0.75)
    assert math.isclose(eval_is_sorted([[5, 4, 3, 2, 1]], 1, None, None, None)[0], 1.0)
    assert math.isclose(eval_is_sorted([[5, 4, 3, 2, 1]], 0, None, None, None)[0], 0.0)
    assert math.isclose(eval_is_sorted([[5, ]], 0, None, None, None)[0], 1.0)
    assert math.isclose(eval_is_sorted([[5, ]], 1, None, None, None)[0], 0.0)
    assert math.isclose(eval_is_sorted([[]], 0, None, None, None)[0], 1.0)
    assert math.isclose(eval_is_sorted([[]], 1, None, None, None)[0], 0.0)


def eval_merge_elem(input, actual, extra_function_params, log_file, verbose):
    elem = input[0]
    data = input[1]
    for i in range(1, len(data)):
        assert data[i-1] <= data[i]
    expect = data + [elem] 
    expect.sort()
    return evaluate_list_of_ints(actual, expect)


def eval_sort(input, actual, extra_function_params, log_file, verbose):
    expect = copy.deepcopy(input[0])
    expect.sort()
    return evaluate_list_of_ints(actual, expect)


# ================================== EXACT evals voor testen van laagjes ==================


def eval_exact_inc(input, actual, extra_function_params, log_file, verbose):
    expect = input[0] + 1
    error = 0 if expect == actual else 1
    return error,


def eval_exact_inc2(input, actual, extra_function_params, log_file, verbose):
    expect = input[0] + 2
    error = 0 if expect == actual else 1
    return error,


def eval_exact_inc3(input, actual, extra_function_params, log_file, verbose):
    expect = input[0] + 3
    error = 0 if expect == actual else 1
    return error,


def eval_exact_inc4(input, actual, extra_function_params, log_file, verbose):
    expect = input[0] + 4
    error = 0 if expect == actual else 1
    return error,


def eval_exact_inc5(input, actual, extra_function_params, log_file, verbose):
    expect = input[0] + 5
    error = 0 if expect == actual else 1
    return error,


def eval_exact_add(input, actual, extra_function_params, log_file, verbose):
    expect = input[0] + input[1]
    error = 0 if expect == actual else 1
    return error,


def eval_exact_add_and_inc(input, actual, extra_function_params, log_file, verbose):
    expect = (input[0] + input[1]) + 1
    error = 0 if expect == actual else 1
    return error,


def eval_exact_inc_and_add(input, actual, extra_function_params, log_file, verbose):
    expect = (input[0] + 1) + (input[1] + 1)
    error = 0 if expect == actual else 1
    return error,


def eval_exact_add3(input, actual, extra_function_params, log_file, verbose):
    expect = input[0] + input[1] + input[2]
    error = 0 if expect == actual else 1
    return error,


def eval_get_diag1_cell(input, actual, extra_function_params, log_file, verbose):
    board, i = input
    expect = board[i][i]
    error = 0 if expect == actual else 1
    return error,


def eval_get_diag2_cell(input, actual, extra_function_params, log_file, verbose):
    board, i = input
    expect = board[i][len(board)-1-i]
    error = 0 if expect == actual else 1
    return error,


# ============================================== INTERFACE ====================


def evaluate_code(actual_code_str, expected_code_str):
    def count_equal_prefix_length(str1, str2):
        n_eq = 0
        for i in range(min(len(str1), len(str2))):
            if str1[i] != str2[i]:
                break
            n_eq += 1
        return n_eq
    return len(expected_code_str) - count_equal_prefix_length(actual_code_str, expected_code_str)


def compute_eval_vectors(example_inputs, actual_outputs, evaluation_function, log_file, verbose, penalise_non_reacting_models=False):
    '''Returns list of evaluation vectors, to be weighted dynamically using compute_weighted_sum'''
    eval_vectors = []
    if type(evaluation_function) == type(""):
        function_name, extra_function_params = evaluation_function, []
    else:
        function_name, extra_function_params = evaluation_function
    global sum_errors
    eval_function = eval(function_name)
    if verbose >= 4:
        log_file.write(f"compute_error_vectors({function_name})\n")
    assert len(example_inputs) == len (actual_outputs)
    domain_output_set = set()
    for example_input, actual_output in zip(example_inputs, actual_outputs):
        domain_output_set.add(recursive_tuple(actual_output))
        eval_vector = eval_function(example_input, actual_output, extra_function_params, log_file, verbose)
        eval_vector = np.array(eval_vector).astype(float)
        if sum_errors.shape[0] != eval_vector.shape[0]:
            sum_errors = np.zeros_like(eval_vector)
        sum_errors += eval_vector
        if verbose >= 4:
            log_file.write(f"    {eval_vector} = error(input={example_input}, output={actual_output})\n")
        eval_vectors.append(eval_vector)
    if penalise_non_reacting_models:
        if len(domain_output_set) == 1:
            global weights
            if verbose >= 3:
                log_file.write(f"penalise_non_reacting_models\n")
                log_file.write(f"    domain_output_set {str(domain_output_set)}\n")
                log_file.write(f"    len(domain_output_set) {len(domain_output_set)}\n")
                vs = [np.sum(v*weights) for v in eval_vectors]
                log_file.write(f"    old model evals {str(vs)}\n")
            max_eval = eval_vectors[0]
            for eval_vector in eval_vectors:
                if np.sum(max_eval) < np.sum(eval_vector):
                    max_eval = eval_vector
            for eval_vector in eval_vectors:
                eval_vector[...] = max_eval
            if verbose >= 3:
                vs = [np.sum(v*weights) for v in eval_vectors]
                log_file.write(f"    new model evals {str(vs)}\n")
    return eval_vectors


def compute_weighted_sums(eval_vectors):
    weighted_sums = []
    global weights
    if weights.shape[0] != eval_vectors[0].shape[0]:
        weights = np.ones((eval_vectors[0].shape[0])) / eval_vectors[0].shape[0]
    for eval_vector in eval_vectors:
        weighted_sums.append(np.sum(eval_vector * weights))
    return weighted_sums


def compute_weighted_error(example_inputs, actual_outputs, evaluation_function, log_file, verbose, penalise_non_reacting_models=False):
    eval_vectors = compute_eval_vectors(example_inputs, actual_outputs, evaluation_function, log_file, verbose)
    weighted_error = sum(compute_weighted_sums(eval_vectors))
    return weighted_error


def init_dynamic_error_weight_adjustment():
    global sum_errors, weights
    n = 6
    sum_errors = np.zeros((n))
    weights = np.ones((n)) / n


def dynamic_error_weight_adjustment(log_file, verbose):
    global sum_errors, weights
    n = len(weights)
    if verbose >= 1:
        log_file.write(f"weights before {weights}, {sum_errors * weights}\n")
    average_weighted_error = np.sum(sum_errors) / n
    for i in range(n):
        if sum_errors[i] > 0:
            weights[i] = average_weighted_error / sum_errors[i]
        else:
            weights[i] = 1.0 / n
    weights /= np.sum(weights)
    if verbose >= 1:
        log_file.write(f"weights after {weights}, {sum_errors * weights}\n")
    sum_errors.fill(0)


if __name__ == "__main__":
    eval_is_sorted_self_test()
