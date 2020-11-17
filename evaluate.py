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
        if sum_board != n * magic_number:
            print("sum_board != n * magic_number")
        error += 0.425 * (n - count_magic_rows) / n
        error += 0.325 * (n - count_magic_cols) / n
        error += 0.25 * (2 - count_magic_diags) / 2
    else:
        is_magic = count_magic_rows == n and count_magic_cols == n and count_magic_diags == 2
        if is_magic:
            error += 1.0
    return error,

    
def compute_count_non_magic(board):
    n = len(board)
    magic_number = (n * (n * n + 1)) // 2
    count, sum_diag1, sum_diag2 = 0, 0, 0
    for row in board:
        if sum(row) != magic_number:
            count += 1
    for i in range(n):
        col = [row[i] for row in board]
        if sum(col) != magic_number:
            count += 1
    for i, row in enumerate(board):
        sum_diag1 += row[i]
        sum_diag2 += row[(n-1) - i]
    if sum_diag1 != magic_number:
        count += 1
    if sum_diag2 != magic_number:
        count += 1
    return count, count / (2 * n + 2)


global eval_is_magic_all_context
eval_is_magic_all_context = None


def eval_is_magic_all_impl(example_inputs, actual_outputs, domain_outputs):
    first_time = True
    first_actual_output, first_domain_output, output_depends_on_input, domain_output_depends_on_input = None, None, False, False
    all_correct_domain, all_correct_type = True, True
    # kijk eerst of de individual er iets van begrepen heeft
    for example_input, actual_output, domain_output in zip(example_inputs, actual_outputs, domain_outputs):
        if type(actual_output) != type(1):
            all_correct_type = False
        elif actual_output not in [0, 1]:
            all_correct_domain = False
        if first_time:
            first_actual_output = actual_output
            first_domain_output = domain_output
            first_time = False
        else:
            if actual_output != first_actual_output:
                output_depends_on_input = True
            if domain_output != first_domain_output:
                domain_output_depends_on_input = True
    if output_depends_on_input and domain_output_depends_on_input:
        # de individual heeft er iets van begrepen
        global eval_is_magic_all_context
        count_negatives, count_postives, sum_count_non_magic, is_magic_list, count_non_magic_list, fraction_non_magic_list, domain_outputs_dict = eval_is_magic_all_context        
        model_evals = []
        count_false_negatives, count_false_positives, sum_count_non_magic_in_false_positives = 0, 0, 0
        for actual_output, is_magic, count_non_magic, fraction_non_magic in zip(actual_outputs, is_magic_list, count_non_magic_list, fraction_non_magic_list):
            model_says_is_magic = bool(actual_output)
            if is_magic:
                if not model_says_is_magic:
                    count_false_negatives += 1
                    model_evals.append(1.0)
                else:
                    model_evals.append(0.0)
            else:
                if model_says_is_magic:
                    count_false_positives += 1
                    sum_count_non_magic_in_false_positives += count_non_magic
                    model_evals.append(fraction_non_magic)
                else:
                    model_evals.append(0.0)
        weighted_error = 0.0
        weighted_error += 0.4 * count_false_negatives / count_postives
        weighted_error += 0.4 * sum_count_non_magic_in_false_positives / sum_count_non_magic
    else:
        # de individual heeft er NIETS van begrepen
        weighted_error = 1.0
        if not output_depends_on_input:            
            if all_correct_domain:
                weighted_error -= 0.02
            elif all_correct_type:
                weighted_error -= 0.01
        else: # not domain_output_depends_on_input
            if all_correct_domain:
                weighted_error -= 0.05
            elif all_correct_type:
                weighted_error -= 0.04
            else:
                weighted_error -= 0.03
        model_evals = [weighted_error for _ in example_inputs]
    return weighted_error, model_evals


def eval_is_magic_all(example_inputs, actual_outputs, extra_function_params, f, debug):
    global eval_is_magic_all_context
    if not eval_is_magic_all_context:
        # eval_is_magic_all wordt duizenden keren aangeroepen.  Reken slechts 1x uit wat de verwachte output is
        count_negatives, count_postives, sum_count_non_magic, is_magic_list = 0, 0, 0, []            
        count_non_magic_list, fraction_non_magic_list = [], []
        for example_input in example_inputs:
            count_non_magic, fraction_non_magic = compute_count_non_magic(example_input[0])
            is_magic = int(count_non_magic == 0)
            is_magic_list.append(is_magic)
            count_non_magic_list.append(count_non_magic)
            fraction_non_magic_list.append(fraction_non_magic)
            if is_magic:
                count_postives += 1
            else:
                count_negatives += 1
                sum_count_non_magic += count_non_magic
        domain_outputs_dict = dict()
        eval_is_magic_all_context = count_negatives, count_postives, sum_count_non_magic, is_magic_list, count_non_magic_list, fraction_non_magic_list, domain_outputs_dict        
    else:
        count_negatives, count_postives, sum_count_non_magic, is_magic_list, count_non_magic_list, fraction_non_magic_list, domain_outputs_dict = eval_is_magic_all_context        
        
    # Is evaluatie voor deze outputs al bekend?
    domain_outputs = (int(bool(actual_output)) for actual_output in actual_outputs)
    if domain_outputs in domain_outputs_dict:
        weighted_error, model_evals = domain_outputs_dict[domain_outputs]
    else:        
        # Nog niet bekend. Bereken evaluatie.
        weighted_error, model_evals = eval_is_magic_all_impl(example_inputs, actual_outputs, domain_outputs)
        domain_outputs_dict[domain_outputs] = weighted_error, model_evals
    
    if f and debug >= 2:
        if debug >= 3:
            f.write(f"actual_outputs {str(actual_outputs)}\n")
            f.write(f" is_magic_list {str(is_magic_list)}\n")
        model_evals_str = " ".join([f"{v:.3f}" for v in model_evals])
        f.write(f"      eval_per_output {model_evals_str} overall_eval {weighted_error:.3f}\n")
    return weighted_error, model_evals


def eval_is_magic_all_old(example_inputs, actual_outputs, extra_function_params, f, debug):
    model_evals = []
    assert len(example_inputs) == len (actual_outputs)
    domain_output_set = set()
    for example_input, actual_output in zip(example_inputs, actual_outputs):
        domain_output_set.add(bool(actual_output))
        v = eval_is_magic(example_input, actual_output, extra_function_params)[0]
        model_evals.append(v)
    if False:
        # penalise non reacting models
        if len(domain_output_set) == 1:
            max_eval = max(model_evals)
            model_evals = [max_eval for _ in model_evals]
    return sum(model_evals), model_evals


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
        print(f"    evaluation {result:.3f}, input {str(input)}, actual_output {str(actual_output)}")
    return result
    
    
def evaluate_all(inputs, actual_outputs, evaluation_function, f, debug):
    function_name, extra_function_params = evaluation_function
    f_all = eval(function_name)
    return f_all(inputs, actual_outputs, extra_function_params, f, debug)

    
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
