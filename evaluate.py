'''Evaluate the error of the actual data with respect to the expected data'''
import sys
import numpy as np
import interpret
import time


# used in dynamic weight adjustment
global sum_weighted_errors, weights
sum_weighted_errors = np.zeros((5))
weights = [0.24, 0.04, 0.34, 0.04, 0.34]
# np.ones((len(sum_weighted_errors))) / len(sum_weighted_errors)


def _extract_numbers_impl(values, depth):
    if depth > 100:
        # skip rest
        return set(), 0        
    if type(values) == type([]):
        result, count = set(), 0
        for item in values:
            item_set, item_count = _extract_numbers_impl(item, depth+1)
            result.update(item_set)
            count += item_count
    else:
        if not(type(values) == type(1)):
            print(values, type(values))
        assert type(values) == type(1)
        result, count = set([values]), 1
    return result, count

    
global longest_values
longest_values = 0
    
    
def extract_numbers(values, label):
    result, count = _extract_numbers_impl(values, 0)
    global longest_values
    if longest_values < count:
        longest_values = count
    return result, count

    
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
    
    
# ============================================== INTERFACE ====================


def evaluate(actual, expect, debug=False):
    '''compute and return error on this output'''
    #print("evaluate start")
    global sum_weighted_errors, weights
    errors = np.zeros_like(weights)
    # error 4: type difference
    if type(actual) != type(expect):
        errors[4] += 1
    if type(actual) == type(1):
        actual = [actual]
    if type(expect) == type(1):
        expect = [expect]
    k = len(expect)
    if k == 0:
        raise RuntimeError("TODO: handle case were expect output is empty")
    # error 0: aantal outputs
    n = 2 if len(actual) < len(expect) else 1.2
    errors[0] = (len(actual) - len(expect)) ** n
    # error 1: hoever zitten de expect getallen van de model getallen af
    actual_numbers, _ = extract_numbers(actual, "evaluate actual")
    expected_numbers, _ = extract_numbers(expect, "evaluate expected")
    for expected_number in expected_numbers:
        errors[1] += _distance_with_closest_numbers(expected_number, actual_numbers) ** 1.5
    # hoever zitten de model getallen van de expect getallen af
    for actual_number in actual_numbers:
        errors[2] += _distance_with_closest_numbers(actual_number, expected_numbers) ** 1.5
    # absolute verschil van de outputs met de gewenste output
    for i in range(len(expect)):
        assert type(expect[i]) == type(1)
        if i < len(actual) and type(actual[i]) == type(1):
            # print(actual[i], expect[i])            
            errors[3] += abs(actual[i] - expect[i]) ** 1.5
        else:
            errors[3] += abs(expect[i]) ** 1.5
    weighted_errors = errors * weights
    sum_weighted_errors += weighted_errors
    if debug:
        print("evaluate")
        print("    actual", actual)
        print("    expect", expect)
        print("    individual errors", errors)
        print("    sum weighted error this sample", np.sum(weighted_errors))
        print("    individual weighted sums errors all samples", sum_weighted_errors)
    #print("evaluate end")
    return np.sum(weighted_errors)


def dynamic_error_weight_adjustment(debug=True):
    global sum_weighted_errors, weights
    n = len(weights)
    if False: # debug:
        print("weights before", weights, sum_weighted_errors, n)
    average_weighted_error = np.sum(sum_weighted_errors) / n
    for i in range(n):
        if sum_weighted_errors[i] > 0:
            weights[i] *= average_weighted_error / sum_weighted_errors[i]
        else:
            weights[i] = 1.0 / n
    weights /= np.sum(weights)
    sum_weighted_errors.fill(0)
    if debug:
        print("weights after", weights)


if __name__ == "__main__":
    file_name = sys.argv[1] if len(sys.argv) >= 2 else "test_evaluate.txt"
    tests = interpret.compile(interpret.load(file_name))
    for actual, expect in tests:
        print(evaluate(actual, expect, debug=True))
    print(sum_weighted_errors)
