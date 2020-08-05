'''Evaluate the error of the actual data with respect to the expected data'''
import sys
import numpy as np
import interpret


# used in dynamic weight adjustment
global sum_weighted_errors, weights
sum_weighted_errors = np.zeros((5))
weights = np.ones((len(sum_weighted_errors))) / len(sum_weighted_errors)


def _distance_with_closest_values(x, values):
    '''distance of x with nearest'''
    if x > 1000000:
        x = 1000000
    if len(values) > 0:
        result = 1000000
        for value in values:
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
    global sum_weighted_errors, weights
    errors = np.zeros_like(weights)
    # error 4: type difference
    if type(actual) != type(expect):
        error[4] += 1
    if type(actual) != type([]):
        actual = [actual]
    if type(expect) != type([]):
        expect = [expect]
    k = len(expect)
    if k == 0:
        raise RuntimeError("TODO: handle case were expect output is empty")
    # error 0: aantal outputs
    n = 2 if len(actual) < len(expect) else 1.2
    errors[0] = (len(actual) - len(expect)) ** n
    # error 1: hoever zitten de expect getallen van de model getallen af
    for output in expect:
        errors[1] += _distance_with_closest_values(output, actual) ** 1.5
    # hoever zitten de model getallen van de expect getallen af
    for output in actual:
        errors[2] += _distance_with_closest_values(output, expect) ** 1.5
    # absolute verschil van de outputs met de gewenste output
    for i in range(len(expect)):
        if i < len(actual):
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
        print("    error", np.sum(weighted_errors))
    return np.sum(weighted_errors)


def dynamic_error_weight_adjustment(debug=True):
    global sum_weighted_errors, weights
    n = len(weights)
    if debug:
        print("weights before", weights, sum_weighted_errors, n)
    average_weighted_error = np.sum(sum_weighted_errors) / n
    for i in range(weights.shape(0)):
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
