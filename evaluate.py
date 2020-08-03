
def evaluate_output(model_output, expected_output, debug=False):
    '''compute and return error on this output'''
    if debug:
        print("evaluate_output")
        print("model_output", model_output)
        print("expected_output", expected_output)
    k = len(expected_output)
    if k == 0:
        raise RuntimeError("TODO: handle case were expected output is empty")
    errors = np.zeros((4))
    # aantal outputs
    errors[0] = (len(model_output) - len(expected_output)) ** 2
    # hoever zitten de expected getallen van de model getallen af
    for output in expected_output:
        errors[1] += distance_with_closest_values(output, model_output) ** 1.5
    # hoever zitten de model getallen van de expected getallen af
    for output in model_output:
        errors[2] += distance_with_closest_values(output, expected_output) ** 1.5
    # absolute verschil van de outputs met de gewenste output
    for i in range(len(expected_output)):
        if i < len(model_output):
            # print(model_output[i], expected_output[i])
            errors[3] += abs(model_output[i] - expected_output[i]) ** 1.5
        else:
            errors[3] += abs(expected_output[i]) ** 1.5
    global sum_weighted_errors, weights
    weighted_errors = errors[:4] * weights[:4]
    sum_weighted_errors[:4] += weighted_errors
    return np.sum(weighted_errors)


def evaluate_program(program_str, hints):
    missed_hints = 0
    for hint in hints:
        if program_str.find(hint) == -1:
            missed_hints += 1
    errors = missed_hints ** 2
    global sum_weighted_errors, weights
    weighted_errors = errors * weights[4]
    sum_weighted_errors[4] += weighted_errors
    # print("progrsam_str", program_str, "hints", hints, "missed_hints", missed_hints)
    return weighted_errors


def evaluate_postcondition(model_output, prev_model_output, input):
    if len(model_output) != 1 or prev_model_output is None or len(prev_model_output) != 1 or len(input) != 1:
        return 0.0
    model_increase = model_output[0] - prev_model_output[0]
    expected_increase = input[0]
    errors = abs(model_increase - expected_increase) ** 1.5
    global sum_weighted_errors, weights
    weighted_errors = errors * weights[5]
    sum_weighted_errors[5] += weighted_errors
    return weighted_errors
    

def dynamic_error_weight_adjustment():
    global dynamic_weight_iteration, sum_weighted_errors, weights
    dynamic_weight_iteration += 1
    if dynamic_weight_iteration >= 1000:
        debug = True
        n = len(weights)
        if debug:
            print("weights before", weights, sum_weighted_errors, n)
        average_weighted_error = np.sum(sum_weighted_errors) / n
        for i in range(6):
            if sum_weighted_errors[i] > 0:
                weights[i] *= average_weighted_error / sum_weighted_errors[i]
            else:
                weights[i] = 1.0 / n
        weights /= np.sum(weights)
        sum_weighted_errors = np.zeros((len(sum_weighted_errors)))
        dynamic_weight_iteration = 1
        if debug:
            print("weights after", weights)
