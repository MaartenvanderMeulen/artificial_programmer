import numpy as np


global dynamic_weights_matrix
dynamic_weights_matrix = None


def compute_normalised_error(raw_error_matrix):
    global dynamic_weights_matrix
    if dynamic_weights_matrix is None or dynamic_weights_matrix.shape != raw_error_matrix.shape:
        dynamic_weights_matrix = np.ones_like(raw_error_matrix)
    return np.sum(dynamic_weights_matrix * raw_error_matrix)


def update_iterations_from_family_difference(reference_family, family):
    global estimated_remaining_iterations_matrix
    for i in range(family.raw_error_matrix.shape[0]):
        for j in range(family.raw_error_matrix.shape[1]):
            v, vref = family.raw_error_matrix[i, j], reference_family[i, j]
            if v < vref:
                estimated_remaining_iterations_matrix[i, j] = (vref - v) / vref


def compute_dynamic_weights(best_raw_error_matrix, estimated_remaining_iterations_matrix):
    global dynamic_weights_matrix
    if dynamic_weights_matrix is None or dynamic_weights_matrix.shape != best_raw_error_matrix.shape:
        dynamic_weights_matrix = np.ones_like(raw_error_matrix)
    dynamic_weights_matrix.fill(1.0)
    for i in range(best_raw_error_matrix.shape[0]):
        for j in range(best_raw_error_matrix.shape[1]):
            if best_raw_error_matrix[i, j] > 0:
                weight = estimated_remaining_iterations_matrix[i, j] / 
                assert math.isclose(weight * best_raw_error_matrix, estimated_remaining_iterations_matrix[i, j])
                dynamic_weights_matrix[i, j] = weight


def update_dynamic_weights(best_family, all_families):
    global prev_best_family, estimated_remaining_iterations_matrix, dynamic_weights_matrix
    if prev_best_family is None:
        prev_best_family = best_family
        estimated_remaining_iterations_matrix = np.ones_like(best_family.raw_error_matrix) * 100
        dynamic_weights_matrix = np.ones_like(best_family.raw_error_matrix)
        return
    if estimated_remaining_iterations_matrix is None or estimated_remaining_iterations_matrix.shape != raw_error_matrix.shape:
        estimated_remaining_iterations_matrix = np.ones_like(best_family.raw_error_matrix, dtype="int")
    if best_family is prev_best_family:
        estimated_remaining_iterations_matrix += 1
    update_iterations_from_family_difference(prev_best_family, best_family)
    for family in all_families:
        if family is not best_family:
            update_iterations_from_family_difference(best_family, family)
    compute_dynamic_weights(best_family.raw_error_matrix, estimated_remaining_iterations_matrix)


def self_test():
    raw_error_matrix = np.array([[4.0, 4.0], [4.0, 4.0]])
    raw_error = compute_raw_error(raw_error_matrix)
    update_avg_raw_error_vector([raw_error_matrix])
    normalised_error_matrix = compute_normalised_error_matrix(raw_error_matrix)
    normalised_error = compute_normalised_error(normalised_error_matrix)
    assert raw_error == 16.0
    assert list(inv_avg_raw_error_vector) == [0.25, 0.25]
    assert [list(normalised_error_matrix[0]), list(normalised_error_matrix[1])] == [[1.0, 1.0], [1.0, 1.0]]
    assert normalised_error == 4.0
    print("selftest ok")

if __name__ == "__main__":
    self_test()