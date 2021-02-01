import math
import numpy as np


global dynamic_weights_matrix, estimated_remaining_iterations_matrix, prev_best_matrix
dynamic_weights_matrix = None
estimated_remaining_iterations_matrix= None
prev_best_matrix = None


def allocate_like(example):
    global estimated_remaining_iterations_matrix, dynamic_weights_matrix
    if estimated_remaining_iterations_matrix is None or estimated_remaining_iterations_matrix.shape != example.shape:
        estimated_remaining_iterations_matrix = np.ones_like(example) * 100
    if dynamic_weights_matrix is None or dynamic_weights_matrix.shape != example.shape:
        dynamic_weights_matrix = np.ones_like(example)

    
def compute_normalised_error(matrix, alpha=2):
    '''alpha > 1 penalises differences between weighted matrix components'''
    global dynamic_weights_matrix
    return np.sum((dynamic_weights_matrix * matrix) ** alpha)


def update_iterations_from_matrix_difference(reference_matrix, matrix):
    global estimated_remaining_iterations_matrix
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v, vref = matrix[i, j], reference_matrix[i, j]
            if v < vref:
                if estimated_remaining_iterations_matrix[i, j] > (vref - v) / vref:
                    estimated_remaining_iterations_matrix[i, j] = (vref - v) / vref


def compute_dynamic_weights(best_matrix):
    global dynamic_weights_matrix, estimated_remaining_iterations_matrix
    dynamic_weights_matrix.fill(1.0)
    for i in range(best_matrix.shape[0]):
        for j in range(best_matrix.shape[1]):
            if best_matrix[i, j] > 0:
                weight = estimated_remaining_iterations_matrix[i, j] / best_matrix[i, j]
                assert math.isclose(weight * best_matrix[i, j], estimated_remaining_iterations_matrix[i, j])
                dynamic_weights_matrix[i, j] = weight


def update_dynamic_weights(best_matrix, all_matrices):
    global prev_best_matrix, estimated_remaining_iterations_matrix, dynamic_weights_matrix
    allocate_like(best_matrix)
    estimated_remaining_iterations_matrix += 1
    if prev_best_matrix is not None:
        update_iterations_from_matrix_difference(prev_best_matrix, best_matrix)
    for matrix in all_matrices:
        if matrix is not best_matrix:
            update_iterations_from_matrix_difference(best_matrix, matrix)
    compute_dynamic_weights(best_matrix)
    prev_best_matrix = best_matrix


def self_test():
    matrix1 = np.array([[4.0, 4.0], [4.0, 4.0]])
    matrix2 = np.array([[2.0, 3.0], [4.0, 5.0]])
    update_dynamic_weights(matrix1, [matrix1, matrix2])
    print(compute_normalised_error(matrix1))
    print(compute_normalised_error(matrix2))
    print("selftest ok")

if __name__ == "__main__":
    self_test()