import math
import numpy as np


global dynamic_weights_matrix, estimated_remaining_iterations_matrix
dynamic_weights_matrix = None
estimated_remaining_iterations_matrix= None


def allocate_like(example):
    global estimated_remaining_iterations_matrix, dynamic_weights_matrix
    if estimated_remaining_iterations_matrix is None or estimated_remaining_iterations_matrix.shape != example.shape:
        estimated_remaining_iterations_matrix = np.ones_like(example) * 100
    if dynamic_weights_matrix is None or dynamic_weights_matrix.shape != example.shape:
        dynamic_weights_matrix = np.ones_like(example)

    
def compute_normalised_error(matrix, alpha):
    '''alpha > 1 penalises differences between weighted matrix components'''
    global dynamic_weights_matrix
    return np.sum((dynamic_weights_matrix * matrix) ** alpha)


def update_iterations_from_matrix_difference(reference_matrix, matrix, divide_vref):
    global estimated_remaining_iterations_matrix
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v, vref = matrix[i, j], reference_matrix[i, j]
            if v < vref:
                iters = vref / (vref - v) if divide_vref else v / (vref - v)
                if estimated_remaining_iterations_matrix[i, j] > iters:
                    # print(i, j, "vref", vref, "v", v, "old iter", estimated_remaining_iterations_matrix[i, j], "new iter", iters)
                    estimated_remaining_iterations_matrix[i, j] = iters 


def compute_dynamic_weights(best_matrix):
    global dynamic_weights_matrix, estimated_remaining_iterations_matrix
    dynamic_weights_matrix.fill(1.0)
    for i in range(best_matrix.shape[0]):
        for j in range(best_matrix.shape[1]):
            if best_matrix[i, j] > 0:
                weight = estimated_remaining_iterations_matrix[i, j] / best_matrix[i, j]
                assert math.isclose(weight * best_matrix[i, j], estimated_remaining_iterations_matrix[i, j])
                dynamic_weights_matrix[i, j] = weight


def update_dynamic_weights(prev_best_matrix, best_matrix, all_matrices):
    global estimated_remaining_iterations_matrix, dynamic_weights_matrix
    allocate_like(best_matrix)
    estimated_remaining_iterations_matrix += 1
    if prev_best_matrix is not None:
        update_iterations_from_matrix_difference(prev_best_matrix, best_matrix, False)
    for matrix in all_matrices:
        if matrix is not best_matrix:
            update_iterations_from_matrix_difference(best_matrix, matrix, True)
    compute_dynamic_weights(best_matrix)
    prev_best_matrix = best_matrix


def self_test():
    matrix2 = np.array([[2.0, 3.0], [4.0, 5.0]])
    matrix1 = np.array([[4.0, 4.0], [4.0, 4.0]])
    print("matrix2", matrix2)
    print("matrix1", matrix1)

    update_dynamic_weights(matrix2, matrix2, [matrix2])
    print("estimated_remaining_iterations_matrix", estimated_remaining_iterations_matrix)
    print("dynamic_weights_matrix", dynamic_weights_matrix)
    print("dynamic_weights_matrix * matrix2", dynamic_weights_matrix * matrix2, np.sum(dynamic_weights_matrix * matrix2))
    print("dynamic_weights_matrix * matrix1", dynamic_weights_matrix * matrix1, np.sum(dynamic_weights_matrix * matrix1))

    update_dynamic_weights(matrix2, matrix1, [matrix1, matrix2])
    print("estimated_remaining_iterations_matrix", estimated_remaining_iterations_matrix)
    print("dynamic_weights_matrix", dynamic_weights_matrix)
    print("dynamic_weights_matrix * matrix2", dynamic_weights_matrix * matrix2, np.sum(dynamic_weights_matrix * matrix2))
    print("dynamic_weights_matrix * matrix1", dynamic_weights_matrix * matrix1, np.sum(dynamic_weights_matrix * matrix1))
    #print("compute_normalised_error(matrix2)", compute_normalised_error(matrix2, 1.0))
    #print("compute_normalised_error(matrix1)", compute_normalised_error(matrix1, 1.0))

    print("end of test output")

if __name__ == "__main__":
    self_test()