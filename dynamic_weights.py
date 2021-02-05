import random
import math
import numpy as np


global dynamic_weights_matrix, estimated_remaining_iterations_matrix, simplify
dynamic_weights_matrix = None
estimated_remaining_iterations_matrix= None


def allocate_like(example):
    global estimated_remaining_iterations_matrix, dynamic_weights_matrix
    if estimated_remaining_iterations_matrix is None or estimated_remaining_iterations_matrix.shape != example.shape:
        estimated_remaining_iterations_matrix = np.ones_like(example) * 100
    if dynamic_weights_matrix is None or dynamic_weights_matrix.shape != example.shape:
        dynamic_weights_matrix = np.ones_like(example)


def compute_normalised_error_matrix(matrix):
    allocate_like(matrix)
    global dynamic_weights_matrix
    return dynamic_weights_matrix * matrix
 

def compute_normalised_error(matrix, alpha):
    '''alpha > 1 penalises differences between weighted matrix components'''
    assert alpha == 1
    allocate_like(matrix)
    global dynamic_weights_matrix
    return np.sum((dynamic_weights_matrix * matrix) ** alpha)


def update_remaining_iterations_v1(prev_best_matrix, best_matrix, all_matrices):
    global estimated_remaining_iterations_matrix
    estimated_remaining_iterations_matrix += 1
    if prev_best_matrix is not None:
        for i in range(estimated_remaining_iterations_matrix.shape[0]):
            for j in range(estimated_remaining_iterations_matrix.shape[1]):
                if prev_best_matrix[i, j] > best_matrix[i, j]:
                    iters = best_matrix[i, j] / (prev_best_matrix[i, j] - best_matrix[i, j])
                    if estimated_remaining_iterations_matrix[i, j] > iters:
                        estimated_remaining_iterations_matrix[i, j] = iters
        estimated_remaining_iterations_matrix[prev_best_matrix == 0] = 1
    for matrix in all_matrices:
        if matrix is not best_matrix:
            for i in range(estimated_remaining_iterations_matrix.shape[0]):
                for j in range(estimated_remaining_iterations_matrix.shape[1]):
                    if best_matrix[i, j] > matrix[i, j]:
                        iters = best_matrix[i, j] / (best_matrix[i, j] - matrix[i, j])
                        if estimated_remaining_iterations_matrix[i, j] > iters:
                            estimated_remaining_iterations_matrix[i, j] = iters
            estimated_remaining_iterations_matrix[matrix == 0] = 1
    estimated_remaining_iterations_matrix[best_matrix == 0] = 0


def update_remaining_iterations_v2(prev_best_matrix, best_matrix, all_matrices):
    global estimated_remaining_iterations_matrix
    estimated_remaining_iterations_matrix += 1
    if prev_best_matrix is not None:
        for i in range(estimated_remaining_iterations_matrix.shape[0]):
            for j in range(estimated_remaining_iterations_matrix.shape[1]):
                if prev_best_matrix[i, j] > best_matrix[i, j]:
                    iters = best_matrix[i, j] / (prev_best_matrix[i, j] - best_matrix[i, j])
                    estimated_remaining_iterations_matrix[i, j] = iters
                elif prev_best_matrix[i, j] < best_matrix[i, j]:
                    iters = best_matrix[i, j] / (best_matrix[i, j] - prev_best_matrix[i, j])
                    estimated_remaining_iterations_matrix[i, j] = iters
    estimated_remaining_iterations_matrix[best_matrix == 0] = 0


def adjust_dynamic_weights_v1(adaptation_speed):
    global estimated_remaining_iterations_matrix, dynamic_weights_matrix
    beta = adaptation_speed
    components = []
    for i in range(estimated_remaining_iterations_matrix.shape[0]):
        for j in range(estimated_remaining_iterations_matrix.shape[1]):
            iters = estimated_remaining_iterations_matrix[i, j]
            components.append([iters, i, j, dynamic_weights_matrix[i, j]])
            if iters == 0 and dynamic_weights_matrix[i, j] > 0.1:
                dynamic_weights_matrix[i, j] /= beta
    components.sort(key=lambda item: item[0] + 1/(1000*item[3]))
    n = (estimated_remaining_iterations_matrix.shape[0] * estimated_remaining_iterations_matrix.shape[1])
    assert n == len(components)
    for iters, i, j, _ in components[:n//3]:
        if iters > 0:
            if iters < components[n//2][0] and dynamic_weights_matrix[i, j] > 0.1:
                dynamic_weights_matrix[i, j] /= beta
    for iters, i, j, _ in components[2*n//3:]:
        if iters > 0:
            dynamic_weights_matrix[i, j] *= beta
    # dynamic_weights_matrix *= n / np.sum(dynamic_weights_matrix)


def adjust_dynamic_weights_v2(adaptation_speed):
    global estimated_remaining_iterations_matrix, dynamic_weights_matrix
    beta = adaptation_speed
    components = []
    for i in range(estimated_remaining_iterations_matrix.shape[0]):
        for j in range(estimated_remaining_iterations_matrix.shape[1]):
            iters = estimated_remaining_iterations_matrix[i, j]
            components.append([iters, i, j, dynamic_weights_matrix[i, j]])
    components.sort(key=lambda item: item[0] + 1/(1000*item[3]))
    n = (estimated_remaining_iterations_matrix.shape[0] * estimated_remaining_iterations_matrix.shape[1])
    assert n == len(components)
    m = 2 # dynamic_weights_matrix.shape[1]
    for iters, i, j, _ in components[:n*1//m]:
        dynamic_weights_matrix[i, j] /= beta
    for iters, i, j, _ in components[n*(m-1)//m:]:
        dynamic_weights_matrix[i, j] *= beta
    # dynamic_weights_matrix *= n / np.sum(dynamic_weights_matrix)
 

def dump_matrix(f, matrix):
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            f.write(f" {matrix[i,j]:.1f}")
        f.write("\n")

    
def dump_dw_matrix(f):
    global dynamic_weights_matrix
    f.write("dw matrix\n")
    dump_matrix(f, dynamic_weights_matrix)

    
def update_dynamic_weights(prev_best_matrix, best_matrix, all_matrices, adaptation_speed):
    allocate_like(best_matrix)    
    update_remaining_iterations_v2(prev_best_matrix, best_matrix, all_matrices)
    adjust_dynamic_weights_v2(adaptation_speed)


def log_info(f):
    global estimated_remaining_iterations_matrix, dynamic_weights_matrix
    msg = " ".join([f"{iter:.0f}" for iter in estimated_remaining_iterations_matrix[-1]])
    if len(msg) > 80:
        msg = msg[:77] + "..."
    f.write("    remaining iterations, last row: " + msg + "\n")
    msg = " ".join([f"{w:.1f}" for w in dynamic_weights_matrix[-1]])
    if len(msg) > 80:
        msg = msg[:77] + "..."
    f.write("    dynamic weights, last row: " + msg + "\n")


def test_result(actual, expected):
    for i in range(actual.shape[0]):
        for j in range(actual.shape[1]):
            if not math.isclose(actual[i, j], expected[i, j]):
                print(actual[i, j], expected[i, j])
            assert math.isclose(actual[i, j], expected[i, j])


def format_matrix(matrix):
    return " ".join([f"{x:.1f}" for x in matrix[-1]])


def self_test():
    matrix2 = np.array([[2.0, 3.0, 4.0, 6.0, 0.0, 3.0, 4.0, 6.0]])
    matrix1 = np.array([[4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0]])
    verbose = False
    if verbose:
        print("=============")
        print("matrix2", matrix2)
        print("matrix1", matrix1)

    update_dynamic_weights(None, matrix2, [matrix2], 1.1)
    test_result(estimated_remaining_iterations_matrix, np.array([[101, 101, 101, 101, 0, 101, 101, 101]]))
    test_result(dynamic_weights_matrix, np.array([[1.0, 1.0, 1.0, 1.0, 1.0/1.1, 1.0, 1.0, 1.0*1.1]]))
    if verbose:
        print("=============")
        print("estimated_remaining_iterations_matrix", format_matrix(estimated_remaining_iterations_matrix))
        print("dynamic_weights_matrix", format_matrix(dynamic_weights_matrix))

    update_dynamic_weights(matrix2, matrix1, [matrix1, matrix2], 1.1)
    test_result(estimated_remaining_iterations_matrix, np.array([[2, 4, 102, 2, 1, 4, 102, 2]]))
    test_result(dynamic_weights_matrix, np.array([[1.0, 1.0, 1.0, 1.0, 1.0/1.1/1.1, 1.0, 1.0*1.1, 1.0*1.1]]))
    if verbose:
        print("=============")
        print("estimated_remaining_iterations_matrix", format_matrix(estimated_remaining_iterations_matrix))
        print("dynamic_weights_matrix", format_matrix(dynamic_weights_matrix))

    update_dynamic_weights(matrix1, matrix1, [matrix1, matrix2], 1.1)
    test_result(estimated_remaining_iterations_matrix, np.array([[2, 4, 103, 3, 1, 4, 103, 3]]))
    test_result(dynamic_weights_matrix, np.array([[1.0, 1.0, 1.0*1.1, 1.0, 1.0/1.1/1.1/1.1, 1.0, 1.0*1.1, 1.0*1.1]]))
    if verbose:
        print("=============")
        print("estimated_remaining_iterations_matrix", format_matrix(estimated_remaining_iterations_matrix))
        print("dynamic_weights_matrix", format_matrix(dynamic_weights_matrix))

    print("test result is OK")

if __name__ == "__main__":
    self_test()