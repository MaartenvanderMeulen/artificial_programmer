import numpy as np


is_free = np.ones((10))
board = np.zeros((3, 3), dtype=int)
done = set()



def still_valid(row, col):
    if np.count_nonzero(board[row, :]) == 3 and np.sum(board[row, :]) != 15:
        return False
    if col == 0:
        if np.count_nonzero(board[:, col]) == 3 and np.sum(board[:, col]) != 15:
            return False
    diag1 = [board[0, 0], board[1, 1], board[2, 2]]
    if np.count_nonzero(np.array(diag1)) == 3 and sum(diag1) != 15:
        return False
    if np.sum(board[:, 0]) > 15 or np.sum(board[0, :]) > 15 or np.sum(board[1, :]) > 15 or np.sum(board[2, :]) > 15:
        return False
    if np.sum(board[0, 0] + board[1, 1] + board[2, 2]) > 15:
        return False
    return True


def magic():
    for nr in range(1, 10):
        if is_free[nr]:
            for row in range(3):
                for col in range(3):
                    if board[row, col] == 0:
                        board[row, col] = nr
                        if still_valid(row, col):
                            is_free[nr] = 0
                            magic()
                            is_free[nr] = 1
                        board[row, col] = 0
    if np.sum(is_free[1:]) == 0:
        if np.sum(board[:, 1]) != 15 or np.sum(board[:, 2]) != 15:
            if tuple(board.flatten()) not in done:
                print(board)
                print()
                done.add(tuple(board.flatten()))
            
    elif np.sum(is_free[1:]) >= 7:
        print(np.sum(is_free))


if __name__ == "__main__":
    is_free[0] = 0
    magic()
