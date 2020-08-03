import sys
import numpy as np


class Helper:
    def are_sums_within_threshold(self, row, col):
        if np.sum(self.board[row, :]) > self.magic_number:
            return False
        if np.sum(self.board[:, col]) > self.magic_number:
            return False
        if row == col:
            if sum([self.board[i, i] for i in range(self.n)]) > self.magic_number:
                return False
        if self.n-1 - row == col:
            if sum([self.board[self.n-1 - i, i] for i in range(self.n)]) > self.magic_number:
                return False
        return True


    def magic_squares_impl(self, location):
        if location == self.n*self.n:
            diag1 = sum([self.board[i, i] for i in range(self.n)])
            diag2 = sum([self.board[n-1 - i, i] for i in range(self.n)])
            if diag1 == self.magic_number and diag2 == self.magic_number:
                print(self.board)
                return 1
            return 0
        result = 0
        row, col = location // n, location % n
        for i in range(n*n):
            if self.is_free[i]:
                self.is_free[i] = False
                number = self.numbers[i]
                self.board[row, col] = number
                if self.are_sums_within_threshold(row, col):
                    result += self.magic_squares_impl(location+1)
                self.is_free[i] = True
                
        self.board[row, col] = 0
        return result
    

def magic_squares(n):
    helper = Helper()    
    helper.n = n
    helper.magic_number = (n * (n*n + 1) // 2)
    helper.board = np.zeros((n, n), dtype="int")
    if n != 4:
        helper.numbers = [i for i in range(1, n*n + 1)]
    else:
        helper.numbers = [
             1, 8, 13, 12,
            15, 10, 3, 6,
             4, 5, 16, 9,
            14, 11, 2, 7]
    helper.is_free = [True for i in range(1, n*n + 1)]
    remaining_numbers = {i for i in range(1, n*n + 1)}
    return helper.magic_squares_impl(0)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    result = magic_squares(n)
    print(result)
