// Call "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build\vcvars64.bat"
// Cl /nologo /MP /W2 /EHsc  /D NDEBUG /O2 /Ob1 /arch:AVX2 /MT *.cpp

#include <stdio.h>
#include <vector>


class Helper {
public:
    int n;
    int magic_number;
    std::vector<std::vector<int>> board;
    std::vector<int> numbers;
    std::vector<int> is_free;
    std::vector<int> sum_row;
    std::vector<int> sum_col;
    int sum_diag1, sum_diag2;
    
    Helper(int n) {
        this->n = n;
        this->magic_number = (n * (n*n + 1) / 2);
        this->board.resize(n);
        for (int i = 0; i < n; ++i) {
            this->board[i].assign(n, 0);
        }
        this->numbers.resize(n*n);
        if (n != 4) {
            for (int i = 0; i < n*n; ++i) {
                this->numbers[i] = i+1;
            }
        } else {
            this->numbers = {
                 1, 8, 13, 12,
                 15, 10, 3, 6,
                 4, 5, 16, 9,
                 14, 11, 2, 7};
        }
        this->is_free.assign(n*n, 1);
        this->sum_row.assign(n, 0);
        this->sum_col.assign(n, 0);
        this->sum_diag1 = 0;
        this->sum_diag2 = 0;
    }
    
    bool are_sums_within_threshold(int row, int col, int number) {
        if (col < this->n - 1) {
            if ((this->sum_row[row] + number) >= this->magic_number) {            
                return false;
            }
        } else {
            if ((this->sum_row[row] + number) != this->magic_number) {            
                return false;
            }
        }
        if (row < this->n-1) {
            if ((this->sum_col[col] + number) >= this->magic_number) {
                return false;
            }
        } else {
            if ((this->sum_col[col] + number) != this->magic_number) {
                return false;
            }
        }
        if (row == col) {
            if (this->sum_diag1 > this->magic_number) {
                return false;
            }
            /*
            if (col < this->n - 1) {
                if (this->sum_diag1 >= this->magic_number) {
                    return false;
                }
            } else {
                if (this->sum_diag1 != this->magic_number) {
                    return false;
                }
            }
            */
        }
        if (this->n-1 - row == col) {
            if (this->sum_diag2 > this->magic_number) {
                return false;
            }
            /*
            if (row < this->n - 1) {
                if (this->sum_diag2 >= this->magic_number) {
                    return false;
                }
            } else {
                if (this->sum_diag2 != this->magic_number) {
                    return false;
                }
            }
            */
        }
        return true;
    }

    void print_board() {
        for (int row = 0; row < this->n; ++row) {
            for (int col = 0; col < this->n; ++col) {
                printf(" %2d", this->board[row][col]);
            }
            printf("\n");
        }            
        printf("=========\n");
    }
    int magic_squares_impl(int location) {
        if (location == this->n*this->n) {
            if (this->sum_diag1 == this->magic_number && this->sum_diag2 == this->magic_number) {
                printf(".");
                //this->print_board();
                return 1;
            }
            printf("o");
            return 0;
        }
        int result = 0;
        int row = location / n, col = location % n;
        for (int i = 0; i < n*n; ++i) {
            if (this->is_free[i]) {
                int number = this->numbers[i];
                if (this->are_sums_within_threshold(row, col, number)) {
                    this->is_free[i] = 0;
                    this->board[row][col] = number;
                    this->sum_row[row] += number;
                    this->sum_col[col] += number;
                    if (row == col) {
                        this->sum_diag1 += number;
                    }
                    if (this->n - 1 - row == col) {
                        this->sum_diag2 += number;
                    }
                    result += this->magic_squares_impl(location+1);
                    this->sum_row[row] -= number;
                    this->sum_col[col] -= number;
                    if (row == col) {
                        this->sum_diag1 -= number;
                    }
                    if (this->n - 1 - row == col) {
                        this->sum_diag2 -= number;
                    }
                    this->is_free[i] = 1;
                }
            }
        }
        return result;
    }
};
    

int magic_squares(int n) {
    Helper helper(n);
    return helper.magic_squares_impl(0);
}


int main(int argc, char* argv[]) {
    auto n = (argc > 1 ? atoi(argv[1]) : 3);
    auto result = magic_squares(n);
    printf("%d\n", result);
}
