# layer 0
def get_n(board):
    return len(board)
    
def get_at(board, row, col):
    return board[row][col]
    
    
#layer 1
def get_row(board, row):
    n = get_n(board)
    return [get_at(board, row, col) for col in range(n)]
 
 
def get_col(board, col):
    n = get_n(board)
    return [get_at(board, row, col) for row in range(n)]
 
 
def get_diag1(board):
    n = get_n(board)
    return [get_at(board, i, i) for i in range(n)]
    

def get_diag2(board):
    n = get_n(board)
    return [get_at(board, n-1 - i, i) for i in range(n)]
    

# layer 2
def compute_sums(board):
    n = len(board[0])
    result = [sum(get_row(board, row)) for row in range(n)]
    result.extend([sum(get_col(board, col)) for col in range(n)])
    result.append(sum(get_diag1(board)))
    result.append(sum(get_diag2(board)))
    return result


# layer 3
def is_magic_square(board):
    n = len(board[0])
    magic_number = (n * (n*n + 1) // 2)
    for value in compute_sums(board):
        if value != magic_number:
            return False
    return True
