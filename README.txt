Experimenten met het genereren van code

Solving problems, layer 1 ...
problem get_col solved after 87853 evaluations by
    (function get_col (board col) (for row board (at row col)))
problem get_diag1 solved after 90900 evaluations by
    (function get_diag1 (board) (for i (len board) (at board i i)))
problem get_diag2 solved after 670206 evaluations by
    (function get_diag2 (board) (for i 3 (sub (at board i (sub (sub 3 3) (sub i 2))) (for i 2 2))))
problem get_magic_number_n solved after 1971042 evaluations by
    (function get_magic_number_n (n) (div (mul (add (div n n) (mul n n)) n) (div 2 1)))
problem are_all_equal_to solved after 3117 evaluations by
    (function are_all_equal_to (values x) (eq values (for values values x)))

Solving problems, layer 2 ...
problem get_row_sums solved after 181 evaluations by
    (function get_row_sums (board) (for row (for row board row) (sum row)))
problem get_col_sums solved after 49966 evaluations by
    (function get_col_sums (board) (for i 3 (sum (get_col board i))))
problem get_diag_sums solved after 89494 evaluations by
    (function get_diag_sums (board) (list (sum (get_diag1 board)) (sum (get_diag2 board))))
problem get_magic_number solved after 14 evaluations by
    (function get_magic_number (board) (get_magic_number_n (len board)))

Solving problems, layer 3 ...
problem get_sums_magic_square solved after 6051 evaluations by
    (function get_sums_magic_square (board) (add (add (get_row_sums board) (get_col_sums board)) (get_diag_sums board)))

Solving problems, layer 4 ...
problem is_magic_square solved after 46 evaluations by
    (function is_magic_square (board) (are_all_equal_to (get_sums_magic_square board) (get_magic_number board)))

total execution time 1651 seconds

