Experimenten met het genereren van code

======================== zondag 23 Augustus 2020 =================================
    mutp = 0.15, cxp = 0.4
    ngen = 70
    select = selTournament, k=3

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

================================= dinsdag 25 Augustus 2020 ===============================
    mutp = 1 - cxp, cxp = 0.5 (leaving it at mutp = 0.15, cxp = 0.4, dus 0.45% is kopie van oude pop, laat alles 4x langer duren!)
    ngen = 30 (leaving it at 70 takes 2x as much time)
    select = selBest (leaving it at selTournament, k=3, maakt niet veel uit, maar selBest is intuitiever)
    dynamic_weights = False (setting it to True simply doubles the # evaluations)
    
Solving problems, layer 1 ...
problem get_col solved after 3464 evaluations by
    (function get_col (board col) (for board board (at (for board board board) col)))
problem get_diag1 solved after 2186 evaluations by
    (function get_diag1 (board) (for i (len board) (at board i i)))
problem get_diag2 solved after 64137 evaluations by
    (function get_diag2 (board) (for i 3 (sub (at board i (sub (at i board i) (sub i 2))) (for 3 2 (for (for 2 board 2) (for i board board) (for 3 3 2))))))
problem get_magic_number_n solved after 52645 evaluations by
    (function get_magic_number_n (n) (div (mul (mul 1 (add (div 2 2) (mul n n))) n) 2))
problem are_all_equal_to solved after 530 evaluations by
    (function are_all_equal_to (values x) (eq values (for values values x)))

Solving problems, layer 2 ...
problem get_row_sums solved after 527 evaluations by
    (function get_row_sums (board) (for board (for row board (sum row)) board))
problem get_col_sums solved after 6902 evaluations by
    (function get_col_sums (board) (get_col (for i 3 (for (for board board i) board (sum (get_col board i)))) (for (sum i) (for board i board) (get_col board i))))
problem get_diag_sums solved after 4604 evaluations by
    (function get_diag_sums (board) (list (sum (get_diag1 board)) (sum (get_diag2 board))))
problem get_magic_number solved after 32 evaluations by
    (function get_magic_number (board) (get_magic_number_n (len board)))

Solving problems, layer 3 ...
problem get_sums_magic_square ...
problem get_sums_magic_square solved after 2131 evaluations by (function get_sums_magic_square (board) (add (add (get_row_sums board) (get_col_sums board)) (get_diag_sums board)))
Solving problems, layer 4 ...
problem is_magic_square ...
problem is_magic_square solved after 151 evaluations by (function is_magic_square (board) (are_all_equal_to (get_sums_magic_square board) (get_magic_number board)))
total execution time 116 seconds

==================================== wo 26 Aug 2020 ==================================

Idee-en: dubbelen eruit; local search

==================================== ma 28 Sep 2020 ==================================

Solving problems, layer 1 ...
problem get_col ...
problem get_col solved after 2092 evaluations by (function get_col (board col) (for board board (at board col)))
problem get_diag1 ...
problem get_diag1 solved after 2552 evaluations by (function get_diag1 (board) (for i (len board) (at board i i)))
problem get_diag2 ...
problem get_diag2 solved after 6121 evaluations by (function get_diag2 (board) (for i (sub 3 board) (at board i (sub 2 i))))
problem get_magic_number_n ...
problem get_magic_number_n solved after 212828 evaluations by (function get_magic_number_n (n) (add (div (mul n (mul n n)) 2) (mul 1 (div (add 1 n) (mul 2 1)))))
problem are_all_equal_to ...
problem are_all_equal_to solved after 2354 evaluations by (function are_all_equal_to (values x) (eq (for (eq values values) (for value values values) value) values))

Solving problems, layer 2 ...
problem get_row_sums ...
problem get_row_sums solved after 182 evaluations by (function get_row_sums (board) (for board board (sum board)))
problem get_col_sums ...
problem get_col_sums solved after 3964 evaluations by (function get_col_sums (board) (for i 3 (sum (get_col board i))))
problem get_diag_sums ...
problem get_diag_sums solved after 2390 evaluations by (function get_diag_sums (board) (list (sum (get_diag1 board)) (sum (get_diag2 board))))
problem get_magic_number ...
problem get_magic_number solved after 2 evaluations by (function get_magic_number (board) (get_magic_number_n (len board)))

Solving problems, layer 3 ...
problem get_sums_magic_square ...
problem get_sums_magic_square solved after 1653 evaluations by (function get_sums_magic_square (board) (add (get_row_sums board) (add (get_col_sums board) (get_diag_sums board))))

Solving problems, layer 4 ...
problem is_magic_square ...
problem is_magic_square solved after 27 evaluations by (function is_magic_square (board) (are_all_equal_to (get_sums_magic_square board) (get_magic_number board)))

total execution time 157 seconds

234165=2092+2552+6121+212828+2354+182+3964+2390+2+1653+27

==================================== 1e local search exp

Solving problems, layer 1 ...
problem get_col ...
problem get_col solved after 2092 evaluations by (function get_col (board col) (for board board (at board col)))
problem get_diag1 ...
problem get_diag1 solved after 2552 evaluations by (function get_diag1 (board) (for i (len board) (at board i i)))
problem get_diag2 ...
problem get_diag2 solved after 13301 evaluations by (function get_diag2 (board) (for i 3 (at board i (sub 2 i))))
problem get_magic_number_n ...
problem get_magic_number_n solved after 5406 evaluations by (function get_magic_number_n (n) (div (mul (div (add n 0) (div (mul n 4) (mul (add 4 0) 1))) (mul (mul 4 (add (mul n n) (mul 1 1))) n)) (div (mul 8 n) n)))
problem are_all_equal_to ...
problem are_all_equal_to solved after 677 evaluations by (function are_all_equal_to (values x) (eq values (for value values x)))
Solving problems, layer 2 ...

problem get_row_sums ...
problem get_row_sums solved after 787 evaluations by (function get_row_sums (board) (for row board (sum row)))
problem get_col_sums ...

die hij niet opelost krijgt ... 

===================================== als we nu beide toevoegen (de oorsponkelijke + de local search), pop --> 600 =========

Solving problems, layer 1 ...
problem get_col ...
problem get_col solved after 6757 evaluations by (function get_col (board col) (for board board (at board col)))
problem get_diag1 ...
problem get_diag1 solved after 5104 evaluations by (function get_diag1 (board) (for i (len board) (at board i i)))
problem get_diag2 ...
problem get_diag2 solved after 10994 evaluations by (function get_diag2 (board) (for i 3 (at board i (sub 2 i))))
problem get_magic_number_n ...
problem get_magic_number_n solved after 12098 evaluations by (function get_magic_number_n (n) (add (div (mul (mul n n) n) (div 2 1)) (div (add n 1) (mul 2 1))))
problem are_all_equal_to ...
problem are_all_equal_to solved after 259 evaluations by (function are_all_equal_to (values x) (eq (for value values x) values))
Solving problems, layer 2 ...
problem get_row_sums ...
problem get_row_sums solved after 773 evaluations by (function get_row_sums (board) (for board board (sum board)))
problem get_col_sums ...
problem get_col_sums solved after 21297 evaluations by (function get_col_sums (board) (for i (sum (for i board 1)) (sum (get_col board i))))
problem get_diag_sums ...
problem get_diag_sums solved after 9219 evaluations by (function get_diag_sums (board) (list (sum (get_diag1 board)) (sum (get_diag2 board))))
problem get_magic_number ...
problem get_magic_number solved after 17 evaluations by (function get_magic_number (board) (get_magic_number_n (len board)))
Solving problems, layer 3 ...
problem get_sums_magic_square ...
problem get_sums_magic_square solved after 7717 evaluations by (function get_sums_magic_square (board) (add (get_row_sums board) (add (get_col_sums board) (get_diag_sums board))))
Solving problems, layer 4 ...
problem is_magic_square ...
problem is_magic_square solved after 257 evaluations by (function is_magic_square (board) (are_all_equal_to (get_sums_magic_square board) (get_magic_number board)))
total execution time 486 seconds
+
74292=6757+5104+10994+12098+259+773+21297+9219+17+7717+57