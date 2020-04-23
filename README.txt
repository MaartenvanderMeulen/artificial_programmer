Artificial programmer.  The AI should write the program that anwers the problem specification.

Run: python calcai.py easy.txt
Run: python calcai.py medium.txt
Run: python calcai.py hard.txt

====================
The solution to the easy problem is: "(for1 i n (print i)))"
The solution to the medium problem is: "((setq x 1) (for1 i n (setq x (mul x i))) (_print x))" # for1 returns last elem or list with elem per iter
                                       "(apply mul (for1 i n i))" # for1 returns list with elem per iter
                                       "(mul (for1 i n i))" # for1 returns list with elem per iter
The solution to the hard problem is: "(for1 i n ((setq x 1) (for1 j i (setq x (mul x j))) (_print x)))"
====================

TODO : solve medium problem
    * maak dynamic weight adjustrment inclusief de hints
    
   
DEAP style str(program)
mul(for1('j', 'n', for1('i', 'n', cons(for1('n', 'j', n), n))), setq(_identifier2identifier('i'), mul(n, for1('n', _identifier2identifier(_identifier2identifier('i')), n))))
    mul(n, setq(i, mul(n, n)))
mul(for1('j', 'n', for1('i', 'n', cons(for1('n', 'j', n), n))), setq(_identifier2identifier('i'), mul(n, for1('n', _identifier2identifier(_identifier2identifier('i')), n))))
mul(for1('j', 'n', for1('i', 'n', cons(for1('n', 'j', n), n))), setq('i', mul(n, for1('n', 'i', n))))
mul(for1('j', 'n', for1('i', 'n', cons(for1('n', 'j', n), n))), setq('i', mul(n, 'i')))
mul(for1('j', 'n', 'j'), setq('i', mul(n, 'i')))
mul('n', setq('i', mul(n, 'i')))

Lisp style
(mul (range1 n))
(apply mul (range1 n))
(setq x 1) (for1 i n (setq x (mul x i))) (print x)
(for1 i n (print