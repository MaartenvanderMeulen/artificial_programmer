Artificial programmer.  The AI should write the program that anwers the problem specification.

Run: python calcai.py easy.txt
Run: python calcai.py medium.txt
Run: python calcai.py hard.txt

====================
The solution to the easy problem is: "for1 i n (_print i))"
The solution to the medium problem is: "((setq x 1) (for1 i n (setq x (mul x i))) (_print x))"
The solution to the hard problem is: "(for1 i n ((setq x 1) (for1 j i (setq x (mul x j))) (_print x)))"
