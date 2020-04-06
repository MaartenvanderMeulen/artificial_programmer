Given a problems specification in text, and with some numeric example inputs and outputs,
the AI should give the formula that anwers the problem specification.

The problem has as answer the formula f for which for each input i f(input(i)) == output(i).
Formula is composed of "+", "-", "*", "/", "(", ")", and possibly also "sin", "cos", and the various x0-xn = input(i).
Example formula's :
    x*x + 2 : list of inputs [[1,], [2,], [3,], ]; list out corresponding outputs [3, 6, 11,]
    1*x0 + 2*x1 + 3*x2 - 4*x0*x1*x2 + 5 

In its simplest form, (no descriptive text), it boils down to symbolic regression analysis.

Run: python calcai.py < easy.txt

====================
The solution to the easy problem is: A + 2B + 3C - ABC
The solution to the medium problem is: A + AB + ABC - C^2 + C/A + 7B/C + sqrt(B)
The solution to the hard problem is: ABC + AB^3 - B/(A+C) + 17
