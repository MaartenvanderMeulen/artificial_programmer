import sys
import math
import calcai


def tokenize(line):
    line = line.lower()
    for sep in ["(", ")", ",", "-", "+", "*", "/", "#"]:
        line = line.replace(sep, " " + sep + " ")
    return [token for token in line.split(" ") if token != '']


class PrefixParser:
    def __init__(self):
        self.end_of_line = ""

    def _first_token(self):
        if len(self.tokens) > 0:
            self.i = 0
            self.token = self.tokens[self.i]
        else:
            self.token = self.end_of_line

    def _next_token(self):
        if len(self.tokens) > self.i + 1:
            self.i += 1
            self.token = self.tokens[self.i]
        else:
            self.token = self.end_of_line

    def _expect_token(self, expected):
        if self.token != expected:
            raise RuntimeError(f"'{expected}' expected instead of '{self.token}'")
        self._next_token()

    def _parse_end_of_line(self):
        if self.token != self.end_of_line:
            raise RuntimeError(f"end of line expected instead of '{self.token}'")

    def _parse_formula(self):
        # print("_parse_formula start", self.token)
        if self.token.isdigit() or self.token in ["a", "b", "c", "zero", "one", "ten"]: # positive number or ABC
            if self.token == "zero":
                result = "0"
            elif self.token == "one":
                result = "1"
            elif self.token == "ten":
                result = "10"
            else:
                result = self.token
            self._next_token()
        elif self.token == "-": # negative number
            self._next_token()
            if not self.token.isdigit():
                raise RuntimeError(f"number expectd after '-' instead of '{self.token}'")
            result = "-" + self.token
            self._next_token()
        else:
            # formula(arg1, ...argn)
            result = [self.token]
            self._next_token()
            self._expect_token("(")
            if self.token != ")":
                while True:
                    result.append(self._parse_formula())
                    if self.token == ",":
                        self._next_token()
                    else:
                        break
            self._expect_token(")")
        # print("_parse_formula result", result, "next token", self.token)           
        return result

    def parse_line(self, line):
        # print(line)
        self.tokens = tokenize(line)
        self._first_token()
        # print(self.tokens)
        if self.token == "#" or self.token == self.end_of_line:
            return None
        formula = self._parse_formula()
        self._expect_token(self.end_of_line)
        return formula


def prefix_to_infix(formula):
    if type(formula) == type([]):
        if formula[0] == "protected_sqrt":
            infix1 = prefix_to_infix(formula[1])
            return "sqrt(" + infix1 + ")"
        else:
            if len(formula) != 3:
                print(formula)
                
            assert len(formula) == 3 # [operator arg1 arg2]
            infix1 = prefix_to_infix(formula[1])
            infix2 = prefix_to_infix(formula[2])
            if formula[0] == "sub" and infix1 == infix2:
                return "0"
            if formula[0] == "mul" and (infix1 == "0" or infix2 == "0"):
                return "0"
            if formula[0] == "protected_div" and infix1 == infix2:
                return "1"
            if formula[0] == "add" and infix1 == "0":
                return infix2
            if formula[0] == "add" and infix2 == "0":
                return infix1
            if formula[0] == "sub" and infix2 == "0":
                return infix1
            if formula[0] == "mul" and infix1 == "1":
                return infix2
            if formula[0] == "mul" and infix2 == "1":
                return infix1
            operator = {"add":" + ", "sub":" - ", "mul":"*", "protected_div":"/", "protected_power":"^"}[formula[0]]
            return "(" + infix1 + operator + infix2 + ")"
    else:
        assert type(formula) == type("")
        return formula


def read_examples(example_file):
    examples = []
    with open(example_file, "r") as f:
        hdr = f.readline()
        for line in f:
            x0, x1, x2, y = (float(s) for s in line.split("\t"))
            examples.append(((x0, x1, x2), y))
    # print(f"{len(examples)} examples, the last is", examples[-1])
    return examples
    
    
def evaluate(formula, a, b, c):
    if type(formula) == type([]):
        if formula[0] == "protected_sqrt":
            arg1 = evaluate(formula[1], a, b, c)
            return calcai.protected_sqrt(arg1)
        else:
            assert len(formula) == 3 # [operator arg1 arg2]
            arg1 = evaluate(formula[1], a, b, c)
            arg2 = evaluate(formula[2], a, b, c)
            if formula[0] == "add":
                return arg1 + arg2
            if formula[0] == "sub":
                return arg1 - arg2
            if formula[0] == "mul":
                return arg1 * arg2
            if formula[0] == "protected_div":
                return calcai.protected_div(arg1, arg2)
            if formula[0] == "protected_power":
                return calcai.protected_power(arg1, arg2)
            raise RuntimeError(f"prefix2infix.py line 150: onbekende operator '{formula[0]}'")
    else:
        assert type(formula) == type("")
        if formula == "a":
            return a
        if formula == "b":
            return b
        if formula == "c":
            return c
        return float(formula)


def compute_rmse(formula, examples):
    se = [(evaluate(formula, x[0], x[1], x[2]) - y)**2 for x, y in examples]
    rmse = math.sqrt(sum(se) / len(examples))
    return rmse
    
    


def convert_prefix_to_infix(f, examples):
    line_parser = PrefixParser()
    line_nr = 0
    for line in f:
        line_nr += 1
        try:
            formula = line_parser.parse_line(line.strip())
            if formula is not None:
                rmse = compute_rmse(formula, examples)
                infix = prefix_to_infix(formula)
                print(rmse, infix)
        except RuntimeError as e:
            print(f"line {line_nr}: {str(e)}")
    

if __name__ == "__main__":
    examples = read_examples("medium.txt")
    with open("tmp_log_medium_long.txt", "r") as f:
        convert_prefix_to_infix(f, examples)

