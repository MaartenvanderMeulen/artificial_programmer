# Inspired by https://deap.readthedocs.io/en/master/examples/gp_symbreg.html
import sys
import operator
# import math
# import random
import numpy
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp
import apl


best_error = None

def evaluate(toolbox, individual):
    program_str = str(individual)
    program = apl.compile_deap(program_str)
    if True: # normal operation
        weighted_error = apl.evaluate_program(program_str, toolbox.hints)
        prev_model_output = None
        for example in toolbox.examples:
            input, expected_output = example
            memory = apl.bind_example(toolbox.input_labels, input)
            model_output = apl.run(program, memory)        
            weighted_error += apl.evaluate_output(model_output, expected_output)
            if False: # for n-faculty problem tests
                weighted_error += apl.evaluate_postcondition(model_output, prev_model_output, input)
            prev_model_output = model_output
            if False:
                print("    ", accumulated_evaluation[-1], model_output)
        # apl.dynamic_error_weight_adjustment()
        if weighted_error == 0:
            # we have the solution! Try to get the shortest solutioh
            penalty = (len(str(program)) - toolbox.len_solution) / 1000 # Penalty for solutions that are longer than needed
            if penalty < 0:
                penalty = 0 
            if penalty > 0.9:
                penalty = 0.9
            weighted_error += penalty
        else:
            weighted_error += 1.0 # so that this weighted_error is always higher than above's penalty for long but correct solutions
    else: # Test if the solution is derivable        
        easy = "for1('i', 'n', _print(_str2element('i')))"
        medium = "apply('mul', for1('i', 'n', _str2element('i')))"
        hard = "for1('i', 'n', _print(apply('mul', for1('j', 'i', _str2element('j')))))"
        expected = hard
        len_equal = 0
        for i in range(min(len(program_str), len(expected))):
            if program_str[i] != expected[i]:
                break
            len_equal += 1
        weighted_error = abs(len(expected) - len_equal) ** 1
    global best_error 
    if best_error is None or best_error > weighted_error:
        best_error = weighted_error
        if True:
            print(program_str, weighted_error)
            if True:
                for example in toolbox.examples:
                    input, expected_output = example
                    memory = apl.bind_example(toolbox.input_labels, input)
                    model_output = apl.run(program, memory)        
                    print(f"          expected output {expected_output}, model_output {model_output}, {len(model_output)}")
    return weighted_error,


'''dummy for DEAP to get typed trees working'''
class Identifier:
    def __init__(self, v):
        self.v = v


class Element:
    def __init__(self, v):
        self.v = v
 

class Function:
    def __init__(self, v):
        self.v = v
 

def for1(i, n, statement):
    return 0


def setq(x, value):
    return 0


def _print(value):
    return 0


def cons(list1, list2):
    return List(list1.v + list2.v)
    

def apply(f1, list1):
    return f1(list1.v)
    

def _int2element(i):
    return [i]
    

def _empty_list():
    return Element([])
    

def _function2function(f):
    return f
    
    
def _str2str(identifier):
    return identifier
    
    
def _str2element(identifier):
    return identifier
    
    
def _int2int(integer):
    return integer
    
    
def _str2int(identifier):
    return int(0)
    
    
def _str2element(identifier):
    return int(0)
    
    
def _int2element(integer):
    return Element(integer)
    
    
def mul(a, b):
    return a * b
    
    
def sub(a, b):
    return a - b
    
    
def add(a, b):
    return a + b
    
    
def div(a, b):
    return a // b
    
    
def initialize_genetic_programming_toolbox(examples, input_labels, len_solution, hints):
    pset = gp.PrimitiveSetTyped("MAIN", [Element], Element) # DEAP's main dimension doesn't matter because the programs are not evaluated by DEAP
    pset.renameArguments(ARG0='n') # NOTE: make sure arity of MAIN is matched here
    
    # str
    pset.addPrimitive(_str2str, [str], str, name="_str2str" ) # dummy operator having a Identifier as result
    predefined_variables = ["i", "j", "x", "n"]
    for variable in predefined_variables:
        pset.addTerminal(variable, str)
    for label in input_labels:
        if label not in predefined_variables:
            pset.addTerminal(label, str)
            assert label.isidentifier()

    # int
    # pset.addTerminal(1, int)
    # pset.addPrimitive(_int2int, [str], int, name="_int2int" )    
    # pset.addPrimitive(_str2int, [str], int, name="_str2int" )    
    
    # function
    pset.addTerminal("mul", Function)
    pset.addPrimitive(_function2function, [Function], Function, name="_function2function" )        
    
    # Element
    pset.addTerminal(1, Element)
    pset.addTerminal(2, Element)
    pset.addTerminal("_empty_list", Element)
    pset.addPrimitive("_element2element", [Element], Element, name="_element2element" )        
    pset.addPrimitive("_str2element", [str], Element, name="_str2element" )        
    # pset.addPrimitive("_int2element", [int], Element, name="_int2element" )        
    pset.addPrimitive(for1, [str, str, Element], Element, name="for1")
    pset.addPrimitive(mul, [Element, Element], Element, name="mul" )
    # pset.addPrimitive(sub, [Element, Element], Element, name="sub" )
    pset.addPrimitive(add, [Element, Element], Element, name="add" )
    pset.addPrimitive(div, [Element, Element], Element, name="div" )
    pset.addPrimitive(_print, [Element], Element, name="_print" )
    pset.addPrimitive(setq, [str, Element], Element, name="setq")
    pset.addPrimitive(cons, [Element, Element], Element, name="cons" )
    pset.addPrimitive(apply, [Function, Element], Element, name="apply" )        
    
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    toolbox.examples = examples
    toolbox.input_labels = input_labels
    toolbox.len_solution = len_solution
    toolbox.hints = hints
    toolbox.register("evaluate", evaluate, toolbox)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=10))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=10))
    return toolbox
    

def calc_ai(toolbox, pop_size, generations):
    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)    
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("min", numpy.min)
    pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.1, generations, stats=mstats,
                                   halloffame=hof, verbose=False)
    program_str = str(hof[0])
    program_str = str(apl.compile_deap(program_str))
    return program_str, hof[0].fitness.values[0]


def main(examples_file):
    examples, input_labels, len_solution, hints = apl.get_examples(examples_file)
    toolbox = initialize_genetic_programming_toolbox(examples, input_labels, len_solution, hints)
    hops, pop_size, generations = 20, 600, 200
    print(f"hops={hops}, pop_size={pop_size}, generations={generations}, units={hops*pop_size*generations}")
    global best_error
    best_error = None
    for hop in range(hops):
        solution_str, error = calc_ai(toolbox, pop_size, generations)
        if best_error is None or best_error > error:
            best_error = error
        print(f"hop {hop+1}, error {error:.3f}: {solution_str}")
        if best_error == 0:
            return


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"at least one argument, the file with the examples, expected")
        exit(2)
    main(sys.argv[1])