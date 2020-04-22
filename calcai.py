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
        for example in toolbox.examples:
            input, expected_output = example
            memory = apl.bind_example(toolbox.input_labels, input)
            model_output = apl.run(program, memory)        
            weighted_error += apl.evaluate_output(model_output, expected_output)
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
        expected = "cons(setq('x', 1), cons(for1('i', 'n', setq('x', mul(_identifier2integer('x'), _identifier2integer('i')))), _print(_identifier2integer('x'))))"
        len_equal = 0
        for i in range(min(len(program_str), len(expected))):
            if program_str[i] != expected[i]:
                break
            len_equal += 1
        weighted_error = (len(expected) - len_equal) ** 1
    global best_error 
    if best_error is None or best_error > weighted_error:
        best_error = weighted_error
        if True:
            print("evaluate, program_str", program_str, "weighted_error", weighted_error)
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
 

def for1(i, n, statement):
    return 0


def setq(x, value):
    return 0


def _print(value):
    return 0


def cons(list1, list2):
    return List(list1.v + list2.v)
    

def _empty_list():
    return List([])
    

def _identifier2identifier(identifier):
    return identifier
    
    
def _integer2integer(integer):
    return integer
    
    
def _identifier2integer(identifier):
    return int(0)
    
    
def _identifier2element(identifier):
    return int(0)
    
    
def _integer2element(integer):
    return Element(integer)
    
    
def mul(a, b):
    return a * b
    
    
def initialize_genetic_programming_toolbox(examples, input_labels, len_solution, hints):
    pset = gp.PrimitiveSetTyped("MAIN", [int], int) # DEAP's main dimension doesn't matter because the programs are not evaluated by DEAP
    pset.renameArguments(ARG0='n') # NOTE: make sure arity of MAIN is matched here
    
    # str
    pset.addPrimitive(_identifier2identifier, [str], str, name="_identifier2identifier" ) # dummy operator having a Identifier as result
    predefined_variables = ["i", "j", "x", "n"]
    for variable in predefined_variables:
        pset.addTerminal(variable, str)
    for label in input_labels:
        if label not in predefined_variables:
            pset.addTerminal(label, str)
            assert label.isidentifier()

    # int
    pset.addTerminal(1, int)
    pset.addPrimitive(_identifier2integer, [str], int, name="_identifier2integer" )    
    pset.addPrimitive(mul, [int, int], int, name="mul" )
    pset.addPrimitive(_print, [int], int, name="_print" )
    pset.addPrimitive(for1, [str, str, int], int, name="for1")
    pset.addPrimitive(setq, [str, int], int, name="setq")
    pset.addPrimitive(cons, [int, int], int, name="cons" )
        
    
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