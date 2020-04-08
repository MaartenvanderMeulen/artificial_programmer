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


best_error = 1e9

def evaluate(toolbox, individual):    
    program_str = str(individual)
    program = apl.compile_deap(program_str)    
    accumulated_evaluation = []
    for example in toolbox.examples:
        input, expected_output = example
        memory = apl.bind_example(toolbox.input_labels, input)
        model_output = apl.run(program, memory)        
        accumulated_evaluation.append(apl.evaluate_output(model_output, expected_output))
        # print("    ", accumulated_evaluation[-1], model_output)
    error = apl.convert_accumulated_evaluation_into_error(accumulated_evaluation)
    if False:
        global best_error 
        if best_error > error:
            best_error = error
            print(error, str(program))
            if False:
                for example in toolbox.examples:
                    input, expected_output = example
                    memory = apl.bind_example(toolbox.input_labels, input)
                    model_output = apl.run(program, memory)        
                    e = apl.evaluate_output(model_output, expected_output, True)
                    print("    ", e, model_output)
    penalty = (len(str(program)) - toolbox.len_solution) / 1000 # Penalty for solutions that are longer than needed
    if penalty < 0:
        penalty = 0 
    error += penalty
    return error,


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
    
    
def initialize_genetic_programming_toolbox(examples, input_labels, len_solution):
    pset = gp.PrimitiveSetTyped("MAIN", [], Element) # DEAP's main dimension doesn't matter because the programs are not evaluated by DEAP
    
    # Element
    pset.addPrimitive(for1, [str, int, Element], Element, name="for1")
    pset.addPrimitive(setq, [str, Element], Element, name="setq")
    pset.addPrimitive(_print, [Element], Element, name="_print" )
    pset.addPrimitive(cons, [Element, Element], Element, name="cons" )
    pset.addPrimitive(_identifier2element, [str], Element, name="_identifier2element" )
    pset.addPrimitive(_integer2element, [int], Element, name="_integer2element" )
    pset.addTerminal(_empty_list, Element, name="_empty_list()" )

    # str
    pset.addPrimitive(_identifier2identifier, [str], str, name="_identifier2identifier" ) # dummy operator having a Identifier as result
    predefined_variables = ["i", "j", "x", "n", ]
    for variable in predefined_variables:
        pset.addTerminal(variable, str)
    for label in input_labels:
        if label not in predefined_variables:
            pset.addTerminal(label, str)

    # int
    pset.addTerminal(1, int)
    pset.addPrimitive(_integer2integer, [int], int, name="_integer2integer" )    
    pset.addPrimitive(_identifier2integer, [str], int, name="_identifier2integer" )    
        
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
    examples, input_labels, len_solution = apl.get_examples(examples_file)
    toolbox = initialize_genetic_programming_toolbox(examples, input_labels, len_solution)
    hops, pop_size, generations = 20, 600, 200
    print(f"hops={hops}, pop_size={pop_size}, generations={generations}, units={hops*pop_size*generations}")
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