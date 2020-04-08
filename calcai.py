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


def evaluate(toolbox, individual):    
    program_str = str(individual)
    program = apl.compile_deap(program_str)
    accumulated_evaluation = []
    for example in toolbox.examples:
        input, expected_output = example
        memory = apl.bind_example(toolbox.input_labels, example)
        model_output = apl.run(program, memory)
        print("    ", memory, model_output)
        accumulated_evaluation.append(apl.evaluate_output(model_output, expected_output))
    error = apl.convert_accumulated_evaluation_into_error(accumulated_evaluation)
    penalty = len(program_str) / 1000 # Penalty for solutions that are longer than needed
    return error + penalty,


def for1(i, n, statement):
    '''dummy for DEAP'''
    return 0


def setq(x, value):
    '''dummy for DEAP'''
    return 0


def _print(value):
    '''dummy for DEAP'''
    return 0


class Identifyer:
    def __init__(self, v):
        self.v = v
 

class Integer:
    def __init__(self, v):
        self.v = v
 

class Statement:
    def __init__(self, v):
        self.v = v
 

class List:
    def __init__(self, v):
        self.v = v
 

def initialize_genetic_programming_toolbox(examples, input_labels):
    pset = gp.PrimitiveSet("MAIN", 1) # DEAP's main dimension doesn't matter because the programs are not evaluated by DEAP
    pset.addPrimitive(for1, 3, name="for1" )
    pset.addPrimitive(setq, 2, name="setq" )
    pset.addPrimitive(_print, 1, name="_print" )
    predefined_variables = ["i", "j", "x", "n", ]
    for variable in predefined_variables:
        pset.addTerminal(variable)
    for label in input_labels:
        if label not in predefined_variables:
            pset.addTerminal(label)
        
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    toolbox.examples = examples
    toolbox.input_labels = input_labels
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
    # print("pop[0]", pop[0], "log[0]", log[[0])
    return str(hof[0]), hof[0].fitness.values


def main(examples_file):
    examples, input_labels = apl.get_examples(examples_file)
    toolbox = initialize_genetic_programming_toolbox(examples, input_labels)
    hops, pop_size, generations = 1000, 600, 200
    print(f"hops={hops}, pop_size={pop_size}, generations={generations}, units={hops*pop_size*generations}")
    best_error = None
    for hop in range(hops):
        solution_str, error = calc_ai(toolbox, pop_size, generations)
        if best_error is None or best_error > error:
            best_error = error
            print(f"hop {hop+1}, error {error:.3f}: {solution_str}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"at least one argument, the file with the examples, expected")
        exit(2)
    main(sys.argv[1])