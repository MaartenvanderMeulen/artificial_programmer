# Inspired by https://deap.readthedocs.io/en/master/examples/gp_symbreg.html
import sys
import operator
import numpy
import copy
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp
import interpret
import evaluate
import numpy as np


def evaluate_individual(toolbox, individual):
    program_str = str(individual)
    #print("DEBUG 18:", program_str)
    program = interpret.compile_deap(program_str, toolbox.functions)
    weighted_error = 0.0 # evaluate.evaluate_program(program_str, toolbox.hints)
    for example in toolbox.examples:
        input, expected_output = example
        variables = interpret.bind_params(toolbox.input_labels, input)
        model_output = interpret.run(program, variables, toolbox.functions)
        weighted_error += evaluate.evaluate(model_output, expected_output)
    return weighted_error,


def f():
    '''Dummy function for DEAP'''
    return None


def initialize_genetic_programming_toolbox(examples, input_labels, functions):
    pset = gp.PrimitiveSet("MAIN", len(input_labels))
    # print("DEBUG 36", input_labels)
    if len(input_labels) > 0:
        pset.renameArguments(ARG0=input_labels[0])
    if len(input_labels) > 1:
        pset.renameArguments(ARG1=input_labels[1])
    if len(input_labels) > 2:
        pset.renameArguments(ARG2=input_labels[2])
    assert len(input_labels) <= 3

    pset.addTerminal(0)
    pset.addTerminal(1)
    predefined_variables = ["i", "j", "x", "n"]
    for variable in predefined_variables:
        if variable not in input_labels:
            pset.addTerminal(variable)
    for function in interpret.get_build_in_functions():
        param_types = interpret.get_build_in_function_param_types(function)
        arity = sum([1 for t in param_types if t in [1, "v"]])
        pset.addPrimitive(f, arity, name=function)
    for function, (params, code) in functions.items():
        arity = len(params)
        pset.addPrimitive(f, arity, name=function)

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    toolbox.examples = examples
    toolbox.input_labels = input_labels
    toolbox.functions = functions
    toolbox.register("evaluate", evaluate_individual, toolbox)
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
    code_str = str(hof[0])
    code = interpret.compile_deap(code_str, toolbox.functions)
    return code_str, code, hof[0].fitness.values[0]


def get_examples(example_file):
    '''Examples are in a tab delimited file, with the columns '''
    examples = []
    with open(example_file, "r") as f:
        solution = f.readline().rstrip()
        hints = f.readline().rstrip().lower().split("\t")
        hdr = f.readline().rstrip().lower().split("\t")
        input_labels = [label for label in hdr if label not in ["", "output",]]
        input_dimension = len(input_labels)
        for line in f:
            values = [int(s) for s in line.rstrip().lower().split("\t")]
            examples.append((values[:input_dimension], values[input_dimension:]))
    return examples, input_labels


def solve_by_new_function(problem, functions):
    problem_name, examples = problem
    nparams = max([len(example[0]) for example in examples])
    params = ["n", "m", "v", "w", "k",]
    input_labels = copy.copy(params[:nparams])
    while len(input_labels) < nparams:
        input_labels.append(f"n{len(input_labels)}")
    code = main(examples, input_labels, functions)
    if not code:
        return None
    return ["function", problem_name, input_labels, code]


def main(examples, input_labels, functions):
    toolbox = initialize_genetic_programming_toolbox(examples, input_labels, functions)
    hops, pop_size, generations = 20, 300, 20
    # print(f"DEBUG 124 : hops={hops}, pop_size={pop_size}, generations={generations}, units={hops*pop_size*generations}")
    for hop in range(hops):
        code_str, code, error = calc_ai(toolbox, pop_size, generations)
        # print(f"DEBUG 127 : hop {hop+1}, error {error:.3f}: deap_str({code_str})")
        if error == 0:
            return code
    return None


if __name__ == "__main__":
    examples_file = sys.argv[1] if len(sys.argv) > 1 else "easy.txt"
    examples, input_labels = get_examples(examples_file)
    main(examples, input_labels, dict())