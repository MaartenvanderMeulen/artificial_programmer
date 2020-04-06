# Inspired by https://deap.readthedocs.io/en/master/examples/gp_symbreg.html
# That example has extensive explanations.
import sys
import operator
import math
import random
import numpy
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp
import prefix2infix

def protected_div(left, right):
    '''division operator that has a normal result when dividing by zero'''
    try:
        return float(left / right)
    except:
        return 0 
        
        
def protected_sqrt(x):
    '''sqrt operator that has a normal result when sqrt negative numbers'''
    try:
        return float(math.sqrt(x)) if x >= 0 else 0.0
    except:
        return 1e17
    
        
def protected_power(left, right):
    '''power operator that has a normal result when left is a negative numbers'''
    try:
        return float(left ** right) if left > 0 else 0.0
    except:
        return 1e17
    
        
def protected_sqr(x):
    '''sqr operator that has a normal result when overflowing'''
    try:
        return float(x ** 2)
    except:
        return 1e17
    
        
def rmse(toolbox, individual):
    # Transform the tree expression in a callable function
    model = toolbox.compile(expr=individual)
    # Evaluate with the root of mean squared error (RMSE)
    se = []
    for x, y in toolbox.examples:
        try:
            se.append(protected_sqr(model(x[0], x[1], x[2]) - y))
        except:
            se.append(1e17)
    rmse = math.sqrt(sum(se) / len(toolbox.examples))
    return rmse


evaluate_count = 0

def evaluate(toolbox, individual):    
    global evaluate_count
    evaluate_count += 1
    penalty = len(str(individual)) / 1000 # Penalty for solutions that are longer than needed
    return rmse(toolbox, individual) + penalty,


def get_examples():
    examples = []
    hdr = sys.stdin.readline()
    for line in sys.stdin:
        x0, x1, x2, y = (float(s) for s in line.split("\t"))
        examples.append(((x0, x1, x2), y))
    # print(f"{len(examples)} examples, the last is", examples[-1])
    return examples


def initialize_genetic_programming_toolbox(examples):
    pset = gp.PrimitiveSet("MAIN", 3)
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(protected_div, 2)
    pset.addPrimitive(protected_sqrt, 1)
    pset.addPrimitive(protected_power, 2)
    # pset.addPrimitive(operator.neg, 1)
    # pset.addPrimitive(math.cos, 1)
    # pset.addPrimitive(math.sin, 1)
    pset.addEphemeralConstant("randdigit", lambda: random.randint(2,9))
    pset.addTerminal(0.0, name="zero")
    pset.addTerminal(1.0, name="one")
    pset.addTerminal(10.0, name="ten")
    
    pset.renameArguments(ARG0='A')
    pset.renameArguments(ARG1='B')
    pset.renameArguments(ARG2='C')
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    toolbox.examples = examples
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
    _, _ = algorithms.eaSimple(pop, toolbox, 0.5, 0.1, generations, stats=mstats,
                                   halloffame=hof, verbose=False)
    return hof[0]


def main():
    # random.seed(318)
    examples = get_examples()
    toolbox = initialize_genetic_programming_toolbox(examples)
    prefix_parser = prefix2infix.PrefixParser()
    hops, pop_size, generations = 1000, 600, 200
    print(f"hops={hops}, pop_size={pop_size}, generations={generations}, units={hops*pop_size*generations}")
    best_error = 1e19
    for hop in range(hops):
        solution = calc_ai(toolbox, pop_size, generations)
        formula = prefix_parser.parse_line(str(solution))
        solution_str = prefix2infix.prefix_to_infix(formula)
        error = rmse(toolbox, solution)
        error2 = prefix2infix.compute_rmse(formula, examples)
        assert math.isclose(error, error2)
        if best_error > error:
            best_error = error
            print(f"hop {hop+1}, error {error:.3f}: {solution_str}")


if __name__ == "__main__":
    main()