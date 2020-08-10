# Inspired by https://deap.readthedocs.io/en/master/examples/gp_symbreg.html
import sys
import operator
import numpy
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp
import interpret
import evaluate
import numpy as np


global one_time_initialisation, best_error
one_time_initialisation = True
best_error = 1e9



def evaluate_individual(toolbox, individual):
    program_str = str(individual)
    program = interpret.compile_deap(program_str, toolbox.functions)
    weighted_error = 0.0
    for input in toolbox.example_inputs:
        variables = interpret.bind_params(toolbox.formal_params, input)
        model_output = interpret.run(program, variables, toolbox.functions)
        weighted_error += evaluate.evaluate(input, model_output, toolbox.evaluation_functions, False)
    global best_error
    if best_error > weighted_error:
        best_error = weighted_error
        print("DEBUG GA_SEARCH_DEAP 34 : best error", best_error, interpret.convert_code_to_str(program))
        if False:
            for input in toolbox.example_inputs:
                variables = interpret.bind_params(toolbox.formal_params, input)
                model_output = interpret.run(program, variables, toolbox.functions)
                error = evaluate.evaluate(input, model_output, toolbox.evaluation_functions, False)
    return weighted_error,


def f():
    '''Dummy function for DEAP'''
    return None


def initialize_genetic_programming_toolbox(problem, functions):
    problem_name, formal_params, example_inputs, evaluation_functions, hints = problem
    pset = gp.PrimitiveSet("MAIN", len(formal_params))
    for i, param in enumerate(formal_params):
        rename_cmd = f'pset.renameArguments(ARG{i}="{param}")'
        eval(rename_cmd)
    pset.addTerminal(0)
    pset.addTerminal(1)
    predefined_variables = ["n", "m", "v", "w", "k", "i", "j", "x"]
    for variable in predefined_variables:
        if variable not in formal_params:
            pset.addTerminal(variable)
    for function in interpret.get_build_in_functions():
        if function in hints:
            param_types = interpret.get_build_in_function_param_types(function)
            arity = sum([1 for t in param_types if t in [1, "v"]])
            pset.addPrimitive(f, arity, name=function)
    for function, (params, code) in functions.items():
        if function in hints:
            arity = len(params)
            pset.addPrimitive(f, arity, name=function)
    global one_time_initialisation
    if one_time_initialisation:
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)
        one_time_initialisation = False
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    toolbox.formal_params = formal_params
    toolbox.example_inputs = example_inputs
    toolbox.evaluation_functions = evaluation_functions
    toolbox.functions = functions
    toolbox.register("evaluate", evaluate_individual, toolbox)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=10))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=10))
    return toolbox


def ga_search(toolbox, pop_size, generations):
    pop = toolbox.population(n=pop_size)
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("min", numpy.min)
    cxpb, mutpb = 0.5, 0.1 # p(mating), p(mutation)
    hof = tools.HallOfFame(1)
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb, mutpb, generations, stats=mstats,
                                   halloffame=hof, verbose=False)
    error = hof[0].fitness.values[0]
    code = interpret.compile_deap(str(hof[0]), toolbox.functions)
    code_str = interpret.convert_code_to_str(code)
    return code_str, code, error


def solve_by_new_function(problem, functions):
    problem_name, params, example_inputs, evaluation_functions, hints = problem
    toolbox = initialize_genetic_programming_toolbox(problem, functions)
    hops, pop_size, generations = 20, 500, 100
    print(f"DEBUG GA_SEARCH 131 : hops={hops}, pop_size={pop_size}, generations={generations}")
    global best_error
    best_error = 1e9
    for hop in range(hops):
        code_str, code, error = ga_search(toolbox, pop_size, generations)        
        print(f"DEBUG GA_SEARCH 134 : hop {hop+1}, error {error:.3f}: {code_str}")
        if error == 0:
            return ["function", problem_name, params, code]
        evaluate.dynamic_error_weight_adjustment(False)
    return None


def test_evaluation():
    functions = interpret.get_functions("functions.txt")
    evaluation_functions = [["eval_sums_rows_cols_diags", []]]
    params = ["board", ]
    example_inputs = [
        [[[1, 2, 3], [4, 5, 6], [7, 8, 9]]],
        [[[9, 8, 7], [6, 5, 4], [3, 2, 1]]],
        [[[2, 9, 4], [7, 5, 3], [6, 1, 8]]],
        ]
    print("params", params)
    for program_str in [
            #"(at board col)", 
            #"(for row board (at row 0))",
            #"(for row board (at row col))",
            #"(at board 0)", 
            #"((at board 0 2) (at board 1 1) 2)", 
            #"(for i (len board) (at board i i))",
            #"(for i (len board) (at board i (sub (len board) 1 i)))",
            "(add (for row board (sum row)) (for col (len board) (sum (get_col board col))) ((sum (get_diag1 board))) ((sum (get_diag2 board))))",
            ]:
        print(program_str)
        program = interpret.compile(program_str)
        print(program)
        sum_error = 0
        for input in example_inputs:
            print("    params", params)
            print("    input", input)
            variables = interpret.bind_params(params, input)
            print("    variables", variables)
            model_output = interpret.run(program, variables, functions)
            print("    model_output", model_output)
            error = evaluate.evaluate(input, model_output, evaluation_functions, False)
            print("    error", error)
            sum_error += error
        print("    ", sum_error)


if __name__ == "__main__":
    test_evaluation()
