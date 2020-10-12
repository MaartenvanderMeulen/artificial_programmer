# Inspired by https://deap.readthedocs.io/en/master/examples/gp_symbreg.html
# it is used by search.py
import sys
import operator
import numpy
import random
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp
import interpret
import evaluate
import numpy as np
import local_search


global one_time_initialisation, eval_count
one_time_initialisation = True
eval_count = 0


def evaluate_individual(toolbox, individual):
    global eval_count
    eval_count += 1
    deap_str = str(individual)
    code = interpret.compile_deap(deap_str, toolbox.functions)
    code_str = interpret.convert_code_to_str(code)
    if False: # debug
        print("type(individual)", type(individual))
        print("deap_str", deap_str)
        print("code", code)
        print("code_str", code_str)
    if toolbox.monkey_mode: # if the solution can be found in monkey mode, the real search could in theory find it also
        weighted_error = evaluate.evaluate_code(code_str, toolbox.solution_code_str)
        if weighted_error == 0.0:
            print("check the evaluation of the solution (should be error 0)")
            for input in toolbox.example_inputs:
                variables = interpret.bind_params(toolbox.formal_params, input)
                model_output = interpret.run(code, variables, toolbox.functions)
                _ = evaluate.evaluate(input, model_output, toolbox.evaluation_functions, True)
    else:
        weighted_error = 0.0
        for input in toolbox.example_inputs:
            variables = interpret.bind_params(toolbox.formal_params, input)
            model_output = interpret.run(code, variables, toolbox.functions)
            weighted_error += evaluate.evaluate(input, model_output, toolbox.evaluation_functions, False)
    return weighted_error,


def f():
    '''Dummy function for DEAP'''
    return None


def initialize_genetic_programming_toolbox(problem, functions):
    problem_name, formal_params, example_inputs, evaluation_functions, hints, layer = problem
    int_hints, var_hints, func_hints, solution_hints = hints
    pset = gp.PrimitiveSet("MAIN", len(formal_params))
    for i, param in enumerate(formal_params):
        rename_cmd = f'pset.renameArguments(ARG{i}="{param}")'
        # print("DEBUG 62 rename_cmd", rename_cmd)
        eval(rename_cmd)
    for c in int_hints:
        pset.addTerminal(c)
    for variable in var_hints:
        if variable not in formal_params:
            # print("DEBUG 68 variable", variable)
            pset.addTerminal(variable)
    for function in interpret.get_build_in_functions():
        if function in func_hints:
            param_types = interpret.get_build_in_function_param_types(function)
            arity = sum([1 for t in param_types if t in [1, "v", []]])
            pset.addPrimitive(f, arity, name=function)
    for function, (params, code) in functions.items():
        if function in func_hints:
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
    
    toolbox.formal_params = formal_params
    toolbox.example_inputs = example_inputs
    toolbox.evaluation_functions = evaluation_functions
    toolbox.functions = functions
    toolbox.pset = pset
    toolbox.solution_code_str = interpret.convert_code_to_str(solution_hints)
    
    if False:
        toolbox.register("select", tools.selTournament, tournsize=3)
    else:
        toolbox.register("select", tools.selBest)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=6))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=6))
    return toolbox


def add_if_unique(toolbox, ind, individuals):
    deap_str = str(ind)
    if len(toolbox.ind_str_set) >= toolbox.pop_size + toolbox.nchildren and deap_str in toolbox.ind_str_set:
        return False
    toolbox.ind_str_set.add(deap_str)
    individuals.append(ind)
    ind.fitness.values = evaluate_individual(toolbox, ind)
    return True


def generate_initial_population(toolbox):
    evaluate.init_dynamic_error_weight_adjustment()        
    toolbox.ind_str_set = set()
    population = []
    while len(population) < toolbox.pop_size:
        ind = toolbox.population(n=1)[0]
        if False:
            if add_if_unique(toolbox, ind, population) and ind.fitness.values[0] == 0.0:
                return None, ind
        if True:
            ind = local_search.local_search(toolbox, ind)
            if add_if_unique(toolbox, ind, population) and ind.fitness.values[0] == 0.0:
                return None, ind
    return population, None


def generate_offspring(toolbox, population):
    offspring = []
    while len(offspring) < toolbox.nchildren:
        op_choice = random.random()
        if op_choice < toolbox.pcrossover: # Apply crossover
            ind, ind2 = list(map(toolbox.clone, random.sample(population, 2)))
            ind, ind2 = toolbox.mate(ind, ind2)
        else: # Apply mutation
            ind = toolbox.clone(random.choice(population))
            ind, = toolbox.mutate(ind)
        if False:
            if add_if_unique(toolbox, ind, offspring) and ind.fitness.values[0] == 0.0:
                return None, ind
        if True:
            ind = local_search.local_search(toolbox, ind)
            if add_if_unique(toolbox, ind, offspring) and ind.fitness.values[0] == 0.0:
                return None, ind
    return offspring, None


def update_dynamic_weighted_evaluation(toolbox, individuals):
    if toolbox.dynamic_weights:
        evaluate.dynamic_error_weight_adjustment()        
        for ind in individuals:        
            ind.fitness.values = evaluate_individual(toolbox, ind)


def ga_seach_impl(toolbox):
    population, solution = generate_initial_population(toolbox)
    if solution:
        return solution, 0
    update_dynamic_weighted_evaluation(toolbox, population)
    for gen in range(toolbox.ngen):
        offspring, solution = generate_offspring(toolbox, population)
        if solution:
            return solution, gen+1
        population[:] = toolbox.select(population + offspring, toolbox.pop_size)
        update_dynamic_weighted_evaluation(toolbox, population)
    best = min(population, key=lambda item: item.fitness.values[0])
    return best, toolbox.ngen+1
    
    
def ga_search(toolbox):
    best, gen = ga_seach_impl(toolbox)
    code = interpret.compile_deap(str(best), toolbox.functions)
    code_str = interpret.convert_code_to_str(code)
    error = best.fitness.values[0]
    return code, code_str, error, gen


def solve_by_new_function(problem, functions):
    problem_name, params, example_inputs, evaluation_functions, hints, layer = problem
    toolbox = initialize_genetic_programming_toolbox(problem, functions)
    toolbox.monkey_mode = False
    toolbox.dynamic_weights = False # not toolbox.monkey_mode
    hops = 100
    result = None
    # for nchildren in [int(pop_size * 0.25), int(pop_size * 0.5), int(pop_size * 0.75), int(pop_size * 1.0)]:
    # for ngen in [20, 30, 40, 50, 60, 70, 80, 90, 100]:
    # for cxpb in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    # for pop_size in [300, 400, 500, 600, 700, 800, 900, 1000]:
    if True:
        toolbox.pop_size, toolbox.pcrossover, toolbox.ngen = 300, 0.5, 30
        toolbox.nchildren, toolbox.pmutations = toolbox.pop_size, 1.0 - toolbox.pcrossover
        global eval_count
        eval_count = 0
        for hop in range(hops):
            code, code_str, error, gen = ga_search(toolbox)        
            if error == 0:
                result = ["function", problem_name, params, code]
                result_str = interpret.convert_code_to_str(result)
                print("problem", problem_name, f"solved after {eval_count} evaluations by", result_str)
                break
            if False:
                print(f"hop {hop+1}, error {error:.3f}: {code_str}")
                if toolbox.monkey_mode:
                    print(f"    {code_str}")
                    print(f"    {toolbox.solution_code_str}")
    return result
