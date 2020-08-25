# Inspired by https://deap.readthedocs.io/en/master/examples/gp_symbreg.html
# python search.py functions.txt problem_col.txt
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


global one_time_initialisation, best_error, eval_count
one_time_initialisation = True
best_error = 1e9
eval_count = 0


def evaluate_individual(toolbox, individual):
    global eval_count
    eval_count += 1
    deap_str = str(individual)
    code = interpret.compile_deap(deap_str, toolbox.functions)
    code_str = interpret.convert_code_to_str(code)
    if toolbox.monkey_mode:
        weighted_error = evaluate.evaluate_code(code_str, toolbox.solution_code_str)
    else:
        weighted_error = 0.0
        for input in toolbox.example_inputs:
            variables = interpret.bind_params(toolbox.formal_params, input)
            model_output = interpret.run(code, variables, toolbox.functions)
            weighted_error += evaluate.evaluate(input, model_output, toolbox.evaluation_functions, False)
    global best_error
    if best_error > weighted_error:
        best_error = weighted_error
        #print("DEBUG GA_SEARCH_DEAP 34 : best error", best_error, code_str)
        if toolbox.monkey_mode and best_error == 0.0:
            for input in toolbox.example_inputs:
                variables = interpret.bind_params(toolbox.formal_params, input)
                model_output = interpret.run(code, variables, toolbox.functions)
                error = evaluate.evaluate(input, model_output, toolbox.evaluation_functions, True)
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
        eval(rename_cmd)
    for c in int_hints:
        pset.addTerminal(c)
    for variable in var_hints:
        if variable not in formal_params:
            pset.addTerminal(variable)
    for function in interpret.get_build_in_functions():
        if function in func_hints:
            param_types = interpret.get_build_in_function_param_types(function)
            arity = sum([1 for t in param_types if t in [1, "v"]])
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
    toolbox.register("compile", gp.compile, pset=pset)
    toolbox.formal_params = formal_params
    toolbox.example_inputs = example_inputs
    toolbox.evaluation_functions = evaluation_functions
    toolbox.functions = functions
    toolbox.solution_code_str = interpret.convert_code_to_str(solution_hints)
    toolbox.register("evaluate", evaluate_individual, toolbox)
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


def varOr(population, toolbox, lambda_, cxpb, mutpb):
    offspring = []
    while len(offspring) < lambda_:
        op_choice = random.random()
        if op_choice < cxpb: # Apply crossover
            ind1, ind2 = list(map(toolbox.clone, random.sample(population, 2)))
            ind1, ind2 = toolbox.mate(ind1, ind2)
            del ind1.fitness.values
            offspring.append(ind1)
        else: # Apply mutation
            ind = toolbox.clone(random.choice(population))
            ind, = toolbox.mutate(ind)
            del ind.fitness.values
            offspring.append(ind)
    return offspring


def eaMuPlusLambda(toolbox, mu, lambda_, cxpb, mutpb, ngen):
    '''adapted from DEAP'''
    debug = False
    
    # Generate initial population
    population = toolbox.population(n=mu)

    # Evaluate initial population
    evaluate.init_dynamic_error_weight_adjustment()        
    fitnesses = toolbox.map(toolbox.evaluate, population)
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit
        if debug:
            deap_str = str(ind)
            code = interpret.compile_deap(deap_str, toolbox.functions)
            code_str = interpret.convert_code_to_str(code)
            print("initial", code_str, fit[0])
        if ind.fitness.values[0] == 0.0:
            return ind, 0
    if toolbox.dynamic_weights:
        evaluate.dynamic_error_weight_adjustment(False)        
        fitnesses = toolbox.map(toolbox.evaluate, population)
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

    # Generational process
    for gen in range(ngen):
        # Generate offspring
        offspring = varOr(population, toolbox, lambda_, cxpb, mutpb)

        # Evaluate offspring
        fitnesses = toolbox.map(toolbox.evaluate, offspring)
        for ind, fit in zip(offspring, fitnesses):
            ind.fitness.values = fit
            if debug:
                deap_str = str(ind)
                code = interpret.compile_deap(deap_str, toolbox.functions)
                code_str = interpret.convert_code_to_str(code)
                print("offspring gen", gen, code_str, fit[0])
            if ind.fitness.values[0] == 0.0:
                return ind, gen+1

        # Select the next generation population
        population[:] = toolbox.select(population + offspring, mu)

        if toolbox.dynamic_weights:
            # dynamic weight adjustment on whole pop
            evaluate.dynamic_error_weight_adjustment(False)        
            fitnesses = toolbox.map(toolbox.evaluate, population)
            for ind, fit in zip(population, fitnesses):
                ind.fitness.values = fit
        
        # progress
        #best = min(population, key=lambda item: item.fitness.values[0])
        #print(gen, best.fitness.values[0])
        
    #exit()
    best = min(population, key=lambda item: item.fitness.values[0])
    return best, ngen+1
    
    
def ga_search(toolbox, pop_size, nchildren, cxpb, mutpb, generations): # cxpb=p(mating), mutpb=p(mutation)
    best, gen = eaMuPlusLambda(toolbox, pop_size, nchildren, cxpb, mutpb, generations)
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
    pop_size, cxpb, generations = 300, 0.5, 30
    # for nchildren in [int(pop_size * 0.25), int(pop_size * 0.5), int(pop_size * 0.75), int(pop_size * 1.0)]:
    # for generations in [20, 30, 40, 50, 60, 70, 80, 90, 100]:
    # for cxpb in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    # for pop_size in [300, 400, 500, 600, 700, 800, 900, 1000]:
    if True:
        nchildren, mutpb = pop_size, 1.0 - cxpb
        global eval_count
        eval_count = 0
        #count_found = 0
        #for tries in range(20):
        if True:
            global best_error
            best_error = 1e9
            result = None
            for hop in range(hops):
                code, code_str, error, gen = ga_search(toolbox, pop_size, nchildren, cxpb, mutpb, generations)        
                if error == 0:
                    result = ["function", problem_name, params, code]
                    result_str = interpret.convert_code_to_str(result)
                    print("problem", problem_name, f"solved after {eval_count} evaluations by", result_str)
                    #print("gen", gen)
                    #count_found += 1
                    break
                if False:
                    print(f"hop {hop+1}, error {error:.3f}: {code_str}")
                    if toolbox.monkey_mode:
                        print(f"    {code_str}")
                        print(f"    {toolbox.solution_code_str}")
        #eff = count_found / eval_count
        #print(f"hops={hops}, pop_size={pop_size}, nchildren={nchildren}, cx={cxpb:.2f}, mut={mutpb:.2f}, generations={generations}, eff {count_found} / {eval_count} = {eff}")
    return result
