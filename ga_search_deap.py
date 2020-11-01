# Inspired by https://deap.readthedocs.io/en/master/examples/gp_symbreg.html
# it is used by search.py
import random
import copy
import interpret
import evaluate
import math
import time
from deap import gp #  gp.PrimitiveSet, gp.genHalfAndHalf, gp.PrimitiveTree, gp.genFull


global total_eval_count
total_eval_count = 0


def recursive_tuple(value):
    if type(value) == type(1):
        return value
    assert type(value) == type([])
    return tuple([recursive_tuple(v) for v in value])


def evaluate_individual(toolbox, individual, debug=0):
    if time.time() >= toolbox.t0 + toolbox.max_seconds:
        toolbox.f.write(f"Stopped after {round(time.time()-toolbox.t0)} seconds\n") 
        exit()
    deap_str = str(individual)
    if deap_str in toolbox.eval_cache:
        eval, model_output = toolbox.eval_cache[deap_str]
        individual.model_output = model_output
        return eval
    toolbox.eval_count += 1
    global total_eval_count
    total_eval_count += 1
    code = interpret.compile_deap(deap_str, toolbox.functions)
    if toolbox.monkey_mode: # if the solution can be found in monkey mode, the real search could in theory find it also
        code_str = interpret.convert_code_to_str(code)
        weighted_error = evaluate.evaluate_code(code_str, toolbox.solution_code_str)
        individual.model_output = ()
        if weighted_error == 0.0:
            sum_v = 0
            for input in toolbox.example_inputs:
                variables = interpret.bind_params(toolbox.formal_params, input)
                model_output = interpret.run(code, variables, toolbox.functions)
                sum_v += evaluate.evaluate(input, model_output, toolbox.evaluation_functions, 1)
            assert sum_v == 0
    else:
        weighted_error = 0.0
        model_outputs = []
        prev_model_output, model_reacts_on_input = None, False
        for input in toolbox.example_inputs:
            variables = interpret.bind_params(toolbox.formal_params, input)
            model_output = interpret.run(code, variables, toolbox.functions)
            model_outputs.append(model_output)
            if model_output != prev_model_output:
                model_reacts_on_input = True
                prev_model_output = model_output
            v = evaluate.evaluate(input, model_output, toolbox.evaluation_functions, debug)
            weighted_error += v
        individual.model_output = recursive_tuple(model_outputs)
        if weighted_error > 0 and not model_reacts_on_input:
            # always the same wrong answer, penalize that heavily.
            weighted_error += 10 + 10*weighted_error

    toolbox.eval_cache[deap_str] = weighted_error, individual.model_output
    return weighted_error


def f():
    '''Dummy function for DEAP'''
    return None


class Toolbox(object):
    def __init__(self, problem, functions):
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
                arity = sum([1 for t in param_types if t in [1, "v", []]])
                pset.addPrimitive(f, arity, name=function)
        for function, (params, code) in functions.items():
            if function in func_hints:
                arity = len(params)
                pset.addPrimitive(f, arity, name=function)
        # toolbox = base.Toolbox()
        self.formal_params = formal_params
        self.example_inputs = example_inputs
        self.evaluation_functions = evaluation_functions
        self.functions = functions
        self.pset = pset
        self.solution_code_str = interpret.convert_code_to_str(solution_hints)
        self.eval_cache = dict()
        self.ind_str_set = set()


def best_of_n(population, n):
    inds = random.sample(population, n)
    ind = min(inds, key=lambda ind: ind.eval)
    return ind



def update_dynamic_weighted_evaluation(toolbox, individuals):
    if toolbox.dynamic_weights:
        evaluate.dynamic_error_weight_adjustment()
        for ind in individuals:
            ind.eval = evaluate_individual(toolbox, ind)


def write_population(toolbox, population, label):
    if toolbox.verbose >= 2:
        toolbox.f.write(f"write_population {label}\n")
        for i, ind in enumerate(population):
            if ind.eval < 2:
                if False:
                    toolbox.f.write(f"individual {i} ")
                    write_path(toolbox, ind)
                if True:
                    deap_str = str(ind)
                    #code = interpret.compile_deap(deap_str, toolbox.functions)
                    #code_str = interpret.convert_code_to_str(code)
                    toolbox.f.write(f"    ind {i} {ind.eval} {len(ind)} {deap_str}\n")
        toolbox.f.write("\n")
        toolbox.f.flush()


def write_path(toolbox, ind, indent=0):
    if toolbox.verbose >= 1:
        indent_str = "".join(['  ' for i in range(indent)])
        operator_str = ["", "mutatie", "crossover"][len(ind.parents)]
        deap_str = str(ind)
        code = interpret.compile_deap(deap_str, toolbox.functions)
        code_str = interpret.convert_code_to_str(code)
        toolbox.f.write(f"{indent_str}{code_str} {ind.eval:.3f} {operator_str}\n")
        for parent in ind.parents:
            write_path(toolbox, parent, indent+1)


def generate_initial_population(toolbox):
    evaluate.init_dynamic_error_weight_adjustment()
    toolbox.ind_str_set = set()
    population = []
    retry_count = 0
    while len(population) < toolbox.pop_size[0]:
        ind = gp.PrimitiveTree(gp.genHalfAndHalf(pset=toolbox.pset, min_=2, max_=4))
        ind.deap_str = str(ind)
        ind.id = f"init{len(population)}"
        if ind.deap_str in toolbox.ind_str_set:
            if retry_count < toolbox.child_creation_retries:
                retry_count += 1
                continue
            else:
                break
        retry_count = 0
        toolbox.ind_str_set.add(ind.deap_str)
        ind.parents = []
        ind.eval = evaluate_individual(toolbox, ind)
        if ind.eval == 0.0:
            return None, ind
        population.append(ind)
    return population, None


def copy_individual(ind):
    consistency_check_ind(ind)
    copy_ind = gp.PrimitiveTree(list(ind[:]))
    copy_ind.deap_str = copy.deepcopy(ind.deap_str)
    copy_ind.parents = [parent for parent in ind.parents]
    copy_ind.eval = ind.eval
    copy_ind.model_output = copy.deepcopy(ind.model_output)
    copy_ind.id = "copy" + ind.id
    consistency_check_ind(copy_ind)
    return copy_ind


def cxOnePoint(toolbox, parent1, parent2):
    if len(parent1) < 2 or len(parent2) < 2:
        # No crossover on single node tree
        print("len(parent1) < 2 or len(parent2) < 2")
        return None
    child = copy_individual(parent1)
    child.parents = [parent1, parent2]
    child.id = "cx" + parent1.id + "," + parent2.id
    index1 = random.randrange(0, len(parent1))
    index2 = random.randrange(0, len(parent2))
    slice1 = parent1.searchSubtree(index1)
    slice2 = parent2.searchSubtree(index2)
    child[slice1] = parent2[slice2]
    child.deap_str = str(child)
    if child.deap_str in toolbox.ind_str_set:
        print("child.deap_str in toolbox.ind_str_set", child.deap_str)
        return None
    child.eval = evaluate_individual(toolbox, child)
    return child


def crossover_with_local_search(toolbox, parent1, parent2):
    if len(parent1) < 2 or len(parent2) < 2:
        # No crossover on single node tree
        return None
    if parent1.eval > parent2.eval or (parent1.eval == parent2.eval and len(parent1) > len(parent2)):
        parent1, parent2 = parent2, parent1
    indexes1 = [i for i in range(len(parent1))]
    indexes2 = [i for i in range(len(parent2))]
    random.shuffle(indexes1)
    random.shuffle(indexes2)
    best = None
    for index2 in indexes2:
        slice2 = parent2.searchSubtree(index2)
        expr2 = parent2[slice2]
        for index1 in indexes1:
            child = copy_individual(parent1)
            child.parents = [parent1, parent2]
            slice1 = child.searchSubtree(index1)
            child[slice1] = expr2
            if len(child) <= toolbox.max_individual_size:
                child.deap_str = str(child)
                if child.deap_str not in toolbox.ind_str_set:
                    child.eval = evaluate_individual(toolbox, child)
                    if parent1.eval > child.eval or (parent1.eval == child.eval and len(parent1) > len(child)):
                        print("DEBUG 227")
                        return child
                    if best is None or best.eval > child.eval or (best.eval == child.eval and len(best) > len(child)):
                        best = child
    return best


def mutUniform(toolbox, parent, expr, pset):
    child = copy_individual(parent)
    child.parents = [parent,]
    index = random.randrange(0, len(child))
    slice_ = child.searchSubtree(index)
    type_ = child[index].ret
    child[slice_] = expr(pset=pset, type_=type_)
    child.deap_str = str(child)
    if child.deap_str in toolbox.ind_str_set:
        return None
    child.eval = evaluate_individual(toolbox, child)
    return child


def replace_subtree_at_best_location(toolbox, parent, expr):
    indexes = [i for i in range(len(parent))]
    random.shuffle(indexes)
    best = None
    for index in indexes:
        child = copy_individual(parent)
        child.parents = [parent,]
        slice1 = child.searchSubtree(index)
        child[slice1] = expr
        if len(child) <= toolbox.max_individual_size:
            child.deap_str = str(child)
            if child.deap_str not in toolbox.ind_str_set:
                child.eval = evaluate_individual(toolbox, child)
                if parent.eval > child.eval or (parent.eval == child.eval and len(parent) > len(child)):
                    return child
                if best is None or best.eval > child.eval or (best.eval == child.eval and len(best) > len(child)):
                    best = child
    return best


def select_parents(toolbox, population):
    for i, ind in enumerate(population):
        print(f"DEBUG 272    ind {i} {ind.eval} {len(ind)} {ind.deap_str} {str(ind)}\n")
    if not toolbox.inter_family_cx_taboo or len(toolbox.model_outputs_dict) == 1:
        return [best_of_n(population, 2), best_of_n(population, 2)]
    # The parents must have different model_output
    for key, value in toolbox.model_outputs_dict.items():
        print("DEBUG 277", key, len(value))
    group1, group2 = random.sample(list(toolbox.model_outputs_dict), 2)
    print("DEBUG 279", str(group1), str(group2))
    group1, group2 = toolbox.model_outputs_dict[group1], toolbox.model_outputs_dict[group2]
    print("DEBUG 281A", type(group1), type(group2))
    print("DEBUG 281", str(group1), str(group2))
    parent1, parent2 = random.sample(group1, 1), random.sample(group2, 1) # all individuals in the group have the same eval
    print("parent1", parent1.deap_str)
    print("parent2", parent2.deap_str)
    exit()
    return parent1, parent2


def generate_offspring(toolbox, population, nchildren):
    offspring = []
    expr_mut = lambda pset, type_: gp.genFull(pset=pset, min_=0, max_=2, type_=type_)
    retry_count = 0  
    cxp_count, mutp_count = 0, 0    
    cx_count, mut_count = 0, 0    
    while len(offspring) < nchildren:
        op_choice = random.random()
        if op_choice < toolbox.pcrossover: # Apply crossover
            cxp_count += 1
            parent1, parent2 = select_parents(toolbox, population)
            if toolbox.parachute_level == 0:
                child = cxOnePoint(toolbox, parent1, parent2)
            else:
                child = crossover_with_local_search(toolbox, parent1, parent2)
            if child:
                cx_count += 1
        else: # Apply mutation
            mutp_count += 1
            parent = best_of_n(population, 2)
            if toolbox.parachute_level == 0:
                child = mutUniform(toolbox, parent, expr=expr_mut, pset=toolbox.pset)
            else:
                expr = gp.genFull(pset=toolbox.pset, min_=0, max_=2)
                child = replace_subtree_at_best_location(toolbox, parent, expr)
            if child:
                mut_count += 1
        if child is None:
            if retry_count < toolbox.child_creation_retries:
                retry_count += 1
                continue
            else:
                break
        retry_count = 0
        assert child.eval is not None
        if child.eval == 0.0:
            return None, child
        assert child.deap_str == str(child)
        assert child.deap_str not in toolbox.ind_str_set
        toolbox.ind_str_set.add(child.deap_str)
        offspring.append(child)
    print("cx_count, mut_count", cx_count, mut_count, "cxp_count, mutp_count", cxp_count, mutp_count, "toolbox.parachute_level", toolbox.parachute_level)
    return offspring, None


def refresh_toolbox_from_population(toolbox, population):
    toolbox.ind_str_set = {str(ind) for ind in population} # refresh set
    toolbox.model_outputs_dict = dict()
    for ind in population:
        if ind.model_output not in toolbox.model_outputs_dict:
            toolbox.model_outputs_dict[ind.model_output] = []
        toolbox.model_outputs_dict[ind.model_output].append(ind)


def consistency_check_ind(ind):
    if ind is not None:
        assert hasattr(ind, "deap_str")
        assert hasattr(ind, "parents")
        assert hasattr(ind, "eval")
        assert hasattr(ind, "model_output")
        assert ind.deap_str == str(ind)
        assert ind.eval is not None


def consistency_check(inds):
    for ind in inds:
        consistency_check_ind(ind)


def ga_search_impl(toolbox):
    population, solution = generate_initial_population(toolbox)
    consistency_check(population)
    if solution:
        return solution, 0
    update_dynamic_weighted_evaluation(toolbox, population)
    population.sort(key=lambda item: item.eval)
    refresh_toolbox_from_population(toolbox, population)
    gen = 0
    for toolbox.parachute_level in range(len(toolbox.ngen)):
        while gen < toolbox.ngen[toolbox.parachute_level]:
            code_str = interpret.convert_code_to_str(interpret.compile_deap(str(population[0]), toolbox.functions))
            toolbox.f.write(f"start generation {gen:2d} best eval {population[0].eval:.5f} {code_str}\n")
            #evaluate_individual(toolbox, population[0], debug=True)
            write_population(toolbox, population, f"generation {gen}, pop at start")
            offspring, solution = generate_offspring(toolbox, population, toolbox.nchildren[toolbox.parachute_level])
            if solution:
                return solution, gen+1
            consistency_check(offspring)
            if len(offspring) != toolbox.nchildren[toolbox.parachute_level]:
                toolbox.f.write(f"{len(offspring)} offspring\n")
            population += offspring
            population.sort(key=lambda item: item.eval)
            population[:] = population[:toolbox.pop_size[toolbox.parachute_level]]
            consistency_check(population)
            update_dynamic_weighted_evaluation(toolbox, population)
            refresh_toolbox_from_population(toolbox, population)
            gen += 1
    write_population(toolbox, population, "pop final")
    best = population[0]
    return best, gen+1


def solve_by_new_function(problem, functions, f, verbose):
    problem_name, params, example_inputs, evaluation_functions, hints, layer = problem
    toolbox = Toolbox(problem, functions)
    toolbox.monkey_mode = False
    toolbox.dynamic_weights = False # not toolbox.monkey_mode
    toolbox.max_individual_size = 40 # max of len(individual).  Don't make individuals larger than it
    toolbox.inter_family_cx_taboo = True
    toolbox.child_creation_retries = 99
    toolbox.max_seconds = 600
    result = None
    toolbox.f = f
    toolbox.verbose = verbose
    toolbox.pcrossover = 0.5
    toolbox.pmutations = 1.0 - toolbox.pcrossover
    toolbox.pop_size, toolbox.ngen = [16, 8], [4, 30]
    toolbox.nchildren = toolbox.pop_size
    for hop in range(1):
        toolbox.ind_str_set = set()
        toolbox.eval_cache = dict()
        toolbox.eval_count = 0
        toolbox.t0 = time.time()
        best, gen = ga_search_impl(toolbox)
        seconds = time.time() - toolbox.t0
        if best.eval == 0:
            deap_str = str(best)
            code = interpret.compile_deap(deap_str, toolbox.functions)
            code_str = interpret.convert_code_to_str(code)
            result = ["function", problem_name, params, code]
            result_str = interpret.convert_code_to_str(result)
            f.write(f"problem {problem_name} solved after {toolbox.eval_count} evaluations, {seconds} seconds, by {result_str}\n")
            assert evaluate_individual(toolbox, best, 1) == 0
            write_path(toolbox, best)
            break
        else:
            f.write(f"problem {problem_name} failed after {toolbox.eval_count} evaluations, {seconds} seconds\n")
        
    return result
