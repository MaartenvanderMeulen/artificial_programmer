# Inspired by https://deap.readthedocs.io/en/master/examples/gp_symbreg.html
# it is used by search.py
import os
import random
import copy
import interpret
import evaluate
from evaluate import recursive_tuple
import math
import time
import json
from deap import gp #  gp.PrimitiveSet, gp.genHalfAndHalf, gp.PrimitiveTree, gp.genFull, gp.from_string


global parents_fraction
parents_fraction = 1.0 # fractie van de oude populatie samen te voegen met de offspring

def evaluate_individual_impl(toolbox, ind, debug=0):
    deap_str = ind.deap_str
    assert deap_str == str(ind)
    toolbox.eval_count += 1
    code = interpret.compile_deap(deap_str, toolbox.functions)
    toolbox.functions[toolbox.problem_name] = [toolbox.formal_params, code]
    if toolbox.monkey_mode: # if the solution can be found in monkey mode, the real search could in theory find it also
        code_str = interpret.convert_code_to_str(code)
        weighted_error = evaluate.evaluate_code(code_str, toolbox.solution_code_str)
        ind.model_outputs = ()
        ind.model_evals = (weighted_error,)
        if weighted_error == 0.0:
            # now check that this also evaluates 
            ind.model_outputs = []
            for input in toolbox.example_inputs:
                model_output = interpret.run([toolbox.problem_name] + input, dict(), toolbox.functions)
                ind.model_outputs.append(model_output)        
            weighted_error, ind.model_evals = evaluate.evaluate_all(toolbox.example_inputs, ind.model_outputs, toolbox.evaluation_function, toolbox.f, debug, toolbox.penalise_non_reacting_models)
            if weighted_error != 0:
                weighted_error, ind.model_evals = evaluate.evaluate_all(toolbox.example_inputs, ind.model_outputs, toolbox.evaluation_function, toolbox.f, 4, toolbox.penalise_non_reacting_models)
            assert weighted_error == 0
    else:
        t0 = time.time()
        ind.model_outputs = []
        for input in toolbox.example_inputs:
            model_output = interpret.run([toolbox.problem_name] + input, dict(), toolbox.functions, debug=toolbox.verbose >= 5)
            ind.model_outputs.append(model_output)        
        toolbox.t_interpret += time.time() - t0
        t0 = time.time()
        weighted_error, ind.model_evals = evaluate.evaluate_all(toolbox.example_inputs, ind.model_outputs, toolbox.evaluation_function, toolbox.f, debug, toolbox.penalise_non_reacting_models)
        assert math.isclose(weighted_error, sum(ind.model_evals))
        toolbox.t_eval += time.time() - t0
        ind.model_outputs = recursive_tuple(ind.model_outputs)
        if weighted_error == 0.0 and len(ind) > len(toolbox.solution_deap_ind) and toolbox.optimise_solution_length:
            weighted_error += (len(ind) - len(toolbox.solution_deap_ind)) / 1000.0
    return weighted_error


def evaluate_individual(toolbox, individual, debug=0):
    if time.time() >= toolbox.t0 + toolbox.max_seconds:
        raise RuntimeWarning("out of time")
    deap_str = individual.deap_str
    assert deap_str == str(individual)
    toolbox.eval_lookup_count += 1
    if deap_str in toolbox.eval_cache: #  TODO handle dynamic weighting that changes the evaluation
        eval, individual.model_outputs, individual.model_evals = toolbox.eval_cache[deap_str]
        assert eval < 0.1 or math.isclose(eval, sum(individual.model_evals))
        return eval
    weighted_error = evaluate_individual_impl(toolbox, individual, debug)
    toolbox.eval_cache[deap_str] = weighted_error, individual.model_outputs, individual.model_evals
    return weighted_error


def f():
    '''Dummy function for DEAP'''
    return None


class Toolbox(object):
    def __init__(self, problem, functions):
        problem_name, formal_params, example_inputs, evaluation_function, hints, _ = problem
        int_hints, var_hints, func_hints, solution_hints = hints
        pset = gp.PrimitiveSet("MAIN", len(formal_params))
        for i, param in enumerate(formal_params):
            rename_cmd = f'pset.renameArguments(ARG{i}="{param}")'
            eval(rename_cmd)
        for constant in int_hints: 
            pset.addTerminal(constant)
        for variable in var_hints:
            if variable not in formal_params:
                pset.addTerminal(variable)
        for function in interpret.get_build_in_functions():
            if function in func_hints:
                param_types = interpret.get_build_in_function_param_types(function)
                arity = sum([1 for t in param_types if t in [1, "v", []]])
                pset.addPrimitive(f, arity, name=function)
        for function, (params, _) in functions.items():
            if function in func_hints:
                arity = len(params)
                pset.addPrimitive(f, arity, name=function)
        # add recursive call if in func_hints
        if problem_name in func_hints:
            arity = len(formal_params)
            pset.addPrimitive(f, arity, name=problem_name)
            dummy_code = 0
            functions[problem_name] = [formal_params, dummy_code]
        self.problem_name = problem_name
        self.formal_params = formal_params
        self.example_inputs = example_inputs
        self.evaluation_function = evaluation_function
        self.functions = functions
        self.pset = pset
        self.eval_cache = dict()
        self.ind_str_set = set()        
        self.solution_code_str = interpret.convert_code_to_str(solution_hints) # for monkey test
        deap_str = interpret.convert_code_to_deap_str(solution_hints, self)
        self.solution_deap_ind = gp.PrimitiveTree.from_string(deap_str, pset) # for finding shortest solution
        if deap_str != str(self.solution_deap_ind):
            print("deap_str1", deap_str)
            print("deap_str2", str(self.solution_deap_ind))
            raise RuntimeError(f"Check if function hints '{str(func_hints)}' contain all functions of solution hint '{str(solution_hints)}'")


def best_of_n(population, n):
    inds = random.sample(population, n) # sample always returns a list
    ind = min(inds, key=lambda ind: ind.eval)
    return ind



def update_dynamic_weighted_evaluation(toolbox, individuals):
    if toolbox.dynamic_weights:
        evaluate.dynamic_error_weight_adjustment(toolbox.f, toolbox.verbose)
        for ind in individuals:
            ind.eval = evaluate_individual(toolbox, ind)


def log_population(toolbox, population, label):
    if toolbox.verbose >= 2:
        toolbox.f.write(f"write_population {label}\n")
        for i, ind in enumerate(population):
            toolbox.f.write(f"    ind {i} {ind.eval:.3f} {len(ind)} {ind.deap_str}\n")
            if toolbox.gen == 0:
                evaluate_individual_impl(toolbox, ind, toolbox.verbose)
            if toolbox.verbose == 2 and i >= 10:
                break
        toolbox.f.write("\n")
        toolbox.f.flush()


def write_population(file_name, population, functions):
    with open(file_name, "w") as f:
        f.write("(\n")
        for ind in population:
            code = interpret.compile_deap(ind.deap_str, functions)
            code_str = interpret.convert_code_to_str(code)
            f.write(f"    {code_str} # {ind.eval:.3f}\n")
        f.write(")\n")


def write_final_population(toolbox, population):
    with open(toolbox.pop_file, "w") as f:
        f.write("(\n")
        for ind in population:
            code = interpret.compile_deap(ind.deap_str, toolbox.functions)
            code_str = interpret.convert_code_to_str(code)
            f.write(f"    {code_str} # {ind.eval:.3f}\n")
        f.write(")\n")


def write_path(toolbox, ind, indent=0):
    if toolbox.verbose >= 1:
        indent_str = "".join(['  ' for i in range(indent)])
        operator_str = ["", "mutatie", "crossover"][len(ind.parents)]
        code = interpret.compile_deap(ind.deap_str, toolbox.functions)
        code_str = interpret.convert_code_to_str(code)
        toolbox.f.write(f"{indent_str}{code_str} {ind.eval:.3f} {operator_str}\n")
        evaluate_individual_impl(toolbox, ind, toolbox.verbose)
        for parent in ind.parents:
            write_path(toolbox, parent, indent+1)


def generate_initial_population_impl(toolbox):
    toolbox.ind_str_set = set()
    population = []
    retry_count = 0
    while len(population) < toolbox.pop_size[0]:
        ind = gp.PrimitiveTree(gp.genHalfAndHalf(pset=toolbox.pset, min_=2, max_=4))
        ind.deap_str = str(ind)
        if ind.deap_str in toolbox.ind_str_set or len(ind) > toolbox.max_individual_size:
            if retry_count < toolbox.child_creation_retries:
                retry_count += 1
                continue
            else:
                break
        retry_count = 0
        toolbox.ind_str_set.add(ind.deap_str)
        if ind.deap_str not in toolbox.all_generations_ind_str_set:
            toolbox.all_generations_ind_str_set.add(ind.deap_str)
            toolbox.all_generations_ind.append(ind)
        ind.parents = []
        ind.eval = evaluate_individual(toolbox, ind)
        if ind.eval == 0.0:
            return None, ind
        population.append(ind)
    return population, None


def read_old_populations(old_populations_folder, prefix):
    old_pops = []
    for filename in os.listdir(old_populations_folder):
        if filename[:len(prefix)] == prefix:
            old_pop = interpret.compile(interpret.load(old_populations_folder + "/" + filename))
            if len(old_pop) > 0:
                old_pops.append(old_pop)
    return old_pops


def deap_len_of_code(code):
    return sum([deap_len_of_code(x) for x in code]) if type(code) == type([]) else 1


def load_initial_population_impl(toolbox, old_pops):
    toolbox.ind_str_set = set()
    population = []
    count_skipped = 0
    for old_pop in old_pops:
        k = toolbox.pop_size[0] // len(old_pops)
        # take sample of size k from the old population
        codes = random.sample(old_pop, k=k) if k < len(old_pop) else old_pop
        for code in codes:
            deap_str = interpret.convert_code_to_deap_str(code, toolbox)
            ind = gp.PrimitiveTree.from_string(deap_str, toolbox.pset)
            ind.deap_str = str(ind)
            assert len(ind) == deap_len_of_code(code)
            assert deap_str == ind.deap_str
            if ind.deap_str in toolbox.ind_str_set or len(ind) > toolbox.max_individual_size:
                count_skipped += 1
                continue
            toolbox.ind_str_set.add(ind.deap_str)
            if ind.deap_str not in toolbox.all_generations_ind_str_set:
                toolbox.all_generations_ind_str_set.add(ind.deap_str)
                toolbox.all_generations_ind.append(ind)
            ind.parents = []
            ind.eval = evaluate_individual(toolbox, ind)
            if ind.eval == 0.0:
                return None, ind
            population.append(ind)
    return population, None


def analyse_population_impl(toolbox, old_pops):
    population = []
    count_deap_str = dict()
    count_model_outputs = dict()
    count_eval = dict()
    for old_pop in old_pops:
        for code in old_pop:
            deap_str = interpret.convert_code_to_deap_str(code, toolbox)
            ind = gp.PrimitiveTree.from_string(deap_str, toolbox.pset)
            ind.deap_str = str(ind)
            ind.parents = []
            ind.eval = evaluate_individual(toolbox, ind)
            population.append(ind)

            assert len(ind) == deap_len_of_code(code)
            assert deap_str == ind.deap_str
            if deap_str not in count_deap_str:
                count_deap_str[deap_str] = 0
            count_deap_str[deap_str] += 1
            if ind.model_outputs not in count_model_outputs:
                count_model_outputs[ind.model_outputs] = 0
            count_model_outputs[ind.model_outputs] += 1
            if ind.eval not in count_eval:
                count_eval[ind.eval] = 0
            count_eval[ind.eval] += 1
    with open(f"{toolbox.output_folder}/analysis.txt", "w") as f:
        f.write(f"count\tstr\n")
        count_deap_str = [(key, value) for key, value in count_deap_str.items()]
        count_deap_str.sort(key=lambda item: -item[1])
        for key, value in count_deap_str:
            f.write(f"{value}\t{key}\n")

        f.write(f"\ncount\tmodel_output\n")
        count_model_outputs = [(key, value) for key, value in count_model_outputs.items()]
        count_model_outputs.sort(key=lambda item: -item[1])
        for key, value in count_model_outputs:
            f.write(f"{value}\t{key}\n")

        f.write(f"\ncount\teval\n")
        count_eval = [(key, value) for key, value in count_eval.items()]
        count_eval.sort(key=lambda item: -item[1])
        for key, value in count_eval:
            f.write(f"{value}\t{key}\n")
    return None


def generate_initial_population(toolbox):
    if toolbox.analyse_best:
        bests = read_old_populations(toolbox.old_populations_folder, "best")
        analyse_population_impl(toolbox, bests)    
        exit()
    evaluate.init_dynamic_error_weight_adjustment()
    if toolbox.new_initial_population:
        return generate_initial_population_impl(toolbox)
    else:
        old_pops = read_old_populations(toolbox.old_populations_folder, "pop")
        return load_initial_population_impl(toolbox, old_pops)


def copy_individual(ind):
    consistency_check_ind(ind)
    copy_ind = gp.PrimitiveTree(list(ind[:]))
    copy_ind.deap_str = copy.deepcopy(ind.deap_str)
    copy_ind.parents = [parent for parent in ind.parents]
    copy_ind.eval = ind.eval
    copy_ind.model_outputs = copy.deepcopy(ind.model_outputs)
    copy_ind.model_evals = copy.deepcopy(ind.model_evals)
    consistency_check_ind(copy_ind)
    return copy_ind


def cxOnePoint(toolbox, parent1, parent2):
    if len(parent1) < 2 or len(parent2) < 2:
        # No crossover on single node tree
        return None
    child = copy_individual(parent1)
    child.parents = [parent1, parent2] if toolbox.keep_path else []
    index1 = random.randrange(0, len(parent1))
    index2 = random.randrange(0, len(parent2))
    slice1 = parent1.searchSubtree(index1)
    slice2 = parent2.searchSubtree(index2)
    child[slice1] = parent2[slice2]
    child.deap_str = str(child)
    if child.deap_str in toolbox.ind_str_set or len(child) > toolbox.max_individual_size:
        return None
    child.eval = evaluate_individual(toolbox, child)
    if child.eval in toolbox.taboo_set:
        return None
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
            child.parents = [parent1, parent2] if toolbox.keep_path else []
            slice1 = child.searchSubtree(index1)
            child[slice1] = expr2
            if len(child) <= toolbox.max_individual_size:
                child.deap_str = str(child)
                if child.deap_str not in toolbox.ind_str_set:
                    child.eval = evaluate_individual(toolbox, child)
                    if False:
                        if parent1.eval > child.eval or (parent1.eval == child.eval and len(parent1) > len(child)):
                            return child
                    if best is None or best.eval > child.eval or (best.eval == child.eval and len(best) > len(child)):
                        if child.eval not in toolbox.taboo_set:
                            best = child
    return best


def mutUniform(toolbox, parent, expr, pset):
    child = copy_individual(parent)
    child.parents = [parent,] if toolbox.keep_path else []
    index = random.randrange(0, len(child))
    slice_ = child.searchSubtree(index)
    type_ = child[index].ret
    child[slice_] = expr(pset=pset, type_=type_)
    child.deap_str = str(child)
    if child.deap_str in toolbox.ind_str_set or len(child) > toolbox.max_individual_size:
        return None
    child.eval = evaluate_individual(toolbox, child)
    if child.eval in toolbox.taboo_set:
        return None
    return child


def replace_subtree_at_best_location(toolbox, parent, expr):
    indexes = [i for i in range(len(parent))]
    random.shuffle(indexes)
    best = None
    for index in indexes:
        child = copy_individual(parent)
        child.parents = [parent,] if toolbox.keep_path else []
        slice1 = child.searchSubtree(index)
        child[slice1] = expr
        if len(child) <= toolbox.max_individual_size:
            child.deap_str = str(child)
            if child.deap_str not in toolbox.ind_str_set:
                child.eval = evaluate_individual(toolbox, child)
                if False:
                    if parent.eval > child.eval or (parent.eval == child.eval and len(parent) > len(child)):
                        return child
                if best is None or best.eval > child.eval or (best.eval == child.eval and len(best) > len(child)):
                    if child.eval not in toolbox.taboo_set:
                        best = child
    return best


def select_parents(toolbox, population):
    if toolbox.parent_selection_strategy == 0 or len(toolbox.model_outputs_dict) == 1:
        return [best_of_n(population, toolbox.best_of_n_cx), best_of_n(population, toolbox.best_of_n_cx)]
    if toolbox.parent_selection_strategy == 1:        
        # First attempt with toolbox.inter_family_cx_taboo.  Its just unlikely, not a taboo.
        group1, group2 = random.sample(list(toolbox.model_outputs_dict), 2) # sample always returns a list
        group1, group2 = toolbox.model_outputs_dict[group1], toolbox.model_outputs_dict[group2]
        index1, index2 = random.randrange(0, len(group1)), random.randrange(0, len(group2))
        parent1, parent2 = group1[index1], group2[index2] # all individuals in the group have the same eval
        return parent1, parent2
        
    # second attempt, hint of Victor:
    # * hoe beter de evaluatie hoe meer kans.  p_fitness = (1 - eval) ** alpha
    # * hoe verder de outputs uit elkaar liggen hoe meer kans.  p_complementair = verschillende_outputs(parent1, parent2) / aantal_outputs
    # * p = (p_fitness(parent1) * p_fitness(parent2)) ^ alpha * p_complementair(parent1, parent2) ^ beta
    best_p = -1
    assert toolbox.best_of_n_cx > 0
    max_eval = max([ind.eval for ind in population])
    for _ in range(toolbox.best_of_n_cx):
        # select two parents
        group1, group2 = random.sample(list(toolbox.model_outputs_dict), 2) # sample always returns a list
        group1, group2 = toolbox.model_outputs_dict[group1], toolbox.model_outputs_dict[group2]
        index1, index2 = random.randrange(0, len(group1)), random.randrange(0, len(group2))
        parent1, parent2 = group1[index1], group2[index2] # all individuals in the group have the same eval
        # sort
        if parent1.eval > parent2.eval or (parent1.eval == parent2.eval and len(parent1) > len(parent2)):
            parent1, parent2 = parent2, parent1
        # compute p        
        assert parent1.eval < 0.1 or math.isclose(parent1.eval, sum(parent1.model_evals))
        assert parent2.eval < 0.1 or math.isclose(parent2.eval, sum(parent2.model_evals))
        p_fitness1 = (1 - parent1.eval/(max_eval*1.1))
        p_fitness2 = (1 - parent2.eval/(max_eval*1.1))
        assert 0 <= p_fitness1 and p_fitness1 <= 1
        assert 0 <= p_fitness2 and p_fitness2 <= 1
        estimate_improvement = sum([eval1-eval2 for eval1, eval2 in zip(parent1.model_evals, parent2.model_evals) if eval1 > eval2])
        assert estimate_improvement >= 0
        p_complementair = estimate_improvement / parent1.eval
        assert 0 <= p_complementair and p_complementair <= 1
        p = ((p_fitness1 * p_fitness2)) + (p_complementair * toolbox.beta)
        assert p >= 0
        if best_p < p:
            best_parent1, best_parent2, best_p = parent1, parent2, p
    return best_parent1, best_parent2


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
            parent = best_of_n(population, toolbox.best_of_n_mut)
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
        toolbox.ind_str_set.add(child.deap_str)
        if child.deap_str not in toolbox.all_generations_ind_str_set:
            toolbox.all_generations_ind_str_set.add(child.deap_str)
            toolbox.all_generations_ind.append(child)
        offspring.append(child)
    return offspring, None


def refresh_toolbox_from_population(toolbox, population):
    toolbox.ind_str_set = {ind.deap_str for ind in population} # refresh set after deletion of non-fit individuals
    toolbox.model_outputs_dict = dict()
    for ind in population:
        if ind.model_outputs not in toolbox.model_outputs_dict:
            toolbox.model_outputs_dict[ind.model_outputs] = []
        toolbox.model_outputs_dict[ind.model_outputs].append(ind)


def consistency_check_ind(ind):
    if ind is not None:
        assert hasattr(ind, "deap_str")
        assert hasattr(ind, "parents")
        assert hasattr(ind, "eval")
        assert hasattr(ind, "model_outputs")
        assert ind.deap_str == str(ind)
        assert ind.eval is not None


def consistency_check(inds):
    for ind in inds:
        consistency_check_ind(ind)


def erase_suboptimum_parents(toolbox, population):
    best_eval = population[0].eval
    toolbox.taboo_set.add(best_eval)
    population = [ind for ind in population if ind.eval > best_eval]
    if len(population) == 0:
        raise RuntimeWarning("grote opschudding gooit hele populatie weg...")
    return population


def ga_search_impl(toolbox):
    if toolbox.final_pop_file:
        write_population(toolbox.final_pop_file, [], toolbox.functions)
    try:
        toolbox.taboo_set = set()
        population, solution = generate_initial_population(toolbox)
        if solution:
            return solution, 0
        toolbox.parachute_count = len(population)
        consistency_check(population)
        update_dynamic_weighted_evaluation(toolbox, population)
        population.sort(key=lambda item: item.eval)
        refresh_toolbox_from_population(toolbox, population)
        toolbox.gen = 0
        global parents_fraction
        prev_best = 1e9
        stuck_count = 0
        grote_opschudding_count = 0
        for toolbox.parachute_level in range(len(toolbox.ngen)):
            while toolbox.gen < toolbox.ngen[toolbox.parachute_level]:

                # track if we are stuck
                if math.isclose(population[0].eval, prev_best):
                    parents_fraction = min(1.0, parents_fraction / 2) # stuck, gooi parents weg
                    stuck_count += 1
                else:
                    # verbetering!
                    parents_fraction = 1.0 # reset 
                    stuck_count = 0
                prev_best = population[0].eval
                if stuck_count >= 5:
                    # Erase all parents with this value BEFORE making the children, ERASING it from the gene pool
                    population = erase_suboptimum_parents(toolbox, population)
                if toolbox.f and toolbox.verbose >= 1:
                    # code_str = interpret.convert_code_to_str(interpret.compile_deap(population[0].deap_str, toolbox.functions))
                    toolbox.f.write(f"gen {toolbox.gen:2d} best {population[0].eval:7.3f} pf {parents_fraction:.3f} sc {stuck_count} goc {grote_opschudding_count} taboo {str(toolbox.taboo_set)}\n") #  {code_str}\n")
                log_population(toolbox, population, f"generation {toolbox.gen}, pop at start")
                offspring, solution = generate_offspring(toolbox, population, toolbox.nchildren[toolbox.parachute_level])
                if solution:
                    return solution, toolbox.gen+1
                consistency_check(offspring)
                if len(offspring) != toolbox.nchildren[toolbox.parachute_level]:
                    toolbox.f.write(f"{len(offspring)} offspring\n")
                if toolbox.parachute_level == 0:
                    toolbox.parachute_offspring_count += len(offspring)
                else:
                    toolbox.normal_offspring_count += len(offspring)
                if parents_fraction < 1.0:
                    population = random.sample(population, k=int(len(population)*parents_fraction))
                population += offspring
                population.sort(key=lambda item: item.eval)
                population[:] = population[:toolbox.pop_size[toolbox.parachute_level]]
                consistency_check(population)
                update_dynamic_weighted_evaluation(toolbox, population)
                refresh_toolbox_from_population(toolbox, population)
                toolbox.gen += 1
    except RuntimeWarning as e:
        toolbox.f.write("RuntimeWarning: " + str(e) + "\n")
    if toolbox.final_pop_file:
        write_population(toolbox.final_pop_file, population, toolbox.functions)
    best = population[0]
    return best, toolbox.gen+1


def count_cx_mut(ind):
    count_cx, count_mut = 0, 0
    if len(ind.parents) == 2:
        count_cx += 1
    elif len(ind.parents) == 1:
        count_mut += 1
    for parent in ind.parents:
        cx, mut = count_cx_mut(parent)
        count_cx += cx
        count_mut += mut
    return count_cx, count_mut

    
def compute_cx_fraction(best):
    cx, mut = count_cx_mut(best)
    return cx / (cx + mut) if cx + mut > 0 else 0.0
    

def basinhopper(toolbox):
    for _ in range(toolbox.hops):
        toolbox.ind_str_set = set()
        toolbox.all_generations_ind_str_set = set()
        toolbox.all_generations_ind = []
        toolbox.eval_cache = dict()
        toolbox.eval_count = 0
        toolbox.eval_lookup_count = 0
        toolbox.parachute_count = 0
        toolbox.parachute_offspring_count = 0
        toolbox.normal_offspring_count = 0
        toolbox.t0 = time.time()
        toolbox.t_interpret = 0
        toolbox.t_eval = 0

        best, gen = ga_search_impl(toolbox)
        if toolbox.best_ind_file:
            write_population(toolbox.best_ind_file, [best], toolbox.functions)
        seconds = round(time.time() - toolbox.t0)
        if best.eval == 0:
            code = interpret.compile_deap(best.deap_str, toolbox.functions)
            result = ["function", toolbox.problem_name, toolbox.problem_params, code]
            # cx_perc = round(100*compute_cx_fraction(best))
            toolbox.f.write(f"solved\t{toolbox.problem_name}\t{seconds}\tsec")
            if toolbox.extensive_statistics:                
                toolbox.f.write(f"\t{gen}\tgen\t{len(best)}\tlen")
                toolbox.f.write(f"\t{toolbox.eval_count}\tevals")
                toolbox.f.write(f"\t{toolbox.eval_lookup_count}\telc\t{toolbox.parachute_count}\tpac")
                toolbox.f.write(f"\t{toolbox.parachute_offspring_count}\tpoc\t{toolbox.normal_offspring_count}\tnoc")
                toolbox.f.write(f"\t{len(toolbox.all_generations_ind_str_set)}\tagu")
            toolbox.f.write(f"\t{best.deap_str}")
            toolbox.f.write(f"\n")
            if toolbox.verbose >= 1:
                score = evaluate_individual_impl(toolbox, best, 4)
                assert score == 0
            if toolbox.verbose >= 1:
                write_path(toolbox, best)
            return result
        else:
            toolbox.f.write(f"stopped\t{toolbox.problem_name}\t{gen}\tgen\t{seconds}\tseconds\n")
        toolbox.f.flush()
        
    toolbox.f.write(f"failed\t{toolbox.problem_name}\n")
    return None

    
def solve_by_new_function(problem, functions, f, params):
    toolbox = Toolbox(problem, functions)
    toolbox.problem_name, toolbox.problem_params, _, _, _, _ = problem
    toolbox.monkey_mode = False
    toolbox.dynamic_weights = False # not toolbox.monkey_mode
    toolbox.child_creation_retries = 99
    toolbox.f = f
    if len(toolbox.solution_deap_ind) > 0:
        f.write(f"solution hint length {len(toolbox.solution_deap_ind)}\n")

    # tunable params
    toolbox.verbose = params["verbose"]
    toolbox.max_seconds = params["max_seconds"]
    toolbox.pop_size = params["pop_size"]
    toolbox.nchildren = params["nchildren"]
    toolbox.ngen = params["ngen"]
    toolbox.max_individual_size = params["max_individual_size"]
    toolbox.pcrossover = params["pcrossover"]
    toolbox.pmutations = 1.0 - toolbox.pcrossover
    toolbox.best_of_n_mut = params["best_of_n_mut"]
    toolbox.best_of_n_cx = params["best_of_n_cx"]
    toolbox.parent_selection_strategy = params["parent_selection_strategy"]
    toolbox.beta = params["weight_complementairity"]
    toolbox.penalise_non_reacting_models = params["penalise_non_reacting_models"]
    toolbox.hops = params["hops"]
    toolbox.output_folder = params["output_folder"]
    toolbox.final_pop_file = None # params["output_folder"] + "/pop_" + str(params["seed"]) + ".txt"
    toolbox.all_ind_file = None # params["output_folder"] + "/ind_" + str(params["seed"]) + ".txt"
    toolbox.best_ind_file = None # params["output_folder"] + "/best_" + str(params["seed"]) + ".txt"
    toolbox.new_initial_population = params["new_initial_population"]
    if not toolbox.new_initial_population:
        toolbox.old_populations_folder = params["old_populations_folder"]
        toolbox.analyse_best = params["analyse_best"]
    else:
        toolbox.analyse_best = False
    toolbox.optimise_solution_length = params["optimise_solution_length"]
    toolbox.extensive_statistics = params["extensive_statistics"]
    toolbox.keep_path = params["keep_path"]
    
    # search
    toolbox.all_generations_ind = []
    if toolbox.all_ind_file:
        write_population(toolbox.all_ind_file, toolbox.all_generations_ind, toolbox.functions)
    result = basinhopper(toolbox)
    if toolbox.all_ind_file:
        write_population(toolbox.all_ind_file, toolbox.all_generations_ind, toolbox.functions)

    return result