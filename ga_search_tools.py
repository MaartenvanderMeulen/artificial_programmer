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


def evaluate_individual_impl(toolbox, ind, debug=0):
    deap_str = ind.deap_str
    assert deap_str == str(ind)
    if toolbox.eval_count >= toolbox.max_evaluations:
        raise RuntimeWarning("max evaluations reached")
    toolbox.eval_count += 1
    code = interpret.compile_deap(deap_str, toolbox.functions)
    toolbox.functions[toolbox.problem_name] = [toolbox.formal_params, code]
    if toolbox.monkey_mode: # if the solution can be found in monkey mode, the real search could in theory find it also
        code_str = interpret.convert_code_to_str(code)
        weighted_error = evaluate.evaluate_code(code_str, toolbox.solution_code_str)
        model_outputs = ()
        model_evals = (weighted_error,)
        if weighted_error == 0.0:
            # now check that this also evaluates 
            model_outputs = []
            for input in toolbox.example_inputs:
                model_output = interpret.run([toolbox.problem_name] + input, dict(), toolbox.functions)
                model_outputs.append(model_output)        
            weighted_error, model_evals = evaluate.evaluate_all(toolbox.example_inputs, model_outputs, toolbox.evaluation_function, toolbox.f, debug, toolbox.penalise_non_reacting_models)
            if weighted_error != 0:
                weighted_error, model_evals = evaluate.evaluate_all(toolbox.example_inputs, model_outputs, toolbox.evaluation_function, toolbox.f, 4, toolbox.penalise_non_reacting_models)
            assert weighted_error == 0
    else:
        t0 = time.time()
        model_outputs = []
        for input in toolbox.example_inputs:
            model_output = interpret.run([toolbox.problem_name] + input, dict(), toolbox.functions, debug=toolbox.verbose >= 5)
            model_outputs.append(model_output)        
        toolbox.t_interpret += time.time() - t0
        t0 = time.time()
        weighted_error, model_evals = evaluate.evaluate_all(toolbox.example_inputs, model_outputs, toolbox.evaluation_function, toolbox.f, debug, toolbox.penalise_non_reacting_models)
        assert math.isclose(weighted_error, sum(model_evals))
        toolbox.t_eval += time.time() - t0
        model_outputs = recursive_tuple(model_outputs)
        if weighted_error == 0.0 and len(ind) > len(toolbox.solution_deap_ind) and toolbox.optimise_solution_length:
            weighted_error += (len(ind) - len(toolbox.solution_deap_ind)) / 1000.0
    return weighted_error, model_outputs, model_evals


def evaluate_individual(toolbox, individual, debug=0):
    if time.time() >= toolbox.t0 + toolbox.max_seconds:
        raise RuntimeWarning("out of time")
    deap_str = individual.deap_str
    assert deap_str == str(individual)
    toolbox.eval_lookup_count += 1
    if deap_str in toolbox.eval_cache: #  TODO handle dynamic weighting that changes the evaluation
        individual.family_index = toolbox.eval_cache[deap_str]
        eval, model_outputs, model_evals = toolbox.families_list[individual.family_index]
        assert eval < 0.1 or math.isclose(eval, sum(model_evals))
        return eval
    weighted_error, model_outputs, model_evals = evaluate_individual_impl(toolbox, individual, debug)
    if model_outputs in toolbox.families_dict:
        individual.family_index = toolbox.families_dict[model_outputs]
    else:
        individual.family_index = len(toolbox.families_list)
        toolbox.families_list.append((weighted_error, model_outputs, model_evals))
        toolbox.families_dict[model_outputs] = individual.family_index
    toolbox.eval_cache[deap_str] = individual.family_index
    return weighted_error


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
        #indent_str = "\t" * indent
        # operator_str = ["", "mutatie", "crossover"][len(ind.parents)]
        code = interpret.compile_deap(ind.deap_str, toolbox.functions)
        code_str = interpret.convert_code_to_str(code)
        if False:
            if indent:
                toolbox.f.write(f"parent\t{ind.eval:.3f}\tcode\t{code_str}\n")
            else:
                toolbox.f.write(f"child\t{ind.eval:.3f}\tcode\t{code_str}\n")
        else:
            toolbox.f.write(f"{code_str} {ind.eval:.3f} \n")
        # evaluate_individual_impl(toolbox, ind, toolbox.verbose)
        if indent == 0:
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
        ind.parents = []
        ind.eval = evaluate_individual(toolbox, ind)
        if ind.eval == 0.0:
            return None, ind
        population.append(ind)
    return population, None


def read_old_populations(toolbox, old_populations_folder, prefix):
    old_pops = []
    for filename in os.listdir(old_populations_folder):
        if filename[:len(prefix)] == prefix:
            if toolbox.old_populations_samplesize != 1 or filename == f"{prefix}_{toolbox.seed}.txt":
                old_pop = interpret.compile(interpret.load(old_populations_folder + "/" + filename))
                if len(old_pop) > 0:
                    old_pops.append(old_pop)
                elif toolbox.old_populations_samplesize == 1:
                    toolbox.f.write("RuntimeWarning: stopped because no set covering needed, 0 evals\n")
                    exit()
    if toolbox.old_populations_samplesize != 1:
        old_pops = random.sample(old_pops, k=toolbox.old_populations_samplesize)
    return old_pops


def deap_len_of_code(code):
    return sum([deap_len_of_code(x) for x in code]) if type(code) == type([]) else 1


def load_initial_population_impl(toolbox, old_pops):
    toolbox.ind_str_set = set()
    population = []
    count_skipped = 0
    for old_pop in old_pops:
        if len(old_pop) > 0:
            k = max(1, toolbox.pop_size[0] // len(old_pops))
            # take sample of size k from the old population
            codes = random.sample(old_pop, k=k) if k < len(old_pop) else old_pop
            for code in codes:
                if hasattr(code, "family_index"):
                    # old_pop is list of gp individuals
                    ind = code
                else:
                    # old_pop is list of lists/ints/strings making 
                    deap_str = interpret.convert_code_to_deap_str(code, toolbox)
                    ind = gp.PrimitiveTree.from_string(deap_str, toolbox.pset)
                    ind.deap_str = str(ind)
                    assert len(ind) == deap_len_of_code(code)
                    assert deap_str == ind.deap_str
                if ind.deap_str in toolbox.ind_str_set or len(ind) > toolbox.max_individual_size:
                    count_skipped += 1
                    continue
                toolbox.ind_str_set.add(ind.deap_str)
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
            model_outputs = toolbox.families_list[ind.family_index][1]
            if model_outputs not in count_model_outputs:
                count_model_outputs[model_outputs] = 0
            count_model_outputs[model_outputs] += 1
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


def generate_initial_population(toolbox, old_pops=None):
    if toolbox.analyse_best:
        bests = read_old_populations(toolbox, toolbox.old_populations_folder, "best")
        analyse_population_impl(toolbox, bests)    
        exit()
    evaluate.init_dynamic_error_weight_adjustment()
    if toolbox.new_initial_population:
        population, solution = generate_initial_population_impl(toolbox)
    else:
        # zorg dat de bron van de populatie niet uitmaakt voor de eindtoestand
        # dit was nodig om bij interne opschuddingen hetzelfde resultaat te krijgen als bij set covering op 1 vastloper
        toolbox.reset() # zorgt voor reproduceerbare toolbox
        if old_pops:
            # old_pops is list of individuals
            population, solution = load_initial_population_impl(toolbox, old_pops)
        else:
            # 
            old_pops = read_old_populations(toolbox, toolbox.old_populations_folder, "pop")
            # old_pops is list of deap strings
            population, solution = load_initial_population_impl(toolbox, old_pops)
        random.seed(toolbox.seed) # zorgt voor reproduceerbare state
    return population, solution


def copy_individual(toolbox, ind):
    consistency_check_ind(toolbox, ind)
    copy_ind = gp.PrimitiveTree(list(ind[:]))
    copy_ind.deap_str = ind.deap_str
    copy_ind.parents = [parent for parent in ind.parents]
    copy_ind.eval = ind.eval
    copy_ind.family_index = ind.family_index
    consistency_check_ind(toolbox, copy_ind)
    return copy_ind


def cxOnePoint(toolbox, parent1, parent2):
    if len(parent1) < 2 or len(parent2) < 2:
        # No crossover on single node tree
        return None
    child = copy_individual(toolbox, parent1)
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
            child = copy_individual(toolbox, parent1)
            child.parents = [parent1, parent2] if toolbox.keep_path else []
            slice1 = child.searchSubtree(index1)
            child[slice1] = expr2
            if len(child) <= toolbox.max_individual_size:
                child.deap_str = str(child)
                if child.deap_str not in toolbox.ind_str_set:
                    child.eval = evaluate_individual(toolbox, child)
                    if best is None or best.eval > child.eval or (best.eval == child.eval and len(best) > len(child)):
                        if child.eval not in toolbox.taboo_set:
                            best = child
    return best


def mutUniform(toolbox, parent, expr, pset):
    child = copy_individual(toolbox, parent)
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
        child = copy_individual(toolbox, parent)
        child.parents = [parent,] if toolbox.keep_path else []
        slice1 = child.searchSubtree(index)
        child[slice1] = expr
        if len(child) <= toolbox.max_individual_size:
            child.deap_str = str(child)
            if child.deap_str not in toolbox.ind_str_set:
                child.eval = evaluate_individual(toolbox, child)
                if best is None or best.eval > child.eval or (best.eval == child.eval and len(best) > len(child)):
                    if child.eval not in toolbox.taboo_set:
                        best = child
    return best


def refresh_toolbox_from_population(toolbox, population):
    toolbox.ind_str_set = {ind.deap_str for ind in population} # refresh set after deletion of non-fit individuals
    toolbox.current_families_dict = dict()
    for ind in population:
        if ind.family_index not in toolbox.current_families_dict:
            toolbox.current_families_dict[ind.family_index] = []
        toolbox.current_families_dict[ind.family_index].append(ind)


def consistency_check_ind(toolbox, ind):
    if ind is not None:
        assert hasattr(ind, "deap_str")
        assert hasattr(ind, "parents")
        assert hasattr(ind, "eval")
        assert hasattr(ind, "family_index")
        assert ind.deap_str == str(ind)
        assert ind.eval is not None
        model_evals = toolbox.families_list[ind.family_index][2]
        assert ind.eval < 0.1 or math.isclose(ind.eval, sum(model_evals))


def consistency_check(toolbox, inds):
    for ind in inds:
        consistency_check_ind(toolbox, ind)


def write_seconds(toolbox, seconds):
    file_name = toolbox.params["output_folder"] + "/time_" + str(toolbox.params["seed"]) + ".txt"
    with open(file_name, "w") as f:
        f.write(f"{seconds}\n")


