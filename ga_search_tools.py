# Inspired by https://deap.readthedocs.io/en/master/examples/gp_symbreg.html
# it is used by search.py
import os
import random
import copy
import math
import time
import json
import numpy as np

import interpret
import evaluate
from evaluate import recursive_tuple
from cpp_coupling import get_cpp_handle, run_on_all_inputs

from deap import gp #  gp.PrimitiveSet, gp.genHalfAndHalf, gp.PrimitiveTree, gp.genFull, gp.from_string


def test_against_python_interpreter(toolbox, cpp_model_outputs, ind):
    t0 = time.time()
    deap_str = ind.deap_str
    assert deap_str == str(ind)
    code = interpret.compile_deap(deap_str, toolbox.functions)
    toolbox.functions[toolbox.problem_name] = [toolbox.formal_params, code]
    py_model_outputs = []
    for input in toolbox.example_inputs:
        model_output = interpret.run([toolbox.problem_name] + input, dict(), toolbox.functions, debug=toolbox.verbose >= 5)
        py_model_outputs.append(model_output)        
    toolbox.t_py_interpret += time.time() - t0
    if py_model_outputs != cpp_model_outputs:
        run_on_all_inputs(toolbox.cpp_handle, ind, debug=2)
        print("py_model_outputs", py_model_outputs)
        print("cpp_model_outputs", cpp_model_outputs)
        print("toolbox.eval_count OK", toolbox.eval_count - 1)
        code_str = interpret.convert_code_to_str(code)
        print(code_str)
    assert py_model_outputs == cpp_model_outputs


def eval_monkey_mode(toolbox, ind, debug):
    '''if the solution can be found in monkey mode, the real search could in theory find it also'''
    deap_str = ind.deap_str
    assert deap_str == str(ind)
    code = interpret.compile_deap(deap_str, toolbox.functions)
    toolbox.functions[toolbox.problem_name] = [toolbox.formal_params, code]
    code_str = interpret.convert_code_to_str(code)
    weighted_error = evaluate.evaluate_code(code_str, toolbox.solution_code_str)
    model_outputs = (weighted_error,)
    model_evals = (weighted_error,)
    if weighted_error == 0.0:
        # now check that this also evaluates 
        model_outputs_tmp = []
        for input in toolbox.example_inputs:
            model_output_tmp = interpret.run([toolbox.problem_name] + input, dict(), toolbox.functions)
            model_outputs_tmp.append(model_output_tmp)        
        weighted_error_tmp, model_evals_tmp = evaluate.evaluate_all(toolbox.example_inputs, model_outputs_tmp, toolbox.evaluation_function, toolbox.f, debug, toolbox.penalise_non_reacting_models)
        if weighted_error_tmp != 0:
            weighted_error_tmp, model_evals_tmp = evaluate.evaluate_all(toolbox.example_inputs, model_outputs_tmp, toolbox.evaluation_function, toolbox.f, 4, toolbox.penalise_non_reacting_models)
        assert weighted_error_tmp == 0
    return weighted_error, model_outputs, model_evals


class Family:
    def __init__(self, model_outputs, eval_vectors):
        self.model_outputs = model_outputs
        self.eval_vectors = eval_vectors
        self.model_evals = evaluate.compute_weighted_sums(self.eval_vectors)
        self.weighted_error = sum(self.model_evals)

    def dynamic_weights_update(self):
        self.model_evals = evaluate.compute_weighted_sums(self.eval_vectors)
        self.weighted_error = sum(self.model_evals)


def add_family(toolbox, model_outputs_tuple, eval_vectors):
    family_index = len(toolbox.families_list)
    toolbox.families_dict[model_outputs_tuple] = family_index
    toolbox.families_list.append(Family(model_outputs_tuple, eval_vectors))
    return family_index


def forced_reevaluation_of_individual_for_debugging(toolbox, ind, debug_level):
    '''Assigns family index to the individual'''
    model_outputs = run_on_all_inputs(toolbox.cpp_handle, ind)
    weighted_error = evaluate.compute_weighted_error(toolbox.example_inputs, model_outputs, toolbox.evaluation_function, \
            toolbox.f, debug_level, toolbox.penalise_non_reacting_models)
    return weighted_error
    


def evaluate_individual_impl(toolbox, ind, debug=0):
    '''Assigns family index to the individual'''
    if toolbox.eval_count >= toolbox.max_evaluations:
        raise RuntimeWarning("max evaluations reached")
    toolbox.eval_count += 1
    if toolbox.monkey_mode: 
        # weighted_error, model_outputs, model_evals = eval_monkey_mode(toolbox, ind, debug)
        pass # TODO : fix this again
    else:
        t0 = time.time()
        cpp_model_outputs = run_on_all_inputs(toolbox.cpp_handle, ind)
        toolbox.t_cpp_interpret += time.time() - t0
        model_outputs = cpp_model_outputs
        if False:
            test_against_python_interpreter(toolbox, cpp_model_outputs, ind)
        t0 = time.time()
        model_outputs_tuple = recursive_tuple(model_outputs)
        if model_outputs_tuple in toolbox.families_dict:
            ind.family_index = toolbox.families_dict[model_outputs_tuple]
        else:
            eval_vectors = evaluate.compute_eval_vectors(toolbox.example_inputs, model_outputs, toolbox.evaluation_function, \
                toolbox.f, debug, toolbox.penalise_non_reacting_models)
            ind.family_index = add_family(toolbox, model_outputs_tuple, eval_vectors)
        toolbox.t_eval += time.time() - t0


def evaluate_individual(toolbox, individual, debug=0):
    if time.time() >= toolbox.t0 + toolbox.max_seconds:
        raise RuntimeWarning("out of time")
    deap_str = individual.deap_str
    assert deap_str == str(individual)
    toolbox.eval_lookup_count += 1
    if deap_str in toolbox.eval_cache:
        individual.family_index = toolbox.eval_cache[deap_str]
    else:
        evaluate_individual_impl(toolbox, individual, debug)
        toolbox.eval_cache[deap_str] = individual.family_index
    family = toolbox.families_list[individual.family_index]
    weighted_error = family.weighted_error
    if weighted_error == 0.0 and len(individual) > len(toolbox.solution_deap_ind) and toolbox.optimise_solution_length:
        weighted_error += (len(individual) - len(toolbox.solution_deap_ind)) / 1000.0
    return weighted_error


def best_of_n(population, n):
    inds = random.sample(population, n) # sample always returns a list
    ind = min(inds, key=lambda ind: ind.eval)
    return ind



def update_dynamic_weighted_evaluation(toolbox, individuals):
    if toolbox.dynamic_weights:
        if len(toolbox.families_list) > 0:
            n_error_dimensions = len(toolbox.families_list[0].eval_vectors[0])
            avg_errors = np.zeros((n_error_dimensions))
            for i, family in enumerate(toolbox.families_list):
                if i in toolbox.current_families_dict:
                    for vec in family.eval_vectors:
                        avg_errors += np.array(vec)
            avg_errors /= len(toolbox.families_list)
            if toolbox.prev_avg_errors is not None:
                toolbox.f.write(f"update_dynamic_weights, n {n_error_dimensions}\n")
                toolbox.f.write(f"{toolbox.prev_avg_errors} prev avg errors\n")
                toolbox.f.write(f"{avg_errors} avg errors\n")
                evaluate.dynamic_error_weight_adjustment(toolbox.f, toolbox.verbose, toolbox.prev_avg_errors, avg_errors)
                for family in toolbox.families_list:
                    family.dynamic_weights_update()
                for ind in individuals:
                    ind.eval = toolbox.families_list[ind.family_index].weighted_error
                individuals.sort(key=toolbox.sort_ind_key)
            toolbox.prev_avg_errors = avg_errors


def log_population(toolbox, population, label):
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
    if len(old_pops) == 0:
        toolbox.f.write(f"RuntimeError: no {prefix}* files in folder {old_populations_folder}\n")
        exit(1)
    if toolbox.old_populations_samplesize != 1:
        if toolbox.old_populations_samplesize < len(old_pops):
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
                population.append(ind)
    return population, None


def analyse_population_impl(toolbox, old_pops):
    families = dict()
    for old_pop in old_pops:
        for code in old_pop:
            deap_str = interpret.convert_code_to_deap_str(code, toolbox)
            ind = gp.PrimitiveTree.from_string(deap_str, toolbox.pset)
            ind.deap_str = str(ind)
            assert deap_str == ind.deap_str
            assert len(ind) == deap_len_of_code(code)
            ind.parents = []
            ind.eval = evaluate_individual(toolbox, ind) # required to set ind.family_index
            if ind.family_index not in families:
                families[ind.family_index] = [ind.family_index, ind.deap_str, 0]
            families[ind.family_index][2] += 1
            if len(families[ind.family_index][1]) > len(ind.deap_str):
                families[ind.family_index][1] = ind.deap_str
    del ind, deap_str
    filename = f"{toolbox.output_folder}/analysis.txt"
    print("output in", filename)
    with open(filename, "w") as f:
        f.write(f"{'  error':7} family_index individuals {'shortest_individual':60} {'last_output':30} {'last_error':30}\n")
        data = [value for key, value in families.items()]
        data.sort(key=lambda item: -toolbox.families_list[item[0]].weighted_error)
        sum_count = 0
        sum_error_vector = np.zeros_like(toolbox.families_list[0].eval_vectors[-1])
        for family_index, deap_str, count in data:
            error = toolbox.families_list[family_index].weighted_error
            outputs = str(toolbox.families_list[family_index].model_outputs[-1])
            error_vector = toolbox.families_list[family_index].eval_vectors[-1]
            sum_error_vector += np.array(error_vector)
            error_vector = " ".join([f"{v:5.3f}" for v in error_vector])
            sum_count += count
            f.write(f"{error:7.3f} {family_index:12d} {count:11d} {deap_str[:60]:60} {outputs[:30]:30} {error_vector[:30]:30}\n")
        sum_error_vector = " ".join([f"{v/len(data):5.3f}" for v in sum_error_vector])
        f.write(f"{' ':7} {' ':12} {sum_count:11} {' ':60} {' ':30} {sum_error_vector}\n")


def generate_initial_population(toolbox, old_pops=None):
    if toolbox.analyse_best:
        # best pop uit pop maken kan met :
        # for s in `seq 1000 1 1999` ; do echo '(' > best_$s.txt ; if grep '\#' pop_$s.txt | head -1 >> best_$s.txt ; then echo $s ; fi ; echo ')' >> best_$s.txt ; done
        # daarna de lege best files identificeren met
        # for s in `seq 1000 1 1999` ; do grep -L '\#' best_$s.txt ; done
        # en dan weghalen.
        bests = read_old_populations(toolbox, toolbox.old_populations_folder, "pop")
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
        print("ga_search_tools, 292, random.seed(", toolbox.seed, ")")
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


def get_family_size(toolbox, ind):
    n = 0
    c = toolbox.current_families_dict
    if ind.family_index in c:
        n += len(c[ind.family_index])
    o = toolbox.offspring_families_dict
    if ind.family_index in o:
        n += len(o[ind.family_index])
    return n


def is_improvement(toolbox, ind, best):
    if best is None:
        return True
    if best.eval != ind.eval:
        return best.eval > ind.eval
    # now they have equal .eval
    if False:
        best_family_size = get_family_size(toolbox, best)
        ind_family_size = get_family_size(toolbox, ind)
        if best_family_size != ind_family_size:
            return best_family_size > ind_family_size
        # now they have equal family size
    return len(best) > len(ind)


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
                    if is_improvement(toolbox, child, best):
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
                if is_improvement(toolbox, child, best):
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
    toolbox.offspring_families_dict = dict()


def consistency_check_ind(toolbox, ind):
    if ind is not None:
        assert hasattr(ind, "deap_str")
        assert hasattr(ind, "parents")
        assert hasattr(ind, "eval")
        assert hasattr(ind, "family_index")
        assert ind.deap_str == str(ind)
        assert ind.eval is not None
        model_evals = toolbox.families_list[ind.family_index].model_evals
        assert math.isclose(ind.eval, sum(model_evals))


def consistency_check(toolbox, inds):
    for ind in inds:
        consistency_check_ind(toolbox, ind)


def write_seconds(toolbox, seconds, label):
    file_name = toolbox.params["output_folder"] + "/time_" + str(toolbox.params["seed"]) + ".txt"
    with open(file_name, "a") as f:
        f.write(f"{label} {seconds}\n")


