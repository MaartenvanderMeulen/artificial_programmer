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
import dynamic_weights
from evaluate import recursive_tuple
import cpp_coupling

from deap import gp #  gp.PrimitiveSet, gp.genHalfAndHalf, gp.PrimitiveTree, gp.genFull, gp.from_string


def make_pp_str(ind):
    if True:
        result = " ".join([x.name if isinstance(x, gp.Primitive) else str(x.value) for x in ind])
    else:
        result = str(ind)
    return result


def test_against_python_interpreter(toolbox, cpp_model_outputs, ind):
    t0 = time.time()
    code = interpret.compile_deap(str(ind), toolbox.functions)
    toolbox.functions[toolbox.problem_name] = [toolbox.formal_params, code]
    py_model_outputs = []
    for input in toolbox.example_inputs:
        model_output = interpret.run([toolbox.problem_name] + input, dict(), toolbox.functions, debug=toolbox.verbose >= 5)
        py_model_outputs.append(model_output)        
    toolbox.t_py_interpret += time.time() - t0
    if py_model_outputs != cpp_model_outputs:
        cpp_coupling.run_on_all_inputs(toolbox.cpp_handle, ind, debug=2)
        print("py_model_outputs", py_model_outputs)
        print("cpp_model_outputs", cpp_model_outputs)
        print("toolbox.eval_count OK", toolbox.eval_count - 1)
        code_str = interpret.convert_code_to_str(code)
        print(code_str)
    assert py_model_outputs == cpp_model_outputs


class Family:
    def __init__(self, family_index, raw_error_matrix):
        self.family_index = family_index
        self.raw_error_matrix = raw_error_matrix
        self.raw_error = evaluate.compute_raw_error(self.raw_error_matrix)
        self.update_normalised_error()

    def update_normalised_error(self):
        self.normalised_error = dynamic_weights.compute_normalised_error(self.raw_error_matrix, 1.0)


def forced_reevaluation_of_individual_for_debugging(toolbox, ind, debug_level):
    '''Assigns family index to the individual'''
    model_outputs = cpp_coupling.run_on_all_inputs(toolbox.cpp_handle, ind)
    raw_error_matrix = evaluate.compute_raw_error_matrix(toolbox.example_inputs, model_outputs, toolbox.error_function, \
        toolbox.f, toolbox.verbose)
    raw_error = evaluate.compute_raw_error(raw_error_matrix)
    return raw_error


def check_error_matrices(raw_error_matrix_py, raw_error_matrix_cpp):
    #print("debug line 68, python, check_error_matrices")
    assert raw_error_matrix_py.shape == raw_error_matrix_cpp.shape
    for i in range(raw_error_matrix_py.shape[0]):
        for j in range(raw_error_matrix_py.shape[1]):
            if not math.isclose(raw_error_matrix_py[i, j], raw_error_matrix_cpp[i, j]):
                print("raw_error_matrix_py", raw_error_matrix_py)
                print("raw_error_matrix_cpp", raw_error_matrix_cpp)
                print("debug line 72:", "i", i, "j", j, "py", raw_error_matrix_py[i, j], "cpp", raw_error_matrix_cpp[i, j])
            assert math.isclose(raw_error_matrix_py[i, j], raw_error_matrix_cpp[i, j])


def evaluate_individual_impl(toolbox, ind, debug=0):
    # cpp implementatie
    raw_error_matrix_cpp, model_outputs_cpp = cpp_coupling.compute_error_matrix(toolbox.cpp_handle, ind, toolbox.penalise_non_reacting_models)

    # piggyback test : vergelijk uitkomst cpp met python implementatie
    if True:
        model_outputs_py = cpp_coupling.run_on_all_inputs(toolbox.cpp_handle, ind)
        raw_error_matrix_py = evaluate.compute_raw_error_matrix(toolbox.example_inputs, model_outputs_py, toolbox.error_function, \
            toolbox.f, debug, toolbox.penalise_non_reacting_models)
        check_error_matrices(raw_error_matrix_py, raw_error_matrix_cpp)

    # bepaling family key
    if toolbox.family_key_is_error_matrix:
        family_key = tuple(raw_error_matrix_cpp.flatten())
    else:
        family_key = model_outputs_cpp

    # bepaling family
    if family_key in toolbox.families_dict:
        family_index = toolbox.families_dict[family_key]
        ind.fam = toolbox.families_list[family_index]
    else:
        family_index = len(toolbox.families_list)
        toolbox.families_dict[family_key] = family_index
        ind.fam = Family(family_index, raw_error_matrix_cpp)
        toolbox.families_list.append(ind.fam)


def evaluate_individual(toolbox, individual, pp_str, debug):
    assert type(pp_str) == type("") and type(debug) == type(1)
    toolbox.eval_lookup_count += 1
    if pp_str in toolbox.pp_str_to_family_index_dict:
        family_index = toolbox.pp_str_to_family_index_dict[pp_str]
        individual.fam = toolbox.families_list[family_index]
    else:
        if time.time() >= toolbox.t0 + toolbox.max_seconds:
            raise RuntimeWarning("out of time")
        if toolbox.eval_count >= toolbox.max_evaluations:
            raise RuntimeWarning("max evaluations reached")
        toolbox.eval_count += 1
        evaluate_individual_impl(toolbox, individual, debug)
        toolbox.pp_str_to_family_index_dict[pp_str] = individual.fam.family_index


def best_of_n(population, n):
    inds = random.sample(population, n) # sample always returns a list
    ind = min(inds, key=lambda ind: ind.fam.normalised_error)
    return ind


def log_population(toolbox, population, label):
    toolbox.f.write(f"write_population {label}\n")
    for i, ind in enumerate(population):
        toolbox.f.write(f"    ind {i} {ind.fam.raw_error:.3f} {len(ind)} {ind(ind)}\n")
    toolbox.f.write("\n")
    toolbox.f.flush()


def write_population(file_name, population, functions):
    with open(file_name, "w") as f:
        f.write("(\n")
        for ind in population:
            code = interpret.compile_deap(str(ind), functions)
            code_str = interpret.convert_code_to_str(code)
            f.write(f"    {code_str} # {ind.fam.raw_error:.3f}\n")
        f.write(")\n")


def write_final_population(toolbox, population):
    with open(toolbox.pop_file, "w") as f:
        f.write("(\n")
        for ind in population:
            code = interpret.compile_deap(str(ind), toolbox.functions)
            code_str = interpret.convert_code_to_str(code)
            f.write(f"    {code_str} # {ind.fam.raw_error:.3f}\n")
        f.write(")\n")


def write_path(toolbox, ind):
    index = ind.fam.family_index
    toolbox.f.write(f"family_index {index}\n")
    results = []
    for key, value in toolbox.cx_count_dict.items():
        index_a, index_b = key
        assert index_a <= index_b
        if index == index_a or index == index_b:
            results.append([index_a , index_b, value])
    results.sort(key=lambda item : -item[2])
    for index_a, index_b, count in results:
        assert index_a <= index_b
        key = (index_a, index_b)
        if key in toolbox.cx_child_dict:
            count = len(toolbox.cx_child_dict[key])
            toolbox.f.write(f"toolbox.cx_child_dict[({index_a},{index_b})] = {count}\n")
            for key, value in toolbox.cx_child_dict[key].items():
                toolbox.f.write(f"    child_fam[{key}] = {value}\n")
    code = interpret.compile_deap(str(ind), toolbox.functions)
    code_str = interpret.convert_code_to_str(code)
    toolbox.f.write(f"{code_str} # {ind.fam.raw_error:.3f} len {len(ind)}\n")


def compute_complementairity(fam1, fam2):
    raw_error_matrix1 = fam1.raw_error_matrix
    raw_error_matrix2 = fam2.raw_error_matrix
    raw_improvement = raw_error_matrix1 - raw_error_matrix2
    complementairity = np.sum(raw_improvement[raw_improvement > 0])
    complementairity /= fam1.raw_error
    assert complementairity >= 0
    if complementairity > 1: # fix any rounding issues: the
        complementairity = 1 # max complementairity is that the whole parent1.fam.raw_error is removed
    return complementairity


def pz(toolbox, index_a, index_b):
    count_in_pop, count_out_pop = 0, 0
    if (index_a, index_b) in toolbox.cx_child_dict:
        for key, value in toolbox.cx_child_dict[(index_a, index_b)].items():
            if key in toolbox.current_families_dict:
                count_in_pop += value
            else:
                count_out_pop += value
    if (index_b, index_a) in toolbox.cx_child_dict:
        for key, value in toolbox.cx_child_dict[(index_b, index_a)].items():
            if key in toolbox.current_families_dict:
                count_in_pop += value
            else:
                count_out_pop += value
    n_missing = max(0, 10 - count_in_pop - count_out_pop)
    result = (count_out_pop + n_missing + 1) / (count_in_pop + count_out_pop + n_missing + 1)
    assert result > 0
    return result


def compute_fam_cx_fitness(toolbox, fam1, fam2):
    # compute p        
    p_fitness1 = max(0, min((1 - fam1.raw_error/(toolbox.max_raw_error*1.1)), 1))
    p_fitness2 = max(0, min((1 - fam2.raw_error/(toolbox.max_raw_error*1.1)), 1))
    # assert 0 <= p_fitness1 and p_fitness1 <= 1
    # assert 0 <= p_fitness2 and p_fitness2 <= 1
    p_complementair = compute_complementairity(fam1, fam2)

    assert 0 <= p_complementair and p_complementair <= 1
    p1 = ((p_fitness1 * p_fitness2)) + (p_complementair * toolbox.parent_selection_weight_complementairity)

    index_a, index_b = fam1.family_index, fam2.family_index
    if index_a > index_b:
        index_a, index_b = index_b, index_a
    key = (index_a, index_b)
    assert index_a <= index_b
    if key in toolbox.cx_count_dict:
        count_cx = toolbox.cx_count_dict[key]
    else:
        count_cx = 0
    if count_cx > 0:
        x2 = count_cx + 1
        y2 = toolbox.parent_selection_weight_cx_count
        x3 = pz(toolbox, fam1.family_index, fam2.family_index) 
        y3 = toolbox.parent_selection_weight_p_out_of_pop
        return p1, x2, x3, p1/(x2**y2)*(x3**y3)
    else:
        return p1, 0, 0, 0


def write_cx_info(toolbox):
    results = []
    sum_cx_count = 0
    for key, cx_count in toolbox.cx_count_dict.items():
        index_a, index_b = key
        assert index_a <= index_b
        results.append([index_a , index_b, cx_count])
        assert key in toolbox.cx_child_dict
        cx_child_count = 0
        for _, child_count in toolbox.cx_child_dict[key].items():
            cx_child_count += child_count
        assert cx_count == cx_child_count
        sum_cx_count += cx_count
    sum_cx_childs_count = 0
    for key, cx_childs in toolbox.cx_child_dict.items():
        for _, child_count in cx_childs.items():
            sum_cx_childs_count += child_count
    toolbox.f.write(f"cx_info, sum_cx_count {sum_cx_count}\n")
    assert sum_cx_count == sum_cx_childs_count
    results.sort(key=lambda item : -item[2])
    for i, (index_a, index_b, cx_count) in enumerate(results):
        key = (index_a, index_b)
        if i < 10:
            p1, p2, p3, p4 = compute_fam_cx_fitness(toolbox, toolbox.families_list[index_a], toolbox.families_list[index_b])
            toolbox.f.write(f"toolbox.cx_child_dict[({index_a},{index_b})] = {cx_count}, fitness {p1},{p2},{p3},{p4}\n")
            childs = []
            for child_index, child_count in toolbox.cx_child_dict[key].items():
                childs.append([child_index, child_count])
            childs.sort(key=lambda item: -item[1])
            for child_index, child_count in childs:
                if child_count > 0:
                    toolbox.f.write(f"    child_family = {child_index}; count = {child_count}\n")


def generate_initial_population_impl(toolbox):
    toolbox.ind_str_set = set()
    population = []
    retry_count = 0
    while len(population) < toolbox.pop_size[0]:
        ind = gp.PrimitiveTree(gp.genHalfAndHalf(pset=toolbox.pset, min_=2, max_=4))
        pp_str = make_pp_str(ind)
        if pp_str in toolbox.ind_str_set or len(ind) > toolbox.max_individual_size:
            if retry_count < toolbox.child_creation_retries:
                retry_count += 1
                continue
            else:
                break
        retry_count = 0
        toolbox.ind_str_set.add(pp_str)
        evaluate_individual(toolbox, ind, pp_str, 0)
        population.append(ind)
    return population


def read_old_populations(toolbox, old_populations_folder, prefix):
    old_pops = []
    filenames = []
    for filename in os.listdir(old_populations_folder):
        if filename[:len(prefix)] == prefix:
            id = int(filename[len(prefix)+1:len(prefix)+1+4])
            if id // toolbox.old_populations_samplesize == toolbox.seed // toolbox.old_populations_samplesize:
                filenames.append(filename)
    filenames.sort()
    for filename in filenames:
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
                    assert deap_str == str(ind)
                    pp_str = make_pp_str(ind)
                    assert len(ind) == deap_len_of_code(code)
                if pp_str in toolbox.ind_str_set or len(ind) > toolbox.max_individual_size:
                    count_skipped += 1
                    continue
                toolbox.ind_str_set.add(ind.pp_str)
                evaluate_individual(toolbox, ind, pp_str, 0)
                population.append(ind)
    return population


def analyse_population_impl(toolbox, old_pops):
    families = dict()
    count = 0
    for old_pop in old_pops:
        for code in old_pop:
            count += 1
            deap_str = interpret.convert_code_to_deap_str(code, toolbox)
            ind = gp.PrimitiveTree.from_string(deap_str, toolbox.pset)
            assert deap_str == str(ind)
            pp_str = make_pp_str(ind)
            assert len(ind) == deap_len_of_code(code)
            evaluate_individual(toolbox, ind, pp_str, 0) # required to set ind.fam
            family_index = ind.fam.family_index
            if family_index not in families:
                families[family_index] = [family_index, pp_str, 0]
            families[family_index][2] += 1
            if len(families[family_index][1]) > len(pp_str):
                families[family_index][1] = pp_str
    del ind, deap_str
    filename = f"{toolbox.output_folder}/analysis.txt"
    print(f"anaysis of {count} individuals in {len(old_pops)} pops : result in {filename}")
    with open(filename, "w") as f:
        f.write(f"{'rawerr':8} {'norerr':8} findex fsize {'last_output':48} {'last_raw_error':48} {'last_nor_error':48}\n")
        data = [value for key, value in families.items()]
        data.sort(key=lambda item: -toolbox.families_list[item[0]].weighted_error)
        sum_count = 0
        sum_raw_error_vector = np.zeros_like(toolbox.families_list[0].raw_error_matrix[-1])
        for family_index, _, count in data:
            raw_error = toolbox.families_list[family_index].raw_error
            raw_error_vector = toolbox.families_list[family_index].raw_error_matrix[-1]
            sum_raw_error_vector += np.array(raw_error_vector)
            raw_error_vector = " ".join([f"{v:5.3f}" for v in raw_error_vector])
            sum_count += count
            nor_error = toolbox.families_list[family_index].normalised_error
            f.write(f"{raw_error:8.3f} {nor_error:8.3f} {family_index:6d} {count:5d} ")
            f.write(f" {raw_error_vector[:48]:48}")
            f.write(f"\n")
        sum_raw_error_vector = " ".join([f"{v/len(data):5.3f}" for v in sum_raw_error_vector])
        f.write(f"{' ':8} {' ':8} {len(toolbox.families_list):6} {sum_count:5} {'last_output':48} ")
        f.write(f"{sum_raw_error_vector:48}\n")


def generate_initial_population(toolbox, old_pops=None):
    if toolbox.analyse_best:
        # best pop uit pop maken kan met :
        # for s in `seq 1000 1 1999` ; do echo '(' > best_$s.txt ; if grep '\#' pop_$s.txt | head -1 >> best_$s.txt ; then echo $s ; fi ; echo ')' >> best_$s.txt ; done
        # daarna de lege best files weghalen
        # for s in `seq 1000 1 1999` ; do grep -L '\#' best_$s.txt ; done > x
        # rm `cat x`
        bests = read_old_populations(toolbox, toolbox.old_populations_folder, "best")
        analyse_population_impl(toolbox, bests)    
        exit()
    if toolbox.new_initial_population:
        population = generate_initial_population_impl(toolbox)
    else:
        # zorg dat de bron van de populatie niet uitmaakt voor de eindtoestand
        # dit was nodig om bij interne opschuddingen hetzelfde resultaat te krijgen als bij set covering op 1 vastloper
        toolbox.reset() # zorgt voor reproduceerbare toolbox
        if old_pops:
            # old_pops is list of individuals
            population = load_initial_population_impl(toolbox, old_pops)
        else:
            # 
            old_pops = read_old_populations(toolbox, toolbox.old_populations_folder, "pop")
            # old_pops is list of deap strings
            population = load_initial_population_impl(toolbox, old_pops)
        random.seed(toolbox.seed) # zorgt voor reproduceerbare state
    return population


def copy_individual(toolbox, ind):
    copy_ind = gp.PrimitiveTree(list(ind[:]))
    copy_ind.fam = ind.fam
    return copy_ind


def cxOnePoint(toolbox, parent1, parent2):
    if len(parent1) < 2 or len(parent2) < 2:
        # No crossover on single node tree
        return None, None
    child = copy_individual(toolbox, parent1)
    index1 = random.randrange(0, len(parent1))
    index2 = random.randrange(0, len(parent2))
    slice1 = parent1.searchSubtree(index1)
    slice2 = parent2.searchSubtree(index2)
    child[slice1] = parent2[slice2]
    pp_str = make_pp_str(child)
    if pp_str in toolbox.ind_str_set or len(child) > toolbox.max_individual_size:
        return None, None
    evaluate_individual(toolbox, child, pp_str, 0)
    return child, pp_str


def is_improvement(toolbox, ind, best):
    if best is None:
        return True
    if best.fam.normalised_error != ind.fam.normalised_error:
        return best.fam.normalised_error > ind.fam.normalised_error
    # now they have equal .normalised_error
    return len(best) > len(ind)


def crossover_with_local_search(toolbox, parent1, parent2):
    if len(parent1) < 2 or len(parent2) < 2:
        # No crossover on single node tree
        return None, None

    # sorting
    t_evaluate = toolbox.t_eval
    if parent1.fam.raw_error > parent2.fam.raw_error or (parent1.fam.raw_error == parent2.fam.raw_error and len(parent1) > len(parent2)):
        parent1, parent2 = parent2, parent1

    # local sesarch
    t0 = time.time()
    indexes1 = [i for i in range(len(parent1))]
    indexes2 = [i for i in range(len(parent2))]
    random.shuffle(indexes1)
    random.shuffle(indexes2)
    best, best_pp_str = None, None
    for index2 in indexes2:
        slice2 = parent2.searchSubtree(index2)
        expr2 = parent2[slice2]
        for index1 in indexes1:
            child = copy_individual(toolbox, parent1)
            slice1 = child.searchSubtree(index1)
            child[slice1] = expr2
            if len(child) <= toolbox.max_individual_size:
                pp_str = make_pp_str(child)
                if pp_str not in toolbox.ind_str_set:
                    evaluate_individual(toolbox, child, pp_str, 0)
                    if is_improvement(toolbox, child, best):
                        best, best_pp_str = child, pp_str
    t_evaluate = toolbox.t_eval - t_evaluate
    toolbox.t_cx_local_search += time.time() - t0 - t_evaluate

    # cx_count administration
    index_a, index_b = parent1.fam.family_index, parent2.fam.family_index
    if index_a > index_b:
        index_a, index_b = index_b, index_a
    key = (index_a, index_b)
    assert index_a <= index_b
    if key not in toolbox.cx_count_dict:
        toolbox.cx_count_dict[key] = 0
    toolbox.cx_count_dict[key] += 1
    if key not in toolbox.cx_child_dict:
        toolbox.cx_child_dict[key] = {-1:0} # -1 is failed cx's
    child_dict = toolbox.cx_child_dict[key]
    if best:
        if best.fam.family_index not in child_dict:
            child_dict[best.fam.family_index] = 0
        child_dict[best.fam.family_index] += 1
    else:
        child_dict[-1] += 1

    return best, best_pp_str


def mutUniform(toolbox, parent, expr, pset):
    child = copy_individual(toolbox, parent)
    index = random.randrange(0, len(child))
    slice_ = child.searchSubtree(index)
    type_ = child[index].ret
    child[slice_] = expr(pset=pset, type_=type_)
    pp_str = make_pp_str(child)
    if pp_str in toolbox.ind_str_set or len(child) > toolbox.max_individual_size:
        return None, None
    evaluate_individual(toolbox, child, pp_str, 0)
    return child, pp_str


def replace_subtree_at_best_location(toolbox, parent, expr):
    indexes = [i for i in range(len(parent))]
    random.shuffle(indexes)
    best = None
    for index in indexes:
        child = copy_individual(toolbox, parent)
        slice1 = child.searchSubtree(index)
        child[slice1] = expr
        if len(child) <= toolbox.max_individual_size:
            pp_str = make_pp_str(child)
            if pp_str not in toolbox.ind_str_set:
                evaluate_individual(toolbox, child, pp_str, 0)
                if is_improvement(toolbox, child, best):
                    best = child
    pp_str = None if best is None else make_pp_str(best) 
    return best, pp_str


def refresh_toolbox_from_population(toolbox, population, population_is_sorted):
    t0 = time.time()
    if not population_is_sorted:
        # population.sort(key=toolbox.sort_ind_key) # influences reproducability with older runs
        pass
    toolbox.ind_str_set = {make_pp_str(ind) for ind in population} # refresh set after deletion of non-fit individuals
    toolbox.current_families_dict = dict()
    for ind in population:
        family_index = ind.fam.family_index        
        if family_index not in toolbox.current_families_dict:
            toolbox.current_families_dict[family_index] = []
        toolbox.current_families_dict[family_index].append(ind)
    if toolbox.dynamic_weights:
        raw_error_matrix_list = []
        if False:
            for index, _ in toolbox.current_families_dict.items():
                family = toolbox.families_list[index]
                raw_error_matrix_list.append(family.raw_error_matrix)
        else:
            for family in toolbox.families_list:
                raw_error_matrix_list.append(family.raw_error_matrix)
        best_raw_error_matrix = population[0].fam.raw_error_matrix
        dynamic_weights.update_dynamic_weights(toolbox.prev_best_raw_error_matrix, best_raw_error_matrix, \
            raw_error_matrix_list, toolbox.dynamic_weights_adaptation_speed)
        dynamic_weights.log_info(toolbox.f)
        toolbox.prev_best_raw_error_matrix = best_raw_error_matrix
        for family in toolbox.families_list:
            family.update_normalised_error()
    # always sort!
    population.sort(key=toolbox.sort_ind_key)
    toolbox.t_refresh += time.time() - t0


def consistency_check_ind(toolbox, ind):
    if ind is not None:
        assert hasattr(ind, "fam")
        assert ind.fam is not None

        assert not hasattr(ind, "deap_str")
        assert not hasattr(ind, "pp_str")
        assert not hasattr(ind, "parents")
        assert not hasattr(ind, "eval")
        assert not hasattr(ind, "raw_error") # voor weergave aan mens
        assert not hasattr(ind, "normalised_error") # voor vergelijken in local search en voor sorteren populatie
        assert not hasattr(ind, "family_index")


def consistency_check(toolbox, inds):
    t0 = time.time()
    for ind in inds:
        consistency_check_ind(toolbox, ind)
    toolbox.t_consistency_check += time.time() - t0


def write_seconds(toolbox, timings, header):
    file_name = toolbox.params["output_folder"] + "/time_" + str(toolbox.params["seed"]) + ".txt"
    with open(file_name, "a") as f:
        total = 0
        for t, label in timings:
            total += t
        if total > 0:
            f.write(f"\n{header}\n")
            totalp = 0
            for t, label in timings:
                if t > 0:
                    f.write(f"{t:.0f}\tsec\t{round(100*t/total)}\t%\t{label}\n")
                totalp += round(100*t/total)
            f.write(f"{total:.0f}\tsec\t{totalp}\t%\ttotal\n")


def write_timings(toolbox, header):
    if hasattr(toolbox, "t_total"):
        timings = [
            (toolbox.t_total, "t_total"),
            ]
        write_seconds(toolbox, timings, header)

    timings = [
        (toolbox.t_init, "init(total)"),
        (toolbox.t_offspring, "offspring"),
        (toolbox.t_refresh, "refresh"),
        (toolbox.t_consistency_check, "consistency_check"),
        ]
    timings.sort(key=lambda item: -item[0])
    write_seconds(toolbox, timings, header + " breakdown of total")

    timings = [
        (toolbox.t_select_parents, "t_select_parents"),
        (toolbox.t_cx, "t_cx"),
        (toolbox.t_mut, "t_mut"),
        ]
    timings.sort(key=lambda item: -item[0])
    write_seconds(toolbox, timings, header + " breakdown of offspring")

    timings = [
        (toolbox.t_eval, "t_eval"),
        ]
    timings.sort(key=lambda item: -item[0])
    write_seconds(toolbox, timings, header + " t_eval")

    timings = [
        (toolbox.t_cpp_interpret, "cpp_interpret"),
        (toolbox.t_py_interpret, "py_interpret"),
        (toolbox.t_error, "error"),
        ]
    timings.sort(key=lambda item: -item[0])
    write_seconds(toolbox, timings, header + " breakdown of t_eval")



