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
from cpp_coupling import get_cpp_handle, run_on_all_inputs

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
        run_on_all_inputs(toolbox.cpp_handle, ind, debug=2)
        print("py_model_outputs", py_model_outputs)
        print("cpp_model_outputs", cpp_model_outputs)
        print("toolbox.eval_count OK", toolbox.eval_count - 1)
        code_str = interpret.convert_code_to_str(code)
        print(code_str)
    assert py_model_outputs == cpp_model_outputs


class Family:
    def __init__(self, model_outputs, raw_error_matrix):
        self.model_outputs = model_outputs
        self.raw_error_matrix = raw_error_matrix
        self.raw_error = evaluate.compute_raw_error(self.raw_error_matrix)
        self.update_normalised_error()

    def update_normalised_error(self):
        self.normalised_error = dynamic_weights.compute_normalised_error(self.raw_error_matrix, 1.0)


def forced_reevaluation_of_individual_for_debugging(toolbox, ind, debug_level):
    '''Assigns family index to the individual'''
    model_outputs = run_on_all_inputs(toolbox.cpp_handle, ind)
    raw_error_matrix = evaluate.compute_raw_error_matrix(toolbox.example_inputs, model_outputs, toolbox.error_function, \
        toolbox.f, toolbox.verbose)
    raw_error = evaluate.compute_raw_error(raw_error_matrix)
    return raw_error


def evaluate_individual_impl(toolbox, ind, debug=0):
    '''Assigns family index to the individual'''
    if toolbox.eval_count >= toolbox.max_evaluations:
        raise RuntimeWarning("max evaluations reached")
    toolbox.eval_count += 1

    t0 = time.time()
    model_outputs = run_on_all_inputs(toolbox.cpp_handle, ind)
    toolbox.t_cpp_interpret += time.time() - t0
    if False:
        test_against_python_interpreter(toolbox, model_outputs, ind)

    t0 = time.time()
    model_outputs_tuple = recursive_tuple(model_outputs)
    if model_outputs_tuple in toolbox.families_dict:
        ind.family_index = toolbox.families_dict[model_outputs_tuple]
    else:
        raw_error_matrix = evaluate.compute_raw_error_matrix(toolbox.example_inputs, model_outputs, toolbox.error_function, \
            toolbox.f, debug, toolbox.penalise_non_reacting_models)
        ind.family_index = len(toolbox.families_list)
        toolbox.families_dict[model_outputs_tuple] = ind.family_index
        toolbox.families_list.append(Family(model_outputs, raw_error_matrix))
    toolbox.t_error += time.time() - t0


def evaluate_individual(toolbox, individual, debug=0):
    if time.time() >= toolbox.t0 + toolbox.max_seconds:
        raise RuntimeWarning("out of time")
    t0 = time.time()
    pp_str = individual.pp_str
    assert pp_str == make_pp_str(individual)
    toolbox.eval_lookup_count += 1
    if pp_str in toolbox.pp_str_to_family_index_dict:
        individual.family_index = toolbox.pp_str_to_family_index_dict[pp_str]
    else:
        evaluate_individual_impl(toolbox, individual, debug)
        toolbox.pp_str_to_family_index_dict[pp_str] = individual.family_index
    family = toolbox.families_list[individual.family_index]
    assert type(family.raw_error) == type(0.0)
    individual.raw_error = family.raw_error
    assert type(family.normalised_error) == np.float64
    individual.normalised_error = family.normalised_error
    toolbox.t_eval += time.time() - t0


def best_of_n(population, n):
    inds = random.sample(population, n) # sample always returns a list
    ind = min(inds, key=lambda ind: ind.normalised_error)
    return ind



def log_population(toolbox, population, label):
    toolbox.f.write(f"write_population {label}\n")
    for i, ind in enumerate(population):
        toolbox.f.write(f"    ind {i} {ind.raw_error:.3f} {len(ind)} {ind(ind)}\n")
    toolbox.f.write("\n")
    toolbox.f.flush()


def write_population(file_name, population, functions):
    with open(file_name, "w") as f:
        f.write("(\n")
        for ind in population:
            code = interpret.compile_deap(str(ind), functions)
            code_str = interpret.convert_code_to_str(code)
            f.write(f"    {code_str} # {ind.raw_error:.3f}\n")
        f.write(")\n")


def write_final_population(toolbox, population):
    with open(toolbox.pop_file, "w") as f:
        f.write("(\n")
        for ind in population:
            code = interpret.compile_deap(str(ind), toolbox.functions)
            code_str = interpret.convert_code_to_str(code)
            f.write(f"    {code_str} # {ind.raw_error:.3f}\n")
        f.write(")\n")


def write_path(toolbox, ind, indent=0):
    #indent_str = "\t" * indent
    # operator_str = ["", "mutatie", "crossover"][len(ind.parents)]
    code = interpret.compile_deap(str(ind), toolbox.functions)
    code_str = interpret.convert_code_to_str(code)
    if False:
        if indent:
            toolbox.f.write(f"parent\t{ind.raw_error:.3f}\tcode\t{code_str}\n")
        else:
            toolbox.f.write(f"child\t{ind.raw_error:.3f}\tcode\t{code_str}\n")
    else:
        toolbox.f.write(f"{code_str} # {ind.raw_error:.3f} len {len(ind)}\n")

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
        ind.pp_str = make_pp_str(ind)
        if ind.pp_str in toolbox.ind_str_set or len(ind) > toolbox.max_individual_size:
            if retry_count < toolbox.child_creation_retries:
                retry_count += 1
                continue
            else:
                break
        retry_count = 0
        toolbox.ind_str_set.add(ind.pp_str)
        ind.parents = []
        evaluate_individual(toolbox, ind)
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
                    ind.pp_str = make_pp_str(ind)
                    assert len(ind) == deap_len_of_code(code)
                if ind.pp_str in toolbox.ind_str_set or len(ind) > toolbox.max_individual_size:
                    count_skipped += 1
                    continue
                toolbox.ind_str_set.add(ind.pp_str)
                ind.parents = []
                evaluate_individual(toolbox, ind)
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
            ind.pp_str = make_pp_str(ind)
            assert len(ind) == deap_len_of_code(code)
            ind.parents = []
            evaluate_individual(toolbox, ind) # required to set ind.family_index
            if ind.family_index not in families:
                families[ind.family_index] = [ind.family_index, ind.pp_str, 0]
            families[ind.family_index][2] += 1
            if len(families[ind.family_index][1]) > len(ind.pp_str):
                families[ind.family_index][1] = ind.pp_str
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
            outputs = str(toolbox.families_list[family_index].model_outputs[-1])
            raw_error_vector = toolbox.families_list[family_index].raw_error_matrix[-1]
            sum_raw_error_vector += np.array(raw_error_vector)
            raw_error_vector = " ".join([f"{v:5.3f}" for v in raw_error_vector])
            sum_count += count
            nor_error = toolbox.families_list[family_index].normalised_error
            f.write(f"{raw_error:8.3f} {nor_error:8.3f} {family_index:6d} {count:5d} {outputs[:48]:48}")
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
    #consistency_check_ind(toolbox, ind)
    copy_ind = gp.PrimitiveTree(list(ind[:]))
    copy_ind.pp_str = ind.pp_str
    copy_ind.parents = [parent for parent in ind.parents]
    copy_ind.raw_error = ind.raw_error
    copy_ind.normalised_error = ind.normalised_error
    copy_ind.family_index = ind.family_index
    #consistency_check_ind(toolbox, copy_ind)
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
    child.pp_str = make_pp_str(child)
    if child.pp_str in toolbox.ind_str_set or len(child) > toolbox.max_individual_size:
        return None
    evaluate_individual(toolbox, child)
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
    if best.normalised_error != ind.normalised_error:
        return best.normalised_error > ind.normalised_error
    # now they have equal .normalised_error
    return len(best) > len(ind)


def crossover_with_local_search(toolbox, parent1, parent2):
    if len(parent1) < 2 or len(parent2) < 2:
        # No crossover on single node tree
        return None
    t0 = time.time()
    t_evaluate = toolbox.t_eval
    if parent1.raw_error > parent2.raw_error or (parent1.raw_error == parent2.raw_error and len(parent1) > len(parent2)):
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
                child.pp_str = make_pp_str(child)
                if child.pp_str not in toolbox.ind_str_set:
                    evaluate_individual(toolbox, child)
                    if is_improvement(toolbox, child, best):
                        best = child
    t_evaluate = toolbox.t_eval - t_evaluate
    toolbox.t_cx_local_search += time.time() - t0 - t_evaluate
    return best


def mutUniform(toolbox, parent, expr, pset):
    child = copy_individual(toolbox, parent)
    child.parents = [parent,] if toolbox.keep_path else []
    index = random.randrange(0, len(child))
    slice_ = child.searchSubtree(index)
    type_ = child[index].ret
    child[slice_] = expr(pset=pset, type_=type_)
    child.pp_str = make_pp_str(child)
    if child.pp_str in toolbox.ind_str_set or len(child) > toolbox.max_individual_size:
        return None
    evaluate_individual(toolbox, child)
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
            child.pp_str = make_pp_str(child)
            if child.pp_str not in toolbox.ind_str_set:
                evaluate_individual(toolbox, child)
                if is_improvement(toolbox, child, best):
                    best = child
    return best


def refresh_toolbox_from_population(toolbox, population, population_is_sorted):
    t0 = time.time()
    if not population_is_sorted:
        # population.sort(key=toolbox.sort_ind_key) # influences reproducability with older runs
        pass
    toolbox.ind_str_set = {ind.pp_str for ind in population} # refresh set after deletion of non-fit individuals
    toolbox.current_families_dict = dict()
    for ind in population:
        if ind.family_index not in toolbox.current_families_dict:
            toolbox.current_families_dict[ind.family_index] = []
        toolbox.current_families_dict[ind.family_index].append(ind)
    toolbox.offspring_families_dict = dict()
    if toolbox.dynamic_weights:
        raw_error_matrix_list = []
        if False:
            for index, _ in toolbox.current_families_dict.items():
                family = toolbox.families_list[index]
                raw_error_matrix_list.append(family.raw_error_matrix)
        else:
            for family in toolbox.families_list:
                raw_error_matrix_list.append(family.raw_error_matrix)
        best_raw_error_matrix = toolbox.families_list[population[0].family_index].raw_error_matrix
        dynamic_weights.update_dynamic_weights(toolbox.prev_best_raw_error_matrix, best_raw_error_matrix, \
            raw_error_matrix_list, toolbox.dynamic_weights_adaptation_speed)
        dynamic_weights.log_info(toolbox.f)
        toolbox.prev_best_raw_error_matrix = best_raw_error_matrix
        for family in toolbox.families_list:
            family.update_normalised_error()
        for ind in population:
            family = toolbox.families_list[ind.family_index]
            ind.raw_error = family.raw_error
            ind.normalised_error = family.normalised_error
    # always sort!
    population.sort(key=toolbox.sort_ind_key)
    toolbox.t_refresh += time.time() - t0


def consistency_check_ind(toolbox, ind):
    if ind is not None:
        assert not hasattr(ind, "deap_str")
        assert hasattr(ind, "pp_str")
        assert hasattr(ind, "parents")
        assert not hasattr(ind, "eval")
        assert hasattr(ind, "raw_error") # voor weergave aan mens
        assert hasattr(ind, "normalised_error") # voor vergelijken in local search en voor sorteren populatie
        assert hasattr(ind, "family_index")
        assert ind.pp_str == make_pp_str(ind)
        assert ind.raw_error is not None
        assert ind.normalised_error is not None
        assert math.isclose(ind.raw_error, toolbox.families_list[ind.family_index].raw_error)
        assert math.isclose(ind.normalised_error, toolbox.families_list[ind.family_index].normalised_error)


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



