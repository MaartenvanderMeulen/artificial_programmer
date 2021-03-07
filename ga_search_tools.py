# Inspired by https://deap.readthedocs.io/en/master/examples/gp_symbreg.html
# it is used by search.py
import os
import random
import copy
import math
import time
import json
import numpy as np
from autopep8 import fix_code

import interpret
import evaluate
import dynamic_weights
from evaluate import recursive_tuple
import cpp_coupling

from deap import gp #  gp.PrimitiveSet, gp.genHalfAndHalf, gp.PrimitiveTree, gp.genFull, gp.from_string


def make_pp_str(ind):
    return " ".join([x.name for x in ind])


def get_fam_info(fam):
    return f"<{fam.family_index}> error {fam.raw_error:.3f}, age_in_pop {fam.age_in_population}"


def get_ind_info(ind):
    return f"[{ind.id}] age {ind.age}, len {len(ind)}, {get_fam_info(ind.fam)}"


def get_ind_node_id(ind):
    return f"<{ind.fam.family_index}>"


def myfix_code(ind):
    return fix_code(str(ind).replace("for", "FOR"))

def test_against_python_interpreter(toolbox, cpp_model_outputs, ind):
    code = interpret.compile_deap(str(ind), toolbox.functions)
    toolbox.functions[toolbox.problem_name] = [toolbox.formal_params, code]
    py_model_outputs = []
    for input in toolbox.example_inputs:
        model_output = interpret.run([toolbox.problem_name] + input, dict(), toolbox.functions, debug=toolbox.verbose >= 5)
        py_model_outputs.append(model_output)        
    if py_model_outputs != cpp_model_outputs:
        cpp_coupling.run_on_all_inputs(toolbox.cpp_handle, ind, debug=2)
        print("py_model_outputs", py_model_outputs)
        print("cpp_model_outputs", cpp_model_outputs)
        print("toolbox.eval_count OK", toolbox.eval_count - 1)
        code_str = interpret.convert_code_to_str(code)
        print(code_str)
    assert py_model_outputs == cpp_model_outputs


class Family:
    def __init__(self, family_index, raw_error_matrix, representative):
        self.family_index = family_index
        self.raw_error_matrix = raw_error_matrix
        self.raw_error = evaluate.compute_raw_error(self.raw_error_matrix)
        self.representative = representative
        self.age = 0
        self.age_in_population = 0
        self.update_normalised_error()

    def update_normalised_error(self):
        self.normalised_error = dynamic_weights.compute_normalised_error(self.raw_error_matrix, 1.0)


def forced_reevaluation_of_individual_for_debugging(toolbox, ind, debug_level):
    '''Assigns family index to the individual'''
    cpp_model_outputs = cpp_coupling.run_on_all_inputs(toolbox.cpp_handle, ind)
    test_against_python_interpreter(toolbox, cpp_model_outputs, ind)
    raw_error_matrix = evaluate.compute_raw_error_matrix(toolbox.example_inputs, cpp_model_outputs, toolbox.error_function, \
        toolbox.f, debug_level, False)
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
    raw_error_matrix_cpp, family_key = cpp_coupling.compute_error_matrix(toolbox.cpp_handle, ind, \
        toolbox.penalise_non_reacting_models, toolbox.families_dict, toolbox.family_key_is_error_matrix)

    # bepaling family
    if family_key in toolbox.families_dict:
        family_index = toolbox.families_dict[family_key]
        ind.fam = toolbox.families_list[family_index]
        if ind.fam.representative is None or len(ind.fam.representative) > len(ind):
            #if ind.fam.raw_error <= toolbox.max_raw_error_for_family_db:
            #    toolbox.f.write(f"at gen {toolbox.real_gen}, found shorter representative of {get_fam_info(ind.fam)}\n")
            ind.fam.representative = ind # keep shortest representative
    else:
        family_index = len(toolbox.families_list)
        toolbox.families_dict[family_key] = family_index
        ind.fam = Family(family_index, raw_error_matrix_cpp, ind)
        toolbox.families_list.append(ind.fam)
        #if ind.fam.raw_error <= toolbox.max_raw_error_for_family_db:
        #    toolbox.f.write(f"at gen {toolbox.real_gen}, new fam {get_fam_info(ind.fam)}\n")
        if ind.fam.raw_error <= toolbox.max_raw_error_for_family_db:
            toolbox.new_families_list.append(ind.fam)
        # piggyback test : vergelijk uitkomst cpp met python implementatie
        if False:
            model_outputs_py = cpp_coupling.run_on_all_inputs(toolbox.cpp_handle, ind)
            raw_error_matrix_py = evaluate.compute_raw_error_matrix(toolbox.example_inputs, model_outputs_py, toolbox.error_function, \
                toolbox.f, debug, toolbox.penalise_non_reacting_models)
            check_error_matrices(raw_error_matrix_py, raw_error_matrix_cpp)



def evaluate_individual(toolbox, individual, pp_str, debug):
    assert type(pp_str) == type("") and type(debug) == type(1)
    if pp_str in toolbox.pp_str_to_family_index_dict:
        family_index = toolbox.pp_str_to_family_index_dict[pp_str]
        individual.fam = toolbox.families_list[family_index]
    else:
        toolbox.eval_count += 1
        evaluate_individual_impl(toolbox, individual, debug)
        toolbox.pp_str_to_family_index_dict[pp_str] = individual.fam.family_index


def best_of_n(population, n):
    inds = random.sample(population, n) # sample always returns a list
    ind = min(inds, key=lambda ind: ind.fam.normalised_error)
    return ind


def write_population(file_name, population, functions):
    with open(file_name, "w") as f:
        f.write("(\n")
        for ind in population:
            code = interpret.compile_deap(str(ind), functions)
            code_str = interpret.convert_code_to_str(code)
            f.write(f"    {code_str}")
            if ind.fam:
                f.write(f" # {ind.fam.raw_error:.1f}")
            f.write(f"\n")
        f.write(")\n")


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
    key = (index_a, index_b)
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
    if toolbox.count_cx > 0:
        toolbox.f.write(f"fraction cx into current pop {toolbox.count_cx_into_current_pop / toolbox.count_cx:.3f}\n")
    results = []
    sum_cx_count = 0
    for key, cx_count in toolbox.cx_count_dict.items():
        index_a, index_b = key
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


def write_cx_graph(toolbox):
    filename = f"{toolbox.output_folder}/cx_{toolbox.id_seed}.txt"
    with open(filename, "w") as f:
        for (a, b), childs in toolbox.cx_child_dict.items():
            for c, n in childs.items():
                if c >= 0: # -1 are the failed axb cx's
                    f.write(f"p{a}  p{b} c{c} {n}\n")


def generate_initial_population_impl(toolbox):
    toolbox.ind_str_set = set()
    population = []
    retry_count = 0
    while len(population) < toolbox.pop_size[0]:
        ind = gp.PrimitiveTree(gp.genHalfAndHalf(pset=toolbox.pset, min_=2, max_=4))
        ind.age = 0
        ind.id = toolbox.get_unique_id()
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


def remove_file(filename):
    if os.path.exists(filename):
        os.remove(filename)


def read_old_populations(toolbox, old_populations_folder, prefix):
    if toolbox.update_fam_db or toolbox.analyse_best:
        print(f"reading files in {old_populations_folder} ...")
    old_pops = []
    filenames = []
    for filename in os.listdir(old_populations_folder):
        if filename[:len(prefix)] == prefix:
            id = int(filename[len(prefix)+1:len(prefix)+1+4])
            if id // toolbox.old_populations_samplesize == toolbox.id_seed // toolbox.old_populations_samplesize:
                filenames.append(filename)
    filenames.sort()
    for filename in filenames:
        if toolbox.old_populations_samplesize != 1 or filename == f"{prefix}_{toolbox.id_seed}.txt":
            old_pop = interpret.compile(interpret.load(old_populations_folder + "/" + filename))
            if len(old_pop) > 0:
                old_pops.append(old_pop)
            elif toolbox.old_populations_samplesize == 1:
                toolbox.f.write("RuntimeWarning: stopped because no set covering needed, 0 evals\n")
                exit()
    if toolbox.old_populations_samplesize != 1:
        if toolbox.old_populations_samplesize < len(old_pops):
            old_pops = random.sample(old_pops, k=toolbox.old_populations_samplesize)
    if toolbox.update_fam_db or toolbox.analyse_best:
        print(f"    {len(old_pops)} files with content")
    return old_pops


def extract_best_fam_from_cx_file(cx_file):
    best_fam = None
    with open(cx_file, "r") as f:
        for line in f:
            parts = line.strip().lower().split(" ")
            ab, c, n = int(parts[0]), int(parts[1]), int(parts[2])
            if best_fam is None or best_fam > ab:
                best_fam = ab
            if best_fam is None or best_fam > c:
                best_fam = c
    return best_fam


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
                    ind.age = 0
                    ind.id = toolbox.get_unique_id()
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


def analyse_vastlopers_via_cx_files_and_family_db(toolbox):
    print(f"analysis starts, reading files ...")
    families = dict()
    file_count = 0
    with open(f"{toolbox.output_folder}/f4.txt", "w") as g:
        for filename in os.listdir(toolbox.old_populations_folder):
            if filename.startswith("cx"):
                family_index = extract_best_fam_from_cx_file(toolbox.old_populations_folder + "/" + filename)
                if family_index > 0 and os.system(f"grep -q 'solved' {toolbox.old_populations_folder}/log_{filename[3:7]}.txt") == 0:
                    print(f"run {filename[3:7]} is solved via mutations")
                    family_index = 0
                if family_index == 4:
                    g.write(f"{filename[3:7]}\n")
                if family_index < len(toolbox.families_list):
                    file_count += 1
                    if family_index not in families:
                        families[family_index] = []
                    families[family_index].append(int(filename[3:7]))
                else:
                    print(filename, "skipped")
    families = [(index, aantal) for index, aantal in families.items()]
    families.sort(key=lambda item: -toolbox.families_list[item[0]].raw_error)
    filename = f"{toolbox.output_folder}/analysis.txt"
    print(f"writing anaysis result of {file_count} files in {filename} ...")
    with open(filename, "w") as f:
        f.write(f"{'error':5} fam_id count  {'last_raw_error':16}\n")
        sum_count = 0
        for family_index, runs in families:
            raw_error = toolbox.families_list[family_index].raw_error
            m = toolbox.families_list[family_index].raw_error_matrix
            msg = ""
            for i in range(m.shape[0]):
                msg += "|"
                for j in range(m.shape[1]):
                    msg += "1" if m[i, j] > 0 else "."
            f.write(f"{raw_error:5.1f} {family_index:6d} {len(runs):5d}  {msg}\n")
            sum_count += len(runs)
            if len(runs) == 1:
                f.write(f"    run_ids {str(runs)}\n")
        f.write(f"{' ':5} {len(toolbox.families_list):6} {sum_count:5}\n")
    print(f"anaysis done")


def analyse_vastlopers_via_best_files_no_family_db(toolbox):
    print(f"analysis starts, reading files ...")
    families = dict()
    file_count = 0
    for filename in os.listdir(toolbox.old_populations_folder):
        if filename.startswith("best"):
            best_list = interpret.compile(interpret.load(toolbox.old_populations_folder + "/" + filename))
            if len(best_list) > 0:
                file_count += 1
                assert len(best_list) == 1
                code = best_list[0]
                deap_str = interpret.convert_code_to_deap_str(code, toolbox)
                ind = gp.PrimitiveTree.from_string(deap_str, toolbox.pset)                    
                ind.age = 0
                ind.id = toolbox.get_unique_id()
                pp_str = make_pp_str(ind)
                evaluate_individual(toolbox, ind, pp_str, 0)
                key = ind.fam.raw_error
                if key not in families:
                    families[key] = ([], ind.fam.raw_error, ind.fam.raw_error_matrix)
                id_seed = int(filename[5:9])
                families[key][0].append(id_seed)
    families = [(raw_error, raw_error_matrix, seeds) for key, (seeds, raw_error, raw_error_matrix) in families.items()]
    families.sort(key=lambda item: -item[0])
    filename = f"{toolbox.output_folder}/analysis.txt"
    print(f"writing anaysis result of {file_count} files in {filename} ...")
    with open(filename, "w") as f:
        f.write(f"{'error':5} count  {'last_raw_error'}\n")
        sum_count = 0
        for raw_error, raw_error_matrix, seeds in families:
            msg = ""
            for i in range(raw_error_matrix.shape[0]):
                msg += "|"
                for j in range(raw_error_matrix.shape[1]):
                    msg += "1" if raw_error_matrix[i, j] > 0 else "."
            f.write(f"{raw_error:5.1f} {len(seeds):5d}  {msg}\n")
            sum_count += len(seeds)
        f.write(f"{' ':5} {sum_count:5}\n")
    filename = f"{toolbox.output_folder}/sorted_seeds.txt"
    print(f"writing sorted seeds in {filename} ...")
    with open(filename, "w") as f:
        for raw_error, raw_error_matrix, seeds in families:
            f.write(f"{raw_error:5.1f} {len(seeds):5d}")
            seeds.sort()
            for id_seed in seeds:
                f.write(f" {id_seed}")
            f.write(f"\n")
    print(f"anaysis done")


def compute_p_cx_c0_db(toolbox):
    print(f"p(cx-->c0) starts, reading files ...")
    p_cx_c0_db = dict()
    for filename in os.listdir(toolbox.old_populations_folder):
        if filename.startswith("cx"):
            with open(toolbox.old_populations_folder + "/" + filename, "r") as f:
                for line in f:
                    parts = line.strip().lower().split(" ")
                    p1, p2, c, n = int(parts[0][1:]), int(parts[2][1:]), int(parts[3][1:]), int(parts[4])
                    if p1 < len(toolbox.families_list) and p2 < len(toolbox.families_list):
                        if (p1, p2) not in p_cx_c0_db:
                            p_cx_c0_db[(p1, p2)] = [0, 0]
                        if c > 1:
                            c = 1
                        p_cx_c0_db[(p1, p2)][c] += n
    filename = f"{toolbox.output_folder}/p_cx_c0_db.txt"
    with open(filename, "w") as f:
        for (p1, p2), counts in p_cx_c0_db.items():
            if counts[0] > 0:
                f.write(f"{p1} {p2} {counts[0]} {counts[0] + counts[1]}\n")
    print(f"p(cx-->c0) done, DB written in {filename}")


def read_p_cx_c0_db(toolbox):
    toolbox.p_cx_c0_db = dict()
    toolbox.p_family_in_cx_c0_db = dict()
    with open(toolbox.p_cx_c0_db_file, "r") as f:
        for line in f:
            parts = line.strip().lower().split(" ")
            index1, index2, p_cx_c0 = int(parts[0]), int(parts[1]), float(parts[2])
            toolbox.p_cx_c0_db[(index1, index2)] = p_cx_c0
            for index in [index1, index2]:
                if index not in toolbox.p_family_in_cx_c0_db:
                    toolbox.p_family_in_cx_c0_db[index] = 0.0
                toolbox.p_family_in_cx_c0_db[index] = max(toolbox.p_family_in_cx_c0_db[index], p_cx_c0)


def read_family_db(toolbox):
    # toolbox.f.write("reading families db, please have some patience\n")
    if toolbox.update_fam_db or toolbox.analyse_best or toolbox.compute_p_cx_c0:
        print(f"reading families db in {toolbox.fam_db_file} ...")
        t0 = time.time()
    families = interpret.compile(interpret.load(toolbox.fam_db_file))
    if toolbox.update_fam_db or toolbox.analyse_best or toolbox.compute_p_cx_c0:
        print(f"    {round(time.time() - t0)} seconds for reading {len(families)} families")
        t0 = time.time()
    for code in families:
        deap_str = interpret.convert_code_to_deap_str(code, toolbox)
        ind = gp.PrimitiveTree.from_string(deap_str, toolbox.pset)                    
        ind.age = 0
        ind.id = toolbox.get_unique_id()
        pp_str = make_pp_str(ind)
        evaluate_individual(toolbox, ind, pp_str, 0)
        if toolbox.clear_representatives_after_reading_family_db:
            # Prevent that the extra short DB snippets will influence the search : remove the code
            ind.fam.representative = None # only the family NUMBER may be used
    toolbox.new_families_list = []
    if toolbox.update_fam_db or toolbox.analyse_best or toolbox.compute_p_cx_c0:
        elapsed = round(time.time() - t0)
        if elapsed > 0:
            print(f"    {elapsed} seconds for processing, {round(len(toolbox.families_list)/elapsed)} families/sec")
    toolbox.t0 = time.time() # discard time lost by reading in the family db


def update_fams(toolbox, newfams_list):
    print("merging new families ...")
    len_at_start = len(toolbox.families_list)
    t0 = time.time()
    for i, newfams in enumerate(newfams_list):
        if time.time() >= t0 + 10:
            t0 = time.time()
            print("    progress", i, "/", len(newfams_list))
        for representative in newfams:
            deap_str = interpret.convert_code_to_deap_str(representative, toolbox)
            representative = gp.PrimitiveTree.from_string(deap_str, toolbox.pset)
            representative.age = 0
            representative.id = toolbox.get_unique_id()
            pp_str = make_pp_str(representative)
            evaluate_individual(toolbox, representative, pp_str, 0)
    all_families = [family.representative for family in toolbox.families_list if family.raw_error <= toolbox.max_raw_error_for_family_db]
    all_families.sort(key=lambda item: item.fam.raw_error)
    write_population(toolbox.fam_db_file + ".update", all_families, toolbox.functions)
    len_at_end = len(toolbox.families_list)
    print(f"    {len_at_end - len_at_start} new families added to the db.  Total now {len_at_end}.")


def generate_initial_population(toolbox, old_pops=None):
    if toolbox.analyse_best:
        analyse_vastlopers_via_best_files_no_family_db(toolbox)
        exit()
    read_family_db(toolbox)
    if toolbox.update_fam_db:
        newfams = read_old_populations(toolbox, toolbox.old_populations_folder, "newfam")
        update_fams(toolbox, newfams)
        exit()
    if toolbox.analyse_cx:
        analyse_vastlopers_via_cx_files_and_family_db(toolbox)
        exit()
    if toolbox.compute_p_cx_c0:
        compute_p_cx_c0_db(toolbox)
        exit()
    read_p_cx_c0_db(toolbox)
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
        random.seed(toolbox.random_seed) # zorgt voor reproduceerbare state
    return population


def copy_individual(toolbox, ind):
    copy_ind = gp.PrimitiveTree(list(ind[:]))
    copy_ind.fam = ind.fam
    copy_ind.age = 0
    copy_ind.id = toolbox.get_unique_id()
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


global debug_index1, debug_index2
debug_index1, debug_index2 = 0, 0

def crossover_with_local_search(toolbox, parent1, parent2, do_shuffle=True, debug=0):
    if len(parent1) < 2 or len(parent2) < 2:
        # No crossover on single node tree
        return None, None

    # local sesarch
    indexes1 = [i for i in range(len(parent1))]
    indexes2 = [i for i in range(len(parent2))]
    if do_shuffle:
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
                    if not toolbox.in_near_solution_area or child.fam.family_index not in toolbox.offspring_families_set:
                        if is_improvement(toolbox, child, best):
                            best, best_pp_str = child, pp_str

    # cx_count administration
    index_a, index_b = parent1.fam.family_index, parent2.fam.family_index
    key = (index_a, index_b)
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
        if best.fam.family_index in toolbox.current_families_dict:
            toolbox.count_cx_into_current_pop += 1
        toolbox.count_cx += 1
    else:
        child_dict[-1] += 1

    if best and toolbox.in_near_solution_area:
        repr = best.fam.representative
        if repr is not None:
            slice_best = best.searchSubtree(0)
            slice_repr = repr.searchSubtree(0)
            best[slice_best] = repr[slice_repr]

    # near solution debugging
    if best and toolbox.in_near_solution_area and best.fam.raw_error <= toolbox.max_raw_error_for_family_db:
        toolbox.near_solution_families_set.add(best.fam.family_index)

    # escape info
    if best:
        best.msg = f"at gen {toolbox.real_gen}, {get_ind_info(best)} = cx({get_ind_info(parent1)},{get_ind_info(parent2)})"
        toolbox.graph.add_cx(get_ind_node_id(parent1), get_ind_node_id(parent2), get_ind_node_id(best), toolbox.real_gen)
        if toolbox.stuck_count > 50 and best.fam.raw_error < toolbox.population[0].fam.raw_error:
            toolbox.escape_counter += 1
            escape_id = toolbox.escape_counter
            toolbox.f.write(f"escape {escape_id}\n")
            toolbox.f.write(f"escape {escape_id} via crossover, stuck_count {toolbox.stuck_count}\n")
            toolbox.f.write(f"escape {escape_id} pop[0] {get_ind_info(toolbox.population[0])}\n")
            toolbox.f.write(f"escape {escape_id}\n")
            toolbox.f.write(f"escape {escape_id} child {best.msg}\n")
            toolbox.f.write(f"escape {escape_id}\n")
            toolbox.f.write(f"escape {escape_id} {str(best)}\n")
            toolbox.f.write(f"escape {escape_id}\n")
            toolbox.f.write(f"escape {escape_id} parent1 {parent1.msg}\n")
            toolbox.f.write(f"escape {escape_id}\n")
            toolbox.f.write(f"escape {escape_id} {str(parent1)}\n")
            toolbox.f.write(f"escape {escape_id}\n")
            toolbox.f.write(f"escape {escape_id} parent2 {parent2.msg}\n")
            toolbox.f.write(f"escape {escape_id}\n")
            toolbox.f.write(f"escape {escape_id} {str(parent2)}\n")
            toolbox.f.write(f"escape {escape_id}\n")
            toolbox.graph.write_tree_to_dst(toolbox.f, get_ind_node_id(best), f"escape{escape_id}", toolbox.real_gen)
        else:
            f1, f2 = parent1.fam.family_index, parent2.fam.family_index
            toolbox.f.write(f"at gen {toolbox.real_gen}, [{best.id}] = {get_ind_info(best)} = cx [{parent1.id}]<{f1}> [{parent2.id}]<{f2}>\n")
            toolbox.f.write(f"at gen {toolbox.real_gen}, [{best.id}] = {str(best)}\n")
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
    n = int(toolbox.mut_local_search * len(indexes))
    indexes = indexes[:n]
    best = None
    for index in indexes:
        child = copy_individual(toolbox, parent)
        slice1 = child.searchSubtree(index)
        child[slice1] = expr
        if len(child) <= toolbox.max_individual_size:
            pp_str = make_pp_str(child)
            if pp_str not in toolbox.ind_str_set:
                evaluate_individual(toolbox, child, pp_str, 0)
                if not toolbox.in_near_solution_area or child.fam.family_index not in toolbox.offspring_families_set:                    
                    if is_improvement(toolbox, child, best):                        
                        best = child
    if best and toolbox.in_near_solution_area:
        repr = best.fam.representative
        if repr is not None:
            slice_best = best.searchSubtree(0)
            slice_repr = repr.searchSubtree(0)
            best[slice_best] = repr[slice_repr]

    pp_str = None if best is None else make_pp_str(best) 
    if best:        
        expr_str = str(expr)
        best.msg = f"at gen {toolbox.real_gen}, {get_ind_info(best)} = mut({get_ind_info(parent)},{expr_str})"
        toolbox.graph.add_mut(get_ind_node_id(parent), expr_str, get_ind_node_id(best), toolbox.real_gen)
        if toolbox.stuck_count > 50 and best.fam.raw_error < toolbox.population[0].fam.raw_error:
            toolbox.escape_counter += 1
            escape_id = toolbox.escape_counter
            toolbox.f.write(f"escape {escape_id}\n")
            toolbox.f.write(f"escape {escape_id} via mutatie, stuck_count {toolbox.stuck_count}\n")
            toolbox.f.write(f"escape {escape_id} pop[0] {get_ind_info(toolbox.population[0])}\n")
            toolbox.f.write(f"escape {escape_id}\n")
            toolbox.f.write(f"escape {escape_id} child {best.msg}\n")
            toolbox.f.write(f"escape {escape_id}\n")
            toolbox.f.write(f"escape {escape_id} {str(best)}\n")
            toolbox.f.write(f"escape {escape_id}\n")
            toolbox.f.write(f"escape {escape_id} parent {parent.msg}\n")
            toolbox.f.write(f"escape {escape_id}\n")
            toolbox.f.write(f"escape {escape_id} {str(parent)}\n")
            toolbox.f.write(f"escape {escape_id}\n")
            toolbox.graph.write_tree_to_dst(toolbox.f, get_ind_node_id(best), f"escape{escape_id}", toolbox.real_gen)
        else:
            f1 = parent.fam.family_index
            toolbox.f.write(f"at gen {toolbox.real_gen}, [{best.id}] = {get_ind_info(best)} = mut [{parent.id}]<{f1}>\n")
            toolbox.f.write(f"at gen {toolbox.real_gen}, [{best.id}] = mut expr {expr_str}\n")
            toolbox.f.write(f"at gen {toolbox.real_gen}, [{best.id}] = {str(best)}\n")
    if False:
        if toolbox.population[0].fam.family_index == 4:
            child = copy_individual(toolbox, parent)
            slice1 = child.searchSubtree(0)
            child[slice1] = expr
            if False:
                pp_str = make_pp_str(child)
                evaluate_individual(toolbox, child, pp_str, 0)
            else:
                child.fam = None
            if best and best.fam.raw_error < toolbox.population[0].fam.raw_error:
                toolbox.good_muts.append(child)
            else:
                toolbox.bad_muts.append(child)
    return best, pp_str


def refresh_toolbox_from_population(toolbox, population, population_is_sorted):
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
    for fam in toolbox.families_list:
        if fam.family_index in toolbox.current_families_dict:
            fam.age_in_population += 1
        else:
            fam.age_in_population = 0
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


def consistency_check_ind(toolbox, ind):
    if ind is not None:
        assert hasattr(ind, "fam")
        assert ind.fam is not None
        assert hasattr(ind, "age")
        assert ind.age >= 0

        assert not hasattr(ind, "deap_str")
        assert not hasattr(ind, "pp_str")
        assert not hasattr(ind, "parents")
        assert not hasattr(ind, "eval")
        assert not hasattr(ind, "raw_error") # voor weergave aan mens
        assert not hasattr(ind, "normalised_error") # voor vergelijken in local search en voor sorteren populatie
        assert not hasattr(ind, "family_index")


def consistency_check(toolbox, inds):
    for ind in inds:
        consistency_check_ind(toolbox, ind)
