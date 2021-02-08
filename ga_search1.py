# Inspired by https://deap.readthedocs.io/en/master/examples/gp_symbreg.html
# it is used by search.py
import os
import random
import copy
import math
import time
import json
import numpy as np

from deap import gp #  gp.PrimitiveSet, gp.genHalfAndHalf, gp.PrimitiveTree, gp.genFull, gp.from_string

import interpret
import evaluate
from evaluate import recursive_tuple
from ga_search_tools import write_population, consistency_check, log_population
from ga_search_tools import best_of_n, generate_initial_population, generate_initial_population_impl
from ga_search_tools import refresh_toolbox_from_population, write_cx_info
from ga_search_tools import load_initial_population_impl, evaluate_individual, consistency_check_ind
from ga_search_tools import crossover_with_local_search, cxOnePoint, mutUniform, replace_subtree_at_best_location
from ga_search_tools import compute_complementairity, pz
import dynamic_weights


def sample_fam_cx_fitness(toolbox, family1_members, family2_members):
    index1, index2 = random.randrange(0, len(family1_members)), random.randrange(0, len(family2_members))
    parent1, parent2 = family1_members[index1], family2_members[index2]

    # sort
    if parent1.fam.raw_error > parent2.fam.raw_error or (parent1.fam.raw_error == parent2.fam.raw_error and len(parent1) > len(parent2)):
        parent1, parent2 = parent2, parent1
    # compute p        
    p_fitness1 = (1 - parent1.fam.raw_error/(toolbox.max_raw_error*1.1))
    p_fitness2 = (1 - parent2.fam.raw_error/(toolbox.max_raw_error*1.1))
    assert 0 <= p_fitness1 and p_fitness1 <= 1
    assert 0 <= p_fitness2 and p_fitness2 <= 1
    p_complementair = compute_complementairity(parent1.fam, parent2.fam)

    assert 0 <= p_complementair and p_complementair <= 1
    p = ((p_fitness1 * p_fitness2)) + (p_complementair * toolbox.parent_selection_weight_complementairity)

    index_a, index_b = parent1.fam.family_index, parent2.fam.family_index
    if index_a > index_b:
        index_a, index_b = index_b, index_a
    key = (index_a, index_b)
    assert index_a <= index_b
    if key in toolbox.cx_count_dict:
        count_cx = toolbox.cx_count_dict[key]
    else:
        count_cx = 0
    if count_cx > 0:
        x = count_cx + 1
        y = toolbox.parent_selection_weight_cx_count
        p /= x ** y
        #toolbox.f.write(f"count_cx({parent1.fam.family_index},{parent2.fam.family_index}) = {x} ** {y} = {x ** y}\n")
        x = pz(toolbox, parent1.fam.family_index, parent2.fam.family_index) 
        y = toolbox.parent_selection_weight_p_out_of_pop
        p *= x ** y
        #toolbox.f.write(f"pz({parent1.fam.family_index},{parent2.fam.family_index}) = {x} ** {y} = {x ** y}\n")
    assert p >= 0
    return p, parent1, parent2


def is_cx_zero_count(toolbox, key):
    return key not in toolbox.cx_count_dict or toolbox.cx_count_dict[key] == 0


def prepare_combinations_families_with_cx_count_zero(toolbox, population):
    toolbox.family_combinations_with_cx_count_zero = []
    if False:
        if toolbox.parachute_level > 0:
            max_n_combinations = toolbox.max_combinations_families_with_cx_count_zero
            combinations_done = set()
            for index_a, family1_members in toolbox.current_families_dict.items():
                for index_b, family2_members in toolbox.current_families_dict.items():
                    if index_a < index_b:
                        key = (index_a, index_b)
                        if key not in combinations_done:
                            combinations_done.add(key)
                            if is_cx_zero_count(toolbox, key):
                                p, parent1, parent2 = sample_fam_cx_fitness(toolbox, family1_members, family2_members)
                                if len(parent1) > 1 and len(parent2) > 1:
                                    toolbox.family_combinations_with_cx_count_zero.append([p, parent1, parent2, index_a, index_b])
            toolbox.family_combinations_with_cx_count_zero.sort(key=lambda item: -item[0])
            toolbox.family_combinations_with_cx_count_zero = toolbox.family_combinations_with_cx_count_zero[:max_n_combinations]
            if len(toolbox.family_combinations_with_cx_count_zero) > 0:
                if toolbox.verbose >= 3:
                    toolbox.f.write(f"prepare_combinations_families_with_cx_count_zero, len {len(toolbox.family_combinations_with_cx_count_zero)}\n")
                    if False:
                        for p, parent1, parent2, index_a, index_b in toolbox.family_combinations_with_cx_count_zero[:100]:
                            toolbox.f.write(f"    index_a {index_a}, index_b, {index_b}, complementarity {p}\n")


def select_families_with_cx_count_zero(toolbox):
    if toolbox.parachute_level > 0:
        for p, parent1, parent2, index_a, index_b in toolbox.family_combinations_with_cx_count_zero:
            if is_cx_zero_count(toolbox, (index_a, index_b)):
                if False: # toolbox.verbose >= 1:
                    toolbox.f.write(f"select_families_with_cx_count_zero index_a {index_a}, index_b {index_b}, p {p}\n")
                return parent1, parent2
    return None, None


def select_parents(toolbox, population):
    if toolbox.parent_selection_strategy == 0 or len(toolbox.current_families_dict) == 1:
        return [best_of_n(population, toolbox.best_of_n_cx), best_of_n(population, toolbox.best_of_n_cx)]
    if toolbox.parachute_level > 0:
        best_parent1, best_parent2 = select_families_with_cx_count_zero(toolbox)
        if best_parent1:
            return best_parent1, best_parent2
    best_p = -1
    assert toolbox.best_of_n_cx > 0
    for _ in range(toolbox.best_of_n_cx):
        # select two parents
        family1_index, family2_index = random.sample(list(toolbox.current_families_dict), 2) # sample always returns a list
        family1_members, family2_members = toolbox.current_families_dict[family1_index], toolbox.current_families_dict[family2_index]
        p, parent1, parent2 = sample_fam_cx_fitness(toolbox, family1_members, family2_members)
        # keep the best couple
        if best_p < p:
            best_parent1, best_parent2, best_p = parent1, parent2, p
    return best_parent1, best_parent2


def analyse_parents(toolbox, population):
    escapes_count = 0
    for parent1 in population:
        for parent2 in population:
            child = crossover_with_local_search(toolbox, parent1, parent2)
            if child and child.fam.raw_error < 77.61253 - 0.00001:
                escapes_count += 1
    return escapes_count


def generate_offspring(toolbox, population, nchildren):
    t0_offspring = time.time()
    offspring = []
    expr_mut = lambda pset, type_: gp.genFull(pset=pset, min_=0, max_=2, type_=type_)
    retry_count = 0  
    all_escapes_count = 0   
    only_cx = False
    if False:
        do_special_experiment = False # math.isclose(population[0].fam.raw_error, 77.61253, abs_tol=0.00001)
        if do_special_experiment:
            only_cx = True
            all_escapes_count = analyse_parents(toolbox, population)
            offspring_escapes_count = 0
    toolbox.max_raw_error = max([ind.fam.raw_error for ind in population])
    prepare_combinations_families_with_cx_count_zero(toolbox, population)
    while len(offspring) < nchildren:
        op_choice = random.random()
        if op_choice < toolbox.pcrossover or only_cx: # Apply crossover
            t0_select_parents = time.time()
            parent1, parent2 = select_parents(toolbox, population)
            toolbox.t_select_parents += time.time() - t0_select_parents
            t0_cx = time.time()
            if toolbox.parachute_level == 0:
                child, pp_str = cxOnePoint(toolbox, parent1, parent2)
            else:
                child, pp_str = crossover_with_local_search(toolbox, parent1, parent2)
            toolbox.t_cx += time.time() - t0_cx
        else: # Apply mutation
            t0_mut = time.time()
            parent = best_of_n(population, toolbox.best_of_n_mut)
            if toolbox.parachute_level == 0:
                child, pp_str = mutUniform(toolbox, parent, expr=expr_mut, pset=toolbox.pset)
            else:
                expr = gp.genFull(pset=toolbox.pset, min_=0, max_=2)
                child, pp_str = replace_subtree_at_best_location(toolbox, parent, expr)
            toolbox.t_mut += time.time() - t0_mut
        if child is None:
            if retry_count < toolbox.child_creation_retries:
                retry_count += 1
                continue
            else:
                break
        retry_count = 0
        assert child.fam is not None
        toolbox.ind_str_set.add(pp_str)
        offspring.append(child)
        if False:
            if do_special_experiment and child.fam.raw_error < 77.61253 - 0.00001:
                offspring_escapes_count += 1
    if False:
        if do_special_experiment:
            expected_offspring_escapes_count = nchildren * all_escapes_count / (len(population) ** 2)
            toolbox.f.write(f"{all_escapes_count}\t{offspring_escapes_count}\t{expected_offspring_escapes_count}\n")
    toolbox.t_offspring += time.time() - t0_offspring
    #consistency_check(toolbox, offspring)
    return offspring


def track_stuck(toolbox, population):
    # track if we are stuck
    if population[0].fam.family_index in toolbox.prev_family_index:                        
        toolbox.stuck_count += 1
        if toolbox.max_observed_stuck_count < toolbox.stuck_count:
            toolbox.max_observed_stuck_count = toolbox.stuck_count
        if toolbox.stuck_count > toolbox.max_stuck_count:            
            raise RuntimeWarning("max stuck count exceeded")
        if toolbox.stuck_count >= toolbox.stuck_count_for_opschudding:            
            if toolbox.count_opschudding >= toolbox.max_reenter_parachuting_phase:
                # toolbox.f.write("max reenter_parachuting_phase exceeded (skipped)\n")
                pass
            else:
                toolbox.f.write(f"reenter_parachuting_phase {toolbox.count_opschudding} < {toolbox.max_reenter_parachuting_phase}\n")
                toolbox.parachute_level = 0
                toolbox.gen = 0
                toolbox.prev_family_index = set()
                toolbox.stuck_count = 0
                toolbox.count_opschudding += 1
                refresh_toolbox_from_population(toolbox, population, True)
    else:
        toolbox.stuck_count = 0
        toolbox.count_opschudding = 0
        # verandering t.o.v. vorige iteratie
    toolbox.prev_family_index.add(population[0].fam.family_index)


def log_info(toolbox, population):

    toolbox.f.write(f"gen {toolbox.real_gen}")
    done = set()
    msg = ""
    for ind in population:
        i = ind.fam.family_index
        fam = ind.fam
        if i not in done:
            done.add(i)
            size = len(toolbox.current_families_dict[i])
            msg += f" ({i},{fam.normalised_error:.0f},{fam.raw_error:.0f},{size})"
    if len(msg) > 150:
        msg = msg[:(150-3)] + "..."
    # toolbox.f.write(f" (idx,dw,raw,#)")
    toolbox.f.write(msg)
    toolbox.f.write(f"\n")
    if False:
        toolbox.f.write(f"best_ind.raw_error_matrix\n")
        i = population[0].fam.family_index
        dynamic_weights.dump_matrix(toolbox.f, toolbox.families_list[i].raw_error_matrix)
        dynamic_weights.dump_dw_matrix(toolbox.f)
        for i in [38, 671]:
            if i in done:
                toolbox.f.write(f"family[{i}].raw_error_matrix\n")
                dynamic_weights.dump_matrix(toolbox.f, toolbox.families_list[i].raw_error_matrix)
                toolbox.f.write(f"family[{i}].dw_error_matrix\n")
                dw_error_matrix = dynamic_weights.compute_normalised_error_matrix(toolbox.families_list[i].raw_error_matrix)
                dynamic_weights.dump_matrix(toolbox.f, dw_error_matrix)


def ga_search_impl(toolbox):
    if toolbox.final_pop_file: # clear the file to avoid confusion with older output
        write_population(toolbox.final_pop_file, [], toolbox.functions)
    try:
        t0 = time.time()
        toolbox.gen = 0
        toolbox.real_gen = 0
        population = [] # generate_initial_population may throw exception
        population = generate_initial_population(toolbox)
        consistency_check(toolbox, population)
        refresh_toolbox_from_population(toolbox, population, False)
        toolbox.t_init = time.time() - t0
        toolbox.t_eval = 0
        toolbox.t_error = 0
        toolbox.t_cpp_interpret = 0
        toolbox.t_py_interpret = 0
        toolbox.prev_family_index = set()
        toolbox.stuck_count, toolbox.count_opschudding = 0, 0
        toolbox.parachute_level = 0
        toolbox.max_observed_stuck_count = 0
        while toolbox.parachute_level < len(toolbox.ngen):
            while toolbox.gen < toolbox.ngen[toolbox.parachute_level]:
                track_stuck(toolbox, population)
                if toolbox.f and toolbox.verbose >= 1:
                    log_info(toolbox, population)
                offspring = generate_offspring(toolbox, population, toolbox.nchildren[toolbox.parachute_level])
                fraction = toolbox.parents_keep_fraction[toolbox.parachute_level]
                if fraction < 1:
                    population = random.sample(population, k=int(len(population)*fraction))
                population += offspring
                population.sort(key=toolbox.sort_ind_key)
                population[:] = population[:toolbox.pop_size[toolbox.parachute_level]]
                consistency_check(toolbox, population)
                refresh_toolbox_from_population(toolbox, population, True)
                write_cx_info(toolbox)
                if toolbox.is_solution(population[0]): # do this after refresh, for debugging refresh
                    return population[0], toolbox.real_gen + 1
                toolbox.gen += 1
                toolbox.real_gen += 1
            toolbox.parachute_level += 1
    except RuntimeWarning as e:
        toolbox.f.write("RuntimeWarning: " + str(e) + "\n")
    if toolbox.final_pop_file: # write the set covering input files
        write_population(toolbox.final_pop_file, population, toolbox.functions)
    return (population[0] if len(population) > 0 else None), toolbox.real_gen + 1


