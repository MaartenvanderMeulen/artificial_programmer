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
from ga_search_tools import write_population, consistency_check
from ga_search_tools import best_of_n, generate_initial_population, generate_initial_population_impl
from ga_search_tools import refresh_toolbox_from_population, write_cx_graph
from ga_search_tools import load_initial_population_impl, evaluate_individual, consistency_check_ind
from ga_search_tools import crossover_with_local_search, cxOnePoint, mutUniform, replace_subtree_at_best_location
from ga_search_tools import compute_complementairity, pz, remove_file
import dynamic_weights


def sample_fam_cx_fitness(toolbox, family1_members, family2_members):
    index1, index2 = random.randrange(0, len(family1_members)), random.randrange(0, len(family2_members))
    parent1, parent2 = family1_members[index1], family2_members[index2]

    # compute p        
    p_fitness1 = (1 - parent1.fam.raw_error/(toolbox.max_raw_error*1.1))
    p_fitness2 = (1 - parent2.fam.raw_error/(toolbox.max_raw_error*1.1))
    assert 0 <= p_fitness1 and p_fitness1 <= 1
    assert 0 <= p_fitness2 and p_fitness2 <= 1
    p_complementair = compute_complementairity(parent1.fam, parent2.fam)

    assert 0 <= p_complementair and p_complementair <= 1
    p = ((p_fitness1 * p_fitness2)) + (p_complementair * toolbox.parent_selection_weight_complementairity)

    index_a, index_b = parent1.fam.family_index, parent2.fam.family_index
    key = (index_a, index_b)
    if key in toolbox.cx_count_dict:
        count_cx = toolbox.cx_count_dict[key]
    else:
        count_cx = 0
    if count_cx > 0:
        x = count_cx + 1
        y = toolbox.parent_selection_weight_cx_count
        p /= x ** y
        x = pz(toolbox, parent1.fam.family_index, parent2.fam.family_index) 
        y = toolbox.parent_selection_weight_p_out_of_pop
        p *= x ** y
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


def search_for_solution(toolbox, population, cx_children):
    n = len(population)
    threshold = population[n//2].fam.raw_error
    offspring = []
    count = 0
    cx_candidates = []
    for _, parents1 in toolbox.current_families_dict.items():
        for _, parents2 in toolbox.current_families_dict.items():
            key = (parents1[0].fam.family_index, parents2[0].fam.family_index)
            if key not in toolbox.cx_count_dict:
                parent1, parent2 = parents1[-1], parents2[-1]
                p = sample_fam_cx_fitness(toolbox, [parent1], [parent2])[0]
                cx_candidates.append((parent1, parent2, p))
    cx_candidates.sort(key=lambda item: -item[2])
    for parent1, parent2, _ in cx_candidates:
        count += 1
        child, _child_pp_str = crossover_with_local_search(toolbox, parent1, parent2)
        if child and child.fam.raw_error < threshold:
            offspring.append(child)
            if len(offspring) >= cx_children:
                break
    return offspring


def generate_offspring(toolbox, population, nchildren):
    offspring = []
    toolbox.max_raw_error = max([ind.fam.raw_error for ind in population])
    do_default_cx = True
    do_200x200 = population[0].fam.family_index <= 4
    if do_200x200:
        toolbox.parents_keep_fraction[toolbox.parachute_level] = 1.0 # 3.0 / 4.0
        toolbox.pop_size[toolbox.parachute_level] = 200
        if True:
            do_default_cx = False
            n = len(toolbox.current_families_dict)
            toolbox.f.write(f"Start {n}x{n} search\n")
            offspring = search_for_solution(toolbox, population, round(nchildren*toolbox.pcrossover))
            toolbox.f.write(f"    {n}x{n} search found {len(offspring)} improvements\n")

    expr_mut = lambda pset, type_: gp.genFull(pset=pset, min_=toolbox.mut_min_height, max_=toolbox.mut_max_height, type_=type_)
    retry_count = 0  
    prepare_combinations_families_with_cx_count_zero(toolbox, population)
    while len(offspring) < nchildren:
        op_choice = random.random()
        if op_choice < toolbox.pcrossover and do_default_cx: # Apply crossover
            parent1, parent2 = select_parents(toolbox, population)
            if toolbox.parachute_level == 0:
                child, pp_str = cxOnePoint(toolbox, parent1, parent2)
            else:
                child, pp_str = crossover_with_local_search(toolbox, parent1, parent2)
        else: # Apply mutation
            parent = best_of_n(population, toolbox.best_of_n_mut)
            if toolbox.parachute_level == 0:
                child, pp_str = mutUniform(toolbox, parent, expr=expr_mut, pset=toolbox.pset)
            else:
                expr = gp.genFull(pset=toolbox.pset, min_=toolbox.mut_min_height, max_=toolbox.mut_max_height)
                child, pp_str = replace_subtree_at_best_location(toolbox, parent, expr)
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
    return offspring


def track_stuck(toolbox, population):
    # track if we are stuck
    if population[0].fam.family_index in toolbox.prev_family_index:                        
        toolbox.stuck_count += 1
        if toolbox.max_observed_stuck_count < toolbox.stuck_count:
            toolbox.max_observed_stuck_count = toolbox.stuck_count
            toolbox.ootb_at_msc = (toolbox.count_cx - toolbox.count_cx_into_current_pop) / toolbox.count_cx
            toolbox.fams_at_msc = len(toolbox.current_families_dict)
            toolbox.error_at_msc = toolbox.population[0].fam.raw_error
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
    msg = f"gen {toolbox.real_gen} best {population[0].fam.raw_error:.1f}"
    toolbox.f.write(msg)
    if True:
        done = set()
        msg = ""
        for ind in population:
            i = ind.fam.family_index
            if i <= 177166:
                if i not in done:
                    done.add(i)
                    #size = len(toolbox.current_families_dict[i])
                    msg += f" f{i}"
        if len(msg) > 150:
            msg = msg[:(150-3)] + "..."
        toolbox.f.write(msg)    
    toolbox.f.write(f"\n")


def check_other_stop_criteria(toolbox):
    if time.time() >= toolbox.t0 + toolbox.max_seconds:
        raise RuntimeWarning("max time reached")
    if toolbox.eval_count >= toolbox.max_evaluations:
        raise RuntimeWarning("max evaluations reached")
    if toolbox.stuck_count >= toolbox.max_stuck_count:            
        raise RuntimeWarning("max stuck count reached")


def ga_search_impl_core(toolbox):
    toolbox.gen = 0
    toolbox.real_gen = 0
    toolbox.prev_family_index = set()
    toolbox.stuck_count, toolbox.count_opschudding = 0, 0
    toolbox.parachute_level = 0
    toolbox.max_observed_stuck_count = 0
    toolbox.count_cx_into_current_pop, toolbox.count_cx = 0, 1 # starting at 1 is easier lateron
    toolbox.population = generate_initial_population(toolbox)
    consistency_check(toolbox, toolbox.population)
    refresh_toolbox_from_population(toolbox, toolbox.population, False)
    while toolbox.parachute_level < len(toolbox.ngen):
        while toolbox.gen < toolbox.ngen[toolbox.parachute_level]:
            check_other_stop_criteria(toolbox)
            track_stuck(toolbox, toolbox.population)
            if toolbox.f and toolbox.verbose >= 1:
                log_info(toolbox, toolbox.population)
            offspring = generate_offspring(toolbox, toolbox.population, toolbox.nchildren[toolbox.parachute_level])
            fraction = toolbox.parents_keep_fraction[toolbox.parachute_level]
            if fraction < 1:
                toolbox.population = random.sample(toolbox.population, k=int(len(toolbox.population)*fraction))
            trim_families = toolbox.population[0].fam.family_index <= 4
            if trim_families:
                toolbox.population = []
                for _, inds in toolbox.current_families_dict.items():
                    toolbox.population.append(inds[-1])
            toolbox.population += offspring
            toolbox.population.sort(key=toolbox.sort_ind_key)
            toolbox.population[:] = toolbox.population[:toolbox.pop_size[toolbox.parachute_level]]
            consistency_check(toolbox, toolbox.population)
            refresh_toolbox_from_population(toolbox, toolbox.population, True)
            if toolbox.is_solution(toolbox.population[0]): # do this after refresh, for debugging refresh
                return
            toolbox.gen += 1
            toolbox.real_gen += 1
        toolbox.parachute_level += 1


def ga_search_impl(toolbox):
    if toolbox.final_pop_file: # clear the file to avoid confusion with older output
        remove_file(toolbox.final_pop_file)
    if toolbox.new_fam_file: # clear the file to avoid confusion with older output
        remove_file(toolbox.new_fam_file)
    if toolbox.good_muts_file: # clear the file to avoid confusion with older output
        remove_file(toolbox.good_muts_file)
    if toolbox.bad_muts_file: # clear the file to avoid confusion with older output
        remove_file(toolbox.bad_muts_file)
    toolbox.population = []
    toolbox.good_muts = []
    toolbox.bad_muts = []
    try:

        ga_search_impl_core(toolbox)

    except RuntimeWarning as e:
        toolbox.f.write("RuntimeWarning: " + str(e) + "\n")
    if toolbox.final_pop_file: # write the input files for "samenvoegen"
        write_population(toolbox.final_pop_file, toolbox.population, toolbox.functions)
    new_families = [family.representative for family in toolbox.new_families_list]
    if toolbox.new_fam_file and len(new_families) > 0: # write the new families
        write_population(toolbox.new_fam_file, new_families, toolbox.functions)
    if toolbox.write_cx_graph:
        write_cx_graph(toolbox)
    if toolbox.good_muts_file:
        write_population(toolbox.good_muts_file, toolbox.good_muts, toolbox.functions)
    if toolbox.bad_muts_file:
        write_population(toolbox.bad_muts_file, toolbox.bad_muts, toolbox.functions)
    toolbox.ga_search_impl_return_value = (toolbox.population[0] if len(toolbox.population) > 0 else None), toolbox.real_gen + 1
    return toolbox.ga_search_impl_return_value
