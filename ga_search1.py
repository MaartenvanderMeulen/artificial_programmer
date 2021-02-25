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
from ga_search_tools import compute_complementairity, pz, remove_file, get_fam_info, get_ind_info
from ga_search_tools import forced_reevaluation_of_individual_for_debugging
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


#global i5607804, i5756632, i5707508, i5292746
#i5607804, i5756632, i5707508, i5292746 = None, None, None, None

def search_for_solution(toolbox, population, cx_children):
    #global i5607804, i5756632, i5707508, i5292746
    #n = len(population)
    # threshold = population[-1].fam.raw_error
    # threshold = population[n//2].fam.raw_error
    offspring = []
    count = 0
    cx_candidates = []
    if False:
        for _, parents1 in toolbox.current_families_dict.items():
            for _, parents2 in toolbox.current_families_dict.items():
                key = (parents1[0].fam.family_index, parents2[0].fam.family_index)
                parent1, parent2 = parents1[-1], parents2[-1]
                f1, f2 = parent1.fam.family_index, parent2.fam.family_index
                if (f1, f2) in toolbox.p_cx_c0_db:
                    if toolbox.p_cx_c0_db[(f1, f2)] == 1.0:
                        if False:
                            cx_candidates.append((parent1, parent2, toolbox.p_cx_c0_db[(f1, f2)]))
                        toolbox.f.write(f"detected potential escape (p_cx_c0==1) via : <{f1}> x <{f2}>\n")
                        child, _child_pp_str = crossover_with_local_search(toolbox, parent1, parent2, do_shuffle=False, debug=3)
                        toolbox.f.write(f"but crossover <{f1}> x <{f2}> has resulting error {child.fam.raw_error}\n")
                        exit()
        if len(cx_candidates) > 0:
            sum_p = sum([p for _, _, p in cx_candidates])
            toolbox.f.write(f"nxn finds {len(cx_candidates)} cx candidates mentioned in p_cx_c0_db.  Sum of p is {sum_p:.2f}\n")
        if len(cx_candidates) == 0:
            # TODO add cx omdat de parents betrokken zijn in p_cx_c0_db
            pass

    if len(cx_candidates) == 0:
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
        if child: #  and child.fam.raw_error < threshold:
            offspring.append(child)
            if len(offspring) >= cx_children:
                break
    return offspring


def generate_offspring(toolbox, population, nchildren):
    offspring = []
    toolbox.max_raw_error = max([ind.fam.raw_error for ind in population])
    toolbox.debug_pp_str = ""
    do_default_cx = True
    if population[0].fam.raw_error <= toolbox.near_solution_threshold:
        if not toolbox.in_near_solution_area:
            toolbox.in_near_solution_area = True
            for index, _inds in toolbox.current_families_dict.items():
                if toolbox.families_list[index].raw_error <= toolbox.max_raw_error_for_family_db:
                    toolbox.near_solution_families_set.add(index)

            toolbox.parents_keep_fraction[toolbox.parachute_level] = 1.0 # 3.0 / 4.0
            toolbox.pop_size[toolbox.parachute_level] = toolbox.near_solution_pop_size
            toolbox.max_individual_size = toolbox.near_solution_max_individual_size
            if not toolbox.params["use_one_random_seed"]:
                random.seed(toolbox.params["seed2"])

            toolbox.cx_count = dict()
            toolbox.cx_child_count = dict()
            if False:
                toolbox.pp_str_to_family_index_dict = dict()
                toolbox.family_list = []
                toolbox.new_families_list = []
                new_dict = dict()
                for new_index, (_old_index, inds) in enumerate(toolbox.current_families_dict.items()):
                    fam = inds[0].fam
                    toolbox.family_list.append(fam)
                    toolbox.new_families_list.append(fam)
                    fam.family_index = new_index
                    new_dict[new_index] = inds
                toolbox.current_families_dict = new_dict
        # toolbox.pcrossover = 0.3
        if False:
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
                if toolbox.use_family_representatives_for_mutation:
                    family = random.choice(toolbox.families_list)
                    mutation = family.representative
                else:
                    mutation = gp.genFull(pset=toolbox.pset, min_=toolbox.mut_min_height, max_=toolbox.mut_max_height)                
                    mutation = gp.PrimitiveTree(mutation)                    
                    mutation.fam = None
                if toolbox.use_crossover_for_mutations:
                    child, pp_str = crossover_with_local_search(toolbox, parent, mutation)
                else:
                    child, pp_str = replace_subtree_at_best_location(toolbox, parent, mutation)
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
        toolbox.graph.clear()
        # verandering t.o.v. vorige iteratie
    toolbox.prev_family_index.add(population[0].fam.family_index)


def log_info(toolbox, population):
    msg = f"gen {toolbox.real_gen} error {population[0].fam.raw_error:.1f}"
    toolbox.f.write(msg)
    if True:
        p_cx_c0 = 0.0
        for p1, _ in toolbox.current_families_dict.items():
            for p2, _ in toolbox.current_families_dict.items():
                if (p1, p2) in toolbox.p_cx_c0_db:
                    if p_cx_c0 < toolbox.p_cx_c0_db[(p1, p2)]:
                        p_cx_c0 = toolbox.p_cx_c0_db[(p1, p2)]
        toolbox.f.write(f" p_cx_c0 {p_cx_c0:.2f}")
    if True:
        avg_ind_len = sum([len(ind) for ind in toolbox.population]) / len(toolbox.population)
        toolbox.f.write(f" avg_ind_len {avg_ind_len:.1f}")
    if False:
        msg = ""
        for _, inds in toolbox.current_families_dict.items():
            i = round(inds[0].fam.family_index)
            n = len(inds)
            msg += f" {n}"
        toolbox.f.write(msg)    
    toolbox.f.write(f"\n")
    if False:
        for _, inds in toolbox.current_families_dict.items():
            toolbox.f.write(f"    current fam {get_fam_info(inds[0].fam)}\n")
        for ind in population:
            toolbox.f.write(f"    current pop ind {get_ind_info(ind)}\n")
    if True:
        msg = f"atgen {toolbox.real_gen} families"
        for _, inds in toolbox.current_families_dict.items():
            i = round(inds[0].fam.family_index)
            msg += f" {i}"
        toolbox.f.write(msg + "\n")    


def check_other_stop_criteria(toolbox):
    if time.time() >= toolbox.t0 + toolbox.max_seconds and toolbox.stuck_count >= 5:
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
    toolbox.in_near_solution_area = False
    toolbox.population = generate_initial_population(toolbox)
    consistency_check(toolbox, toolbox.population)
    refresh_toolbox_from_population(toolbox, toolbox.population, False)
    while toolbox.parachute_level < len(toolbox.ngen):
        while toolbox.gen < toolbox.ngen[toolbox.parachute_level]:
            for ind in toolbox.population:
                ind.age += 1
            for fam in toolbox.families_list:
                fam.age += 1
            track_stuck(toolbox, toolbox.population)
            if toolbox.f and toolbox.verbose >= 1:
                log_info(toolbox, toolbox.population)
            check_other_stop_criteria(toolbox)
            offspring = generate_offspring(toolbox, toolbox.population, toolbox.nchildren[toolbox.parachute_level])
            fraction = toolbox.parents_keep_fraction[toolbox.parachute_level]
            if toolbox.stuck_count < toolbox.parents_keep_all_duration:
                fraction = 1
            if fraction < 1:
                if toolbox.parents_keep_fraction_per_family:
                    toolbox.population = []
                    for _, inds in toolbox.current_families_dict.items():
                        toolbox.population.extend(random.sample(inds, k=max(1,round(len(inds)*fraction))))
                else:
                    toolbox.population = random.sample(toolbox.population, k=int(len(toolbox.population)*fraction))
            if toolbox.in_near_solution_area:
                # trim families
                toolbox.population = []
                for index, inds in toolbox.current_families_dict.items():
                    if True:
                        ind = inds[-1]
                        rep = toolbox.families_list[index].representative
                        slice_ind = ind.searchSubtree(0)
                        slice_rep = rep.searchSubtree(0)
                        ind[slice_ind] = rep[slice_rep]
                        toolbox.population.append(ind)
                    else:
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
    if toolbox.near_solution_families_file: # clear the file to avoid confusion with older output
        remove_file(toolbox.near_solution_families_file)
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
    near_solution_families = [toolbox.families_list[index].representative for index in toolbox.near_solution_families_set]
    if toolbox.near_solution_families_file and len(near_solution_families) > 0: # write the near_solution_families families
        write_population(toolbox.near_solution_families_file, near_solution_families, toolbox.functions)
    if toolbox.write_cx_graph:
        write_cx_graph(toolbox)
    if toolbox.good_muts_file:
        write_population(toolbox.good_muts_file, toolbox.good_muts, toolbox.functions)
    if toolbox.bad_muts_file:
        write_population(toolbox.bad_muts_file, toolbox.bad_muts, toolbox.functions)
    toolbox.ga_search_impl_return_value = (toolbox.population[0] if len(toolbox.population) > 0 else None), toolbox.real_gen + 1
    return toolbox.ga_search_impl_return_value
