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
from ga_search_tools import write_population, consistency_check, log_population
from ga_search_tools import best_of_n, generate_initial_population, generate_initial_population_impl
from ga_search_tools import update_dynamic_weighted_evaluation, refresh_toolbox_from_population
from ga_search_tools import load_initial_population_impl, evaluate_individual, consistency_check_ind
from ga_search_tools import crossover_with_local_search, cxOnePoint, mutUniform, replace_subtree_at_best_location

from deap import gp #  gp.PrimitiveSet, gp.genHalfAndHalf, gp.PrimitiveTree, gp.genFull, gp.from_string


def select_parents(toolbox, population):
    if toolbox.parent_selection_strategy == 0 or len(toolbox.current_families_dict) == 1:
        return [best_of_n(population, toolbox.best_of_n_cx), best_of_n(population, toolbox.best_of_n_cx)]
    if toolbox.parent_selection_strategy == 1:        
        # First attempt with toolbox.inter_family_cx_taboo.  Its just unlikely, not a taboo.
        list_of_keys = list(toolbox.current_families_dict)
        group1, group2 = random.sample(list_of_keys, 2) # sample always returns a list
        group1, group2 = toolbox.current_families_dict[group1], toolbox.current_families_dict[group2]
        index1, index2 = random.randrange(0, len(group1)), random.randrange(0, len(group2))
        parent1, parent2 = group1[index1], group2[index2] # all individuals in the group have the same eval
        return parent1, parent2
        
    # second attempt, hint of Victor:
    # * hoe beter de evaluatie hoe meer kans.  p_fitness = (1 - eval) ** alpha
    # * hoe verder de outputs uit elkaar liggen hoe meer kans.
    # p_complementair = verschillende_outputs(parent1, parent2) / aantal_outputs
    # * p = (p_fitness(parent1) * p_fitness(parent2)) ^ alpha * p_complementair(parent1, parent2) ^ beta
    best_p = -1
    assert toolbox.best_of_n_cx > 0
    max_eval = max([ind.eval for ind in population])
    for _ in range(toolbox.best_of_n_cx):
        # select two parents
        group1, group2 = random.sample(list(toolbox.current_families_dict), 2) # sample always returns a list
        group1, group2 = toolbox.current_families_dict[group1], toolbox.current_families_dict[group2]
        index1, index2 = random.randrange(0, len(group1)), random.randrange(0, len(group2))
        parent1, parent2 = group1[index1], group2[index2] # all individuals in the group have the same eval
        # sort
        if parent1.eval > parent2.eval or (parent1.eval == parent2.eval and len(parent1) > len(parent2)):
            parent1, parent2 = parent2, parent1
        # compute p        
        model_evals1 = toolbox.families_list[parent1.family_index].model_evals
        model_evals2 = toolbox.families_list[parent2.family_index].model_evals
        p_fitness1 = (1 - parent1.eval/(max_eval*1.1))
        p_fitness2 = (1 - parent2.eval/(max_eval*1.1))
        if p_fitness1 < 0 or 1 < p_fitness1:
            print("p_fitness1", p_fitness1)
        assert 0 <= p_fitness1 and p_fitness1 <= 1
        if p_fitness2 < 0 or 1 < p_fitness2:
            print("p_fitness2", p_fitness2)
        assert 0 <= p_fitness2 and p_fitness2 <= 1
        estimate_improvement = sum([eval1-eval2 for eval1, eval2 in zip(model_evals1, model_evals2) if eval1 > eval2])
        assert estimate_improvement >= 0
        p_complementair = estimate_improvement / parent1.eval
        assert 0 <= p_complementair and p_complementair <= 1
        p = ((p_fitness1 * p_fitness2)) + (p_complementair * toolbox.beta)
        assert p >= 0
        if best_p < p:
            best_parent1, best_parent2, best_p = parent1, parent2, p
    return best_parent1, best_parent2


def analyse_parents(toolbox, population):
    escapes_count = 0
    for parent1 in population:
        for parent2 in population:
            child = crossover_with_local_search(toolbox, parent1, parent2)
            if child and child.eval < 77.61253 - 0.00001:
                escapes_count += 1
    return escapes_count


def generate_offspring(toolbox, population, nchildren):
    offspring = []
    expr_mut = lambda pset, type_: gp.genFull(pset=pset, min_=0, max_=2, type_=type_)
    retry_count = 0  
    cxp_count, mutp_count = 0, 0    
    cx_count, mut_count = 0, 0 
    all_escapes_count = 0   
    toolbox.keep_path = False
    only_cx = False
    if False:
        do_special_experiment = False # math.isclose(population[0].eval, 77.61253, abs_tol=0.00001)
        if do_special_experiment:
            only_cx = True
            all_escapes_count = analyse_parents(toolbox, population)
            toolbox.keep_path = True
            offspring_escapes_count = 0
    while len(offspring) < nchildren:
        op_choice = random.random()
        if op_choice < toolbox.pcrossover or only_cx: # Apply crossover
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
        offspring.append(child)
        if False:
            if do_special_experiment and child.eval < 77.61253 - 0.00001:
                if False:
                    toolbox.f.write(f"child\t{child.eval:.3f}\tcode\t{child.deap_str}\n")
                    for parent in child.parents:
                        toolbox.f.write(f"parent\t{parent.eval:.3f}\tcode\t{parent.deap_str}\n")
                if False:
                    toolbox.f.write(f"{child.eval:.3f}\t")
                    child.parents.sort(key=lambda item: item.eval)
                    for parent in child.parents:
                        toolbox.f.write(f"\t{parent.eval:.3f}")
                    toolbox.f.write(f"\n")
                offspring_escapes_count += 1
    if False:
        if do_special_experiment:
            expected_offspring_escapes_count = nchildren * all_escapes_count / (len(population) ** 2)
            toolbox.f.write(f"{all_escapes_count}\t{offspring_escapes_count}\t{expected_offspring_escapes_count}\n")
    consistency_check(toolbox, offspring)
    return offspring, None


def track_stuck(toolbox, population):
    # track if we are stuck
    if population[0].eval == toolbox.prev_eval:                        
        toolbox.stuck_count += 1
        if toolbox.stuck_count >= toolbox.max_stuck_count + 1:            
            if "reenter_parachuting_phase" not in toolbox.metaevolution_strategies or toolbox.count_opschudding > 0:
                raise RuntimeWarning("max stuck count exceeded")
            if "reenter_parachuting_phase" in toolbox.metaevolution_strategies:
                toolbox.f.write("reenter_parachuting_phase\n")
                toolbox.parachute_level = 0
                toolbox.gen = 0
                toolbox.prev_eval = 1e9
                toolbox.stuck_count = 0
                toolbox.count_opschudding += 1
                toolbox.reset()
                for ind in population:
                    ind.parents = []
                    ind.eval = evaluate_individual(toolbox, ind)
                    consistency_check_ind(toolbox, ind)
                refresh_toolbox_from_population(toolbox, population)
    elif toolbox.best_eval > population[0].eval:
        # betere oplossing dan ooit gevonden!
        toolbox.best_eval = population[0].eval
        toolbox.stuck_count = 0
    else:
        # verandering t.o.v. vorige iteratie
        pass
    toolbox.prev_eval = population[0].eval


def ga_search_impl(toolbox):
    if toolbox.final_pop_file: # clear the file to avoid confusion with older output
        write_population(toolbox.final_pop_file, [], toolbox.functions)
    try:
        population, solution = [], None # generate_initial_population may throw exception
        population, solution = generate_initial_population(toolbox)
        if solution:
            return solution, 0
        consistency_check(toolbox, population)
        update_dynamic_weighted_evaluation(toolbox, population)
        population.sort(key=lambda item: item.eval)
        refresh_toolbox_from_population(toolbox, population)
        toolbox.prev_eval = 1e9
        toolbox.best_eval = 1e9
        toolbox.stuck_count, toolbox.count_opschudding = 0, 0
        toolbox.parachute_level = 0
        toolbox.gen = 0
        toolbox.real_gen = 0
        while toolbox.parachute_level < len(toolbox.ngen):
            while toolbox.gen < toolbox.ngen[toolbox.parachute_level]:
                track_stuck(toolbox, population)
                if toolbox.f and toolbox.verbose >= 1:
                    count_best = sum([1 for ind in population if ind.eval == population[0].eval])
                    toolbox.f.write(f"gen {toolbox.real_gen:2d} best {population[0].eval:7.3f} ")
                    toolbox.f.write(f"sc {toolbox.stuck_count:2d} count_best {count_best:4d} {population[0].deap_str[:120]}\n")
                if toolbox.verbose >= 3:
                    log_population(toolbox, population, f"generation {toolbox.real_gen}, pop at start")
                offspring, solution = generate_offspring(toolbox, population, toolbox.nchildren[toolbox.parachute_level])
                if solution:
                    return solution, toolbox.real_gen + 1
                if toolbox.verbose >= 4:
                    log_population(toolbox, offspring, f"generation {toolbox.real_gen}, offspring")
                fraction = toolbox.parents_keep_fraction[toolbox.parachute_level]
                if fraction < 1:
                    population = random.sample(population, k=int(len(population)*fraction))
                population += offspring
                population.sort(key=lambda item: item.eval)
                population[:] = population[:toolbox.pop_size[toolbox.parachute_level]]
                consistency_check(toolbox, population)
                update_dynamic_weighted_evaluation(toolbox, population)
                refresh_toolbox_from_population(toolbox, population)
                toolbox.gen += 1
                toolbox.real_gen += 1
            toolbox.parachute_level += 1
    except RuntimeWarning as e:
        toolbox.f.write("RuntimeWarning: " + str(e) + "\n")
    if toolbox.final_pop_file: # write the set covering input files
        write_population(toolbox.final_pop_file, population, toolbox.functions)
    return (population[0] if len(population) > 0 else None), toolbox.real_gen + 1


