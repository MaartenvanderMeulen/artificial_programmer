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
from ga_search_tools import write_population, consistency_check, log_population
from ga_search_tools import best_of_n, generate_initial_population, generate_initial_population_impl, refresh_toolbox_from_population
from ga_search_tools import update_dynamic_weighted_evaluation
from ga_search_tools import crossover_with_local_search, cxOnePoint, mutUniform, replace_subtree_at_best_location


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
    # * hoe verder de outputs uit elkaar liggen hoe meer kans.  p_complementair = verschillende_outputs(parent1, parent2) / aantal_outputs
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
        model_evals1 = toolbox.families_list[parent1.family_index][2]
        model_evals2 = toolbox.families_list[parent2.family_index][2]
        assert parent1.eval < 0.1 or math.isclose(parent1.eval, sum(model_evals1))
        assert parent2.eval < 0.1 or math.isclose(parent2.eval, sum(model_evals2))
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
    offspring.sort(key=lambda item: item.eval)
    if False:
        if do_special_experiment:
            expected_offspring_escapes_count = nchildren * all_escapes_count / (len(population) ** 2)
            toolbox.f.write(f"{all_escapes_count}\t{offspring_escapes_count}\t{expected_offspring_escapes_count}\n")
    return offspring, None


def apply_taboo_metaevolution(toolbox, population):
    taboo_value = population[0].eval
    if "taboo_set" in toolbox.metaevolution_strategies:
        toolbox.taboo_set.add(taboo_value)
    elif "taboo_value" in toolbox.metaevolution_strategies:
        toolbox.taboo_set = set([taboo_value])
    new_population = [ind for ind in population if ind.eval != taboo_value]
    if len(new_population) == 0:
        return population
    return new_population


def apply_mix_with_leftovers_metaevolution(toolbox, population, stay_fraction, replace_style):
    popN = toolbox.pop_size[toolbox.parachute_level]
    assert 0 <= stay_fraction and stay_fraction <= 1
    new_population = population[:int(popN*stay_fraction)]
    if replace_style == "random":
        k = popN - len(new_population)
        if k <= len(toolbox.leftovers):
            new_population += random.sample(toolbox.leftovers, k=k)
        else:
            print("gen", toolbox.gen, "len(leftovers)", len(toolbox.leftovers), "stay_fraction", stay_fraction, "k", k, "len(new_population)", len(new_population))
            new_population += toolbox.leftovers
    else:
        new_population += toolbox.leftovers[:popN - len(new_population)]
    return new_population


def ga_search_impl(toolbox):
    if toolbox.final_pop_file:
        write_population(toolbox.final_pop_file, [], toolbox.functions)
    try:
        toolbox.taboo_set = set()
        population, solution = generate_initial_population(toolbox)
        if solution:
            return solution, 0
        toolbox.parachute_count = len(population)
        consistency_check(toolbox, population)
        update_dynamic_weighted_evaluation(toolbox, population)
        population.sort(key=lambda item: item.eval)
        refresh_toolbox_from_population(toolbox, population)
        toolbox.gen = 0
        toolbox.best_eval = 1e9
        toolbox.prev_eval = 1e9
        stuck_count, stuck_count_since_last_meta_evolution = 0, 0
        meta_evolution_count = 0
        for toolbox.parachute_level in range(len(toolbox.ngen)):
            popN = toolbox.pop_size[toolbox.parachute_level]
            while toolbox.gen < toolbox.ngen[toolbox.parachute_level]:
                if False:
                    do_special_experiment = False # math.isclose(population[0].eval, 77.61253, abs_tol=0.00001)
                    if do_special_experiment and population[0].eval < 77.61253 - 0.00001:
                        toolbox.f.write(f"exit because of escape\n")
                        exit()
                # track if we are stuck
                if population[0].eval == toolbox.prev_eval:                        
                    if stuck_count >= toolbox.max_stuck_count:
                        raise RuntimeWarning("max stuck count exceeded")
                    stuck_count += 1
                    stuck_count_since_last_meta_evolution += 1
                else:
                    # verandering!
                    stuck_count = 0
                    if toolbox.best_eval > population[0].eval:
                        # verbetering!
                        toolbox.best_eval = population[0].eval
                        meta_evolution_count = 0
                        stuck_count_since_last_meta_evolution = 0
                toolbox.prev_eval = population[0].eval
                if False:
                    if do_special_experiment:
                        if stuck_count > 5:
                            toolbox.leftovers.sort(key=lambda item: item.eval)
                            c = analyse_parents(toolbox, population)
                            if c > 0:
                                toolbox.f.write(f"exit because 200x200 analysis on pop returns > 0\n")
                                exit()
                            toolbox.f.write(f"0\n")
                            c1 = analyse_parents(toolbox, apply_mix_with_leftovers_metaevolution(toolbox, population, 0.5, "random"))
                            toolbox.f.write(f"{c1}\n")
                            c2 = analyse_parents(toolbox, apply_mix_with_leftovers_metaevolution(toolbox, population, 0.33, "random"))
                            toolbox.f.write(f"{c2}\n")
                            c3 = analyse_parents(toolbox, apply_mix_with_leftovers_metaevolution(toolbox, population, 0.1, "random"))
                            toolbox.f.write(f"xx\t{c1}\t{c2}\t{c3}\t\n")
                            toolbox.f.write(f"exit special experiment\n")
                            exit()
                if stuck_count_since_last_meta_evolution >= 5 and len(toolbox.metaevolution_strategies) > 0:
                    if meta_evolution_count >= 5:
                        raise RuntimeWarning(f"5x meta_evolution did not work")
                    meta_evolution_count += 1
                    stuck_count_since_last_meta_evolution = 0
                    if "taboo_set" in toolbox.metaevolution_strategies or "taboo_value" in toolbox.metaevolution_strategies:
                        # Erase all parents with this value BEFORE making the children, ERASING it from the gene pool
                        population = apply_taboo_metaevolution(toolbox, population)
                    if "mix_with_leftovers" in toolbox.metaevolution_strategies:
                        population = apply_mix_with_leftovers_metaevolution(toolbox, population, 0.5, "random")
                    refresh_toolbox_from_population(toolbox, population)

                if toolbox.f and toolbox.verbose >= 1:
                    # code_str = interpret.convert_code_to_str(interpret.compile_deap(population[0].deap_str, toolbox.functions))
                    toolbox.f.write(f"gen {toolbox.gen:2d} best {population[0].eval:7.3f} sc {stuck_count} goc {meta_evolution_count} taboo {str(toolbox.taboo_set)}\n") #  {code_str}\n")
                log_population(toolbox, population, f"generation {toolbox.gen}, pop at start")
                offspring, solution = generate_offspring(toolbox, population, toolbox.nchildren[toolbox.parachute_level])
                if solution:
                    return solution, toolbox.gen+1
                consistency_check(toolbox, offspring)
                if len(offspring) != toolbox.nchildren[toolbox.parachute_level]:
                    toolbox.f.write(f"{len(offspring)} offspring\n")
                if toolbox.parachute_level == 0:
                    toolbox.parachute_offspring_count += len(offspring)
                else:
                    toolbox.normal_offspring_count += len(offspring)
                childN = toolbox.nchildren[toolbox.parachute_level]
                parents_fraction = max(0, 1 - childN / popN)
                if popN == 200:
                    assert math.isclose(parents_fraction, 0.5)
                population = random.sample(population, k=int(len(population)*parents_fraction))
                population += offspring
                population.sort(key=lambda item: item.eval)
                if stuck_count == 0:
                    toolbox.leftovers += population[popN:]
                population[:] = population[:popN]
                consistency_check(toolbox, population)
                update_dynamic_weighted_evaluation(toolbox, population)
                refresh_toolbox_from_population(toolbox, population)
                toolbox.gen += 1
    except RuntimeWarning as e:
        toolbox.f.write("RuntimeWarning: " + str(e) + "\n")
    if toolbox.final_pop_file:
        write_population(toolbox.final_pop_file, population, toolbox.functions)
    best = population[0]
    return best, toolbox.gen+1


