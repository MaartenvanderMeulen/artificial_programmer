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

def cx(toolbox, parent1, parent2, final):
    '''return used_parents, next_gen_parent, child '''
    if parent1.eval > parent2.eval:
        parent1, parent2 = parent2, parent1
    if toolbox.parachute_level == 0:
        child = cxOnePoint(toolbox, parent1, parent2)
    else:
        child = crossover_with_local_search(toolbox, parent1, parent2)
    if child is not None:
        if child.eval < parent1.eval:
            return [parent1, parent2, ], [], child
        elif child.eval < parent2.eval:
            return [parent1, parent2, ], [parent1,], child
        elif final:
            return [parent1, parent2, ], [parent1, parent2], child
    return None, None, None


def mut(toolbox, parent, expr_mut, final):
    '''return used_parents, next_gen_parent, child '''
    if toolbox.parachute_level == 0:
        child = mutUniform(toolbox, parent, expr=expr_mut, pset=toolbox.pset)
    else:
        expr = gp.genFull(pset=toolbox.pset, min_=0, max_=2)
        child = replace_subtree_at_best_location(toolbox, parent, expr)
    if child is not None:
        if child.eval < parent.eval:
            return [parent], [], child
        elif final:
            return [parent], [parent], child
    return None, None, None


def generate_next_generation_impl(toolbox, usable_parents, n_cx_children, n_mut_children, \
        cx_children, mut_children, next_gen_parents, final):
    result = False
    n_cx_children -= len(cx_children)
    if 2 * n_cx_children > len(usable_parents):
        n_cx_children = len(usable_parents) // 2
    n_mut_children -= len(mut_children)
    if n_mut_children > len(usable_parents):
        n_mut_children = len(usable_parents)
    expr_mut = lambda pset, type_: gp.genFull(pset=pset, min_=0, max_=2, type_=type_)
    if n_cx_children > 0 and 2 * n_cx_children >= n_mut_children:
        # select 2 * n_cx_children uit usable_parents
        assert len(usable_parents) >= 2 * n_cx_children
        if 2 * n_cx_children > len(usable_parents):
            cx_parents = usable_parents
            n_cx_children = len(usable_parents) // 2
        else:
            cx_parents = random.sample(usable_parents, k=2 * n_cx_children)
        for i in range(n_cx_children):
            used_parents, next_gen_parent, child = cx(toolbox, cx_parents[i], cx_parents[i + n_cx_children], final)
            if child:
                for parent in used_parents:
                    usable_parents.remove(parent)
                next_gen_parents += next_gen_parent
                cx_children.append(child)
                result = True
    elif n_mut_children > 0:
        # select n_mut_children uit usable_parents
        assert len(usable_parents) >= n_mut_children
        mut_parents = random.sample(usable_parents, k=n_mut_children)
        for i in range(n_mut_children):
            used_parents, next_gen_parent, child = mut(toolbox, mut_parents[i], expr_mut, final)
            if child:
                for parent in used_parents:
                    usable_parents.remove(parent)
                next_gen_parents += next_gen_parent
                mut_children.append(child)
                result = True
    return result


def generate_next_generation(toolbox, parents, nchildren):
    nchildren = int(len(parents) / (toolbox.pcrossover + 1))
    n_cx_children = int(nchildren * toolbox.pcrossover)
    n_mut_children = nchildren - n_cx_children
    # print("DEBUG 72: n_cx_children", n_cx_children, "n_mut_children", n_mut_children, "check", 2*n_cx_children+n_mut_children)
    cx_children, mut_children, next_gen_parents = [], [], []
    iter, last_iter = 1, 20
    while iter <= last_iter and len(parents) > 1 and len(cx_children) + len(mut_children) < (n_cx_children + n_mut_children):
        final = (iter >= last_iter * 3 // 4)
        generate_next_generation_impl(toolbox, parents, n_cx_children, n_mut_children, cx_children, mut_children, next_gen_parents, final)
        iter += 1
    next_gen = next_gen_parents + cx_children + mut_children + parents
    next_gen.sort(key=lambda ind: ind.eval)
    count = sum([1 for ind in next_gen if ind.eval == next_gen[0].eval])
    avg = sum([ind.eval for ind in next_gen]) / len(next_gen)
    # print("DEBUG 98: ", f"eval {next_gen[0].eval:.3f}-{next_gen[-1].eval:.3f}, {len(next_gen)} (next_gen) = {len(cx_children)} (cxc) + {len(mut_children)} (mut) + {len(next_gen_parents)} (cxp) + {len(parents)} (p), count top {count}, avg {avg:.6f}, evals {toolbox.eval_count}")
    for ind in next_gen:
        if ind.eval == 0:
            return None, ind
    return  next_gen, None


def ga_search_impl(toolbox):
    try:
        population, solution = generate_initial_population(toolbox)
        if solution:
            return solution, 0
        toolbox.parachute_count = len(population)
        update_dynamic_weighted_evaluation(toolbox, population)
        population.sort(key=lambda item: item.eval)
        refresh_toolbox_from_population(toolbox, population)
        toolbox.gen = 0
        for toolbox.parachute_level in range(len(toolbox.ngen)):
            popN = toolbox.pop_size[toolbox.parachute_level]
            population[:] = population[:popN]
            while toolbox.gen < toolbox.ngen[toolbox.parachute_level]:
                if toolbox.f and toolbox.verbose >= 1:
                    toolbox.f.write(f"gen {toolbox.gen:2d} best {population[0].eval:7.3f} \n")
                log_population(toolbox, population, f"generation {toolbox.gen}, pop at start")
                population, solution = generate_next_generation(toolbox, population, toolbox.nchildren[toolbox.parachute_level])
                if solution:
                    return solution, toolbox.gen+1
                population.sort(key=lambda item: item.eval)
                population[:] = population[:popN]
                update_dynamic_weighted_evaluation(toolbox, population)
                refresh_toolbox_from_population(toolbox, population)
                toolbox.gen += 1
    except RuntimeWarning as e:
        toolbox.f.write("RuntimeWarning: " + str(e) + "\n")
    if toolbox.final_pop_file:
        write_population(toolbox.final_pop_file, population, toolbox.functions)
    best = population[0]
    return best, toolbox.gen+1
