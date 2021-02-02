# Inspired by https://deap.readthedocs.io/en/master/examples/gp_symbreg.html
# it is used by search.py
import os
import random
import copy
import math
import time
import json
import cProfile
import pstats
import sys

from deap import gp #  gp.PrimitiveSet, gp.genHalfAndHalf, gp.PrimitiveTree, gp.genFull, gp.from_string

import interpret
import evaluate
from evaluate import recursive_tuple
import ga_search1
import ga_search_tools
from cpp_coupling import get_cpp_handle, run_on_all_inputs


def f():
    '''Dummy function for DEAP'''
    return None


class Toolbox(object):
    def __init__(self, problem, functions, seed):
        problem_name, formal_params, example_inputs, error_function, hints, _ = problem
        int_hints, var_hints, func_hints, solution_hints = hints
        var_names = formal_params + var_hints
        assert len(set(var_names)) == len(var_names)
        pset = gp.PrimitiveSet("MAIN", len(var_names))
        for i, param in enumerate(var_names):
            rename_cmd = f'pset.renameArguments(ARG{i}="{param}")'
            eval(rename_cmd)
        for constant in int_hints: 
            pset.addTerminal(constant)
        for function in interpret.get_build_in_functions():
            if function in func_hints:
                param_types = interpret.get_build_in_function_param_types(function)
                arity = sum([1 for t in param_types if t in [1, "v", []]])
                #print(function, arity)
                pset.addPrimitive(f, arity, name=function)
        for function, (params, _) in functions.items():
            if function in func_hints:
                arity = len(params)
                pset.addPrimitive(f, arity, name=function)
        # add recursive call if in func_hints
        if problem_name in func_hints:
            arity = len(formal_params)
            pset.addPrimitive(f, arity, name=problem_name)
            dummy_code = 0
            functions[problem_name] = [formal_params, dummy_code]
        self.problem_name = problem_name
        self.formal_params = formal_params
        self.example_inputs = example_inputs
        self.error_function = error_function
        self.functions = functions
        self.pset = pset
        self.solution_code_str = interpret.convert_code_to_str(solution_hints) # for monkey test
        deap_str = interpret.convert_code_to_deap_str(solution_hints, self)
        self.solution_deap_ind = gp.PrimitiveTree.from_string(deap_str, pset) # for finding shortest solution
        if deap_str != str(self.solution_deap_ind):
            print("deap_str1", deap_str)
            print("deap_str2", str(self.solution_deap_ind))
            raise RuntimeError(f"Check if function hints '{str(func_hints)}' contain all functions of solution hint '{str(solution_hints)}'")
        self.cpp_handle = get_cpp_handle(example_inputs, formal_params, var_hints)
        self.seed = seed
        self.reset()

    def reset(self):
        self.pp_str_to_family_index_dict = dict()
        self.families_list = []
        self.families_dict = dict()
        random.seed(self.seed)

    def sort_ind_key(self, ind):
        result = ind.normalised_error
        if self.optimise_solution_length and result == 0.0 and len(ind) > len(self.solution_deap_ind):
            result = 0.001 + (len(ind) - len(self.solution_deap_ind)) / 100000.0
        return result

    def is_solution(self, ind):
        result = ind.normalised_error
        if self.optimise_solution_length and result == 0.0 and len(ind) > len(self.solution_deap_ind):
            return False
        return result == 0.0


def call_ga_search(toolbox):
    '''store return value of ga_search1.ga_search_impl, I don't know how to get them with cProfile runctx'''
    toolbox.ga_search_impl_return_vallue = ga_search1.ga_search_impl(toolbox)


def my_profile(toolbox):
    cProfile.runctx("call_ga_search(toolbox)", globals(), locals(), filename="tmp/stats.txt")
    p = pstats.Stats("tmp/stats.txt")
    p.strip_dirs().sort_stats(pstats.SortKey.CUMULATIVE).print_stats(30)
    # toolbox.t_total = time.time() - toolbox.t0
    # ga_search_tools.write_timings(toolbox, "end of hop")
    return toolbox.ga_search_impl_return_vallue


def basinhopper(toolbox):    
    for _ in range(toolbox.hops):
        toolbox.eval_count = 0
        toolbox.eval_lookup_count = 0
        toolbox.t_cpp_interpret = 0
        toolbox.t_py_interpret = 0
        toolbox.t_error = 0
        toolbox.t_eval = 0 # is t_interpret + t_error
        toolbox.t_init = 0
        toolbox.t_refresh = 0
        toolbox.t_offspring = 0
        toolbox.t_consistency_check = 0
        toolbox.t_select_parents = 0
        toolbox.t_cx = 0
        toolbox.t_mut = 0
        toolbox.t_cx_local_search = 0

        toolbox.t0 = time.time()
        if toolbox.use_cprofile:
            best, gen = my_profile(toolbox)
        else:
            best, gen = ga_search1.ga_search_impl(toolbox)
        toolbox.t_total = time.time() - toolbox.t0
        ga_search_tools.write_timings(toolbox, "end of hop")
        if best and toolbox.best_ind_file:
            ga_search_tools.write_population(toolbox.best_ind_file, [best], toolbox.functions)
        if best and toolbox.is_solution(best):
            code = interpret.compile_deap(str(best), toolbox.functions)
            result = ["function", toolbox.problem_name, toolbox.problem_params, code]
            toolbox.f.write(f"solved\t{gen}\tgen\t{toolbox.eval_count}\tevals\t{toolbox.max_observed_stuck_count}\tmax_sc\n")
            if toolbox.verbose >= 1:
                error = ga_search_tools.forced_reevaluation_of_individual_for_debugging(toolbox, best, 4)
                assert error == 0
            ga_search_tools.write_path(toolbox, best)
            return result
        else:
            toolbox.f.write(f"stopped\t{gen}\tgen\t{toolbox.eval_count}\tevals\t{toolbox.max_observed_stuck_count}\tmax_sc\n")
        toolbox.f.flush()
        
    toolbox.f.write(f"failed\t{toolbox.problem_name}\n")
    return None

    
def solve_by_new_function(problem, functions, f, params):
    f.write(f"sys.getrecursionlimit() {sys.getrecursionlimit()}\n")
    sys.setrecursionlimit(sys.getrecursionlimit() + 500)
    f.write(f"sys.getrecursionlimit() {sys.getrecursionlimit()}\n")

    toolbox = Toolbox(problem, functions, params["seed"])
    toolbox.problem_name, toolbox.problem_params, _, _, _, _ = problem
    toolbox.monkey_mode = False
    toolbox.child_creation_retries = 99
    toolbox.f = f
    if params["verbose"] >= 3 and len(toolbox.solution_deap_ind) > 0:
        f.write(f"solution hint length {len(toolbox.solution_deap_ind)}\n")

    # tunable params
    toolbox.params = params
    toolbox.verbose = params["verbose"]
    toolbox.max_seconds = params["max_seconds"]
    toolbox.max_evaluations = params["max_evaluations"]
    toolbox.max_stuck_count = params["max_stuck_count"]
    toolbox.pop_size = params["pop_size"]
    toolbox.nchildren = params["nchildren"]
    toolbox.parents_keep_fraction = params["parents_keep_fraction"]
    toolbox.ngen = params["ngen"]
    toolbox.max_individual_size = params["max_individual_size"]
    toolbox.pcrossover = params["pcrossover"]
    toolbox.pmutations = 1.0 - toolbox.pcrossover
    toolbox.best_of_n_mut = params["best_of_n_mut"]
    toolbox.best_of_n_cx = params["best_of_n_cx"]
    toolbox.parent_selection_strategy = params["parent_selection_strategy"]
    toolbox.beta = params["weight_complementairity"]
    toolbox.penalise_non_reacting_models = params["penalise_non_reacting_models"]
    toolbox.hops = params["hops"]
    toolbox.output_folder = params["output_folder"]
    toolbox.final_pop_file = params["output_folder"] + "/pop_" + str(params["seed"]) + ".txt"
    toolbox.all_ind_file = None # params["output_folder"] + "/ind_" + str(params["seed"]) + ".txt"
    toolbox.best_ind_file = None # params["output_folder"] + "/best_" + str(params["seed"]) + ".txt"
    toolbox.new_initial_population = params["new_initial_population"]
    if not toolbox.new_initial_population:
        toolbox.old_populations_folder = params["old_populations_folder"]
        toolbox.analyse_best = params["analyse_best"]
        toolbox.old_populations_samplesize = params["old_populations_samplesize"]
    else:
        toolbox.analyse_best = False
    toolbox.optimise_solution_length = params["optimise_solution_length"]
    toolbox.extensive_statistics = params["extensive_statistics"]
    toolbox.keep_path = params["keep_path"]
    toolbox.evolution_strategies = params["evolution_strategies"]
    toolbox.metaevolution_strategies = params["metaevolution_strategies"]
    toolbox.idea_victor = params["idea_victor"]
    toolbox.dynamic_weights = params["dynamic_weights"]
    toolbox.use_cprofile = params["use_cprofile"]
    evaluate.g_w1 = params["w1"]
    evaluate.g_w2a = params["w2a"]
    evaluate.g_w2b = params["w2b"]
    evaluate.g_w3 = params["w3"]
    evaluate.g_w4 = params["w4"]
    evaluate.g_w5 = params["w5"]
    evaluate.g_w6 = params["w6"]
    evaluate.g_w7 = params["w7"]
    evaluate.g_w8 = params["w8"]
    # toolbox.stuck_count_for_opschudding = params["stuck_count_for_opschudding"]
    
    # search
    toolbox.all_generations_ind = []
    if toolbox.all_ind_file:
        ga_search_tools.write_population(toolbox.all_ind_file, toolbox.all_generations_ind, toolbox.functions)
    result = basinhopper(toolbox)
    if toolbox.all_ind_file:
        ga_search_tools.write_population(toolbox.all_ind_file, toolbox.all_generations_ind, toolbox.functions)

    return result