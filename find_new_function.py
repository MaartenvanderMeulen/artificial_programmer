# Inspired by https://deap.readthedocs.io/en/master/examples/gp_symbreg.html
# it is used by search.py
import os
import random
import copy
import math
import time
import json

import interpret
import evaluate
from evaluate import recursive_tuple
import ga_search1
import ga_search_tools
from cpp_coupling import get_cpp_handle, run_on_all_inputs

from deap import gp #  gp.PrimitiveSet, gp.genHalfAndHalf, gp.PrimitiveTree, gp.genFull, gp.from_string


def f():
    '''Dummy function for DEAP'''
    return None


class Toolbox(object):
    def __init__(self, problem, functions, seed):
        problem_name, formal_params, example_inputs, evaluation_function, hints, _ = problem
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
        self.evaluation_function = evaluation_function
        self.functions = functions
        self.pset = pset
        self.solution_code_str = interpret.convert_code_to_str(solution_hints) # for monkey test
        deap_str = interpret.convert_code_to_deap_str(solution_hints, self)
        self.solution_deap_ind = gp.PrimitiveTree.from_string(deap_str, pset) # for finding shortest solution
        if deap_str != str(self.solution_deap_ind):
            print("deap_str1", deap_str)
            print("deap_str2", str(self.solution_deap_ind))
            raise RuntimeError(f"Check if function hints '{str(func_hints)}' contain all functions of solution hint '{str(solution_hints)}'")
        self.taboo_set = set()
        self.cpp_handle = get_cpp_handle(example_inputs, formal_params, var_hints)
        self.seed = seed
        self.reset()

    def reset(self):
        self.eval_cache = dict()
        self.families_list = []
        self.families_dict = dict()
        random.seed(self.seed)


def write_timings(toolbox):
    ga_search_tools.write_seconds(toolbox, toolbox.t_total, "total")
    ga_search_tools.write_seconds(toolbox, toolbox.t_cpp_interpret, "cpp_interpret")
    if toolbox.t_py_interpret > 0:
        ga_search_tools.write_seconds(toolbox, toolbox.t_py_interpret, "py_interpret")
    ga_search_tools.write_seconds(toolbox, toolbox.t_eval, "eval")


def basinhopper(toolbox):
    for _ in range(toolbox.hops):
        toolbox.eval_count = 0
        toolbox.eval_lookup_count = 0
        toolbox.t0 = time.time()
        toolbox.t_py_interpret = 0
        toolbox.t_cpp_interpret = 0
        toolbox.t_eval = 0

        best, gen = ga_search1.ga_search_impl(toolbox)
        if best and toolbox.best_ind_file:
            ga_search_tools.write_population(toolbox.best_ind_file, [best], toolbox.functions)
        toolbox.t_total = time.time() - toolbox.t0
        write_timings(toolbox)
        if best and best.eval == 0:
            code = interpret.compile_deap(best.deap_str, toolbox.functions)
            result = ["function", toolbox.problem_name, toolbox.problem_params, code]
            toolbox.f.write(f"solved\t{toolbox.problem_name}")
            if toolbox.extensive_statistics:                
                toolbox.f.write(f"\t{gen}\tgen\t{len(best)}\tlen")
                toolbox.f.write(f"\t{toolbox.eval_count}\tevals")
                toolbox.f.write(f"\t{toolbox.eval_lookup_count}")
            toolbox.f.write(f"\t{best.deap_str}")
            toolbox.f.write(f"\n")
            if toolbox.verbose >= 1:
                score, _, _ = ga_search_tools.evaluate_individual_impl(toolbox, best, 4)
                assert score == 0
            if toolbox.verbose >= 1:
                ga_search_tools.write_path(toolbox, best)
            return result
        else:
            toolbox.f.write(f"stopped\t{toolbox.problem_name}\t{gen}\tgen\t{toolbox.eval_count}\tevals\n")
        toolbox.f.flush()
        
    toolbox.f.write(f"failed\t{toolbox.problem_name}\n")
    return None

    
def solve_by_new_function(problem, functions, f, params):
    toolbox = Toolbox(problem, functions, params["seed"])
    toolbox.problem_name, toolbox.problem_params, _, _, _, _ = problem
    toolbox.monkey_mode = False
    toolbox.dynamic_weights = False # not toolbox.monkey_mode
    toolbox.child_creation_retries = 99
    toolbox.f = f
    if params["verbose"] >= 1 and len(toolbox.solution_deap_ind) > 0:
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
    
    # search
    toolbox.all_generations_ind = []
    if toolbox.all_ind_file:
        ga_search_tools.write_population(toolbox.all_ind_file, toolbox.all_generations_ind, toolbox.functions)
    result = basinhopper(toolbox)
    if toolbox.all_ind_file:
        ga_search_tools.write_population(toolbox.all_ind_file, toolbox.all_generations_ind, toolbox.functions)

    return result