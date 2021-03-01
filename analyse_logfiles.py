import sys
import os
import json

from deap import gp #  gp.PrimitiveSet, gp.genHalfAndHalf, gp.PrimitiveTree, gp.genFull, gp.from_string

import ga_search_tools
import interpret
import find_new_function
import evaluate


def get_global_id(toolbox, deap_str):
    ind = gp.PrimitiveTree.from_string(deap_str, toolbox.pset)
    pp_str = ga_search_tools.make_pp_str(ind)
    ga_search_tools.evaluate_individual(toolbox, ind, pp_str, 0)
    return ind.fam.family_index


class Family(object):
    def __init__(self, global_fam_id, generation):
        self.global_fam_id = global_fam_id
        self.generation = generation # local
        self.parents_set = set() # local
        self.childs_set = set() # local


class Context(object):
    def __init__(self, toolbox):
        self.toolbox = toolbox
        self.fam_dicts = []
        self.all_relevant_global_ids_set = set()
        self.usage_count = dict()
        self.sum_written = 0
    
    def add_cx(self, fam_dict, code, fam_id, generation, parent1_fam_id, parent2_fam_id):
        if fam_id not in fam_dict:
            global_fam_id = get_global_id(self.toolbox, code)
            fam_dict[fam_id] = Family(global_fam_id, generation)
        if fam_id != parent1_fam_id and fam_id != parent2_fam_id:
            fam = fam_dict[fam_id]
            fam.parents_set.add(parent1_fam_id)
            fam.parents_set.add(parent2_fam_id)
            if parent1_fam_id in fam_dict: # is False families made during the parachuting
                fam_dict[parent1_fam_id].childs_set.add(fam_id)
            if parent2_fam_id in fam_dict: # is False families made during the parachuting
                fam_dict[parent2_fam_id].childs_set.add(fam_id)

    def add_mut(self, fam_dict, code, fam_id, generation, parent1_fam_id):
        if fam_id not in fam_dict:
            global_fam_id = get_global_id(self.toolbox, code)
            fam_dict[fam_id] = Family(global_fam_id, generation)
        if fam_id != parent1_fam_id:
            fam = fam_dict[fam_id]
            fam.parents_set.add(parent1_fam_id)
            if parent1_fam_id in fam_dict: # is False families made during the parachuting
                fam_dict[parent1_fam_id].childs_set.add(fam_id)

    def inc_usage(self, global_id):
        if global_id not in self.usage_count:
            self.usage_count[global_id] = 0
        self.usage_count[global_id] += 1

    def extract_relevant_global_ids(self, fam_dict, solution_id):
        assert solution_id in fam_dict
        self.relevant_ids_for_this_run = set()
        fam = fam_dict[solution_id]
        self.count = 1
        self.relevant_ids_for_this_run.add(fam.global_fam_id)
        self.inc_usage(fam.global_fam_id)
        self.extract_relevant_global_ids_impl(fam_dict, fam)
        self.all_relevant_global_ids_set.update(self.relevant_ids_for_this_run)
        print(len(self.relevant_ids_for_this_run), "families relevant for this run", self.count, "count", len(self.all_relevant_global_ids_set), "all")
        return self.relevant_ids_for_this_run # relevant global ids

    def extract_relevant_global_ids_impl(self, fam_dict, fam):
        for id in fam.parents_set:
            if id in fam_dict:
                pfam = fam_dict[id]
                if pfam.generation < fam.generation:
                    if self.toolbox.families_list[pfam.global_fam_id].raw_error <= 87.7:
                        if pfam.global_fam_id not in self.relevant_ids_for_this_run:
                            self.relevant_ids_for_this_run.add(pfam.global_fam_id)
                            self.inc_usage(pfam.global_fam_id)
                            self.count += 1
                            self.extract_relevant_global_ids_impl(fam_dict, pfam)

    def assign_sorted_global_ids(self):
        print(len(self.all_relevant_global_ids_set), "relevant_global_id's")
        print("sorting on error value")
        relevant_global_ids = []
        for id in self.all_relevant_global_ids_set:
            relevant_global_ids.append((id, self.toolbox.families_list[id].raw_error))
        relevant_global_ids.sort(key=lambda item: item[1])
        self.global_ids_order_dict = dict()
        self.sum_relevant_shares = 0
        for i, (id, _error) in enumerate(relevant_global_ids):
            self.global_ids_order_dict[id] = i
            self.sum_relevant_shares += self.usage_count[id]
        print(len(self.global_ids_order_dict), "relevant_global_id's")
        print(self.sum_relevant_shares, "sum of uses of the shared relevant id's")


    def write_tree(self, filename, fam_dict, solution_id, relevant_ids_for_this_run):
        print(f"write_tree {filename}")
        self.done = set()
        self.all_ids = set()
        n = self.sum_written
        with open(filename, "w") as f:
            self.write_tree_impl(f, fam_dict, solution_id, 0, 99999, relevant_ids_for_this_run)
        print(f"write_tree {filename}, nodes {self.sum_written - n}, len(done) {len(self.done)}, len(all_ids) {len(self.all_ids)}")        

    def get_global_ordered_ids(self, fam_dict, ids_set, relevant_ids_for_this_run):
        ids = []
        for id in ids_set:
            if id in fam_dict:
                global_id = fam_dict[id].global_fam_id                
                if global_id in relevant_ids_for_this_run and global_id in self.global_ids_order_dict:
                    ids.append(self.global_ids_order_dict[global_id])
        for id in ids:
            self.all_ids.add(id)
        return ids

    def get_stuck_count(self, fam_dict, fam, relevant_ids_for_this_run):
        result = 9999
        for id in fam.parents_set:
            if id in fam_dict:
                pfam = fam_dict[id]
                global_id = pfam.global_fam_id                
                if global_id in relevant_ids_for_this_run:
                    if fam.generation > pfam.generation:
                        if result > fam.generation - pfam.generation:
                            result = fam.generation - pfam.generation
        return f"stuck {result}, " if result < 9999 else ""

    def write_tree_impl(self, f, fam_dict, dst, depth, generation, relevant_ids_for_this_run):
        if dst in fam_dict:
            fam = fam_dict[dst]
            if fam.generation <= generation:
                if fam.global_fam_id in self.global_ids_order_dict:
                    global_ordered_dst = self.global_ids_order_dict[fam.global_fam_id]
                    if global_ordered_dst not in self.done:
                        self.done.add(global_ordered_dst)
                        self.all_ids.add(global_ordered_dst)
                        indent = "--" * depth + " " if depth > 0 else ""
                        parent_ids = self.get_global_ordered_ids(fam_dict, fam.parents_set, relevant_ids_for_this_run)
                        parents_str = ", ".join([str(parent_id) for parent_id in parent_ids])
                        childs_ids = self.get_global_ordered_ids(fam_dict, fam.childs_set, relevant_ids_for_this_run)
                        childs_str = ", ".join([str(child_id) for child_id in childs_ids])
                        err = self.toolbox.families_list[fam.global_fam_id].raw_error
                        gen = fam.generation
                        cnt = self.usage_count[fam.global_fam_id]
                        stuck = self.get_stuck_count(fam_dict, fam, relevant_ids_for_this_run)
                        f.write(f"{indent}{global_ordered_dst} ({parents_str}) [{childs_str}] <err {err:.0f}, gen {gen}, {stuck}shared {cnt}>\n")
                        self.sum_written += 1
                        for parent in fam.parents_set:
                            self.write_tree_impl(f, fam_dict, parent, depth+1, fam.generation-1, relevant_ids_for_this_run)


def contains_solution(filename):
    return os.system(f"grep -q solved {filename}") == 0


def is_cx_line(line):
    return line.find("= cx [") >= 0


def is_mut_line(line):
    return line.find("= mut [") >= 0


def parse_cx(line1, line2):
    parts = line1.split(" = ")
    part1parts = parts[1].split(" ")
    fam_id = int(part1parts[5][1:-1])
    assert part1parts[6] == "error"
    error = float(part1parts[7][:-1])
    part2parts = parts[2].split(" ")
    parent1_fam_id = int(part2parts[1].split("<")[1][:-1])
    parent2_fam_id = int(part2parts[2].split("<")[1][:-1])
    generation = int(line1.split(" = ")[0].split(" ")[2][:-1])
    code = line2.split(" = ")[1]
    return code, fam_id, error, generation, parent1_fam_id, parent2_fam_id
    

def parse_mut(line1, line2, line3):
    parts = line1.split(" = ")
    part1parts = parts[1].split(" ")
    fam_id = int(part1parts[5][1:-1])
    assert part1parts[6] == "error"
    error = float(part1parts[7][:-1])
    part2parts = parts[2].split(" ")
    parent1_fam_id = int(part2parts[1].split("<")[1][:-1])
    generation = int(line1.split(" = ")[0].split(" ")[2][:-1])
    code = line3.split(" = ")[1]
    return code, fam_id, error, generation, parent1_fam_id


def read_tree(context, filename):
    print("reading", filename)
    fam_dict = dict()   
    with open(filename, "r") as f:
        for line in f:
            if is_cx_line(line):
                line2 = f.readline()
                code, fam_id, error, generation, parent1_fam_id, parent2_fam_id = parse_cx(line.rstrip(), line2.rstrip())
                context.add_cx(fam_dict, code, fam_id, generation, parent1_fam_id, parent2_fam_id)
                if error == 0.0:
                    return (fam_dict, fam_id)
            elif is_mut_line(line):
                line2 = f.readline()
                line3 = f.readline()
                code, fam_id, error, generation, parent1_fam_id = parse_mut(line.rstrip(), line2.rstrip(), line3.rstrip())
                context.add_mut(fam_dict, code, fam_id, generation, parent1_fam_id)
                if error == 0.0:
                    return (fam_dict, fam_id)
    return (None, None)


def analyse_paths(toolbox):
    folder = toolbox.old_populations_folder
    context = Context(toolbox)
    for filename in os.listdir(folder):
        #filename = "log_1766.txt"
        if filename.startswith("log_"):
            if contains_solution(folder + "/" + filename):
                fam_dict, solution_id = read_tree(context, folder + "/" + filename)
                if fam_dict is not None:
                    relevant_ids_for_this_run = context.extract_relevant_global_ids(fam_dict, solution_id)
                    context.fam_dicts.append((fam_dict, solution_id, filename, relevant_ids_for_this_run))
                    #if len(context.fam_dicts) >= 3:
                    #    break
    context.assign_sorted_global_ids()
    for fam_dict, solution_id, filename, relevant_ids_for_this_run in context.fam_dicts:
        context.write_tree(folder + "/path_" + filename[4:], fam_dict, solution_id, relevant_ids_for_this_run)
    print(context.sum_relevant_shares, "sum of uses of shared relevant id's")
    print(context.sum_written, "sum of nodes written in path files")


def initalise_toolbox(problem, functions, f, params):
    sys.setrecursionlimit(sys.getrecursionlimit() + 500)

    if params["use_one_random_seed"]:    
        random_seed = params["seed"]
        id_seed = params["seed"]
    else:
        random_seed = params["random_seed"]
        id_seed = params["id_seed"]
    toolbox = find_new_function.Toolbox(problem, functions, random_seed, id_seed)
    toolbox.problem_name, toolbox.problem_params, _, _, _, _ = problem
    toolbox.monkey_mode = False
    toolbox.child_creation_retries = 99
    toolbox.f = f
    if len(toolbox.solution_deap_ind) > 0:
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
    if len(toolbox.solution_deap_ind) > toolbox.max_individual_size:
        f.write(f"solution hint length {len(toolbox.solution_deap_ind)} longer than max length {toolbox.max_individual_size}\n")
        exit("max_individual_size to short")
    toolbox.pcrossover = params["pcrossover"]
    toolbox.pmutations = 1.0 - toolbox.pcrossover
    toolbox.best_of_n_mut = params["best_of_n_mut"]
    toolbox.best_of_n_cx = params["best_of_n_cx"]
    toolbox.parent_selection_strategy = params["parent_selection_strategy"]
    toolbox.penalise_non_reacting_models = params["penalise_non_reacting_models"]
    toolbox.hops = params["hops"]
    toolbox.output_folder = params["output_folder"]
    toolbox.final_pop_file = None # params["output_folder"] + "/pop_" + str(id_seed) + ".txt" # for "samenvoegen" runs & 'analyse_best'
    toolbox.best_ind_file = None # params["output_folder"] + "/best_" + str(id_seed) + ".txt" # for 'analyse_best'
    toolbox.good_muts_file = None # params["output_folder"] + "/goodmuts_" + str(id_seed) + ".txt"
    toolbox.bad_muts_file = None # params["output_folder"] + "/badmuts_" + str(id_seed) + ".txt"
    toolbox.fam_db_file = params["family_db_file"]
    toolbox.p_cx_c0_db_file = params["p_cx_c0_db_file"]
    toolbox.near_solution_families_file = params["output_folder"] + "/newfam_" + str(id_seed) + ".txt" # is added later to family DB
    toolbox.update_fam_db = params["update_family_db"]
    toolbox.max_raw_error_for_family_db = params["max_raw_error_for_family_db"]
    toolbox.write_cx_graph = params["write_cx_graph"]
    toolbox.new_initial_population = params["new_initial_population"]
    toolbox.old_populations_folder = params["old_populations_folder"]
    toolbox.analyse_best = params["analyse_best"]
    toolbox.analyse_cx = params["analyse_cx"]
    toolbox.compute_p_cx_c0 = params["compute_p_cx_c0"]
    toolbox.old_populations_samplesize = params["old_populations_samplesize"]
    toolbox.optimise_solution_length = params["optimise_solution_length"]
    toolbox.dynamic_weights = params["dynamic_weights"]
    toolbox.dynamic_weights_adaptation_speed = params["dynamic_weights_adaptation_speed"]
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
    toolbox.stuck_count_for_opschudding = params["stuck_count_for_opschudding"]
    toolbox.max_reenter_parachuting_phase = params["max_reenter_parachuting_phase"]
    toolbox.family_key_is_error_matrix = params["family_key_is_error_matrix"]
    toolbox.parent_selection_weight_complementairity = params["parent_selection_weight_complementairity"]
    toolbox.parent_selection_weight_cx_count = params["parent_selection_weight_cx_count"]
    toolbox.parent_selection_weight_p_out_of_pop = params["parent_selection_weight_p_out_of_pop"]
    toolbox.mut_min_height = params["mut_min_height"]
    toolbox.mut_max_height = params["mut_max_height"]
    toolbox.parents_keep_all_duration = params["parents_keep_all_duration"]
    toolbox.parents_keep_fraction_per_family = params["parents_keep_fraction_per_family"]
    toolbox.use_family_representatives_for_mutation = params["use_family_representatives_for_mutation"]
    toolbox.use_crossover_for_mutations = params["use_crossover_for_mutations"]
    toolbox.mut_local_search = params["mut_local_search"]
    toolbox.near_solution_threshold = params["near_solution_threshold"]
    toolbox.near_solution_pop_size = params["near_solution_pop_size"]
    toolbox.near_solution_max_individual_size = params["near_solution_max_individual_size"]
    toolbox.eval_count = 0

    return toolbox


def main():
    id = "path_a2"
    seed = 1000
    param_file = f"experimenten/params_{id}.txt" 
    if not os.path.exists(param_file):
        exit(f"param file {param_file} does not exist")
    output_folder = f"tmp/{id}"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    with open(param_file, "r") as f:
        params = json.load(f)
    seed += params["seed_prefix"]
    log_file = f"{output_folder}/log_{seed}.txt" 
    if params.get("do_not_overwrite_logfile", False):
        if os.path.exists(log_file):
            exit(0)
    params["param_file"] = param_file
    params["id"] = id
    params["output_folder"] = output_folder
    params["seed"] = seed

    with open(f"{output_folder}/log_{seed}.txt", "w") as log_file:
        if hasattr(log_file, "reconfigure"):
            log_file.reconfigure(line_buffering=True)
        functions_file_name = params["functions_file"]
        problems_file_name = params["problems_file"]
        functions = interpret.get_functions(functions_file_name)
        problems = interpret.compile(interpret.load(problems_file_name))        
        toolbox = initalise_toolbox(problems[0], functions, log_file, params)
        analyse_paths(toolbox)


if __name__ == "__main__":
    main()
