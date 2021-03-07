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
        self.graph = dict()
    
    def add_cx(self, fam_dict, code, fam_id, generation, parent1_fam_id, parent2_fam_id):
        if fam_id not in fam_dict:
            global_fam_id = get_global_id(self.toolbox, code)
            fam_dict[fam_id] = Family(global_fam_id, generation)
        if fam_id != parent1_fam_id and fam_id != parent2_fam_id:
            fam = fam_dict[fam_id]
            fam.parents_set.add(parent1_fam_id)
            fam.parents_set.add(parent2_fam_id)
            if parent1_fam_id in fam_dict: # is False for families made during the parachuting
                fam_dict[parent1_fam_id].childs_set.add(fam_id)
            if parent2_fam_id in fam_dict: # is False for families made during the parachuting
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

    def add_cx_to_total_graph(self, fam_dict, code, fam_id, generation, parent1_fam_id, parent2_fam_id):
        if fam_id not in fam_dict:
            global_fam_id = get_global_id(self.toolbox, code)
            fam_dict[fam_id] = Family(global_fam_id, generation)
            if global_fam_id not in self.graph:
                self.graph[global_fam_id] = []
            self.graph[global_fam_id].append(fam_dict[fam_id])
        fam = fam_dict[fam_id]
        if parent1_fam_id in fam_dict: # is False for families made during the parachuting or in initial population
            p1_fam = fam_dict[parent1_fam_id]
            fam.parents_set.add(p1_fam.global_fam_id)
            p1_fam.childs_set.add(fam.global_fam_id)
        if parent2_fam_id in fam_dict: # is False for families made during the parachuting or in initial population
            p2_fam = fam_dict[parent2_fam_id]
            fam.parents_set.add(p2_fam.global_fam_id)
            p2_fam.childs_set.add(fam.global_fam_id)

    def add_mut_to_total_graph(self, fam_dict, code, fam_id, generation, parent1_fam_id):
        if fam_id not in fam_dict:
            global_fam_id = get_global_id(self.toolbox, code)
            fam_dict[fam_id] = Family(global_fam_id, generation)
            if global_fam_id not in self.graph:
                self.graph[global_fam_id] = []
            self.graph[global_fam_id].append(fam_dict[fam_id])
        fam = fam_dict[fam_id]
        if parent1_fam_id in fam_dict: # is False for families made during the parachuting or in initial population
            p1_fam = fam_dict[parent1_fam_id]
            fam.parents_set.add(p1_fam.global_fam_id)
            p1_fam.childs_set.add(fam.global_fam_id)

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
        #print(len(self.relevant_ids_for_this_run), "families relevant for this run", self.count, "count", len(self.all_relevant_global_ids_set), "all")
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
                        f.write(f"{indent}{global_ordered_dst} ({parents_str}) [{childs_str}] <err {err:.3f}, gen {gen}, {stuck}shared {cnt}>\n")
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
    filenames = []
    for filename in os.listdir(folder):
        if filename.startswith("log_"):
            if contains_solution(folder + "/" + filename):
                filenames.append(filename)
    filenames.sort()
    for filename in filenames:
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


def read_tree_to_total_graph(context, filename):
    print("reading", filename)
    fam_dict = dict()   
    with open(filename, "r") as f:
        for line in f:
            if is_cx_line(line):
                line2 = f.readline()
                code, fam_id, _error, generation, parent1_fam_id, parent2_fam_id = parse_cx(line.rstrip(), line2.rstrip())
                context.add_cx_to_total_graph(fam_dict, code, fam_id, generation, parent1_fam_id, parent2_fam_id)
            elif is_mut_line(line):
                line2 = f.readline()
                line3 = f.readline()
                code, fam_id, _error, generation, parent1_fam_id = parse_mut(line.rstrip(), line2.rstrip(), line3.rstrip())
                context.add_mut_to_total_graph(fam_dict, code, fam_id, generation, parent1_fam_id)


def get_best_next_gfam(context, current_gfam):
    counts = dict()
    for lfam in context.graph[current_gfam.family_index]:
        # bepaal alle kinderen van lfam : met lagere error
        for child_gid in lfam.childs_set:
            child_gfam = context.toolbox.families_list[child_gid]
            if child_gfam.raw_error < current_gfam.raw_error:
                if child_gid not in counts:
                    counts[child_gid] = 0
                counts[child_gid] += 1
    counts = [(gid, count) for gid, count in counts.items()]
    counts.sort(key=lambda item: -item[1])
    if len(counts) > 0:
        print("volgende stuk in hoofdvariant heeft count", counts[0][1])
    best_next_gfam = context.toolbox.families_list[counts[0][0]] if len(counts) > 0 else None
    return best_next_gfam


def count_paths_next_gfam(context, current_gfam, next_gfam):
    count = 0
    for lfam in context.graph[current_gfam.family_index]:
        # bepaal alle kinderen van lfam : met lagere error
        for child_gid in lfam.childs_set:
            child_gfam = context.toolbox.families_list[child_gid]
            if child_gfam is next_gfam:
                count += 1
    return count


def write_main_line(context):
    print("write_main_line")
    mainline_codes = []

    code = "append(sorted_data, elem)"
    mainline_codes.append(code)

    for_str = f"for(i, sorted_data, assign(elem, i))"
    code = f"append({for_str}, elem)"
    mainline_codes.append(code)

    if_then_else_str = f"if_then_else(le(elem, i), assign(elem, i), i)"
    for_str = f"for(i, sorted_data, {if_then_else_str})"
    code = f"append({for_str}, elem)"
    mainline_codes.append(code)

    swap_str = "last3(assign(k, elem), assign(elem, i), k)"
    if_then_else_str = f"if_then_else(le(elem, i), {swap_str}, i)"
    for_str = f"for(i, sorted_data, {if_then_else_str})"
    code = f"append({for_str}, elem)"
    mainline_codes.append(code)

    mainline_gids = [get_global_id(context.toolbox, code) for code in mainline_codes]
    mainline_gfams = [context.toolbox.families_list[gid] for gid in mainline_gids]

    for i in range(len(mainline_gfams) - 1):
        current_gfam = mainline_gfams[i]
        next_gfam = mainline_gfams[i+1]
        count = count_paths_next_gfam(context, current_gfam, next_gfam)
        print(str(current_gfam.representative), f"# error {current_gfam.raw_error:.3f}, mainlines {count}")
    print(str(next_gfam.representative), "# error", next_gfam.raw_error)


def extract_main_line(toolbox):
    folder = toolbox.old_populations_folder
    print(f"extract main line from log files in {folder}")
    context = Context(toolbox)
    filenames = []
    for filename in os.listdir(folder):
        if filename.startswith("log_"):
            if contains_solution(folder + "/" + filename):
                filenames.append(filename)
                #if len(filenames) >= 5:
                #    break
    filenames.sort()
    for filename in filenames:
        read_tree_to_total_graph(context, folder + "/" + filename)
    write_main_line(context)


def main(folder_with_logfiles):
    id = "mainline"
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

    params["param_file"] = param_file
    params["id"] = id
    params["output_folder"] = output_folder
    params["seed"] = seed
    params["old_populations_folder"] = folder_with_logfiles

    functions_file_name = params["functions_file"]
    problems_file_name = params["problems_file"]
    functions = interpret.get_functions(functions_file_name)
    problems = interpret.compile(interpret.load(problems_file_name))        
    toolbox = find_new_function.initialise_toolbox(problems[0], functions, sys.stdout, params)
    if False:
        analyse_paths(toolbox)
    extract_main_line(toolbox)
    


if __name__ == "__main__":
    if len(sys.argv) != 2:
        exit(f"usage {sys.argv[0]} folder_with_logfiles")
    folder_with_logfiles = sys.argv[1]
    main(folder_with_logfiles)
