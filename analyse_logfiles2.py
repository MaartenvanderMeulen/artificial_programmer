import sys
import os
import json

from deap import gp #  gp.PrimitiveSet, gp.genHalfAndHalf, gp.PrimitiveTree, gp.genFull, gp.from_string

import ga_search_tools
import interpret
import find_new_function


def get_global_fam(toolbox, deap_str):
    ind = gp.PrimitiveTree.from_string(deap_str, toolbox.pset)
    pp_str = ga_search_tools.make_pp_str(ind)
    ga_search_tools.evaluate_individual(toolbox, ind, pp_str, 0)
    return ind.fam


class Family(object):
    def __init__(self, global_fam, generation):
        self.gfam = global_fam
        self.global_fam_id = global_fam.family_index
        self.generation = generation # local
        self.local_parents_set = set()


class Context(object):
    def __init__(self, toolbox):
        self.toolbox = toolbox
    
    def add_init_to_total_graph(self, fam_dict, code, fam_id, generation):
        if fam_id not in fam_dict:
            global_fam = get_global_fam(self.toolbox, code)
            fam_dict[fam_id] = Family(global_fam, generation)

    def add_cx_to_total_graph(self, fam_dict, code, fam_id, generation, parent1_fam_id, parent2_fam_id):
        if fam_id not in fam_dict:
            global_fam = get_global_fam(self.toolbox, code)
            fam_dict[fam_id] = Family(global_fam, generation)
        fam = fam_dict[fam_id]
        p1_fam = fam_dict[parent1_fam_id]
        p2_fam = fam_dict[parent2_fam_id]
        if p1_fam.gfam.raw_error > fam.gfam.raw_error and p2_fam.gfam.raw_error > fam.gfam.raw_error:
            fam.local_parents_set.add(parent1_fam_id)

    def add_mut_to_total_graph(self, fam_dict, code, fam_id, generation, parent1_fam_id):
        if fam_id not in fam_dict:
            global_fam = get_global_fam(self.toolbox, code)
            fam_dict[fam_id] = Family(global_fam, generation)
        fam = fam_dict[fam_id]
        p1_fam = fam_dict[parent1_fam_id]
        if p1_fam.gfam.raw_error > fam.gfam.raw_error:
            fam.local_parents_set.add(parent1_fam_id)


def is_init_line(line):
    return line.find("= init") >= 0


def is_cx_line(line):
    return line.find("= cx [") >= 0


def is_mut_line(line):
    return line.find("= mut [") >= 0


def parse_init(line1, line2):
    parts = line1.split(" = ")
    part1parts = parts[1].split(" ")
    fam_id = int(part1parts[5][1:-1])
    assert part1parts[6] == "error"
    error = float(part1parts[7][:-1])
    generation = int(line1.split(" = ")[0].split(" ")[2][:-1])
    code = line2.split(" = ")[1]
    return code, fam_id, error, generation
    

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


def read_to_total_graph(context, filename):
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
            elif is_init_line(line):
                line2 = f.readline()
                code, fam_id, _error, generation = parse_init(line.rstrip(), line2.rstrip())
                context.add_init_to_total_graph(fam_dict, code, fam_id, generation)
    return fam_dict


def write_graph(local_fam_dict, filename):
    with open(filename, "w") as f:        
        for _local_id, local_fam in local_fam_dict.items():
            if len(local_fam.local_parents_set) == 0:
                f.write(f"999999.000 {local_fam.gfam.raw_error:.3f}\n")
            else:
                for local_parent_id in local_fam.local_parents_set:
                    e_str = f"{local_fam_dict[local_parent_id].gfam.raw_error:.3f}"
                    f.write(f"{e_str} {local_fam.gfam.raw_error:.3f}\n")


def extract_main_line(toolbox):
    folder = toolbox.old_populations_folder
    print(f"extract main line from log files in {folder}")
    context = Context(toolbox)
    filenames = []
    for filename in os.listdir(folder):
        if filename.startswith("log_"):
            filenames.append(filename)
    filenames.sort()
    for filename in filenames:        
        local_fam_dict = read_to_total_graph(context, folder + "/" + filename)
        write_graph(local_fam_dict, folder + "/" + (filename.replace("log_", "nx_")))


def main(folder_with_logfiles):
    id = "mainline"
    seed = 1000
    param_file = f"experimenten/params_{id}.txt" 
    print("using param file", param_file)
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

    extract_main_line(toolbox)
    


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"usage   : {sys.argv[0]} folder_with_logfiles")
        print(f"example : {sys.argv[0]} tmp/ad")
        exit(2)
    folder_with_logfiles = sys.argv[1]
    main(folder_with_logfiles)
