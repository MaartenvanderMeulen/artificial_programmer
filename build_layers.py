'''Build layers'''
import os
import sys
import time
import random
import json

import interpret
import evaluate
import find_new_function
import solve_problems


class LayerBuilder(object):
    def __init__(self, old_functions, example_inputs, log_file, verbose):
        self.old_functions = old_functions
        self.example_inputs = example_inputs
        self.log_file = log_file
        self.verbose = verbose
        
    def _compute_tree_depth_and_size(self, model_output):
        tree_depth, tree_size = 1, 1
        if type(model_output) == type([]):
            for item in model_output:
                depth, size = self._compute_tree_depth_and_size(item)
                if tree_depth < 1 + depth:
                    tree_depth = 1 + depth
                tree_size += size
        return tree_depth, tree_size
        
    def _append(self, code_tree, tree_depth, tree_size, free_variables):
        if len(free_variables) > 0:
            if free_variables.isdisjoint(set(self.loop_variables)):
                model_outputs = []
                for input in self.example_inputs:
                    variables = interpret.bind_params(self.parameters, input)
                    model_output = interpret.run(code_tree, variables, self.old_functions)
                    model_outputs.append(model_output)        
                model_outputs_tuple = find_new_function.recursive_tuple(model_outputs)
                if model_outputs_tuple not in self.families:
                    self.families[model_outputs_tuple] = (code_tree, tree_depth, tree_size)
                    self.new_code_trees.append([code_tree, tree_depth, tree_size, free_variables])                
                else:
                    stored_code_tree, stored_tree_depth, stored_tree_size = self.families[model_outputs_tuple]
                    if stored_tree_size > tree_size:
                        self.families[model_outputs_tuple] = (code_tree, tree_depth, tree_size)
                        self.new_code_trees.append([code_tree, tree_depth, tree_size, free_variables])                
                    else:
                        old_code_tree = code_tree
                        code_tree, tree_depth, tree_size = stored_code_tree, stored_tree_depth, stored_tree_size
            else:
                self.new_code_trees.append([code_tree, tree_depth, tree_size, free_variables])
        else:
            variables = dict()
            model_output = interpret.run(code_tree, variables, self.old_functions)
            model_output_tuple = find_new_function.recursive_tuple(model_output)
            if model_output_tuple not in self.constant_expressions:
                self.constant_expressions.add(model_output_tuple)
                tree_depth, tree_size = self._compute_tree_depth_and_size(model_output)
                if tree_size == 1:
                    print("new constant value", model_output_tuple, "depth", tree_depth, "size", tree_size)
                self.new_code_trees.append([model_output, tree_depth, tree_size, set()])
            
    def _generate_all_code_trees_of_depth(self, depth):
        '''Using self.code_trees, which contains all code trees <= depth-1'''
        if depth == 1:
            for variable in self.loop_variables + self.parameters:
                tree_size, free_variables = 1, set([variable])
                self._append(variable, depth, tree_size, free_variables)
            for constant in self.constants:
                tree_size, free_variables = 1, set()
                self._append(constant, depth, tree_size, free_variables)
        else:
            for fname,(params,_) in self.snippets.items():
                if len(params) == 1: # functions like "len", 
                    for code_tree, code_tree_depth, code_tree_size, free_variables in self.code_trees:
                        if 1+code_tree_depth == depth:
                            self._append([fname, code_tree], depth, 1 + code_tree_size, free_variables)
                elif len(params) == 2:
                    for code_tree1, code_tree_depth1, code_tree_size1, free_variables1 in self.code_trees:
                        for code_tree2, code_tree_depth2, code_tree_size2, free_variables2 in self.code_trees:
                            if 1+code_tree_depth1 == depth or 1+code_tree_depth2 == depth:
                                code_tree_size = 1 + code_tree_size1 + code_tree_size2
                                free_variables = free_variables1.union(free_variables2)
                                self._append([fname, code_tree1, code_tree2], depth, code_tree_size, free_variables)
                elif fname == "for":
                    for id in self.loop_variables:
                        for code_tree1, code_tree_depth1, code_tree_size1, free_variables1 in self.code_trees:
                            for code_tree2, code_tree_depth2, code_tree_size2, free_variables2 in self.code_trees:
                                if 1+code_tree_depth1 == depth or 1+code_tree_depth2 == depth:
                                    code_tree_size = 1 + 1 + code_tree_size1 + code_tree_size2
                                    free_variables = free_variables1.union(free_variables2.difference(set([id])))
                                    self._append([fname, id, code_tree1, code_tree2], depth, code_tree_size, free_variables)
                elif fname == "at3":
                    for code_tree1, code_tree_depth1, code_tree_size1, free_variables1 in self.code_trees:
                        for code_tree2, code_tree_depth2, code_tree_size2, free_variables2 in self.code_trees:
                            for code_tree3, code_tree_depth3, code_tree_size3, free_variables3 in self.code_trees:
                                if 1+code_tree_depth1 == depth or 1+code_tree_depth2 == depth or 1+code_tree_depth3 == depth:
                                    code_tree_size = 1 + code_tree_size1 + code_tree_size2 + code_tree_size3
                                    free_variables = free_variables1.union(free_variables2).union(free_variables3)
                                    self._append([fname, code_tree1, code_tree2, code_tree3], depth, code_tree_size, free_variables)
            
    def build_layer(self, max_depth):
        self.constants = [0, 1, 2]
        self.constant_expressions = set()
        self.loop_variables = ["i",]
        self.parameters = ["param0",]
        self.families = dict()
        self.snippets = {fname:(params,code) for fname,(params,code) in self.old_functions.items()}
        for fname in interpret.get_build_in_functions():
            # if fname in ["for", "len", "at3"]:
            if fname not in ["list3", "last3",]:
                count_params = len(interpret.get_build_in_function_param_types(fname))
                formal_params = [f"param{i}" for i in range(count_params)]
                self.snippets[fname] = (formal_params, [fname] + formal_params)    
        for terminal in self.constants + self.loop_variables + self.parameters:
            self.snippets[terminal] = ((), terminal)    
        self.code_trees = []
        for depth in range(1, max_depth+1):
            self.new_code_trees = []
            self._generate_all_code_trees_of_depth(depth)
            self.code_trees += self.new_code_trees
        new_functions = dict()
        for code_tree, _, _, free_variables in self.code_trees:
            if free_variables.isdisjoint(set(self.loop_variables)):
                fname = f"f{len(new_functions)}"
                new_functions[fname] = (self.parameters, code_tree)
        return new_functions


def is_solved_by_new_functions(problem, all_functions, new_functions, log_file, verbose):
    '''Build layers'''
    problem_label, _, example_inputs, evaluation_function, _, _ = problem
    for fname, (_, _) in new_functions.items():
        if solve_problems.is_solved_by_function(example_inputs, evaluation_function, fname, all_functions, log_file, verbose):
            log_file.write(f"solved\t{problem_label}\tby\t{fname}\n")
            return True
    log_file.write(f"failed\t{problem_label}\n")
    return False


def main(param_file):
    '''Build layers'''
    if param_file[:len("experimenten/params_")] != "experimenten/params_" or param_file[-len(".txt"):] != ".txt":
        exit("param file must have format 'experimenten/params_id.txt'")
    id = param_file[len("experimenten/params_"):-len(".txt")]    
    output_folder = f"tmp/{id}"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    with open(param_file, "r") as f:
        params = json.load(f)
    log_file = f"{output_folder}/log.txt" 
    if params.get("do_not_overwrite_logfile", False):
        if os.path.exists(log_file):
            exit(0)

    with open(f"{output_folder}/params.txt", "w") as f:
        # write a copy to the output folder
        json.dump(params, f, sort_keys=True, indent=4)

    with open(log_file, "w") as log_file:
        if hasattr(log_file, "reconfigure"):
            log_file.reconfigure(line_buffering=True)
        functions_file_name = params["functions_file"]
        problems_file_name = params["problems_file"]
        layer_output_file_name = params["layer_output_file_name"]
        verbose = params["verbose"]
        old_functions = interpret.get_functions(functions_file_name)
        problems = interpret.compile(interpret.load(problems_file_name))
        if len(problems) > 1:
            print(f"warning : build_layers : sorry, only first problem in {problems_file_name} is used.")
        _, _, example_inputs, _, _, _ = problems[0]
        start_time = time.time()
        layer_builder = LayerBuilder(old_functions, example_inputs, log_file, verbose)
        new_functions = layer_builder.build_layer(max_depth=4)
        all_functions = old_functions
        for fname, (params, code) in new_functions.items():
            interpret.add_function(["function", fname, params, code], all_functions)
        solved = is_solved_by_new_functions(problems[0], all_functions, new_functions, log_file, verbose)
        print("problem solved", bool(solved), "elapsed", round(time.time() - start_time), "seconds")
        print("writing output file ...")
        with open(layer_output_file_name, "w") as f:
            for fname, (params, code) in new_functions.items():
                f.write(f"#    (function {fname} {interpret.convert_code_to_str(params)} {interpret.convert_code_to_str(code)})\n")
        print("writing output file done")
        return 0 if solved else 1


if __name__ == "__main__":
    '''Build layers'''
    if len(sys.argv) != 2:
        exit(f"Usage: python {sys.argv[0]} paramsfile")
    param_file = sys.argv[1]
    exit(main(param_file))
