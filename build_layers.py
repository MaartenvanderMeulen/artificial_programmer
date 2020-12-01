'''Build layers'''
import os
import sys
import time
import random
import json

import interpret
import evaluate
from find_new_function import recursive_tuple
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
        
    def _compute_free_variables(self, code_tree):
        params, locals = set(), set()
        if type(code_tree) == type([]):
            if len(code_tree) > 0:
                if code_tree[0] in ["for", "var"]:
                    id = code_tree[1]
                    params2, locals2 = self._compute_free_variables(code_tree[2])
                    params3, locals3 = self._compute_free_variables(code_tree[3])
                    params.update(params2)
                    params.update(params3)
                    locals.update(locals2)
                    locals.update(locals3.difference(set([id])))
                else:
                    for item in code_tree:
                        params2, locals2 = self._compute_free_variables(item)
                        params.update(params2)
                        locals.update(locals2)
        elif type(code_tree) == type(""):
            if code_tree in self.parameters:
                params.add(code_tree)
            if code_tree in self.local_variables:
                locals.add(code_tree)
        return params, locals
        
    def _check_consistency(self, code_tree, tree_depth, tree_size, params, unassigned_locals):
        tree_depth2, tree_size2 = self._compute_tree_depth_and_size(code_tree)
        assert tree_depth == tree_depth2
        if tree_size != tree_size2:
            print("code_tree", code_tree, "tree_size", tree_size, "tree_size2", tree_size2)
        assert tree_size == tree_size2
        params2, unassigned_locals2 = self._compute_free_variables(code_tree)
        if params != params2:
            print("code_tree", code_tree, "params", params, "params2", params2)
        assert params == params2
        if unassigned_locals != unassigned_locals2:
            print("code_tree", code_tree, "unassigned_locals", unassigned_locals, "unassigned_locals2 free variables", unassigned_locals2)
        assert unassigned_locals == unassigned_locals2

    def _append(self, code_tree, tree_depth, tree_size, params, unassigned_locals):
        #print("_append", code_tree)
        if len(unassigned_locals) == 0:
            if len(params) == 0 and tree_depth > 1:
                # skip constant code
                pass
            else:
                # We can run this code, do family optimization
                model_outputs = []
                for input in self.example_inputs:
                    variables = interpret.bind_params(self.parameters[:len(input)], input)
                    model_output = interpret.run(code_tree, variables, self.old_functions)
                    model_outputs.append(model_output)        
                model_outputs_tuple = recursive_tuple(model_outputs)
                if model_outputs_tuple not in self.family_size:
                    self.family_size[model_outputs_tuple] = 0
                self.family_size[model_outputs_tuple] += 1
                #print("    family size", self.family_size[model_outputs_tuple])
                if model_outputs_tuple in self.old_families:
                    pass
                else:
                    #print("    new family")
                    if model_outputs_tuple not in self.new_families:
                        self.new_families[model_outputs_tuple] = (code_tree, tree_depth, tree_size, params, unassigned_locals)
                    else:
                        stored_code_tree, stored_tree_depth, stored_tree_size, params, unassigned_locals = self.new_families[model_outputs_tuple]
                        if stored_tree_size > tree_size:
                            self.new_families[model_outputs_tuple] = (code_tree, tree_depth, tree_size, params, unassigned_locals)
                            # print("    found family improvement")
        else:
            # We cannot run this code, no family optimization possible
            self.new_code_trees.append([code_tree, tree_depth, tree_size, params, unassigned_locals])
            # print("    append to new_code_trees because of unassigned_locals: ", unassigned_locals)
            
    def _generate_all_code_trees_of_depth1(self):
        for constant in self.constants:
            self._append(constant, 1, 1, set(), set())
        for param in self.parameters:
            self._append(param, 1, 1, set([param]), set())
        for variable in self.local_variables:
            self._append(variable, 1, 1, set(), set([variable]))
        
    def _generate_all_code_trees_of_depth(self, depth):
        for fname,(params,_) in self.snippets.items():
            if len(params) == 1: # functions like "len", but also lower layer functions with one param
                for code_tree, code_tree_depth1, code_tree_size1, params, unassigned_locals in self.old_code_trees:
                    tree_depth = 1 + max(1, code_tree_depth1)
                    code_tree_size = 1 + 1 + code_tree_size1
                    self._append([fname, code_tree], tree_depth, code_tree_size, params, unassigned_locals)
            elif len(params) == 2:
                for code_tree1, code_tree_depth1, code_tree_size1, params1, unassigned_locals1 in self.old_code_trees:
                    for code_tree2, code_tree_depth2, code_tree_size2, params2, unassigned_locals2 in self.old_code_trees:
                        tree_depth = 1 + max(1, code_tree_depth1, code_tree_depth2)
                        code_tree_size = 1 + 1 + code_tree_size1 + code_tree_size2
                        params =params1.union(params2)
                        unassigned_locals = unassigned_locals1.union(unassigned_locals2)
                        self._append([fname, code_tree1, code_tree2], tree_depth, code_tree_size, params, unassigned_locals)
            elif fname in ["for", "var"]:
                for id in self.local_variables:
                    for code_tree1, code_tree_depth1, code_tree_size1, params1, unassigned_locals1 in self.old_code_trees:
                        for code_tree2, code_tree_depth2, code_tree_size2, params2, unassigned_locals2 in self.old_code_trees:
                            tree_depth = 1 + max(1, 1, code_tree_depth1, code_tree_depth2)
                            code_tree_size = 1 + 1 + 1 + code_tree_size1 + code_tree_size2
                            params =params1.union(params2)
                            unassigned_locals = unassigned_locals1.union(unassigned_locals2.difference(set([id])))
                            self._append([fname, id, code_tree1, code_tree2], tree_depth, code_tree_size, params, unassigned_locals)
                                            
    def _generate_all_code_trees_of_max_depth(self, max_depth):
        for fname,(params,_) in self.snippets.items():
            if False and len(params) == 1: # functions like "len", 
                for code_tree, code_tree_depth1, code_tree_size1, params, unassigned_locals in self.old_code_trees_no_local_variables:
                    if len(params) > 0:
                        tree_depth = 1 + max(1, code_tree_depth1)
                        code_tree_size = 1 + 1 + code_tree_size1
                        self._append([fname, code_tree], tree_depth, code_tree_size, params, unassigned_locals)
            elif False and len(params) == 2:
                if interpret.is_pure_numeric(fname):
                    for code_tree1, code_tree_depth1, code_tree_size1, params1, unassigned_locals1 in self.old_code_trees_for_numeric_functions:
                        for code_tree2, code_tree_depth2, code_tree_size2, params2, unassigned_locals2 in self.old_code_trees_for_numeric_functions:
                            if len(params1) > 0 or len(params2) > 0:
                                tree_depth = 1 + max(1, code_tree_depth1, code_tree_depth2)
                                code_tree_size = 1 + 1 + code_tree_size1 + code_tree_size2
                                params =params1.union(params2)
                                unassigned_locals = unassigned_locals1.union(unassigned_locals2)
                                self._append([fname, code_tree1, code_tree2], tree_depth, code_tree_size, params, unassigned_locals)
                else:
                    for code_tree1, code_tree_depth1, code_tree_size1, params1, unassigned_locals1 in self.old_code_trees_no_local_variables:
                        for code_tree2, code_tree_depth2, code_tree_size2, params2, unassigned_locals2 in self.old_code_trees_no_local_variables:
                            if len(params1) > 0 or len(params2) > 0:
                                tree_depth = 1 + max(1, code_tree_depth1, code_tree_depth2)
                                code_tree_size = 1 + 1 + code_tree_size1 + code_tree_size2
                                params =params1.union(params2)
                                unassigned_locals = unassigned_locals1.union(unassigned_locals2)
                                self._append([fname, code_tree1, code_tree2], tree_depth, code_tree_size, params, unassigned_locals)
            elif False and fname in ["for",]:
                next_time = time.time() + 3
                # loop over all loop variables
                for id in self.local_variables:
                    # simplification : loop over all parameters
                    for code_tree1 in self.parameters:
                        code_tree_depth1, code_tree_size1, unassigned_locals1 = 1, 1, set()
                        # simplification : loop over all bodies which use 'id'
                        for i, (code_tree2, code_tree_depth2, code_tree_size2, params2, unassigned_locals2) in enumerate(self.old_code_trees):
                            if time.time() >= next_time:
                                next_time = time.time() + 3
                                print(i, len(self.old_code_trees))
                            if set([id]) == unassigned_locals2:                                
                                params1 = set([code_tree1])
                                tree_depth = 1 + max(1, 1, code_tree_depth1, code_tree_depth2)
                                code_tree_size = 1 + 1 + 1 + code_tree_size1 + code_tree_size2
                                params =params1.union(params2)
                                unassigned_locals = unassigned_locals1
                                self._append([fname, id, code_tree1, code_tree2], tree_depth, code_tree_size, params, unassigned_locals)
            
    def build_layer(self, max_depth):
        self.constants = [0, 1, []]
        self.local_variables = ["i",]
        self.parameters = ["param0",]
        self.snippets = {fname:(params,code) for fname,(params,code) in self.old_functions.items()}
        for fname in interpret.get_build_in_functions():
            # if fname in ["for", "len", "at3"]:
            # if fname in ["add", "sub", "mul", "div"]:
            if fname not in ["list3", "last3", "at3"]:
                count_params = len(interpret.get_build_in_function_param_types(fname))
                formal_params = [f"param{i}" for i in range(count_params)]
                self.snippets[fname] = (formal_params, [fname] + formal_params)    
        self.old_code_trees = []
        self.old_families = dict()
        self.family_size = dict()
        for depth in range(1, max_depth+1):
            #print("depth", depth)
            #print("#old_code_trees", len(self.old_code_trees))
            #for item in self.old_code_trees:
            #    print("    ", item)
            #print("#old_families", len(self.old_families))
            #for key, value in self.old_families.items():
            #    print("    ", key, value)
            #print("generate")
            self.new_code_trees = []
            self.new_families = dict()
            if depth == 1:
                self._generate_all_code_trees_of_depth1()
            if depth < max_depth:
                self._generate_all_code_trees_of_depth(depth)
            else: # depth == max_depth
                self.old_code_trees_no_local_variables = []
                self.old_code_trees_for_numeric_functions = []
                for item in self.old_code_trees:
                    code_tree, _, _, params, unassigned_locals = item
                    if len(unassigned_locals) == 0:
                        self.old_code_trees_no_local_variables.append(item)
                        if len(params) > 0 or type(code_tree) == type(1):
                            self.old_code_trees_for_numeric_functions.append(item)
                #print("old_code_trees_no_local_variables", self.old_code_trees_no_local_variables)
                #print("old_code_trees_for_numeric_functions", self.old_code_trees_for_numeric_functions)
                #print("#old_families", len(self.old_families))
                self._generate_all_code_trees_of_max_depth(max_depth)
            #print("#old_code_trees", len(self.old_code_trees))
            #print("#old_families", len(self.old_families))
            #print("#new_code_trees", len(self.new_code_trees))
            #for item in self.new_code_trees:
            #    print("    ", item)
            #print("#new_families", len(self.new_families))
            #for key, value in self.new_families.items():
            #    print("    ", key, value)
            self.old_code_trees += self.new_code_trees
            for key, value in self.new_families.items():
                self.old_families[key] = value
                self.old_code_trees.append(value)
                
        self.old_code_trees.sort(key=lambda item: 100*item[1] + item[2] + len(str(item[0]))/100)
        new_functions = dict()
        for code_tree, depth, size, params, unassigned_locals in self.old_code_trees:
            self._check_consistency(code_tree, depth, size, params, unassigned_locals)
            if len(unassigned_locals) == 0:
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
