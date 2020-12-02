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
    def __init__(self, example_inputs, log_file, verbose):
        self.example_inputs = example_inputs
        self.log_file = log_file
        self.verbose = verbose
        self.constants = [0, 1, []]
        self.local_variables = ["i",]
        self.parameters = ["param0", "param1",]
        self.snippets = dict()
        for fname in interpret.get_build_in_functions():
            # if fname in ["for", "len", "at3"]:
            # if fname in ["add", "sub", "mul", "div"]:
            if fname not in ["list3", "last3", "at3"]:
                count_params = len(interpret.get_build_in_function_param_types(fname))
                formal_params = [f"param{i}" for i in range(count_params)]
                self.snippets[fname] = (formal_params, [fname] + formal_params)    
        self.old_families = dict()
        self.family_size = dict()
        self.is_constant_family = dict()
        
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
        self.log_file.write(f"L{self.L}D{self.D} _append({str(code_tree)})\n")
        if time.time() > self.last_time + 10:
            self.last_time = time.time()
            print("_append", code_tree, "new_code_trees size", len(self.new_code_trees), "new_families size", len(self.new_families))            
        # self._check_consistency(code_tree, tree_depth, tree_size, params, unassigned_locals)
        #print("_append", code_tree)
        if len(unassigned_locals) == 0:
            if len(params) == 0 and tree_depth > 1:
                # skip constant code
                pass
            else:
                # We can run this code, do family optimization
                model_outputs = []
                first, first_model_output, all_equal_to_first = True, None, True
                for input in self.example_inputs:
                    variables = interpret.bind_params(self.parameters[:len(input)], input)
                    model_output = interpret.run(code_tree, variables, self.old_functions)
                    if first:
                        first = False
                        first_model_output = model_output
                    elif first_model_output != model_output:
                        all_equal_to_first = False
                    model_outputs.append(model_output)        
                model_outputs_tuple = recursive_tuple(model_outputs)
                if model_outputs_tuple not in self.family_size:
                    self.family_size[model_outputs_tuple] = 0
                self.is_constant_family[model_outputs_tuple] = all_equal_to_first
                self.family_size[model_outputs_tuple] += 1
                #print("    family size", self.family_size[model_outputs_tuple])
                if model_outputs_tuple in self.old_families:
                    # zorg dat families die nog niet op dit Level toegevoegd zijn wel meegenomen worden op regel 
                    if model_outputs_tuple not in self.new_families_this_layer:
                        self.new_families_this_layer[model_outputs_tuple] = self.old_families[model_outputs_tuple]
                    
                else:
                    #print("    new family")
                    if model_outputs_tuple not in self.new_families:
                        self.new_families[model_outputs_tuple] = (code_tree, tree_depth, tree_size, params, unassigned_locals)
                    else:
                        _, _, stored_tree_size, _, _ = self.new_families[model_outputs_tuple]
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
            #print(fname)
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
                    # simplification : use at least "id" in the body
                    for code_tree2, code_tree_depth2, code_tree_size2, params2, unassigned_locals2 in self.old_code_trees:
                        if id in unassigned_locals2:
                            # simplification : range must use a param or free local
                            for code_tree1, code_tree_depth1, code_tree_size1, params1, unassigned_locals1 in self.old_code_trees:
                                if len(params1) > 0 or len(unassigned_locals1) > 0:
                                    tree_depth = 1 + max(1, 1, code_tree_depth1, code_tree_depth2)
                                    code_tree_size = 1 + 1 + 1 + code_tree_size1 + code_tree_size2
                                    params =params1.union(params2)
                                    unassigned_locals = unassigned_locals1.union(unassigned_locals2.difference(set([id])))
                                    self._append([fname, id, code_tree1, code_tree2], tree_depth, code_tree_size, params, unassigned_locals)
                                            
    def _generate_all_code_trees_of_max_depth(self, max_depth):
        for fname,(params,_) in self.snippets.items():
            if len(params) == 1: # functions like "len", 
                for code_tree, code_tree_depth1, code_tree_size1, params, unassigned_locals in self.old_code_trees_no_local_variables:
                    if len(params) > 0:
                        tree_depth = 1 + max(1, code_tree_depth1)
                        code_tree_size = 1 + 1 + code_tree_size1
                        self._append([fname, code_tree], tree_depth, code_tree_size, params, unassigned_locals)
            elif len(params) == 2:
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
            elif fname in ["for", "var", ]:
                # loop over all loop variables
                for id in self.local_variables:
                    # simplification : loop over all bodies which use EXACTLY 'id'
                    for i, (code_tree2, code_tree_depth2, code_tree_size2, params2, unassigned_locals2) in enumerate(self.old_code_trees):
                        if set([id]) == unassigned_locals2:                                
                            # simplification : loop over all ranges which usea parameter
                            for code_tree1, code_tree_depth1, code_tree_size1, params1, unassigned_locals1 in self.old_code_trees_no_local_variables:
                                if len(params1) > 0:
                                    tree_depth = 1 + max(1, 1, code_tree_depth1, code_tree_depth2)
                                    code_tree_size = 1 + 1 + 1 + code_tree_size1 + code_tree_size2
                                    params =params1.union(params2)
                                    unassigned_locals = unassigned_locals1
                                    self._append([fname, id, code_tree1, code_tree2], tree_depth, code_tree_size, params, unassigned_locals)
            
    def build_layer(self, max_depth, old_functions, layer_level):
        print("DEBUG 208 len(old_functions)", len(old_functions))
        self.old_functions = old_functions
        self.L = layer_level
        for fname,(params,code) in self.old_functions.items():
            self.snippets[fname] = (params, code)
        self.last_time = time.time()
        self.old_code_trees = []
        self.new_families_this_layer = dict()
        for depth in range(1, max_depth+1):
            self.D = depth
            print(f"D{depth}")
            self.new_code_trees = []
            self.new_families = dict()
            if depth == 1:
                self._generate_all_code_trees_of_depth1()
            elif depth < max_depth:
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
                self._generate_all_code_trees_of_max_depth(max_depth)
            self.old_code_trees += self.new_code_trees
            print("len(self.new_families)", len(self.new_families))
            for key, value in self.new_families.items():
                self.old_families[key] = value
                self.old_code_trees.append(value)                
            for key, value in self.new_families_this_layer.items():
                self.old_code_trees.append(value)                
        new_functions = dict()
        for key, (code_tree, depth, size, params, unassigned_locals) in self.new_families_this_layer.items():
            self._check_consistency(code_tree, depth, size, params, unassigned_locals)            
            if not self.is_constant_family[key]:
                params_list = list(params)
                params_list.sort()
                if params_list == self.parameters[:len(params_list)]:
                    fname = f"f{len(new_functions)}"
                    new_functions[fname] = (params_list, code_tree)
        print("len(self.new_families_this_layer)", len(self.new_families_this_layer), "len(new_functions)", len(new_functions))
        return new_functions


def compute_solved_all(input_chunks, all_functions, new_functions, log_file, verbose):
    print(f"compute solved_all, # new functions", len(new_functions))
    evaluation_functions_set = set()
    solved = set()
    for example_inputs, evaluation_functions in input_chunks:
        evaluation_functions_set.update(set(evaluation_functions))
        for fname, (_, _) in new_functions.items():
            for evaluation_function in evaluation_functions:
                if solve_problems.is_solved_by_function(example_inputs, evaluation_function, fname, all_functions, log_file, verbose):                    
                    solved.add(evaluation_function)
                    log_file.write(f"solved\t{evaluation_function}\tby\t{fname}\n")
                    print(f"    {fname} is evaluated OK by {evaluation_function}")
    return len(solved) == len(evaluation_functions_set)


def write_layer(layer_output_file_name, new_functions):
    print("writing output file ...")
    with open(layer_output_file_name, "w") as f:
        f.write(f"(\n")
        for fname, (params, code) in new_functions.items():
            f.write(f"    (function {fname} {interpret.convert_code_to_str(params)} {interpret.convert_code_to_str(code)})\n")
        f.write(f")\n")
    print("writing output file done")


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
    functions_file_name = params["functions_file"]
    inputs_file_name = params["inputs_file"]
    verbose = params["verbose"]
    with open(f"{output_folder}/params.txt", "w") as f:
        # write a copy to the output folder
        json.dump(params, f, sort_keys=True, indent=4)
        
    log_file = f"{output_folder}/log.txt" 
    with open(log_file, "w") as log_file:
        if hasattr(log_file, "reconfigure"):
            log_file.reconfigure(line_buffering=True)
        all_functions = interpret.get_functions(functions_file_name)
        input_chunks = interpret.compile(interpret.load(inputs_file_name))
        example_inputs = []
        for inputs_chunk, evaluations_chunk in input_chunks:
            example_inputs += inputs_chunk
        layer_builder = LayerBuilder(example_inputs, log_file, verbose)
        layer_level = 0
        while True:
            layer_level += 1
            start_time = time.time()
            print("Starting layers level", layer_level, "len(all_functions)", len(all_functions))
            new_functions = layer_builder.build_layer(max_depth=3, old_functions=all_functions, layer_level=layer_level)
            if len(new_functions) == 0:
                print("No new functions found at layers level", layer_level)
                return 1
            write_layer(f"{output_folder}/L{layer_level}D3.txt", new_functions)
            for fname, (params, code) in new_functions.items():
                interpret.add_function(["function", fname, params, code], all_functions)
            solved_all = compute_solved_all(input_chunks, all_functions, new_functions, log_file, verbose)
            print("DEBUG 315 len(all_functions)", len(all_functions))
            print("solved all", bool(solved_all), "elapsed", round(time.time() - start_time), "seconds")
        return 0 if solved_all else 1


if __name__ == "__main__":
    '''Build layers'''
    if len(sys.argv) != 2:
        exit(f"Usage: python {sys.argv[0]} paramsfile")
    param_file = sys.argv[1]
    exit(main(param_file))
