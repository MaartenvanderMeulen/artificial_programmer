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
    def __init__(self, example_inputs, log_file, verbose, language_subset):
        self.example_inputs = example_inputs
        self.log_file = log_file
        self.verbose = verbose
        if language_subset == "ims":
            self.language_subset = ["for", "len", "at2", "sub", "eq", "sum", "list2", "extend"]
            self.constants = [0, 1, ]
            self.local_variables = ["i",]
            self.parameters = ["param0", "param1",]
        else:
            raise RuntimeError(f"Unknown language subset {language_subset}")        
        self.old_families = dict()
        self.family_size = dict()
        self.is_constant_family = dict()
        self.old_code_trees = []
        self.first_time = True

    def _generate_all_code_trees_of_depth1(self):
        for constant in self.constants:
            self._append(constant, 1, 1, set(), set())
        for param in self.parameters:
            self._append(param, 1, 1, set([param]), set())
        for variable in self.local_variables:
            self._append(variable, 1, 1, set(), set([variable]))

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

    def _count_use(self, code_tree, usage, level):
        if type(code_tree) == type([]):
            for item in code_tree:
                self._count_use(item, usage, level)
        elif type(code_tree) == type("") and code_tree[0] == "L" and int(code_tree[1:2]) == level:
            if code_tree not in usage:
                usage[code_tree] = 0
            usage[code_tree] += 1
        
    def _append(self, code_tree, tree_depth, tree_size, params, unassigned_locals):
        self.log_file.write(f"L{self.L}D{self.D} _append({str(code_tree)})\n")
        if self.verbose >= 1:
            if time.time() > self.last_time + 10:
                self.last_time = time.time()
                print(f"L{self.L}D{self.D} _append", code_tree, "new_code_trees size", len(self.new_code_trees), "new_families size", len(self.new_families))
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
                if model_outputs_tuple in self.old_families:
                    pass
                else:
                    if model_outputs_tuple not in self.new_families:
                        self.new_families[model_outputs_tuple] = (code_tree, tree_depth, tree_size, params, unassigned_locals)
                    else:
                        _, _, stored_tree_size, _, _ = self.new_families[model_outputs_tuple]
                        if stored_tree_size > tree_size:
                            self.new_families[model_outputs_tuple] = (code_tree, tree_depth, tree_size, params, unassigned_locals)
        else:
            # We cannot run this code, no family optimization possible
            self.new_code_trees.append([code_tree, tree_depth, tree_size, params, unassigned_locals])

    def _generate_all_code_trees_of_depth(self, depth, root_loop_functions, param1_loop, param2_loop, for_bodies, free_locals_allowed, must_use_param):
        if self.verbose >= 2:
            print(f"L{self.L}D{depth} param_loop:")
            for value in param1_loop:
                print(f"L{self.L}D{depth}    ", value[0])
        for fname, arity in root_loop_functions:
            if arity == 1:
                for code_tree, code_tree_depth1, code_tree_size1, params, unassigned_locals in param1_loop:
                    if free_locals_allowed or len(unassigned_locals) == 0:
                        if not must_use_param or len(params) > 0:
                            tree_depth = 1 + max(1, code_tree_depth1)
                            if tree_depth == depth:
                                code_tree_size = 1 + 1 + code_tree_size1
                                self._append([fname, code_tree], tree_depth, code_tree_size, params, unassigned_locals)
            elif arity == 2:
                for code_tree1, code_tree_depth1, code_tree_size1, params1, unassigned_locals1 in param1_loop:
                    for code_tree2, code_tree_depth2, code_tree_size2, params2, unassigned_locals2 in param2_loop:
                        if free_locals_allowed or (len(unassigned_locals1) == 0 and len(unassigned_locals2) == 0):
                            if not must_use_param or len(params1) > 0 or len(params2) > 0:
                                tree_depth = 1 + max(1, code_tree_depth1, code_tree_depth2)
                                if tree_depth == depth:
                                    code_tree_size = 1 + 1 + code_tree_size1 + code_tree_size2
                                    params =params1.union(params2)
                                    unassigned_locals = unassigned_locals1.union(unassigned_locals2)
                                    self._append([fname, code_tree1, code_tree2], tree_depth, code_tree_size, params, unassigned_locals)
            elif fname in ["for", "var"]:
                for id in self.local_variables:                    
                    simplify2 = False # simplification : use at least "id" in the body 
                    for code_tree2, code_tree_depth2, code_tree_size2, params2, unassigned_locals2 in for_bodies:
                        if not simplify2 or (id in unassigned_locals2):
                            unassigned_locals2 = unassigned_locals2.difference(set([id]))
                            if free_locals_allowed or (len(unassigned_locals2) == 0):                                
                                simplify1 = True # simplification : range must use a param and may not use a free local
                                for code_tree1, code_tree_depth1, code_tree_size1, params1, unassigned_locals1 in param1_loop:
                                    if not simplify1 or (len(params1) > 0 and len(unassigned_locals1) == 0):
                                        tree_depth = 1 + max(1, 1, code_tree_depth1, code_tree_depth2)
                                        if tree_depth == depth:
                                            code_tree_size = 1 + 1 + 1 + code_tree_size1 + code_tree_size2
                                            params =params1.union(params2)
                                            unassigned_locals = unassigned_locals1.union(unassigned_locals2.difference(set([id])))
                                            self._append([fname, id, code_tree1, code_tree2], tree_depth, code_tree_size, params, unassigned_locals)

    def _update_root_loop_functions(self):
        self.root_loop_functions = []
        
        for fname in self.language_subset:
            count_params = len(interpret.get_build_in_function_param_types(fname))
            self.root_loop_functions.append((fname, count_params))
        for fname, (params, _) in self.old_functions.items():
            self.root_loop_functions.append((fname, len(params)))
        if self.verbose >= 2:
            print(f"L{self.L} root_loop_functions:")
            if self.verbose >= 2:
                for fname, arity in self.root_loop_functions:
                    print(f"L{self.L}    ", fname)
    
    def build_layer(self, max_depth, old_functions, layer_level):
        self.old_functions = old_functions
        self.L = layer_level
        self._update_root_loop_functions()
        self.new_families_this_layer = dict()
        self.last_time = time.time()
        for depth in range(1, max_depth+1):
            self.D = depth
            self.new_code_trees = []
            self.new_families = dict()
            if depth == 1:
                if self.first_time:
                    self._generate_all_code_trees_of_depth1()
                    self.first_time = False
            elif depth < max_depth:
                root_loop_functions = self.root_loop_functions
                param1_loop = self.old_code_trees
                param2_loop = self.old_code_trees
                for_bodies = self.old_code_trees
                free_locals_allowed = True
                must_use_param = False
                self._generate_all_code_trees_of_depth(depth, root_loop_functions, param1_loop, param2_loop, for_bodies, free_locals_allowed, must_use_param)
            else: # depth == max_depth
                root_loop_functions = self.root_loop_functions
                old_code_trees_no_local_variables = [item for item in self.old_code_trees if len(item[4]) == 0]
                param1_loop = old_code_trees_no_local_variables
                param2_loop = old_code_trees_no_local_variables
                for_bodies = self.old_code_trees
                free_locals_allowed = False
                must_use_param = True
                self._generate_all_code_trees_of_depth(depth, root_loop_functions, param1_loop, param2_loop, for_bodies, free_locals_allowed, must_use_param)
            self.old_code_trees += self.new_code_trees
            if self.verbose >= 2:
                print(f"L{self.L}D{depth} new families:")
                for key, value in self.new_families.items():
                    print(f"L{self.L}D{depth}    ", key, value[0])
            if self.verbose >= 2:
                print(f"L{self.L}D{depth} new code trees:")
                for value in self.new_code_trees:
                    print(f"L{self.L}D{depth}    ", value[0])
            for key, value in self.new_families.items():
                self.old_families[key] = value
                self.old_code_trees.append(value)
                self.new_families_this_layer[key] = value
        new_functions = []
        usage = {fname: 0 for fname, _ in self.old_functions.items() if fname[0] == "L" and int(fname[1]) == self.L-1}
        for key, (code_tree, depth, size, params, unassigned_locals) in self.new_families_this_layer.items():
            self._check_consistency(code_tree, depth, size, params, unassigned_locals)
            if not self.is_constant_family[key]:
                params_0_to_n = self.parameters[:len(params)]
                if params == set(params_0_to_n):
                    if depth >= 3:
                        fname = f"L{self.L}f{len(new_functions)}"
                        new_functions.append([fname, params_0_to_n, code_tree, self.family_size[key]])
                        self._count_use(code_tree, usage, self.L - 1)
        new_functions.sort(key=lambda item: -item[-1])
        return new_functions, usage


def compute_solved_all(input_chunks, all_functions, log_file, verbose):
    if verbose >= 0:
        print(f"compute solved_all:")
    evaluation_functions_set = set()
    solved = set()
    for example_inputs, evaluation_functions in input_chunks:
        arity = len(example_inputs[0])
        evaluation_functions_set.update(set(evaluation_functions))
        for fname, (_, _) in all_functions.items():
            for evaluation_function in evaluation_functions:
                if solve_problems.is_solved_by_function(example_inputs, evaluation_function, fname, all_functions, log_file, verbose):
                    solved.add(evaluation_function)
                    log_file.write(f"solved\t{evaluation_function}\tby\t{fname}\n")
                    if verbose >= 0:
                        print(f"    {fname} is evaluated OK by {evaluation_function}")
    solved_all = len(solved) == len(evaluation_functions_set)
    if verbose >= 0:
        print("    solved all", bool(solved_all))
    return solved_all


def write_layer(layer_output_file_name, new_functions):
    with open(layer_output_file_name, "w") as f:
        f.write(f"(\n")
        if False:
            sizes = [-item[-1] for item in new_functions]
            sizes.sort()
            n = max(10, min(len(new_functions) // 10, 100))
            threshold = sizes[n] if len(sizes) > n else sizes[-1]
        else:
            threshold = 0
        for fname, params, code, family_size in new_functions:
            if family_size >= threshold:
                f.write(f"    (function {fname} {interpret.convert_code_to_str(params)} {interpret.convert_code_to_str(code)}) # {family_size}\n")
        f.write(f")\n")


def write_family_size(output_file_name, new_functions):
    with open(output_file_name, "w") as f:
        for fname, params, code, family_size in new_functions:
            f.write(f"{fname}\t{family_size}\n")


def write_usage(output_file_name, usage):
    usage_list = [ (fname, count) for fname, count in usage.items()]
    usage_list.sort(key = lambda item: item[0])
    with open(output_file_name, "w") as f:
        for fname, count in usage.items():
            f.write(f"{fname}\t{count}\n")


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
    language_subset = params["language_subset"]
    start_level = params["start_level"]
    max_depth = params["max_depth"]
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
        layer_builder = LayerBuilder(example_inputs, log_file, verbose, language_subset)
        for layer_level in range(start_level, start_level+1):
            if verbose >= 0:
                print(f"L{layer_level}")
            new_functions, usage = layer_builder.build_layer(max_depth=max_depth, old_functions=all_functions, layer_level=layer_level)
            if len(new_functions) == 0:
                if verbose >= 0:
                    print(f"L{layer_level} no new functions found")
                return 1
            write_layer(f"{output_folder}/L{layer_level}D3.txt", new_functions)
            write_family_size(f"{output_folder}/L{layer_level}D3_family_size.txt", new_functions)
            write_usage(f"{output_folder}/L{layer_level}D3_usage.txt", usage)
            for fname, params, code, _ in new_functions:
                interpret.add_function(["function", fname, params, code], all_functions)
            solved_all = compute_solved_all(input_chunks, all_functions, log_file, verbose)
        return 0 if solved_all else 1


if __name__ == "__main__":
    '''Build layers'''
    if len(sys.argv) != 2:
        exit(f"Usage: python {sys.argv[0]} paramsfile")
    param_file = sys.argv[1]
    interpret.self_test()
    exit(main(param_file))
