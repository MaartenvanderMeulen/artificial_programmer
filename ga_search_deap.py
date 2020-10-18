# Inspired by https://deap.readthedocs.io/en/master/examples/gp_symbreg.html
# it is used by search.py
import random
import copy
import interpret
import evaluate
import math
from deap import gp #  gp.PrimitiveSet, gp.genHalfAndHalf, gp.PrimitiveTree, gp.genFull


global total_eval_count
total_eval_count = 0


def evaluate_individual(toolbox, individual, debug=0):
    deap_str = str(individual)
    if deap_str in toolbox.eval_cache:
        return toolbox.eval_cache[deap_str]
    toolbox.eval_count += 1
    global total_eval_count
    total_eval_count += 1
    code = interpret.compile_deap(deap_str, toolbox.functions)
    if toolbox.monkey_mode: # if the solution can be found in monkey mode, the real search could in theory find it also
        code_str = interpret.convert_code_to_str(code)
        weighted_error = evaluate.evaluate_code(code_str, toolbox.solution_code_str)
        if weighted_error == 0.0:
            sum_v = 0
            for input in toolbox.example_inputs:
                variables = interpret.bind_params(toolbox.formal_params, input)
                model_output = interpret.run(code, variables, toolbox.functions)
                sum_v += evaluate.evaluate(input, model_output, toolbox.evaluation_functions, 1)
            if sum_v != 0:
                print("There is something wrong")
                
    else:
        weighted_error = 0.0
        prev_model_output, model_reacts_on_input = None, False
        for input in toolbox.example_inputs:
            variables = interpret.bind_params(toolbox.formal_params, input)
            model_output = interpret.run(code, variables, toolbox.functions)
            if model_output != prev_model_output:
                model_reacts_on_input = True
                prev_model_output = model_output
            v = evaluate.evaluate(input, model_output, toolbox.evaluation_functions, debug)
            weighted_error += v
        if weighted_error > 0 and not model_reacts_on_input:
            # always the same wrong answer, penalize that.  
            weighted_error *= 2
            
    toolbox.eval_cache[deap_str] = weighted_error
    return weighted_error


def f():
    '''Dummy function for DEAP'''
    return None


class Toolbox(object):    
    def __init__(self, problem, functions):
        problem_name, formal_params, example_inputs, evaluation_functions, hints, layer = problem
        int_hints, var_hints, func_hints, solution_hints = hints
        pset = gp.PrimitiveSet("MAIN", len(formal_params))
        for i, param in enumerate(formal_params):
            rename_cmd = f'pset.renameArguments(ARG{i}="{param}")'
            # print("DEBUG 62 rename_cmd", rename_cmd)
            eval(rename_cmd)
        for c in int_hints:
            pset.addTerminal(c)
        for variable in var_hints:
            if variable not in formal_params:
                # print("DEBUG 68 variable", variable)
                pset.addTerminal(variable)
        for function in interpret.get_build_in_functions():
            if function in func_hints:
                param_types = interpret.get_build_in_function_param_types(function)
                arity = sum([1 for t in param_types if t in [1, "v", []]])
                pset.addPrimitive(f, arity, name=function)
        for function, (params, code) in functions.items():
            if function in func_hints:
                arity = len(params)
                pset.addPrimitive(f, arity, name=function)            
        # toolbox = base.Toolbox()    
        self.formal_params = formal_params
        self.example_inputs = example_inputs
        self.evaluation_functions = evaluation_functions
        self.functions = functions
        self.pset = pset
        self.solution_code_str = interpret.convert_code_to_str(solution_hints)
        self.eval_cache = dict()


def best_of_n(population, n):
    inds = random.sample(population, n)
    return min(inds, key=lambda ind: ind.evaluation)


def update_dynamic_weighted_evaluation(toolbox, individuals):
    if toolbox.dynamic_weights:
        evaluate.dynamic_error_weight_adjustment()        
        for ind in individuals:        
            ind.evaluation = evaluate_individual(toolbox, ind)
            

def write_population(toolbox, population, label):
    if toolbox.f:
        toolbox.f.write("write_population " + label + "\n")
        for i, ind in enumerate(population):            
            if ind.evaluation < 2:
                if False:
                    toolbox.f.write(f"individual {i} ")
                    write_path(toolbox, ind)
                if True:
                    deap_str = str(ind)
                    #code = interpret.compile_deap(deap_str, toolbox.functions)
                    #code_str = interpret.convert_code_to_str(code)
                    toolbox.f.write(f"    ind {i} {ind.evaluation} {len(ind)} {deap_str}\n")
        toolbox.f.write("\n")
        toolbox.f.flush()


def write_path(toolbox, ind, indent=0):
    if toolbox.f:
        indent_str = "".join(['  ' for i in range(indent)])
        operator_str = ["", "mutatie", "crossover"][len(ind.parents)]
        deap_str = str(ind)
        code = interpret.compile_deap(deap_str, toolbox.functions)
        code_str = interpret.convert_code_to_str(code)
        toolbox.f.write(f"{indent_str}{code_str} {ind.evaluation:.3f} {operator_str}\n")
        for parent in ind.parents:
            write_path(toolbox, parent, indent+1)
        
        
def add_if_unique(toolbox, ind, individuals):
    deap_str = str(ind)
    if deap_str in toolbox.ind_str_set:
        return False
    toolbox.ind_str_set.add(deap_str)
    individuals.append(ind)
    if ind.evaluation is None:
        ind.evaluation = evaluate_individual(toolbox, ind)
    return True


def create_individual(toolbox):
    expr = gp.genHalfAndHalf(pset=toolbox.pset, min_=1, max_=3)
    individual = gp.PrimitiveTree(expr)
    individual.evaluation = evaluate_individual(toolbox, individual)
    return individual
    

def generate_initial_population(toolbox):
    evaluate.init_dynamic_error_weight_adjustment()        
    toolbox.ind_str_set = set()
    population = []    
    for i in range(toolbox.pop_size[0]): 
        ind = create_individual(toolbox)
        ind.parents = []
        if add_if_unique(toolbox, ind, population) and ind.evaluation == 0.0:
            return None, ind
    return population, None


def cxOnePoint(ind1, ind2):
    if len(ind1) < 2 or len(ind2) < 2:
        # No crossover on single node tree
        return ind1
    index1 = random.randrange(0, len(ind1))
    index2 = random.randrange(0, len(ind2))
    slice1 = ind1.searchSubtree(index1)
    slice2 = ind2.searchSubtree(index2)
    ind1[slice1], ind2[slice2] = ind2[slice2], ind1[slice1]
    ind1.evaluation = None
    ind2.evaluation = None
    return ind1


def crossover_with_local_search(toolbox, ind1, ind2):
    if len(ind1) < 2 or len(ind2) < 2:
        # No crossover on single node tree
        return ind1
    if ind1.evaluation > ind2.evaluation or (ind1.evaluation == ind2.evaluation and len(ind1) > len(ind2)):
        ind1, ind2 = ind2, ind1
    indexes1 = [i for i in range(len(ind1))]
    indexes2 = [i for i in range(len(ind2))]
    random.shuffle(indexes1)
    random.shuffle(indexes2)
    count_bloat, count_test, count_non_unique = 0, 0, 0
    best_evaluation, best_slice1, best_expr2, best_len = None, None, None, None
    for index2 in indexes2:
        slice2 = ind2.searchSubtree(index2)
        expr2 = ind2[slice2]
        for index1 in indexes1:
            ind1_tmp = copy.deepcopy(ind1)
            slice1 = ind1_tmp.searchSubtree(index1)
            ind1_tmp[slice1] = expr2
            if len(ind1_tmp) <= toolbox.max_individual_size:
                deap_str = str(ind1_tmp)
                if deap_str not in toolbox.ind_str_set:
                    count_test += 1
                    ind1_tmp.evaluation = evaluate_individual(toolbox, ind1_tmp)
                    if ind1.evaluation > ind1_tmp.evaluation or (ind1.evaluation == ind1_tmp.evaluation and len(ind1) > len(ind1_tmp)):
                        ind1[slice1] = expr2
                        ind1.evaluation = ind1_tmp.evaluation
                        return ind1
                    if best_evaluation is None or best_evaluation > ind1_tmp.evaluation or (best_evaluation == ind1_tmp.evaluation and best_len > len(ind1_tmp)):
                        best_evaluation = ind1_tmp.evaluation
                        best_slice1 = slice1
                        best_expr2 = expr2
                        best_len = len(ind1_tmp)
                else:
                    count_non_unique += 1
            else:
                count_bloat += 1
    assert len(ind1) * len(ind2) == count_test + count_non_unique + count_bloat
    # print(f"crossover anomaly, count_test {count_test}, len1 {len(ind1)}, len2 {len(ind2)}, count_bloat {count_bloat}, count_non_unique {count_non_unique}, current eval {ind1.evaluation}")
    if True:
        if best_evaluation is not None:
            ind1[best_slice1] = best_expr2
            ind1.evaluation = best_evaluation
            return ind1
    return None 


def mutUniform(ind, expr, pset):
    index = random.randrange(0, len(ind))
    slice_ = ind.searchSubtree(index)
    type_ = ind[index].ret
    ind[slice_] = expr(pset=pset, type_=type_)
    ind.evaluation = None
    return ind


def replace_subtree_at_best_location(toolbox, ind_orig, expr):
    indexes = [i for i in range(len(ind_orig))]
    random.shuffle(indexes)
    best_evaluation, best_slice1, best_expr2, best_len = None, None, None, None
    for index in indexes:
        ind_tmp = copy.deepcopy(ind_orig)
        slice1 = ind_tmp.searchSubtree(index)
        ind_tmp[slice1] = expr
        if len(ind_tmp) <= toolbox.max_individual_size:
            deap_str = str(ind_tmp)
            if deap_str not in toolbox.ind_str_set:
                ind_tmp.evaluation = evaluate_individual(toolbox, ind_tmp)
                if ind_orig.evaluation > ind_tmp.evaluation or (ind_orig.evaluation == ind_tmp.evaluation and len(ind_orig) > len(ind_tmp)):
                    ind_orig[slice1] = expr
                    ind_orig.evaluation = ind_tmp.evaluation
                    return ind_orig
                if best_evaluation is None or best_evaluation > ind_tmp.evaluation or (best_evaluation == ind_tmp.evaluation and best_len > len(ind_tmp)):
                    best_evaluation = ind_tmp.evaluation
                    best_slice1 = slice1
                    best_len = len(ind_tmp)
    if True:
        if best_evaluation is not None:
            ind_orig[best_slice1] = expr
            ind_orig.evaluation = best_evaluation
            return ind_orig
    return None


def generate_offspring(toolbox, population, nchildren):
    offspring = []
    expr_mut = lambda pset, type_: gp.genFull(pset=pset, min_=0, max_=2, type_=type_)
    for i in range(nchildren):
        op_choice = random.random()
        if op_choice < toolbox.pcrossover: # Apply crossover
            ind1, ind2 = [best_of_n(population, 2), best_of_n(population, 2)]
            if toolbox.parachute_level == 0:
                ind = cxOnePoint(copy.deepcopy(ind1), copy.deepcopy(ind2))
            else:
                ind = crossover_with_local_search(toolbox, copy.deepcopy(ind1), copy.deepcopy(ind2))
            parents = [ind1, ind2]
        else: # Apply mutation
            ind1 = best_of_n(population, 2)            
            if toolbox.parachute_level == 0:
                ind = mutUniform(copy.deepcopy(ind1), expr=expr_mut, pset=toolbox.pset)
            else:
                expr = gp.genFull(pset=toolbox.pset, min_=0, max_=2)
                ind = replace_subtree_at_best_location(toolbox, copy.deepcopy(ind1), expr)
            parents = [ind1,]
        if ind is not None:
            ind.parents = parents
            if add_if_unique(toolbox, ind, offspring) and ind.evaluation == 0.0:
                return None, ind
    return offspring, None


def ga_search_impl(toolbox):
    population, solution = generate_initial_population(toolbox)
    if solution:
        return solution, 0
    update_dynamic_weighted_evaluation(toolbox, population)
    population.sort(key=lambda item: item.evaluation)            
    gen = 0
    for toolbox.parachute_level in range(len(toolbox.ngen)):
        while gen < toolbox.ngen[toolbox.parachute_level]:            
            print("gen", gen, "fitness", population[0].evaluation, str(population[0]))
            #evaluate_individual(toolbox, population[0], debug=True)
            write_population(toolbox, population, f"generation {gen}, pop at start")
            offspring, solution = generate_offspring(toolbox, population, toolbox.nchildren[toolbox.parachute_level])
            if solution:
                return solution, gen+1
            print(len(offspring), "offspring")
            population += offspring
            population.sort(key=lambda item: item.evaluation)    
            population[:] = population[:toolbox.pop_size[toolbox.parachute_level]]
            toolbox.ind_str_set = {str(ind) for ind in population} # refresh set
            update_dynamic_weighted_evaluation(toolbox, population)
            gen += 1
    write_population(toolbox, population, "pop final")
    best = population[0] # min(population, key=lambda item: item.evaluation)
    return best, gen+1
    
    
def solve_by_new_function(problem, functions):
    problem_name, params, example_inputs, evaluation_functions, hints, layer = problem
    toolbox = Toolbox(problem, functions)
    toolbox.monkey_mode = False
    toolbox.dynamic_weights = False # not toolbox.monkey_mode
    toolbox.max_individual_size = 40 # max of len(individual).  Don't make individuals larger than it
    hops = 3
    result = None
    with open("tmp_log.txt", "w") as f:
        toolbox.f = f
        toolbox.pcrossover = 0.5
        toolbox.pmutations = 1.0 - toolbox.pcrossover
        toolbox.pop_size, toolbox.ngen = [8000, 1000], [4, 1000]
        toolbox.nchildren = toolbox.pop_size
        toolbox.eval_count = 0
        for hop in range(hops):
            best, gen = ga_search_impl(toolbox)        
            if best.evaluation == 0:
                deap_str = str(best)
                code = interpret.compile_deap(deap_str, toolbox.functions)
                code_str = interpret.convert_code_to_str(code)
                result = ["function", problem_name, params, code]
                result_str = interpret.convert_code_to_str(result)
                print("problem", problem_name, f"solved after {toolbox.eval_count} evaluations by", result_str)
                assert evaluate_individual(toolbox, best, 1) == 0
                write_path(toolbox, best)
                break
    return result
