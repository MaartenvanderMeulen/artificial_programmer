# Inspired by https://deap.readthedocs.io/en/master/examples/gp_symbreg.html
# it is used by search.py
import random
import copy
import interpret
import evaluate
from deap import gp #  gp.PrimitiveSet, gp.genHalfAndHalf, gp.PrimitiveTree, gp.genFull


global total_eval_count
total_eval_count = 0


def evaluate_individual(toolbox, individual, debug=False):
    toolbox.eval_count += 1
    global total_eval_count
    total_eval_count += 1
    deap_str = str(individual)
    code = interpret.compile_deap(deap_str, toolbox.functions)
    if toolbox.monkey_mode: # if the solution can be found in monkey mode, the real search could in theory find it also
        code_str = interpret.convert_code_to_str(code)
        weighted_error = evaluate.evaluate_code(code_str, toolbox.solution_code_str)
        if weighted_error == 0.0:
            print("check the evaluation of the solution (should be error 0)")
            for input in toolbox.example_inputs:
                variables = interpret.bind_params(toolbox.formal_params, input)
                model_output = interpret.run(code, variables, toolbox.functions)
                _ = evaluate.evaluate(input, model_output, toolbox.evaluation_functions, True)
    else:
        weighted_error = 0.0
        for input in toolbox.example_inputs:
            variables = interpret.bind_params(toolbox.formal_params, input)
            model_output = interpret.run(code, variables, toolbox.functions)
            weighted_error += evaluate.evaluate(input, model_output, toolbox.evaluation_functions, debug)
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
        if 0 not in int_hints:
            int_hints.append(0)
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
        print("Finding zero ...")
        self.expr_zero = gp.genHalfAndHalf(pset=self.pset, min_=0, max_=0)
        while self.expr_zero[0].name != '0':
            self.expr_zero = gp.genHalfAndHalf(pset=self.pset, min_=0, max_=0)
        assert self.expr_zero[0].name == '0'
        print("Found zero")


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
        for i, ind in enumerate(population[:3]):
            toolbox.f.write(f"individual {i}\n")
            write_path(toolbox.f, ind, toolbox)
            if False:
                deap_str = str(ind)
                code = interpret.compile_deap(deap_str, toolbox.functions)
                code_str = interpret.convert_code_to_str(code)
                toolbox.f.write(f"    ind {i} {ind.evaluation} {code_str}\n")
        toolbox.f.write("\n")
        toolbox.f.flush()


def write_path(f, ind, toolbox, indent=0):
    indent_str = "".join(['  ' for i in range(indent)])
    operator_str = ["", "mutatie", "crossover"][len(ind.parents)]
    deap_str = str(ind)
    code = interpret.compile_deap(deap_str, toolbox.functions)
    code_str = interpret.convert_code_to_str(code)
    f.write(f"{indent_str}{code_str} {ind.evaluation:.3f} {operator_str}\n")
    for parent in ind.parents:
        write_path(f, parent, toolbox, indent+1)
        
        
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
    if False:
        # replace parts that make no difference by 0
        index = 0
        while index < len(individual):
            ind_tmp = copy.deepcopy(individual)
            slice_ = ind_tmp.searchSubtree(index)
            ind_tmp[slice_] = toolbox.expr_zero
            ind_tmp.evaluation = evaluate_individual(toolbox, ind_tmp)
            if individual.evaluation == ind_tmp.evaluation and len(individual) > len(ind_tmp):
                individual[slice_] = toolbox.expr_zero
            index += 1    
    return individual
    

def generate_initial_population(toolbox):
    print("generate intial pop...")
    evaluate.init_dynamic_error_weight_adjustment()        
    toolbox.ind_str_set = set()
    population = []    
    for i in range(toolbox.pop_size[0]): 
        ind = create_individual(toolbox)
        ind.parents = []
        if add_if_unique(toolbox, ind, population) and ind.evaluation == 0.0:
            return None, ind
    print("generate intial pop done")
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
    index2 = random.randrange(0, len(ind2))
    slice2 = ind2.searchSubtree(index2)
    ind1 = replace_subtree_at_best_location(toolbox, ind1, ind2[slice2] )
    return ind1 


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
    for index in indexes[:40]: # very big trees can make the optimalisation very slow
        ind_tmp = copy.deepcopy(ind_orig)
        slice_ = ind_tmp.searchSubtree(index)
        ind_tmp[slice_] = expr
        ind_tmp.evaluation = evaluate_individual(toolbox, ind_tmp)
        if ind_orig.evaluation > ind_tmp.evaluation or (ind_orig.evaluation == ind_tmp.evaluation and len(ind_orig) > len(ind_tmp)):
            ind_orig[slice_] = expr
            ind_orig.evaluation = ind_tmp.evaluation
            break
    return ind_orig


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
            ind.parents = [ind1, ind2]
        else: # Apply mutation
            ind1 = best_of_n(population, 2)            
            if toolbox.parachute_level == 0:
                ind = mutUniform(copy.deepcopy(ind1), expr=expr_mut, pset=toolbox.pset)
            else:
                expr = gp.genFull(pset=toolbox.pset, min_=0, max_=2)
                ind = replace_subtree_at_best_location(toolbox, copy.deepcopy(ind1), expr)
            ind.parents = [ind1,]
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
            #print("gen", gen, "fitness", population[0].evaluation)
            write_population(toolbox, population, f"generation {gen}, pop at start")
            offspring, solution = generate_offspring(toolbox, population, toolbox.nchildren[toolbox.parachute_level])
            if solution:
                return solution, gen+1
            population += offspring
            population.sort(key=lambda item: item.evaluation)            
            population[:] = population[:toolbox.pop_size[toolbox.parachute_level]]
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
    hops = 1
    result = None
    with open("tmp_log.txt", "w") as f:
        toolbox.f = f
        toolbox.pcrossover = 0.5
        toolbox.pmutations = 1.0 - toolbox.pcrossover
        toolbox.pop_size, toolbox.ngen = [8000, 600], [0, 100]
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
                assert evaluate_individual(toolbox, best, False) == 0
                write_path(f, best, toolbox)
                break
    return result
