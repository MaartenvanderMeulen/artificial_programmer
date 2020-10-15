import interpret
import evaluate
import copy
import random
from deap import gp # gp.PrimitiveTree.from_string


def evaluate_individual(toolbox, code):
    weighted_error = 0.0
    for input in toolbox.example_inputs:
        variables = interpret.bind_params(toolbox.formal_params, input)
        model_output = interpret.run(code, variables, toolbox.functions)
        weighted_error += evaluate.evaluate(input, model_output, toolbox.evaluation_functions, False)
    return weighted_error
    

class FindCodeImprovementHelper:
    def __init__(self, toolbox, code):
        self.toolbox = toolbox
        self.code = code
        self.numbers = set()
        self.find_numbers(self.code)

    def find_numbers(self, branch):        
        if type(branch) != type([]):
            return        
        for index, value in enumerate(branch):
            if type(value) == type(1):
                self.numbers.add(value)
            else:
                self.find_numbers(value)

    def find_code_improvement_impl(self, branch):
        if type(branch) != type([]):
            return        
        for index, old_value in enumerate(branch):
            if type(old_value) == type(1):
                for new_value in self.numbers:
                    if old_value != new_value:
                        branch[index] = new_value
                        error = evaluate_individual(self.toolbox, self.code)
                        if self.best_error > error:
                            self.best_error = error
                            self.best_code_change = (branch, index, new_value)
                            if self.best_error < self.initial_error / 2:
                                break
                branch[index] = old_value
            else:
                self.find_code_improvement_impl(old_value)
            if self.best_error < self.initial_error / 2:
                break

    def find_code_improvement(self):
        self.initial_error = evaluate_individual(self.toolbox, self.code)
        self.best_error, self.best_code_change = self.initial_error, None
        self.find_code_improvement_impl(self.code)
    
    def execute_code_change(self, code_change):
        branch, index, new_value = code_change
        branch[index] = new_value
    

def local_search_impl(toolbox, code):
    helper = FindCodeImprovementHelper(toolbox, code)
    helper.find_code_improvement()
    improvement = False
    if helper.best_error < helper.initial_error / 2:
        helper.execute_code_change(helper.best_code_change)        
        improvement = True
    return improvement, code, helper.best_error


def local_search(toolbox, individual):
    if False:
        deap_str_initial = str(individual).replace("'", "")
        code = interpret.compile_deap(deap_str_initial, toolbox.functions)
        improvement, new_code, new_error,  = local_search_impl(toolbox, code)
        if improvement:
            deap_str = interpret.convert_code_to_deap_str(new_code, toolbox)
            ind_orig = individual
            individual = gp.PrimitiveTree.from_string(deap_str, toolbox.pset)
            individual.evaluation = new_error
            individual.parents = [ind_orig]
            deap_str_final = str(individual).replace("'", "")
            if deap_str_initial != deap_str_final and random.random() < 0.0:
                print("deap_str_initial", deap_str_initial)
                print("deap_str_final  ", deap_str_final)
                print()
    return individual


if __name__ == "__main__":
    pass

