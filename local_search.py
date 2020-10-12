import interpret
import evaluate
import copy
from deap import creator


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

    def find_code_improvement_impl(self, branch):
        if type(branch) != type([]):
            return        
        for index, old_value in enumerate(branch):
            if type(old_value) == type(1):
                for new_value in [old_value + 1, old_value - 1]:
                    if new_value >= 0:
                        branch[index] = new_value
                        error = evaluate_individual(self.toolbox, self.code)
                        if self.best_error > error:
                            self.best_error = error
                            self.best_code_change = (branch, index, new_value)
                branch[index] = old_value
            else:
                self.find_code_improvement_impl(old_value)

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
    if True:
        while helper.best_error < helper.initial_error / 2:
            helper.execute_code_change(helper.best_code_change)        
            helper.find_code_improvement()
    return code


def local_search(toolbox, individual):
    deap_str_initial = str(individual).replace("'", "")
    code = interpret.compile_deap(deap_str_initial, toolbox.functions)
    code = local_search_impl(toolbox, code)
    deap_str = interpret.convert_code_to_deap_str(code, toolbox)
    individual = creator.Individual.from_string(deap_str, toolbox.pset)
    deap_str_final = str(individual).replace("'", "")
    if False:
        print("DEBUG local_search 13", deap_str_initial)
        print("DEBUG local_search 15", code)
        print("DEBUG local_search 19", deap_str)
        print("DEBUG local_search 23", deap_str_final)
        assert deap_str_initial == deap_str
        assert deap_str_initial == deap_str_final
    return individual


if __name__ == "__main__":
    pass

