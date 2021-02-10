import os
import sys
import time
import json


class Context(object):
    def __init__(self, n_runs, min_value=0.1, max_value=2.0, step_size=0.1):
        self.best_params_file = "experimenten/params_calbest.txt"
        self.scenario = "c"
        self.working_params_file = f"experimenten/params_{self.scenario}.txt"
        self.min_value = min_value
        self.max_value = max_value
        self.step_size = step_size
        self.best_score = None
        self.first_seed = 1000 + n_runs # zoadat 
        self.n_runs = n_runs

    def read_params(self, paramfile):
        with open(paramfile, "r") as f:
            params = json.load(f)
        return params

    def write_params(self, params, paramfile):
        with open(paramfile, "w") as f:
            json.dump(params, f, sort_keys=True, indent=4)

    def launch_run(self):
        last_seed = self.first_seed + self.n_runs - 1
        cmd = f"rm -f tmp/{self.scenario}/* ; for seed in `seq {self.first_seed} 1 {last_seed}` ; do tsp -n python solve_problems.py $seed {self.scenario} ; done"
        os.system(cmd)

    def wait_for_completion(self):
        count_complete, prev_count_complete = 0, 0
        while count_complete < self.n_runs:
            count_complete = 0
            for i in range(self.n_runs):
                if os.path.exists(f"tmp/{self.scenario}/end_{self.first_seed+i}.txt"):
                    count_complete += 1
            if count_complete < self.n_runs and count_complete > prev_count_complete:
                sys.stdout.write(f" {count_complete}")
                sys.stdout.flush()
                time.sleep(10)
            prev_count_complete = count_complete
        # time.sleep(1) # make sure everything is finished
        sys.stdout.write("\n")

    def get_score(self):
        os.system(f"grep solv tmp/{self.scenario}/log_* | wc -l > tmp/score.txt")
        with open("tmp/score.txt", "r") as f:
            line = f.readline()
            score = int(line)
            return score

    def make_lookup_value(self, value):
        lookup_value = round(1000 * value) # precise to 0.001
        return lookup_value

    def store_value(self, value, score):
        lookup_value = self.make_lookup_value(value)
        self.scores_cache[lookup_value] = score

    def compute_score_impl(self, value):
        lookup_value = self.make_lookup_value(value)
        if lookup_value in self.scores_cache:
            print(self.param_name, "value", value, "score", self.scores_cache[lookup_value], "(from cache lookup)")
            return self.scores_cache[lookup_value]
        print(self.param_name, "value", value, "score", "?")
        params = self.read_params(self.best_params_file)
        params[self.param_name] = value
        self.write_params(params, self.working_params_file)
        self.launch_run()
        self.wait_for_completion()
        score = self.get_score()
        self.scores_cache[lookup_value] = score
        if self.best_score is not None and self.best_score < score:
            print(self.param_name, "value", value, "score", score, "(new best score)")
            self.write_params(params, self.best_params_file)
        else:
            print(self.param_name, "value", value, "score", score)
        return score

    def compute_score(self, param_name, value):
        self.scores_cache = dict()
        self.param_name = param_name
        self.best_value = value
        self.best_score = None
        self.best_score = self.compute_score_impl(value)

    def autocal(self, param_name, best_value):
        print(param_name, "value", best_value, "start calibration of this param, clear score cache")
        self.scores_cache = dict()
        self.param_name = param_name
        self.best_value = best_value
        if self.best_score is None:
            self.best_score = self.compute_score_impl(self.best_value)
        else:
            print(param_name, "value", best_value, "score", self.best_score, "(add to score cache)")
            self.store_value(self.best_value, self.best_score)
        epsilon = 0.001    
        continue_searching = True
        count_improvements = 0
        while continue_searching:
            continue_searching = False
            if self.min_value - epsilon <= self.best_value - self.step_size:
                new_value = self.best_value - self.step_size
                new_score = self.compute_score_impl(new_value)
                if self.best_score < new_score:
                    self.best_score = new_score
                    self.best_value = new_value
                    continue_searching = True
                    count_improvements += 1
                    continue
            if self.max_value + epsilon >= self.best_value + self.step_size:
                new_value = self.best_value + self.step_size
                new_score = context.compute_score_impl(new_value)
                if self.best_score < new_score:
                    self.best_score = new_score
                    self.best_value = new_value
                    continue_searching = True
                    count_improvements += 1
                    continue   
        return count_improvements


if __name__ == "__main__":
    if False:
        context = Context(n_runs)
        for i, param in enumerate([
                #"w3",
                #"w4",
                #"w5",
                #"w2a",
                #"w2b",
                "w8",
                #"w1",
                #"w6",
                #"w7",
                ]):
            params = context.read_params(context.best_params_file)
            # value = params[param]
            value = 0.3
            context.autocal(param, value)
    if False:
        context = Context(n_runs, 1, 500, 1)
        params = context.read_params(context.best_params_file)
        param = "stuck_count_for_opschudding"
        for value in [5, 10, 20, 50000]:
            context.compute_score(param, value)
    if False:
        for n_runs in [3*31, ]:
            print("Start calibration with", n_runs, "runs")
            context = Context(n_runs, 1, 1.1, 0.001)
            params = context.read_params(context.best_params_file)
            param = "dynamic_weights_adaptation_speed"
            for value in [1.001, 1.002, 1.005, ]:
                context.compute_score(param, value)
    if True:
        n_runs = 3*31
        print("Start calibration with", n_runs, "runs")
        context = Context(n_runs, 0.001, 2.0, 0.001)
        params = context.read_params(context.best_params_file)
        param = "parent_selection_weight_cx_count"
        # param = "parent_selection_weight_p_out_of_pop"
        for value in [0.0, 0.001, 0.002, 0.0005, ]:
            context.compute_score(param, value)
        

