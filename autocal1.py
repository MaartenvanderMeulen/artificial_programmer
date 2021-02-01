import os
import sys
import time
import json


class Context(object):
    def __init__(self, n_runs):
        self.best_params_file = "experimenten/params_calbest.txt"
        self.scenario = "c"
        self.working_params_file = f"experimenten/params_{self.scenario}.txt"
        self.min_value = 0.1
        self.max_value = 2.0
        self.precision = 0.1
        self.best_score = None
        self.first_seed = 1000
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
            time.sleep(2) # make sure everything is finished
            prev_count_complete = count_complete
        sys.stdout.write("\n")

    def get_score(self):
        os.system(f"grep solv tmp/{self.scenario}/log_* | wc -l > tmp/score.txt")
        with open("tmp/score.txt", "r") as f:
            line = f.readline()
            score = int(line)
            return score

    def store_value(self, value, score):
        lookup_value = round(value / self.precision)
        self.scores_cache[lookup_value] = score

    def compute_score(self, value):
        print(self.param_name, "value", value, "start compute score")
        lookup_value = round(value / self.precision)
        if lookup_value in self.scores_cache:
            return self.scores_cache[lookup_value]
        params = self.read_params(self.best_params_file)
        params[self.param_name] = value
        self.write_params(params, self.working_params_file)
        self.launch_run()
        self.wait_for_completion()
        score = self.get_score()
        print(self.param_name, "value", value, "score", score)
        self.scores_cache[lookup_value] = score
        if self.best_score is not None and self.best_score < score:
            print(self.param_name, "value", value, "new best score", score, "write_params", self.best_params_file)
            self.write_params(params, self.best_params_file)
        return score

    def autocal(self, param_name, best_value):
        print(param_name, "value", best_value, "start calibration of this param, clear score cache")
        self.scores_cache = dict()
        self.param_name = param_name
        self.best_value = best_value
        if self.best_score is None:
            self.best_score = self.compute_score(self.best_value)
        else:
            print(param_name, "value", best_value, "score", self.best_score, "add to score cache")
            self.store_value(self.best_value, self.best_score)
        epsilon = 0.001    
        continue_searching = True
        count_improvements = 0
        while continue_searching:
            continue_searching = False
            if self.min_value - epsilon <= self.best_value - self.precision:
                print(self.param_name, "best value", self.best_value, "score", self.best_score, "check down", self.precision)
                new_value = self.best_value - self.precision
                new_score = self.compute_score(new_value)
                if self.best_score < new_score:
                    self.best_score = new_score
                    self.best_value = new_value
                    continue_searching = True
                    count_improvements += 1
                    continue
            if self.max_value + epsilon >= self.best_value + self.precision:
                print(self.param_name, "best value", self.best_value, "check up", self.precision)
                new_value = self.best_value + self.precision
                new_score = context.compute_score(new_value)
                if self.best_score < new_score:
                    self.best_score = new_score
                    self.best_value = new_value
                    continue_searching = True
                    count_improvements += 1
                    continue   
        return count_improvements


if __name__ == "__main__":
    for n_runs in [3*31, 3*3*31, 3*3*3*31,]:
        print("Start calibration with", n_runs, "runs")
        context = Context(n_runs)
        continue_searching = True
        while continue_searching:
            continue_searching = False
            for i, param in enumerate([
                    "w3",
                    "w4",
                    "w5",
                    "w2a",
                    "w2b",
                    "w8",
                    "w1",
                    "w6",
                    "w7",
                    ]):
                params = context.read_params(context.best_params_file)
                if context.autocal(param, params[param]):
                    continue_searching = True

