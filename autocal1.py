import os
import sys
import time
import json


class Context(object):
    def __init__(self, n_runs):
        self.best_params_file = "experimenten/params_calbest.txt"
        self.scenario = "c"
        self.working_params_file = f"experimenten/params_{self.scenario}.txt"
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
        sys.stdout.write("\n")

    def get_score(self):
        os.system(f"grep solv tmp/{self.scenario}/log_* | wc -l > tmp/score.txt")
        with open("tmp/score.txt", "r") as f:
            line = f.readline()
            score = int(line)
            return score

    def compute_score_impl(self, value):
        print(self.param_name, "value", value, "score", "?")
        params = self.read_params(self.best_params_file)
        params[self.param_name] = value
        self.write_params(params, self.working_params_file)
        self.launch_run()
        self.wait_for_completion()
        score = self.get_score()
        print(self.param_name, "value", value, "score", score)

    def compute_score(self, param_name, value):
        self.param_name = param_name
        self.compute_score_impl(value)


if __name__ == "__main__":
    S = 20
    os.system(f"tsp -S {S}")
    n = 3
    n_runs = n*S
    print(f"Start calibration with {n}x{S}={n_runs} runs")
    context = Context(n_runs)
    #param = "mut_max_height"
    #for value in [3,]:
    #    context.compute_score(param, value)
    # 3 : 11
    # 2 : 10
    # 1 : 2
    # 0 : 0
    param = "near_solution_pop_size"
    for value in [325, 275]:
        context.compute_score(param, value)
    # 400 : 4
    # 325 : 8
    # 300 : 10
    # 275 : 6
    # 200 : 5
    param = "nchildren"
    for value in [[4000,150], ]:
        context.compute_score(param, value)
    # 150 : 11
    # 125 : 12
    # 100 : 10
    # 75 : 10

        

