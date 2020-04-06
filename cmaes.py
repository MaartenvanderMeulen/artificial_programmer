import copy
from deap import base # conda install -c conda-forge deap
from deap import cma
from deap import creator
from deap import tools
import math
import numpy as np
import random


def cmaes(dim, f, y_target=0.0):
    '''Return x for which f(x) is minimal. Stop early when y-target is reached.'''
    if dim < 2 or 10000 < dim:
        print("nparams value is invalid, must be in [2, 10000]")
        return None
    population_size = max(math.ceil(4 + 3 * math.log(dim) + 0.5), dim // 2) # dim/2 is result of experiments by Maarten on dim > 100
    population_size *= 2
    ngen = 600 # 25 + int(0.2*dim) # result of experiments by Maarten
    print("dimension of problem space ", dim)
    print("population size            ", population_size)
    print("generations                ", ngen)    
    #random.seed(42)
    #np.random.seed(42)    
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    nhops = 10
    for hop in range(nhops):
        centroid = [random.randint(-3, 3) for i in range(dim)]
        strategy = cma.Strategy(centroid=centroid, sigma=0.5, lambda_=population_size)
        toolbox = base.Toolbox()
        toolbox.register("generate", strategy.generate, creator.Individual)
        toolbox.register("update", strategy.update)
        evaluation_count, best_x, best_y = 0, [1]*dim, None
        try:
            for gen in range(ngen):
                population = toolbox.generate()
                if False:
                    for i in range(len(population)):
                        for j in range(dim):
                            population[i][j] = round(population[i][j]*1000) / 1000
                for x in population:
                    y = f(x)
                    x.fitness.values = (y,)
                    evaluation_count += 1
                    if best_y is None or best_y > y:
                        best_x, best_y = copy.deepcopy(x), copy.deepcopy(y)
                        if False:
                            x_str = ", ".join([f"{xi:.6f}" if xi != round(xi) else f"{int(round(xi))}" for xi in best_x])
                            print(f"    gen {gen}, f({x_str}) = {best_y:.6f}")
                if best_y <= y_target:
                    break
                toolbox.update(population)
            x_str = ", ".join([f"{xi:.12f}" if xi != round(xi) else f"{int(round(xi))}" for xi in best_x])
            print(f"evaluations {evaluation_count} evaluations f({x_str}) = {best_y:.12f}")
        except:
            pass
    return best_x, best_y


f_easy_points = (
    (4, 98, 19, -7191),
    (51, 22, 8, -8857),
    (95, 80, 96, -729057),
    (84, 34, 57, -162469),
    (58, 61, 81, -286155),
    (2, 88, 18, -2936),
    (68, 31, 24, -50390),
    (9, 46, 78, -31957),
    (63, 48, 8, -24009),
    (72, 2, 55, -7679),
    (92, 48, 33, -145441),
    (15, 33, 87, -42723),
    (8, 55, 51, -22169),
    (41, 8, 25, -8068),
    (5, 98, 19, -9052),
    (65, 66, 85, -364198),
    (3, 18, 93, -4704),
    (21, 44, 31, -28442),
    (38, 12, 71, -32101),)


f_medium_b_points = (
    (46, 58, 21, 7.615773105834),
    (86, 53, 26, 7.280109889383),
    (45, 59, 87, 7.681145747891),
    (5, 83, 75, 9.110433579135),
    (31, 10, 4, 3.162277660165),
    (34, 88, 51, 9.380831519433),
    (66, 73, 69, 8.544003745075),
    (10, 46, 9, 6.782329983123),
    (55, 9, 80, 2.999999999956),
    (21, 70, 23, 8.366600265377),
    (3, 21, 22, 4.582575694956),
    (46, 43, 69, 6.557438524411),
    (75, 33, 23, 5.744562646563),
    (56, 98, 75, 9.899494937097),
    (92, 25, 88, 5.000000000204),
    (94, 49, 27, 6.999999999738),
    (27, 90, 60, 9.486832980794),
    (16, 37, 53, 6.082762530270),
    (60, 58, 8, 7.615773105867),)


f_medium_b2_points = (
    (46, 58, 21, 2.0865917971),
    (86, 53, 26, 2.0864861634),
    (45, 59, 87, 2.0866107887),
    (5, 83, 75, 2.0869299051),
    (31, 10, 4, 2.0813720966),
    (34, 88, 51, 2.08697457),
    (66, 73, 69, 2.0868223121),
    (10, 46, 9, 2.0863000034),
    (55, 9, 80, 2.0806945863),
    (21, 70, 23, 2.0867840711),
    (3, 21, 22, 2.0846394984),
    (46, 43, 69, 2.0862018241),
    (75, 33, 23, 2.0857470605),
    (56, 98, 75, 2.0870502781),
    (92, 25, 88, 2.08512511),
    (94, 49, 27, 2.0863862506),
    (27, 90, 60, 2.086991052),
    (16, 37, 53, 2.0859581733),
    (60, 58, 8, 2.0865917971),)

def f_easy(x):
    '''The 'easy' puzzle of Victor x0*ABC+x1*AB+x2*AC+...+x7, waarbij x tussen -9 en 9'''
    global f_easy_points
    sum_se = 0.0        
    for A, B, C, y in f_easy_points:
        result = x[0]*A*B*C + x[1]*A*B + x[2]*A*C + x[3]*B*C + x[4]*A + x[5]*B + x[6]*C + x[7]
        if not math.isfinite(result):
            return 1e9
        assert math.isfinite(result)
        sum_se += (y - result) ** 2
    rmse = math.sqrt(sum_se / len(f_easy_points))
    return rmse

    
def f_medium_b(x):
    '''The part of the 'medium' puzzle of Victor that assumed A+AB+ABC-CC+C/A+7B/C/A
    and tries to find x0-x5 for the remaining function of B'''
    global f_medium_b2_points
    sum_se = 0.0        
    for A, B, C, y in f_medium_b2_points:
        # result = (x[0]*B*B + x[1]*B + x[2]) / (x[3]*B*B + x[4]*B + x[5]) # overgespecificeerd
        # result = (x[0]*B + x[1]) / (x[2]*B*B + 1) # f(0.110, 2.197, 0.000) = 0.097908
        result = (x[0]*B + x[1]) / (x[2]*B + 1) # f(0.168, 1.796, 0.009) = 0.044172
        # result = (x[0]*B*B + x[1]*B + x[2]) / (x[3]*B + 1) # f(0.001, 0.263, 1.333, 0.029) = 0.005183
        # result = (x[0]*B*B + x[1]*B + x[2]) / (x[3]*B*B + x[4]*B + 1) # f(0.004731392457, 0.349786398259, 1.089472565483, 0.000147703358, 0.058625369277) = 0.000697
        # result = (x[0]*B*B*B + x[1]*B*B + x[2]*B + x[3]) / (x[4]*B*B + x[5]*B + 1) # f(0.000023456024, 0.013422289241, 0.452771550862, 0.888166060601, 0.000881837251, 0.105654221816) = 0.000134
        # result = 1 / (x[0]*B + x[1]) + (x[2]*B + x[3]) / (x[4]*B + 1) #
        # result = (x[0]*B + x[1]) / (x[2]*B + 1) + x[3]*A + x[4]*C #
        if not math.isfinite(result):
            return 1e9
        assert math.isfinite(result)
        sum_se += (y - result) ** 2
    rmse = math.sqrt(sum_se / len(f_medium_b_points))
    return rmse
    
    
if __name__ == "__main__":    
    if False:
        best_x, best_y = cmaes(8, f_easy, 0.1)
    if True:
        best_x, best_y = cmaes(3, f_medium_b, 0.0)
        print(17/3, 21/11, 19/7)
    
