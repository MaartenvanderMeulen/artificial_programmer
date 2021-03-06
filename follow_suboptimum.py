import os
import math
import sys


def get_final_subopt_valuez(filename, score_to_follow, precision, counts):
    count_enter_subopt, count_miss_subopt, count_stuck_at_subopt, count_leave_subopt, sum_delta, sum_evals, sum_error_min, count_error_min = counts
    local_enter_subopt, local_miss_subopt, local_error_max, local_error_min, local_evals = 0, 0, 0, 1e9, 0
    with open(filename, "r") as f:
        prev_eval = None
        for line in f:
            # line = line.strip().lower().split(" ")
            if len(line) >= 3 and line[:3] ==  "gen":
                # example : gen 124 best 38.75731
                eval = float(line[12:19])
                if local_error_max < eval:
                    local_error_max = eval
                if local_error_min > eval:
                    local_error_min = eval
                if prev_eval is not None:
                    if prev_eval > score_to_follow + precision:
                        if eval > score_to_follow + precision:
                            pass
                        elif eval < score_to_follow - precision:
                            local_miss_subopt += 1
                        else:
                            local_enter_subopt += 1
                    elif prev_eval < score_to_follow - precision:
                        pass
                    else:
                        if eval < score_to_follow - precision:
                            count_leave_subopt += 1
                        else:
                            count_stuck_at_subopt += 1
                prev_eval = eval
            elif len(line) >= 7 and line[:7] == "stopped":
                line = line.strip().lower().split("\t")
                local_evals = int(line[4])
            elif len(line) >= 6 and line[:6] == "solved":
                line = line.strip().lower().split("\t")
                local_evals = int(line[6])
    #if local_enter_subopt + local_miss_subopt != 1:
    #    print("file", filename, "enter+miss", local_enter_subopt, "+", local_miss_subopt)
    if local_error_max > local_error_min and local_evals > 0:
        sum_delta += local_error_max - local_error_min
        sum_evals += local_evals
    if local_error_min < 1e9:
        sum_error_min += local_error_min
        count_error_min += 1
    return count_enter_subopt+local_enter_subopt, count_miss_subopt+local_miss_subopt, count_stuck_at_subopt, count_leave_subopt, sum_delta, sum_evals, sum_error_min, count_error_min



def handle_file(filename, score_to_follow, precision, counts):
    count_enter_subopt, count_miss_subopt, count_stuck_at_subopt, count_leave_subopt, sum_delta, \
        sum_evals, sum_error_min, count_error_min, count_gen, count_already_escaped = counts
    local_error_max, local_error_min, local_evals = 0, 1e9, 0
    local_enter_subopt, local_miss_subopt, local_stuck_at_subopt, local_leave_subopt = 0, 0, 0, 0
    local_high, local_gen, local_already_escaped = 0, 0, 0
    with open(filename, "r") as f:
        prev_eval = None
        for line in f:
            # line = line.strip().lower().split(" ")
            if len(line) >= 3 and line[:3] ==  "gen":
                local_gen += 1
                # example : gen 124 best 38.75731
                eval = float(line[12:19])
                if local_error_max < eval:
                    local_error_max = eval
                if local_error_min > eval:
                    local_error_min = eval
                if prev_eval is not None:
                    if prev_eval > score_to_follow + precision:
                        if eval > score_to_follow + precision:
                            local_high += 1
                        elif eval < score_to_follow - precision:
                            local_miss_subopt += 1
                        else:
                            local_enter_subopt += 1
                    elif prev_eval < score_to_follow - precision:
                        if eval > score_to_follow + precision:
                            local_high += 1
                        elif eval < score_to_follow - precision:
                            local_already_escaped += 1
                        else:
                            local_stuck_at_subopt += 1
                            if local_leave_subopt > 0:
                                local_leave_subopt -= 1
                                local_stuck_at_subopt += 1
                    else:
                        if eval < score_to_follow - precision:
                            local_leave_subopt += 1
                        else:
                            local_stuck_at_subopt += 1
                else:
                    if eval > score_to_follow + precision:
                        local_high += 1
                    elif eval < score_to_follow - precision:
                        local_miss_subopt += 1
                    else:
                        local_enter_subopt += 1

                prev_eval = eval
            elif len(line) >= 7 and line[:7] == "stopped":
                line = line.strip().lower().split("\t")
                local_evals = int(line[4])
            elif len(line) >= 6 and line[:6] == "solved":
                line = line.strip().lower().split("\t")
                local_evals = int(line[6])
    if local_leave_subopt > 1 or local_enter_subopt > 1:
        print(filename, local_high + local_miss_subopt + local_enter_subopt + local_already_escaped + local_leave_subopt + local_stuck_at_subopt, local_gen)
        print(local_high, local_enter_subopt, local_stuck_at_subopt, local_leave_subopt, local_already_escaped, local_miss_subopt)
    assert local_high + local_miss_subopt + local_enter_subopt + local_already_escaped + local_leave_subopt + local_stuck_at_subopt == local_gen
    #if local_enter_subopt + local_miss_subopt != 1:
    #    print("file", filename, "enter+miss", local_enter_subopt, "+", local_miss_subopt)
    if local_error_max > local_error_min and local_evals > 0:
        sum_delta += local_error_max - local_error_min
        sum_evals += local_evals
    if local_error_min < 1e9:
        sum_error_min += local_error_min
        count_error_min += 1
    return count_enter_subopt + local_enter_subopt, \
        count_miss_subopt + local_miss_subopt, \
        count_stuck_at_subopt + local_stuck_at_subopt, \
        count_leave_subopt + local_leave_subopt, \
        sum_delta, sum_evals, sum_error_min, count_error_min, \
        count_gen+local_gen, \
        count_already_escaped + local_already_escaped


def follow_subopt(folder, score_to_follow, precision):
    print("score_to_follow", score_to_follow)
    count_enter_subopt, count_miss_subopt, count_stuck_at_subopt, count_leave_subopt, sum_delta, sum_evals, sum_error_min, \
        count_error_min, count_gen, count_already_escaped = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    counts = count_enter_subopt, count_miss_subopt, count_stuck_at_subopt, count_leave_subopt, sum_delta, sum_evals, \
        sum_error_min, count_error_min, count_gen, count_already_escaped
    count_files = 0
    filenames = []
    for filename in os.listdir(folder):
        if filename[:3] == "log":
            filenames.append(filename)
    filenames.sort()
    for filename in filenames:
        count_files += 1
        counts = handle_file(folder + "/" + filename, score_to_follow, precision, counts)
    count_enter_subopt, count_miss_subopt, count_stuck_at_subopt, count_leave_subopt, sum_delta, sum_evals, \
        sum_error_min, count_error_min, count_gen, count_already_escaped = counts
    print("count enter subopt", count_enter_subopt, "count through subopt", count_miss_subopt)
    print("count stuck at subopt", count_stuck_at_subopt, "count leave subopt", count_leave_subopt, "count escaped subopt", count_already_escaped)
    #print("p enter subopt", count_enter_subopt / (count_enter_subopt + count_miss_subopt))
    print("p stuck at subopt", count_stuck_at_subopt / (count_stuck_at_subopt + count_leave_subopt))
    print("count_files", count_files)
    print("count_gen", count_gen)
    #print("efficiency", sum_delta / sum_evals)
    #print("avg min error", sum_error_min / count_error_min)


if __name__ == "__main__":
    id = sys.argv[1] if len(sys.argv) >= 2 else "acsh"
    follow_subopt(f"tmp/{id}", 18.275, 0.001)
