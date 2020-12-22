import os
import math


def handle_file(filename, score_to_follow, precision, counts):
    count_enter_subopt, count_miss_subopt, count_stuck_at_subopt, count_leave_subopt = counts
    local_enter_subopt, local_miss_subopt = 0, 0
    with open(filename, "r") as f:
        prev_eval = None
        for line in f:
            # line = line.strip().lower().split(" ")

            if len(line) > 0 and line[:3] ==  "gen":
                # example : gen 124 best 38.75731
                eval = float(line[12:19])
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
    if local_enter_subopt + local_miss_subopt != 1:
        print("file", filename, "enter+miss", local_enter_subopt, "+", local_miss_subopt)
    return count_enter_subopt+local_enter_subopt, count_miss_subopt+local_miss_subopt, count_stuck_at_subopt, count_leave_subopt


def follow_subopt(folder, score_to_follow, precision):
    count_enter_subopt, count_miss_subopt, count_stuck_at_subopt, count_leave_subopt = 0, 0, 0, 0
    counts = count_enter_subopt, count_miss_subopt, count_stuck_at_subopt, count_leave_subopt
    count_files = 0
    filenames = []
    for filename in os.listdir(folder):
        if filename[:3] == "log":
            filenames.append(filename)
    filenames.sort()
    for filename in filenames:
        count_files += 1
        counts = handle_file(folder + "/" + filename, score_to_follow, precision, counts)
    count_enter_subopt, count_miss_subopt, count_stuck_at_subopt, count_leave_subopt = counts
    print("count enter subopt", count_enter_subopt, "count through subopt", count_miss_subopt)
    print("p enter subopt", count_enter_subopt / (count_enter_subopt + count_miss_subopt))
    print("p stuck at subopt", count_stuck_at_subopt / (count_stuck_at_subopt + count_leave_subopt))
    print("count stuck at subopt", count_stuck_at_subopt, "count leave subopt", count_leave_subopt)
    print("count_files", count_files)


if __name__ == "__main__":
    follow_subopt("tmp/09ACB", 77.61253, 0.001)