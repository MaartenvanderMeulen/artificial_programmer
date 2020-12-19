import os
import math


def handle_file(filename, score_to_follow, precision, counts):
    count_enter_subopt, count_miss_subopt, count_stuck_at_subopt, count_leave_subopt = counts
    local_enter_subopt, local_miss_subopt = 0, 0
    with open(filename, "r") as f:
        prev_eval = None
        for line in f:
            if line[:len("start generation")] ==  "start generation":
                # example : start generation 124 best eval 38.75731
                line_stripped = line.strip().lower().split(" ")
                eval = None
                for i, part in enumerate(line_stripped):
                    if part == "eval":
                        eval = float(line_stripped[i+1])
                        break
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
    for filename in os.listdir(folder):
        if filename[:3] == "log":
            count_files += 1
            counts = handle_file(folder + "/" + filename, score_to_follow, precision, counts)
    count_enter_subopt, count_miss_subopt, count_stuck_at_subopt, count_leave_subopt = counts
    print("p enter subopt", count_enter_subopt / (count_enter_subopt + count_miss_subopt), "check", count_enter_subopt + count_miss_subopt)
    print("p stuck at subopt", count_stuck_at_subopt / (count_stuck_at_subopt + count_leave_subopt), "check", count_stuck_at_subopt + count_leave_subopt)
    print("count_files", count_files)


if __name__ == "__main__":
    follow_subopt("tmp/09AC", 77.61253, 0.00001)
