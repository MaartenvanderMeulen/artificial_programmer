import os
import math


def handle_file(filename, score_to_follow):
    iter_before, iter_at, iter_after, solved = 0, 0, 0, False
    with open(filename, "r") as f:
        for line in f:
            if line[:len("solved")] ==  "solved":
                solved = True
            if line[:len("start generation")] ==  "start generation":
                # example : start generation 124 best eval 38.75731
                line_stripped = line.strip().lower().split(" ")
                eval = None
                for i, part in enumerate(line_stripped):
                    if part == "eval":
                        eval = float(line_stripped[i+1])
                        break
                if math.isclose(eval, score_to_follow):
                    iter_at += 1
                else:
                    if iter_at:
                        iter_after += 1
                    else:
                        iter_before += 1
    return iter_before, iter_at, iter_after, solved


def follow_subopt(folder, score_to_follow):
    for filename in os.listdir(folder):
        if filename[:3] == "log":
            iter_before, iter_at, iter_after, solved = handle_file(folder + "/" + filename, score_to_follow)
            if solved:
                print(f"{iter_before}\t{iter_at}\t{iter_after}")


if __name__ == "__main__":
    follow_subopt("tmp/09AA", 77.61253)
