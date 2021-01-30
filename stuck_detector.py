import os
import math
import sys


global max_iters_same_family_but_still_solved, count_iters_saved
max_iters_same_family_but_still_solved, count_iters_saved = 0, 0


def handle_file(filename):
    global max_iters_same_family_but_still_solved, count_iters_saved
    print(filename)
    with open(filename, "r") as f:
        print("14", filename)
        prev_family = None
        count_iters_this_family = 0
        max_iters_same_family = 0
        count_iters_since_max = 0
        for line in f:
            if len(line) >= 3 and line[:3] ==  "gen":
                items = line.split(" ")
                assert items[2] == "family_index"
                family = int(items[3])
                if prev_family is not None:
                    if prev_family != family:
                        count_iters_this_family = 0
                prev_family = family
                count_iters_this_family += 1
                if max_iters_same_family < count_iters_this_family:
                    max_iters_same_family = count_iters_this_family
                if max_iters_same_family_but_still_solved < count_iters_this_family or count_iters_since_max > 0:
                    count_iters_since_max += 1
            elif len(line) >= 6 and line[:6] == "solved":
                if max_iters_same_family_but_still_solved < max_iters_same_family:
                    max_iters_same_family_but_still_solved = max_iters_same_family
            elif len(line) >= 6 and line[:6] == "stoppe":
                count_iters_saved += count_iters_since_max


def stuck_detector(folder):
    global max_iters_same_family_but_still_solved, count_iters_saved
    filenames = []
    print("debug 38", folder)
    for filename in os.listdir(folder):
        if filename[:3] == "log":
            filenames.append(filename)
    filenames.sort()
    for filename in filenames:
        handle_file(folder + "/" + filename)
    count_iters_saved = 0
    for filename in filenames:
        handle_file(folder + "/" + filename)
    print("max iterations stuck but still solved", max_iters_same_family_but_still_solved)
    print("count iterations saved", count_iters_saved)
    

if __name__ == "__main__":
    id = sys.argv[1] if len(sys.argv) >= 2 else "aa"
    stuck_detector("tmp/" + id)
