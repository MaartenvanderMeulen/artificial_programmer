import os
import math
import sys


def handle_file(filename, count_non_stuck, count_stuck, count_stopped_early):
    with open(filename, "r") as f:
        prev_eval = None
        count_iters = 0
        for line in f:
            if len(line) >= 3 and line[:3] ==  "gen":
                eval = float(line[12:19])
                if prev_eval is not None:
                    if prev_eval > eval:
                        if count_iters not in count_non_stuck:
                            count_non_stuck[count_iters] = 0
                        count_non_stuck[count_iters] += 1
                        count_iters = 0
                prev_eval = eval
                count_iters += 1
            elif len(line) >= 7 and line[:7] == "stopped":
                if count_iters not in count_stuck:
                    count_stuck[count_iters] = 0
                count_stuck[count_iters] += 1
                if count_iters <= 2:
                    count_stopped_early[0] += 1
                    print(filename, count_iters)
            elif len(line) >= 6 and line[:6] == "solved":
                if count_iters not in count_non_stuck:
                    count_non_stuck[count_iters] = 0
                count_non_stuck[count_iters] += 1


def normalise(count_non_stuck, count_stuck):
    n = sum([value for key, value in count_non_stuck.items()])
    n += sum([value for key, value in count_stuck.items()])
    for key, _ in count_non_stuck.items():
        count_non_stuck[key] /= n
    for key, _ in count_stuck.items():
        count_stuck[key] /= n


def write_to_file(count_dict, file_name):
    count_list =[(key, value) for key, value in count_dict.items()]
    count_list.sort(key=lambda item: item[0])
    with open(file_name, "w") as f:
        for key, value in count_list:
            f.write(f"{key}\t{value}\n")


def stuck_detector(folder):
    count_non_stuck = dict()
    count_stuck = dict()
    count_stopped_early = [0]
    filenames = []
    for filename in os.listdir(folder):
        if filename[:3] == "log":
            filenames.append(filename)
    filenames.sort()
    for filename in filenames:
        handle_file(folder + "/" + filename, count_non_stuck, count_stuck, count_stopped_early)
    
    normalise(count_non_stuck, count_stuck)
    write_to_file(count_non_stuck, "tmp/count_non_stuck.txt")
    write_to_file(count_stuck, "tmp/count_stuck.txt")
    print("count_stopped_early", count_stopped_early)
    


if __name__ == "__main__":
    id = sys.argv[1] if len(sys.argv) >= 2 else "09AC"
    stuck_detector(f"tmp/{id}")
