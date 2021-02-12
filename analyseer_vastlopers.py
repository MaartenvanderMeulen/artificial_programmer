import time
import os

global maxi
maxi = 177175

def read_cx_graph(folder):
    print(f"reading cx_graaf from {folder} ...")
    global maxi
    edge = dict()    
    for filename in os.listdir(folder):
        if filename.startswith("cx"):
            with open(f"{folder}/{filename}", "r") as f:
                for line in f:
                    parts = line.strip().lower().split(" ")
                    ab, c, n = int(parts[0]), int(parts[1]), int(parts[2])
                    if ab <= maxi and c <= maxi:
                        if ab != c:
                            if (ab, c) not in edge:
                                edge[(ab, c)] = 0
                            edge[(ab, c)] += n
    edge_list = [(ab, c, n) for (ab, c), n in edge.items()]
    edge_list.sort(key=lambda item: -item[2])
    return edge_list


def compute_degrees(cx_graph_edge_list, idx):
    in_degree, out_degree_to0 = 0, 0
    for ab, c, n in cx_graph_edge_list:
        if idx == c:
            in_degree += n
        if idx == ab and c == 0:
            out_degree_to0 += n
    return in_degree, out_degree_to0


def compute_in_out_degree(cx_graph_edge_list, vastlopers):
    in_degree, out_degree_to0 = dict(), dict()
    for idx, _ in vastlopers.items():
        in_degree[idx], out_degree_to0[idx] = compute_degrees(cx_graph_edge_list, idx)
    return in_degree, out_degree_to0


def extract_best_fam_from_cx_file(cx_file):
    best_fam = None
    with open(cx_file, "r") as f:
        for line in f:
            parts = line.strip().lower().split(" ")
            ab, c, _ = int(parts[0]), int(parts[1]), int(parts[2])
            if best_fam is None or best_fam > ab:
                best_fam = ab
            if best_fam is None or best_fam > c:
                best_fam = c
    return best_fam


def read_vastlopers(folder):
    print(f"reading vastlopers from {folder} ...")
    global maxi
    vastlopers = dict()
    count = 0
    for filename in os.listdir(folder):
        if filename.startswith("cx"):
            family_index = extract_best_fam_from_cx_file(folder + "/" + filename)
            if family_index <= maxi:
                count += 1
                if family_index not in vastlopers:
                    vastlopers[family_index] = 0
                vastlopers[family_index] += 1
            else:
                print(filename, "skipped")
    return vastlopers


def write_result(filename, vastlopers, in_degree, out_degree_to0):
    print(f"writing anaysis result in {filename} ...")
    vastlopers = [(index, aantal) for index, aantal in vastlopers.items()]
    vastlopers.sort(key=lambda item: -item[0])
    with open(filename, "w") as f:
        f.write(f"fam_id aantal_vastlopers totaal_ingraad totaal_uitgraad_naar_fam0\n")
        sum_count = 0
        for index, count in vastlopers:
            sum_count += count
            f.write(f"{index:6d} {count:17d} {in_degree[index]:14d} {out_degree_to0[index]:14d}\n")
        f.write(f"{' ':6} {sum_count:17}\n")
    print(f"anaysis done")


folder = "tmp/a_cx"
edge_list = read_cx_graph(folder)
vastlopers = read_vastlopers(folder)
in_degree, out_degree_to0 = compute_in_out_degree(edge_list, vastlopers)
write_result(f"{folder}/vastlopers.txt", vastlopers, in_degree, out_degree_to0)
