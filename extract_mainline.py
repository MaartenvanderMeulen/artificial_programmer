import os
import sys


def write_graph(graph, filename):
    print("write result in", filename)
    with open(filename, "w") as f:
        items = []
        for key, parents_dict in graph.items():
            inflow = sum([flow for parent, flow in parents_dict.items()])
            items.append((key, inflow))
        items.sort(key=lambda item: float(item[0]))
        for key, inflow in items:
            parents_dict = graph[key]
            f.write(f"{key}, inflow {inflow}\n")
            subitems = []
            for parent, flow in parents_dict.items():
                subitems.append((parent, flow))
            subitems.sort(key=lambda item: -item[1])
            for parent, flow in subitems:
                f.write(f"    {parent} {flow}\n")


def parse_st(st, graph):
    for s, t in st:
        if t not in graph:
            graph[t] = dict()
        if s not in graph:
            graph[s] = dict()
        if s not in graph[t]:
            graph[t][s] = [0, 0]
        graph[t][s][0] += 1


def read_tree(filename, graph, start_key, stop_key):
    st = []
    with open(filename, "r") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            assert len(parts) == 2
            s, t = parts[0], parts[1]
            assert float(s) > float(t)
            if float(s) > float(start_key):
                continue
            if float(t) < float(stop_key):
                continue
            st.append((s, t))
            if t == stop_key:
                parse_st(st, graph)
                return True
    return False


global best_path, best_flow
best_path, best_flow = [], 0


def compute_incoming_free_capacity(graph, t_key):
    t = graph[t_key]
    incoming_free_capacity = sum([capacity[0] - capacity[1] for parent, capacity in t.items()])
    return incoming_free_capacity


def get_path_with_largest_flow(graph, s_key, t_key):
    global best_path, best_flow
    best_path, best_flow = [], 0
    depth_first_search(graph, s_key, t_key, [[t_key, 999999, ], ], )


def allocate_capacity(total_graph):
    global best_path, best_flow
    if best_flow > 0:
        target_key = best_path[0][0]
        for parent_key, _flow_to_parent in best_path[1:]:
            if target_key not in total_graph:
                total_graph[target_key] = dict()
            if parent_key not in total_graph:
                total_graph[parent_key] = dict()
            if parent_key not in total_graph[target_key]:
                total_graph[target_key][parent_key] = 0
            total_graph[target_key][parent_key] += best_flow
            target_key = parent_key


def depth_first_search(graph, s_key, t_key, path):
    global best_path, best_flow
    parents_ordered_on_free_capacity = [(key, capacity[0] - capacity[1], capacity[1]) for key, capacity in graph[t_key].items()]
    parents_ordered_on_free_capacity.sort(key=lambda item: -item[1])
    for key, free_capacity, _use in parents_ordered_on_free_capacity:
        if float(key) <= float(s_key):
            if free_capacity > best_flow:
                if key == s_key:
                    best_path = path + [[s_key, free_capacity, ], ]
                    best_flow = min([flow for key, flow in best_path])
                else:
                    depth_first_search(graph, s_key, key, path + [[key, free_capacity, ], ])
                    if len(path) > 0 and min([flow for key, flow in path]) <= best_flow:
                        return
            else:
                return # return because they are sorted, the rest has free_capacity <= this free_capacity


def read_and_combine_trees(folder_with_nx_files, s_key, t_key, prefix):
    filenames = []
    for filename in os.listdir(folder_with_nx_files):
        if filename.startswith(prefix):
            filenames.append(filename)
    filenames.sort()
    total_graph = dict()
    for filename in filenames:        
        graph = dict()
        if read_tree(folder_with_nx_files + "/" + filename, graph, s_key, t_key):
            get_path_with_largest_flow(graph, s_key, t_key)
            print(filename, best_flow, best_path)
            del graph
            allocate_capacity(total_graph)
    assert s_key in total_graph
    assert t_key in total_graph
    return total_graph


def main(folder_with_nx_files, s_key, t_key, prefix):
    total_graph = read_and_combine_trees(folder_with_nx_files, s_key, t_key, prefix)
    write_graph(total_graph, folder_with_nx_files + f"/graph_{s_key}_{t_key}.txt")


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print(f"usage   : {sys.argv[0]} folder_with_nx_files s_key t_key file_prefix")
        print(f"example : {sys.argv[0]} tmp/ad 72.893 0.000 nx_")
        exit(2)
    folder_with_nx_files = sys.argv[1]
    s_key, t_key, prefix = sys.argv[2], sys.argv[3], sys.argv[4]
    main(folder_with_nx_files, s_key, t_key, prefix)
