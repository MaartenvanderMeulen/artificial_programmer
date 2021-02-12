import time
import os


folder = "tmp/a"
prefix = "cx"
filenames = []
for filename in os.listdir(folder):
    if filename[:len(prefix)] == prefix:
        filenames.append(filename)
filenames.sort()

maxi = 177175
edge = dict()    
for filename in filenames:
    with open(f"{folder}/{filename}", "r") as f:
        for line in f:
            parts = line.strip().lower().split(" ")
            ab, c, n = int(parts[0]), int(parts[1]), int(parts[2])
            if ab <= maxi and c <= maxi:
                if (ab, c) not in edge:
                    edge[(ab, c)] = 0
                edge[(ab, c)] += n

edge_list = [(ab, c, n) for (ab, c), n in edge.items()]
edge_list.sort(key=lambda item: -item[2])
filename = f"tmp/a_cx.txt"
with open(filename, "w") as f:
    for ab, c, n in edge_list:
        if n > 1:
            f.write(f"{ab} {c} {n}\n")
print("merged cx weggeschreven naar", filename)

print("van dst naar src")
dst = 0
find = True
nodes_done = set([dst])
while find:
    find = False
    for ab, c, n in edge_list:
        if c == dst and ab not in nodes_done and ab > c:
            nodes_done.add(c)
            print(ab, c, n)
            dst = ab
            find = True
            break
print()

print("van src naar dst")
src = 177166
find = True
nodes_done = set([src])
while find:
    find = False
    for ab, c, n in edge_list:
        if ab == src and c not in nodes_done and ab > c:
            nodes_done.add(c)
            print(ab, c, n)
            src = c
            find = True
            break

print("alle voorlopers van 0")
voorlopers = []
for ab, c, n in edge_list:
    if c == 0:
        voorlopers.append((ab, c, n))

print("    ", len(voorlopers))
summary = []
for voorloper in voorlopers:
    sum_incoming_weights = sum([n for ab, c, n in edge_list if c == voorloper[0]])
    summary.append((voorloper[0], sum_incoming_weights, voorloper[2]))
summary.sort(key=lambda item: -item[2])
for ab, iws, ow in summary[:40]:
    print(ab, iws, ow)

    


