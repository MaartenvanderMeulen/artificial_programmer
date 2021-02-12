import networkx as nx

print("inlezen...")
G = nx.read_weighted_edgelist("tmp/a_cx.txt", nodetype=int, create_using=nx.DiGraph)
#for n, nbrs in G.adj.items():
#    for nbr, eattr in nbrs.items():
#        wt = eattr['weight']
#        print(n, nbr, wt)
s, t = 177166, 0
if False:
    print("max flow berekenen...")
    flow_value, flow_dict = nx.maximum_flow(G, s, t, capacity="weight")
    print("flow_value", flow_value)
    for ab, childs_dict in flow_dict.items():
        for c, n in childs_dict.items():
            if n > 30:
                print(ab, c, round(n))
    print("flow_value", flow_value)

# ga langs alle neighbours van '0',
# geef daarvan nodenr, som inkomende gewichten, per uitgaande link: gewicht en target node

