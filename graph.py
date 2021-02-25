import sys


class Graph(object):
    def __init__(self, ):
        self.clear()

    def clear(self):
        self._in_edges = dict()

    def add_cx(self, parent1_id, parent2_id, child_id, generation):
        assert type(parent1_id) == type("")
        assert type(parent2_id) == type("")
        assert type(child_id) == type("")
        assert type(generation) == type(1)
        if child_id != parent1_id and child_id != parent2_id:
            if False:
                cx_id = (parent1_id, parent2_id)
                self._add_edge(parent1_id, cx_id, generation)
                self._add_edge(parent2_id, cx_id, generation)
                self._add_edge(cx_id, child_id, generation)
            else:
                self._add_edge(parent1_id, child_id, generation)
                self._add_edge(parent2_id, child_id, generation)

    def add_mut(self, parent1_id, mut_str, child_id, generation):
        assert type(parent1_id) == type("")
        assert type(mut_str) == type("")
        assert type(child_id) == type("")
        assert type(generation) == type(1)
        if child_id != parent1_id:
            if False:
                mut_str = mut_str.replace(" ", "")
                mut_id = (parent1_id, mut_str)
                self._add_edge(parent1_id, mut_id, generation)
                self._add_edge(mut_str, mut_id, generation)
                self._add_edge(mut_id, child_id, generation)
            else:
                self._add_edge(parent1_id, child_id, generation)

    def write_tree_to_dst(self, f, dst, label, generation):
        self._dst_closed = set()
        self._f = f
        self._label = label
        dst_str = str(dst).replace("'", "").replace(" ", "")
        self._f.write(f"{dst_str} {generation} # {self._label}\n")
        self._tree_to_dst_impl(dst, generation, 1)

    def _add_edge(self, src, dst, generation):
        if dst not in self._in_edges:
            self._in_edges[dst] = dict()
        if src not in self._in_edges[dst]:
            self._in_edges[dst][src] = generation
        else:
            self._in_edges[dst][src] = min(self._in_edges[dst][src], generation)

    def _tree_to_dst_impl(self, dst, generation, depth):
        result = []
        if dst in self._in_edges and dst not in self._dst_closed:
            self._dst_closed.add(dst)
            for src, edge_generation in self._in_edges[dst].items():
                if edge_generation <= generation:
                    indent = "    " * depth
                    src_str = str(src).replace("'", "").replace(" ", "")
                    self._f.write(f"{indent}{src_str} {edge_generation} # {self._label}\n")
                if edge_generation <= generation:
                    self._tree_to_dst_impl(src, edge_generation-1, depth+1)
        return result


def self_test():
    g = Graph()
    g.add_mut("<239>", "elem", "<144>", 277)
    g.add_cx("<144>", "<120>", "<12>", 279)
    g.add_cx("<12>", "<42>", "<0>", 280)
    g.write_tree_to_dst(sys.stdout, "<0>", "escape1")


if __name__ == "__main__":
    self_test()
