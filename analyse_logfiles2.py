import sys
import os


class Individual(object):
    def __init__(self, raw_error, generation, parent_id, pop0_error, median_error, code, cx_mut_init):
        self.raw_error = raw_error
        self.generation = generation
        self.parent_id = parent_id
        self.pop0_error = pop0_error
        self.median_error = median_error
        self.code = code
        self.cx_mut_init = cx_mut_init


class Context(object):
    def __init__(self):
        pass
    
    def add_init_to_total_graph(self, id_dict, raw_error, code, id, generation):
        assert id not in id_dict
        id_dict[id] = Individual(raw_error, generation, None, None, None, code, "init")

    def add_cx_to_total_graph(self, id_dict, raw_error, code, id, generation, parent1_id, pop0_error, median_error):
        assert id not in id_dict
        id_dict[id] = Individual(raw_error, generation, parent1_id, pop0_error, median_error, code, "cx")

    def add_mut_to_total_graph(self, id_dict, raw_error, code, id, generation, parent1_id, pop0_error, median_error):
        assert id not in id_dict
        id_dict[id] = Individual(raw_error, generation, parent1_id, pop0_error, median_error, code, "mut")


def is_init_line(line):
    return line.find("= init") >= 0


def is_cx_line(line):
    return line.find("= cx [") >= 0


def is_mut_line(line):
    return line.find("= mut [") >= 0


def is_gen_summary_line(line):
    return line.startswith("gen ")


def parse_init(line1, line2):
    generation = line1.split(" = ")[0].split(" ")[2][:-1]
    id = line1.split(" = ")[0].split(" ")[3][1:-1]
    error = line1.split(" = ")[1].split(" ")[7][:-1]
    code = line2.split(" = ")[1]
    return code, id, error, generation
    

def parse_cx(line1, line2):
    generation = line1.split(" = ")[0].split(" ")[2][:-1]
    id = line1.split(" = ")[0].split(" ")[3][1:-1]
    error = line1.split(" = ")[1].split(" ")[7][:-1]
    parent1_id = line1.split(" = ")[2].split(" ")[1].split("<")[0][1:-1]
    code = line2.split(" = ")[1]
    return code, id, error, generation, parent1_id
    

def parse_mut(line1, line2, line3):
    generation = line1.split(" = ")[0].split(" ")[2][:-1]
    id = line1.split(" = ")[0].split(" ")[3][1:-1]
    error = line1.split(" = ")[1].split(" ")[7][:-1]
    parent1_id = line1.split(" = ")[2].split(" ")[1].split("<")[0][1:-1]
    code = line3.split(" = ")[1]
    return code, id, error, generation, parent1_id


def parse_gen(line1):
    pop0_error = line1.split(" ")[3]
    median_error = line1.split(" ")[5]
    return pop0_error, median_error


def read_path(context, filename, target_error):
    print("reading", filename)
    id_dict = dict()   
    pop0_error, median_error = None, None
    with open(filename, "r") as f:
        for line in f:
            if is_cx_line(line):
                line2 = f.readline()
                code, id, error, generation, parent1_id = parse_cx(line.rstrip(), line2.rstrip())
                context.add_cx_to_total_graph(id_dict, error, code, id, generation, parent1_id, pop0_error, median_error)
                if error == target_error:
                    return id_dict, id
            elif is_mut_line(line):
                line2 = f.readline()
                line3 = f.readline()
                code, id, error, generation, parent1_id = parse_mut(line.rstrip(), line2.rstrip(), line3.rstrip())
                context.add_mut_to_total_graph(id_dict, error, code, id, generation, parent1_id, pop0_error, median_error)
                if error == target_error:
                    return id_dict, id
            elif is_init_line(line):
                line2 = f.readline()
                code, id, error, generation = parse_init(line.rstrip(), line2.rstrip())
                context.add_init_to_total_graph(id_dict, error, code, id, generation)
                if error == target_error:
                    return id_dict, id
            elif is_gen_summary_line(line):
                pop0_error, median_error = parse_gen(line.rstrip())
    return None, None


def write_path(id_dict, id, filename):
    with open(filename, "w") as f:
        while id:            
            ind = id_dict[id]
            code_str = str(ind.code)
            code_str = code_str.replace(" ", "")
            f.write(f"{ind.generation}\t{ind.raw_error}\t{ind.pop0_error}\t{ind.median_error}\t{ind.cx_mut_init}\t{id}\t{code_str}\n")
            id = ind.parent_id


def extract_main_line(folder_with_logfiles, target_error):
    context = Context()
    filenames = []
    for filename in os.listdir(folder_with_logfiles):
        if filename.startswith("log_"):
            filenames.append(filename)
    filenames.sort()
    for filename in filenames:        
        id_dict, target_id = read_path(context, folder_with_logfiles + "/" + filename, target_error)
        seed = filename[4:8]
        output_filename = folder_with_logfiles + f"/nx_{seed}_{target_error.replace('.', '_')}.txt"
        write_path(id_dict, target_id, output_filename)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"usage   : {sys.argv[0]} folder_with_logfiles target_error")
        print(f"example : {sys.argv[0]} tmp/ad 0.000")
        exit(2)
    folder_with_logfiles = sys.argv[1]
    target_error = sys.argv[2]
    extract_main_line(folder_with_logfiles, target_error)
