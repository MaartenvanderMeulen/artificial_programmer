import ctypes
import platform


class CodeItem (ctypes.Structure):
    _fields_ = [("_type", ctypes.c_int), ("_value", ctypes.c_int), ("_arity", ctypes.c_int)]


def run(code, variables):
    lib_name = "cpp_interpret"
    if platform.system().lower().startswith('lin'):
        lib_name += ".so"
    lib = ctypes.cdll.LoadLibrary(lib_name)
    c_code = (CodeItem * len(code))()
    CODEITEM_INT = 1
    CODEITEM_FCALL = 2
    CODEITEM_VAR = 3
    CODEITEM_LIST = 4
    fcall_index = {'lt':(1,2), 'le':(2,2), 'ge':(3,2), 'gt':(4,2), 'add':(5,2), 'sub':(6,2), 'mul':(7,2), 'div':(8,2)}

    # make symbol table for variables
    index_of_symbol = dict()
    value_at_index = (ctypes.c_int * len(variables))()
    for key, value in variables.items():
        index = len(index_of_symbol)
        value_at_index[index] = value
        index_of_symbol[key] = index
        
    # "compile" code into array of int structs
    for i, item in enumerate(code):
        if type(item) == type(""):
            if item in fcall_index:
                c_code[i]._type = CODEITEM_FCALL
                c_code[i]._value, c_code[i]._arity = fcall_index[item]
            else:
                c_code[i]._type = CODEITEM_VAR
                c_code[i]._value, c_code[i]._arity = index_of_symbol[item], 0
        else:
            assert type(item) == type(1)
            c_code[i]._type = CODEITEM_INT
            c_code[i]._value, c_code[i]._arity = item, 0
    max_output_items = 1000
    n_output_items = ctypes.c_int()
    output = (CodeItem * max_output_items)()
    
    # call C++
    lib.run_code(ctypes.byref(c_code), ctypes.c_int(len(code)), ctypes.byref(value_at_index), ctypes.c_int(max_output_items), ctypes.byref(output), ctypes.byref(n_output_items))
    
    # process output
    n_output_items = n_output_items.value
    for i in range(n_output_items):
        print("python: ", i, output[i]._type, output[i]._value, output[i]._arity)
        

if __name__ == "__main__":
    run(['add', 3, 'mul', 'a', 'b'], {'a':4, 'b':5})
