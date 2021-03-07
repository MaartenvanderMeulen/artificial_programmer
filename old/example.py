import ctypes
import platform


class CodeItem (ctypes.Structure):
    _fields_ = [("_type", ctypes.c_int), ("_value", ctypes.c_int), ("_arity", ctypes.c_int)]


def mycompile(code, symbol_table, c_code, offset):
    ITEM_INT = 1
    ITEM_FCALL = 2
    ITEM_VAR = 3
    ITEM_LIST = 4
    fcall_index = {'lt':(1,2), 'le':(2,2), 'ge':(3,2), 'gt':(4,2), 'add':(5,2), 'sub':(6,2), 'mul':(7,2), 'div':(8,2)}
    # "compile" code into array of structs
    for i, item in enumerate(code):
        if type(item) == type(""):
            if item in fcall_index:
                c_code[offset+i]._type = ITEM_FCALL
                c_code[offset+i]._value, c_code[offset+i]._arity = fcall_index[item]
            else:
                if item not in symbol_table:
                    symbol_table[item] = len(symbol_table)
                c_code[offset+i]._type = ITEM_VAR
                c_code[offset+i]._value, c_code[offset+i]._arity = symbol_table[item], 0
        else:
            assert type(item) == type(1)
            c_code[offset+i]._type = ITEM_INT
            c_code[offset+i]._value, c_code[offset+i]._arity = item, 0
    

def run(code, symbol_table, params):
    lib_name = "cpp_interpret"
    if platform.system().lower().startswith('lin'):
        lib_name += ".so"
    lib = ctypes.cdll.LoadLibrary(lib_name)
    
    assert len(symbol_table) == len(params)
    
    n_param_items = sum([len(param) if type(param) == type([]) else 1 for param in params])
    c_params = (CodeItem * n_param_items)()
    offset = 0
    param_sizes = (ctypes.c_int * len(params))()
    for i, param in enumerate(params):
        mycompile(param, symbol_table, c_params, offset)
        param_sizes[i] = len(param)
        print("python: param", i, " ", "=", param)
        for j in range(len(param)):
            print("    ", c_params[offset+j]._type, c_params[offset+j]._value, c_params[offset+j]._arity)
        offset += len(param)
    print("python: params")
    for j in range(n_param_items):
        print("    ", c_params[j]._type, c_params[j]._value, c_params[j]._arity)
    
    
    print("python: code", code, "len(code)", len(code))
    c_code = (CodeItem * len(code))()
    mycompile(code, symbol_table, c_code, 0)
        
    output_bufsize = 1000
    output_buf = (CodeItem * output_bufsize)()
    n_output = ctypes.c_int()
    n_output.value = 0
    n_local_variables = 0
    
    # call C++
    # int run_non_recursive_level1_function(
    #    int n_params, int* param_sizes, Item* params, // actual params, param[i] = params[sum(param_sizes[:i]):sum(param_sizes[:i+1])
    #    int n_local_variables,
    #    Item* function_body, int function_body_size, // 
    #    int output_bufsize, Item* output_buf, int* n_output)

    lib.run_non_recursive_level1_function( \
        ctypes.c_int(len(params)), ctypes.byref(param_sizes), ctypes.byref(c_params), \
        ctypes.c_int(n_local_variables), \
        ctypes.byref(c_code), ctypes.c_int(len(c_code)), \
        ctypes.c_int(output_bufsize), ctypes.byref(output_buf), ctypes.byref(n_output))
    
    # process output
    n_output = n_output.value
    for i in range(n_output):
        print("python: ", i, output_buf[i]._type, output_buf[i]._value, output_buf[i]._arity)
        

if __name__ == "__main__":
    run(['add', 3, 'mul', 'b', 'a'], {'a':0, 'b':1}, [[3], [4]])
