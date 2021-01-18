import ctypes
import platform


class CodeItem (ctypes.Structure):
    _fields_ = [("_type", ctypes.c_int), ("_value", ctypes.c_int), ("_arity", ctypes.c_int)]


def compile_deap(deap_code, symbol_table):
    c_code = (CodeItem * len(deap_code))()
    ITEM_INT = 1
    ITEM_FCALL = 2
    ITEM_VAR = 3
    fcall_index = {'lt':(1,2), 'le':(2,2), 'ge':(3,2), 'gt':(4,2), 'add':(5,2), 'sub':(6,2), 'mul':(7,2), 'div':(8,2)}
    # "compile" deap code into array of c structs
    for i, item in enumerate(deap_code):
        if type(item) == type(""):
            if item in fcall_index:
                c_code[i]._type = ITEM_FCALL
                c_code[i]._value, c_code[i]._arity = fcall_index[item]
            else:
                if item not in symbol_table:
                    raise_exception_when_symbol_not_in_symbol_table = True
                    if raise_exception_when_symbol_not_in_symbol_table:
                        raise RuntimeError(f"Error: symbol {item} not in symbol table")
                    else:
                        symbol_table[item] = len(symbol_table)
                c_code[i]._type = ITEM_VAR
                c_code[i]._value, c_code[i]._arity = symbol_table[item], 0
        else:
            assert type(item) == type(1)
            c_code[i]._type = ITEM_INT
            c_code[i]._value, c_code[i]._arity = item, 0
    return c_code
    

def create_symbol_table(param_names, local_variable_names):
    symbol_table = dict()
    for symbol in param_names + local_variable_names:
        if symbol not in symbol_table:
            symbol_table[symbol] = len(symbol_table)
    return symbol_table


def convert_data_to_prefix_notation(data):
    result = []
    ITEM_INT = 1
    ITEM_LIST = 4
    # "compile" data into array of structs
    if type(data) == type([]):
        result.append(CodeItem(ITEM_LIST, 0, len(data)))
        for item in data:
            if type(item) == type(1):
                result.append(CodeItem(ITEM_INT, item, 0))
            else:
                assert type(item) == type([])
                result.extend(convert_data_to_prefix_notation(item))
    else:
        result.append(CodeItem(ITEM_INT, item, 0))
    return result
    

def convert_data_in_prefix_notation_to_c(data_in_prefix_notation):
    n = len(data_in_prefix_notation)
    c_data = (CodeItem * n)()
    for i in range(n):
        c_data[i] = data_in_prefix_notation[i]
    return c_data
    

def compile_params(params):
    data_in_prefix_notation = convert_data_to_prefix_notation(params)
    c_params = convert_data_in_prefix_notation_to_c(data_in_prefix_notation)
    c_param_sizes = (ctypes.c_int * len(params))()
    for i, param in enumerate(params):
        c_param_sizes[i] = len(convert_data_to_prefix_notation(param))
    return c_param_sizes, c_params


def compile_inputs(inputs):
    result = []
    for params in inputs:
        result.append(compile_params(params))
    return result


def load_cpp_lib():
    lib_name = "cpp_interpret"
    if platform.system().lower().startswith('lin'):
        lib_name += ".so"
    lib = ctypes.cdll.LoadLibrary(lib_name)
    return lib


def create_ouput_buf():
    output_bufsize = 1000
    output_buf = (CodeItem * output_bufsize)()
    return output_buf, output_bufsize


def convert_c_output_to_python_impl(output_buf, n_output, sp):
    ITEM_INT = 1
    ITEM_LIST = 4
    sp = 1
    if n_output == 0:
        result = None
    elif n_output == 1:
        if output_buf[0]._type == ITEM_INT:
            result = int(output_buf[0]._value)
        else:
            result = []
    else:
        assert output_buf[0]._type == ITEM_LIST
        arity = output_buf[0]._arity
        result = []
        for _ in range(arity):
            if output_buf[sp]._type == ITEM_INT:
                result.append(int(output_buf[sp]._value))
            else:
                subtree, sp = convert_c_output_to_python_impl(output_buf, n_output, sp)
                result.append(subtree)
        assert sp == n_output
    return result, sp


def convert_c_output_to_python(output_buf, n_output):
    sp = 0
    result, sp = convert_c_output_to_python_impl(output_buf, n_output, sp)
    return result 


def run_once(lib, c_param_sizes, c_params, n_local_variables, c_code, output_bufsize, output_buf):
    n_output = ctypes.c_int()
    n_output.value = 0
    lib.run_non_recursive_level1_function( \
        ctypes.c_int(len(c_param_sizes)), ctypes.byref(c_param_sizes), ctypes.byref(c_params), \
        ctypes.c_int(n_local_variables), \
        ctypes.byref(c_code), ctypes.c_int(len(c_code)), \
        ctypes.c_int(output_bufsize), ctypes.byref(output_buf), ctypes.byref(n_output))
    return convert_c_output_to_python(output_buf, n_output.value)


# ======================================== interface ================================================


def get_cpp_handle(inputs, param_names, local_variable_names):
    lib = load_cpp_lib()
    c_inputs = compile_inputs(inputs)
    symbol_table = create_symbol_table(param_names, local_variable_names)
    n_local_variables = len(local_variable_names)
    output_bufsize, output_buf = create_ouput_buf()
    cpp_handle = lib, c_inputs, symbol_table, n_local_variables, output_bufsize, output_buf
    return cpp_handle


def run_on_all_inputs(cpp_handle, deap_code):
    result = []
    lib, c_inputs, symbol_table, n_local_variables, output_bufsize, output_buf = cpp_handle
    c_code = compile_deap(deap_code, symbol_table)
    for c_param_sizes, c_params in c_inputs:
        result.append(run_once(lib, c_param_sizes, c_params, n_local_variables, c_code, output_bufsize, output_buf))

