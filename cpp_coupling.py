import ctypes
import platform

from deap import gp #  gp.PrimitiveSet, gp.genHalfAndHalf, gp.PrimitiveTree, gp.genFull, gp.from_string


class CodeItem (ctypes.Structure):
    _fields_ = [("_type", ctypes.c_int), ("_value", ctypes.c_int), ("_arity", ctypes.c_int)]


def compile_deap(deap_code, symbol_table, get_item_value):
    c_code = (CodeItem * len(deap_code))()
    ITEM_INT = 1
    ITEM_FCALL = 2
    ITEM_VAR = 3
    fcall_index = {
        'lt':(1,2), 'le':(2,2), 'ge':(3,2), 'gt':(4,2), 'add':(5,2), 'sub':(6,2), 'mul':(7,2), 'div':(8,2),
        'eq':(9,2), 'ne':(10,2), 'and':(11,2), 'or':(12,2), 'not':(13,1),
        'first':(14,1), 'rest':(15,1), 'extend':(16,2), 'append':(17,2), 'cons':(18,2), 'len':(19,1),
        'at':(20,2), 'at2':(20,2), 'at3':(20,3), 
        'list':(21,2), 'list1':(21,1), 'list2':(21,2), 'list3':(21,3), 
        'last':(22,2), 'last2':(22,2), 'last3':(22,3), 
        'var':(23,3), 'assign':(24,2),
        'function':(25,4), # 4: func_id, n_params, n_locals, code
        'if':(26,2), 'if_then_else':(26,3),
        'for':(27,3),
        'print':(28,1),
        'assert':(29,1),
        'exit':(30,0),
        'sum':(31,1),
    }
    # "compile" deap code into array of c structs
    for i, item in enumerate(deap_code):
        item = get_item_value(item)
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
    if type(data) == type([]) or type(data) == type(()):
        result.append(CodeItem(ITEM_LIST, 0, len(data)))
        for item in data:
            if type(item) == type(1):
                result.append(CodeItem(ITEM_INT, item, 0))
            else:
                assert type(item) == type([]) or type(item) == type(())
                result.extend(convert_data_to_prefix_notation(item))
    else:
        result.append(CodeItem(ITEM_INT, data, 0))
    return result
    

def convert_data_in_prefix_notation_to_c(data_in_prefix_notation):
    n = len(data_in_prefix_notation)
    c_data = (CodeItem * n)()
    for i in range(n):
        c_data[i] = data_in_prefix_notation[i]
    return c_data
    

def compile_params(params):
    params_data_in_prefix_notation = []
    c_param_sizes = (ctypes.c_int * len(params))()
    for i, param in enumerate(params):
        param_data_in_prefix_notation = convert_data_to_prefix_notation(param)
        c_param_sizes[i] = len(param_data_in_prefix_notation)
        params_data_in_prefix_notation.extend(param_data_in_prefix_notation)
    c_params = convert_data_in_prefix_notation_to_c(params_data_in_prefix_notation)
    return c_param_sizes, c_params


def compile_inputs(inputs):
    result = []
    for params in inputs:
        result.append(compile_params(params))
    return result


def load_cpp_lib():
    lib_name = "cpp_interpret"
    if platform.system().lower().startswith('lin'):
        lib_name = "./" + lib_name + ".so"
    lib = ctypes.cdll.LoadLibrary(lib_name)
    return lib


def create_ouput_buf():
    output_bufsize = 1000
    output_buf = (CodeItem * output_bufsize)()
    return output_buf, output_bufsize


def convert_c_output_to_python_impl(output_buf, n_output, sp):
    if sp >= n_output:
        print("DEBUG : sp >= n_output at line 123")
        return None, sp
    ITEM_INT = 1
    ITEM_LIST = 4
    if output_buf[sp]._type == ITEM_INT:
        result = int(output_buf[sp]._value)
        sp += 1
    else:
        assert output_buf[sp]._type == ITEM_LIST
        arity = output_buf[sp]._arity
        result = []
        sp += 1
        for _ in range(arity):
            subtree, sp = convert_c_output_to_python_impl(output_buf, n_output, sp)
            result.append(subtree)
    return result, sp


def convert_c_output_to_python(output_buf, n_output):
    if n_output == 0:
        result = 0
    else:
        sp = 0
        result, sp = convert_c_output_to_python_impl(output_buf, n_output, sp)
        assert sp == n_output
    #print("convert_c_output_to_python result", result)
    return result 


def call_cpp_interpreter(lib, c_n_params, c_param_sizes, c_params, n_local_variables, c_code, output_bufsize, output_buf, n_output, debug):
    '''In a separate python function to get exact timings on the C++ part via cProfile'''
    lib.run_non_recursive_level1_function( \
        c_n_params, ctypes.byref(c_param_sizes), ctypes.byref(c_params), \
        ctypes.c_int(n_local_variables), \
        ctypes.byref(c_code), ctypes.c_int(len(c_code)), \
        ctypes.c_int(output_bufsize), ctypes.byref(output_buf), ctypes.byref(n_output), ctypes.c_int(debug))


def run_once(lib, c_param_sizes, c_params, n_local_variables, c_code, output_bufsize, output_buf, debug):
    c_n_params = ctypes.c_int(len(c_param_sizes))
    n_output = ctypes.c_int()
    n_output.value = 0
    call_cpp_interpreter(lib, c_n_params, c_param_sizes, c_params, n_local_variables, c_code, output_bufsize, output_buf, n_output, debug)
    return convert_c_output_to_python(output_buf, n_output.value)


# ======================================== interface ================================================


def get_cpp_handle(inputs, param_names, local_variable_names):
    lib = load_cpp_lib()
    c_inputs = compile_inputs(inputs)
    symbol_table = create_symbol_table(param_names, local_variable_names)
    n_local_variables = len(local_variable_names)
    output_buf, output_bufsize = create_ouput_buf()
    cpp_handle = lib, c_inputs, symbol_table, n_local_variables, output_bufsize, output_buf
    return cpp_handle


def run_on_all_inputs(cpp_handle, deap_code, get_item_value=None, debug=0):
    result = []
    lib, c_inputs, symbol_table, n_local_variables, output_bufsize, output_buf = cpp_handle
    if get_item_value is None:
        get_item_value = lambda x : x.name if isinstance(x, gp.Primitive) else x.value
    c_code = compile_deap(deap_code, symbol_table, get_item_value)
    for c_param_sizes, c_params in c_inputs:
        result.append(run_once(lib, c_param_sizes, c_params, n_local_variables, c_code, output_bufsize, output_buf, debug))
    return result


# ======================================== test ================================================


if __name__ == "__main__":
    print("Testing input & outputs conversion")
    inputs = (
            (84, (), ),
            (85, (86, ), ),
            (87, (86, 89, ), ),
        )
    c_inputs = compile_inputs(inputs)
    reconverted_inputs = []
    for c_param_sizes, c_params in c_inputs:
        reconverted_input = []
        n = len(c_param_sizes)
        param_sizes = [c_param_sizes[i] for i in range(n)]
        offset = 0
        params = []
        for i in range(n):
            c_param = (CodeItem * param_sizes[i])()
            for j in range(param_sizes[i]):
                c_param[j] = c_params[offset + j]
            param = [(c_param[j]._type, c_param[j]._value, c_param[j]._arity) for j in range(param_sizes[i])]
            params.append(param)
            reconverted_input.append(convert_c_output_to_python(c_param, param_sizes[i]))
            offset += param_sizes[i]
        assert offset == len(c_params)
        print(i, param_sizes, params)
        reconverted_inputs.append(reconverted_input)
    print(reconverted_inputs)

    print("Testing deap compilation")
    deap_code = [
        "append",
            "for", "i", "sorted_data",
                "if_then_else",
                    "le", "elem", "i",
                    "last3",
                        "assign", "k", "elem",
                        "assign", "elem", "i",
                        "k",
                    "i",
            "elem",
    ]
    print("deap_code", deap_code)
    param_names, local_variable_names = ["elem", "sorted_data"], ["i", "k"]
    symbol_table = create_symbol_table(param_names, local_variable_names)
    get_item_value = (lambda x : x)
    c_code = compile_deap(deap_code, symbol_table, get_item_value=get_item_value)

    print("Testing get_cpp_handle")
    cpp_handle = get_cpp_handle(inputs, param_names, local_variable_names)

    print("Testing run_on_all_inputs")
    outputs = run_on_all_inputs(cpp_handle, deap_code, get_item_value=get_item_value)
    print("outputs", outputs)
    assert outputs == [[84], [85, 86], [86, 87, 89]]

    print("Integration test OK")

