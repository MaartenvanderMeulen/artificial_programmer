import ctypes
import platform
import numpy as np


def run(code, var_index):
    lib_name = "cpp_interpret"
    if platform.system().lower().startswith('lin'):
        lib_name += ".so"
    lib = ctypes.cdll.LoadLibrary(lib_name)
    f = lib.run_code
    _types = np.zeros((len(code)), dtype='int')
    _values = np.zeros((len(code)), dtype='int')
    _arities = np.zeros((len(code)), dtype='int')
    CODEITEM_INT = 1
    CODEITEM_FCALL = 2
    CODEITEM_VAR = 3
    fcall_index = {'lt':(1,2), 'le':(2,2), 'ge':(3,2), 'gt':(4,2), 'add':(5,2), 'sub':(6,2), 'mul':(7,2), 'div':(8,2)}
    var_index = dict()
    for i, item in enumerate(code):
        if type(item) == type(""):
            if item in fcall_index:
                _types[i] = CODEITEM_FCALL
                _values[i], _arities[i] = fcall_index[item]
            else:
                if item not in var_index:
                    var_index[item] = len(var_index)
                _types[i] = CODEITEM_VAR
                _values[i], _arities[i] = var_index[item], 0
        else:
            assert type(item) == type(1)
            _types[i] = CODEITEM_INT
            _values[i], _arities[i] = item, 0
    f(ctypes.c_int(len(code)), ctypes.c_void_p(_types.ctypes.data),
        ctypes.c_void_p(_values.ctypes.data), ctypes.c_void_p(_arities.ctypes.data))


if __name__ == "__main__":
    run(['add', 3, 17])


