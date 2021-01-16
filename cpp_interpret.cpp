// compile on Windows with:
// call "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build\vcvars64.bat"
// Cl /nologo /EHsc /F 8000000 /D NDEBUG /O2 /Ob1 /MD /LD cpp_interpret.cpp
// Uitleg : /LD create DLL; /MD use multithreaded runtime for DLL's
// compile on Linux with:
// g++ -shared -fPIC -O2 -o upstream_area.so upstream_area.cpp

#include <vector>
#include <string>
#include <map>
#include <exception>
#include <string.h> // strncmp
#include <cassert>

using namespace std;


// =================================== constants

static const int ITEM_INT = 1;
static const int ITEM_FCALL = 2;
static const int ITEM_VAR = 3;
static const int ITEM_LIST = 4; // {ITEM_LIST, *, arity} == {ITEM_FCALL, F_LIST, arity}

static const int F_LT = 1;
static const int F_LE = 2;
static const int F_GE = 3;
static const int F_GT = 4;
static const int F_ADD = 5;
static const int F_SUB = 6;
static const int F_MUL = 7;
static const int F_DIV = 8;
static const int F_EQ = 9;
static const int F_NE = 10;
static const int F_AND = 11;
static const int F_OR = 12;
static const int F_NOT = 13;
static const int F_FIRST = 14;
static const int F_REST = 15;
static const int F_EXTEND = 16;
static const int F_APPEND = 17;
static const int F_CONS = 18;
static const int F_LEN = 19;
static const int F_AT = 20;
static const int F_LIST = 21;
static const int F_LAST = 22;
static const int F_VAR = 23;
static const int F_ASSIGN = 24;
static const int F_FUNCTION = 25;
static const int F_IF = 26;
static const int F_FOR = 27;
static const int F_PRINT = 28;
static const int F_ASSERT = 29;
static const int F_EXIT = 30;
static const int F_SUM = 31;



// =================================== types


class SaveFp {
public:
    SaveFp() {}
    SaveFp(FILE* fp) : _fp(fp) {}
    ~SaveFp() {
        if (_manage && _fp) {
            fclose(_fp);
            _fp = 0;
        }
    }
    FILE* _fp = 0;
    bool _manage = true;
};


struct Item {
public:
    int _type;
    int _value;
    int _arity;
    bool operator==(const Item& that) const {
        return this->_type == that._type && this->_value == that._value && this->_arity == that._arity;
    }
    bool operator!=(const Item& that) const {
        return this->_type != that._type || this->_value != that._value || this->_arity != that._arity;
    }
};
typedef vector<Item> List;


struct Function {
public:
    int _param_count;
    int _local_variable_count;
    List _function_body;
};


// =========================================== globals

static int g_count_runs_calls = 0;


// =========================================== Bsic List handling functions


int len(const List& aa) {    
    return (aa.size() > 0 ? 1 + aa[0]._arity : 0);
}


List first(const List& aa) {
    List result;
    if (aa.size() > 1 && aa[0]._arity > 0) {
        int n = 1;
        int i = 1;
        while (n > 0) {
            result.push_back(aa[i]);
            n += aa[i]._arity;
            i++;
            n--;
        }
    }
    return result;
}



List rest(const List& aa) {
    List result;
    if (aa.size() > 1 && aa[0]._arity > 0) {
        // skip first
        int n = 1;
        int i = 1;
        while (n > 0) {
            n += aa[i]._arity;
            i++;
            n--;
        }
        // take the rest
        while (i < int(aa.size())) {
            result.push_back(aa[i]);
            i++;
        }
    }
    return result;
}



List cons(const List& aa, const List& bb) {
    List result;
    if (bb.size() > 0 && bb[0]._type == ITEM_LIST) {
        result.reserve(aa.size() + bb.size());
        result.push_back(bb[0]);
        result[0]._arity++;
        for (int i = 0; i < int(aa.size()); ++i) {
            result.push_back(aa[i]);
        }
        for (int i = 0; i < int(bb.size()); ++i) {
            result.push_back(bb[i]);
        }
    }
    return result;
}


// =========================================== other functions


void print_code_impl(const Item*& item) {
    if (item->_type == ITEM_LIST || item->_type == ITEM_FCALL) {
        printf("(");
        if (item->_type == ITEM_FCALL) {
            printf("f%d", item->_value);
        }            
        int n = item->_arity;
        item++;
        while (n > 0) {
            printf(" ");
            print_code_impl(item);
            n--;
        }
        printf(")");
    } else if (item->_type == ITEM_INT) {
        printf("%d", item->_value);
        item++;
    } else if (item->_type == ITEM_VAR) {
        printf("v%d", item->_value);
        item++;
    } else {
        printf("?");
        item++;
    }
}


void print_code(const List& program) {
    const Item* sp = &program[0];
    print_code_impl(sp);
    printf("\n");
}
void print_code(const Item* program) {
    const Item* sp = program;
    print_code_impl(sp);
    printf("\n");
}
//void log_variables(const vector<int>& variables);
//void log_functions(const map<string, Function>& functions);


inline bool is_true(const List& expr) {
    if (expr.size() == 0
            || (expr.size() == 1 && expr[0]._type == ITEM_INT && expr[0]._value == 0)
            || (expr.size() == 1 && expr[0]._type == ITEM_LIST && expr[0]._arity == 0)
    ) {
        return false;
    }
    return true;
}


bool is_eq(const List& aa, const List& bb) {
    if (aa.size() != bb.size()) {
        return false;
    }
    for (int i = 0; i < int(aa.size()); ++i) {
        if (aa[i] != bb[i]) {
            return false;
        }
    }
    return true;    
}


List extend(const List& aa, const List& bb) {
    List result;
    if (aa[0]._type == ITEM_LIST && bb[0]._type == ITEM_LIST) {
        result.reserve(int(aa.size()) + int(bb.size()) - 1);
        result = aa;
        for (int i = 1; i < int(bb.size()); ++i) {
            result.push_back(bb[i]);
        }
        result[0]._arity += bb[0]._arity;
    }
    return result;
}


List append(const List& aa, const List& bb) {
    List result;
    if (aa[0]._type == ITEM_LIST) {
        result.reserve(aa.size() + bb.size());
        result = aa;
        for (int i = 0; i < int(bb.size()); ++i) {
            result.push_back(bb[i]);
        }
        result[0]._arity += 1;
    }
    return result;
}


List list(const vector<List>& params) {
    List result;
    result = {{ITEM_LIST, 0, 0}};
    for (int i = 0; i < int(params.size()); ++i) {
        result = append(result, params[i]);
    }
    return result;
}


List last(const vector<List>& params) {
    List result;
    if (params.size() > 0) {
        result = params[params.size() - 1];
    }
    return result;
}


List function(const vector<List>& params, vector<Function>& functions) {
    List result;
    printf("TODO function\n");
    return result;    
}


void subtree_impl(List& result, const Item*& data) {
    result.push_back(*data);
    int n = data->_arity;
    while (n > 0) {
        data++;
        subtree_impl(result, data);
        n--;
    }
}


List subtree(const Item* data) {
    List result;
    subtree_impl(result, data);
    return result;    
}


int next_item(const List& tree, int index) {
    int n = tree[index]._arity;
    index++;
    while (n > 0) {
        index = next_item(tree, index);
        n--;
    }
    return index;    
}


int next_item(const Item* tree, int index) {
    int n = tree[index]._arity;
    index++;
    while (n > 0) {
        index = next_item(tree, index);
        n--;
    }
    return index;    
}


List at_impl(const List& data, int index) {
    List result;
    if (index >= 0 && index < data[0]._arity) {
        int sp = 0;
        for (int i = 0; i < index; ++i) {
            sp = next_item(data, sp);
        }
        result = subtree(&data[sp]);
    }
    return result;    
}


List at(const vector<List>& params) {
    List result = params[0];
    for (int dim = 1; dim < params.size(); ++dim) {
        if (params[dim][0]._type == ITEM_INT) {
            result = at_impl(result, params[dim][0]._value);
        } else {
            result.clear();
            break;
        }
    }
    return result;    
}


int compute_sum(const List& aa) {
    int result = 0;
    if (aa.size() > 0 && aa[0]._type == ITEM_LIST) {
        for (int i = 1; i <= aa[0]._arity; ++i) {
            if (aa[i]._type == ITEM_INT) {
                result += aa[i]._value;
            } else {
                return 0;
            }
        }
    }
    return result;    
}


List var(int& sp, Item* program, int program_size,
         vector<List>& variables, vector<Function>& functions, bool debug, int depth);
List for_loop(int& sp, Item* program, int program_size,
         vector<List>& variables, vector<Function>& functions, bool debug, int depth);


List
run_impl(int& sp, Item* program, int program_size,
         vector<List>& variables, vector<Function>& functions, bool debug, int depth
) {
    List result;
    if (depth > 100) {
        throw runtime_error("warning: code depth exceeded");
    }
    g_count_runs_calls += 1;
    if (g_count_runs_calls > 10000) {
        throw runtime_error("warning: code run calls exceeded");
    }
    if (debug) {
        printf("depth %d\n", depth);
        print_code(program);
        //log_variables(variables);
        //log_functions(functions);
        printf("sp %d\n", sp);
        printf("\n");
    }
    assert(sp < program_size);
    if (program[sp]._type == ITEM_FCALL) {
        int func_index = program[sp]._value, arity = program[sp]._arity;
        sp += 1;
        vector<List> params;
        if (// a function of which all parameters has to be evaluated ALWAYS                
                // this exclude : AND, OR and IF because they use Lazy evaluation
                func_index != F_AND && func_index != F_OR && func_index != F_IF
                // and this exclude : FOR because of the special loop semantics
                && func_index != F_VAR && func_index != F_FOR
        ) {
            for (int i = 0; i < arity; ++i) {
                params.push_back(run_impl(sp, program, program_size, variables, functions, debug, depth+1));
            }
        }
        switch (func_index) {
            case F_LT : 
            case F_LE :
            case F_GE :
            case F_GT :
            case F_ADD :
            case F_SUB :
            case F_MUL :
            case F_DIV : {
                assert(arity == 2);
                List& aa = params[0];
                List& bb = params[1];
                if (aa.size() != 1 || aa[0]._type != ITEM_INT || bb.size() != 1 || bb[0]._type != ITEM_INT) {
                    result.push_back({ITEM_INT, 0, 0});
                } else {
                    int a = aa[0]._value, b = bb[0]._value;
                    switch (func_index) {
                        case F_LT : result = {{ITEM_INT, (a < b ? 1 : 0), 0}}; break;
                        case F_LE : result = {{ITEM_INT, (a <= b ? 1 : 0), 0}}; break;
                        case F_GE : result = {{ITEM_INT, (a >= b ? 1 : 0), 0}}; break;
                        case F_GT : result = {{ITEM_INT, (a > b ? 1 : 0), 0}}; break;
                        case F_ADD : result = {{ITEM_INT, a + b, 0}}; break;
                        case F_SUB : result = {{ITEM_INT, a - b, 0}}; break;
                        case F_MUL : {
                            int c = (((long long)(a) * (long long)(b) <= 1000000000) ? a * b : 0);
                            result = {{ITEM_INT, c, 0}};
                            break;
                        }
                        case F_DIV : result = {{ITEM_INT, (b ? a / b : 0), 0}}; break;
                    }
                }
                break;               
            }
            case F_EQ : 
            case F_NE : {
                int eq = is_eq(params[0], params[1]) ? 1 : 0;
                result = {{ITEM_INT, (func_index == F_EQ ? eq : 1-eq), 0}};
                break;
            }
            case F_AND : {
                int count_true = 0, count_false = 0;
                for (int i = 0; i < arity; ++i) {
                    if (!count_false) {
                        if (is_true(run_impl(sp, program, program_size, variables, functions, debug, depth+1))) {
                            count_true++;
                        } else {
                            count_false++;
                        }
                    } else {
                        sp = next_item(program, sp);
                    }
                }
                result = {{ITEM_INT, (count_true > 0 && count_false == 0 ? 1 : 0), 0}};
                break;
            }
            case F_OR : {
                int count_true = 0, count_false = 0;
                for (int i = 0; i < arity; ++i) {
                    if (!count_true) {
                        if (is_true(run_impl(sp, program, program_size, variables, functions, debug, depth+1))) {
                            count_true++;
                        } else {
                            count_false++;
                        }
                    } else {
                        sp = next_item(program, sp);
                    }
                }
                result = {{ITEM_INT, (count_true > 0 ? 1 : 0), 0}};
                break;
            }
            case F_NOT : result = {{ITEM_INT, (is_true(params[0]) ? 0 : 1), 0}}; break;
            case F_FIRST : result = first(params[0]); break;
            case F_REST : result = rest(params[0]); break;
            case F_EXTEND : result = extend(params[0], params[1]); break;
            case F_APPEND : result = append(params[0], params[1]); break;
            case F_CONS : result = cons(params[0], params[1]); break;
            case F_LEN : result = {{ITEM_INT, len(params[0]), 0}}; break;
            case F_AT : result = at(params); break;
            case F_LIST : result = list(params); break;
            case F_LAST : result = last(params); break;
            case F_VAR : result = var(sp, program, program_size, variables, functions, debug, depth); break;
            case F_ASSIGN :
                if (params[0].size() >0 && params[0][0]._type == ITEM_VAR) {
                    variables[params[0][0]._value] = params[1];
                }
                result = params[1];
                break;
            case F_FUNCTION : function(params, functions); break;
            case F_IF :
                if (is_true(run_impl(sp, program, program_size, variables, functions, debug, depth+1))) {
                    result = run_impl(sp, program, program_size, variables, functions, debug, depth+1);
                } else if (arity > 2) {
                    sp = next_item(program, sp);
                    result = run_impl(sp, program, program_size, variables, functions, debug, depth+1);
                } else {
                    result = {{ITEM_INT, 0, 0}};
                }
                break;
            case F_FOR : result = for_loop(sp, program, program_size, variables, functions, debug, depth); break;
            case F_PRINT : result = params[0]; print_code(params[0]); break;
            case F_ASSERT : result = params[0]; assert(is_true(params[0])); break;
            case F_EXIT : exit(0); break;
            case F_SUM : {
                int sum = compute_sum(params[0]);
                result = {{ITEM_INT, sum, 0}};
                break;
            }
        }
    } else if (program[sp]._type == ITEM_INT) {
        assert(program[sp]._arity == 0);
        result = {{ITEM_INT, program[sp]._value, 0}};
        sp += 1;
    } else if (program[sp]._type == ITEM_VAR) {
        assert(program[sp]._arity == 0);
        result = variables[program[sp]._value];
        sp += 1;
        
    } else if (program[sp]._type == ITEM_LIST) {
        int arity = program[sp]._arity;
        sp += 1;
        vector<List> params;
        for (int i = 0; i < arity; ++i) {
            params.push_back(run_impl(sp, program, program_size, variables, functions, debug, depth+1));
        }
        result = list(params);
        if (debug) {
            printf("DEBUG ITEM_LIST, result: ");
            print_code(result);
        }
    } else {
        throw runtime_error("error: cannot parse code snippet");        
    }
    return result;
}


List var(int& sp, Item* program, int program_size,
         vector<List>& variables, vector<Function>& functions, bool debug, int depth
) {
    List result;
    List old_value;
    List variable = run_impl(sp, program, program_size, variables, functions, debug, depth+1);
    List value = run_impl(sp, program, program_size, variables, functions, debug, depth+1);
    if (variable.size() == 1 && variable[0]._type == ITEM_VAR) {
        old_value = variables[variable[0]._value];
        variables[variable[0]._value] = value;
    }
    result = run_impl(sp, program, program_size, variables, functions, debug, depth+1);
    if (variable.size() == 1 && variable[0]._type == ITEM_VAR) {
        variables[variable[0]._value] = old_value;
    }
    return result;
}


List for_loop(int& sp, Item* program, int program_size,
         vector<List>& variables, vector<Function>& functions, bool debug, int depth
) {
    List result;
    List old_value;
    List loop_variable = run_impl(sp, program, program_size, variables, functions, debug, depth+1);
    List steps = run_impl(sp, program, program_size, variables, functions, debug, depth+1);
    if (steps.size() == 1 && steps[0]._type == ITEM_INT) {
        int n = steps[0]._value;
        if (n > 1000) {
            throw runtime_error("warning: for loop max iterations exceeded");
        }
        steps = {{ITEM_LIST, 0, n}};
        for (int i = 0; i < n; ++i) {
            steps.push_back({ITEM_INT, i, 0});
        }
    }
    if (loop_variable.size() == 1 && loop_variable[0]._type == ITEM_VAR) {
        old_value = variables[loop_variable[0]._value];
        variables[loop_variable[0]._value] = {{ITEM_INT, 0, 0}}; // make sure the old value cannot be accessed anymore
    }
    assert(steps[0]._type == ITEM_LIST);
    int index = 0;
    int sp_begin = sp;
    while (index < steps[0]._arity) {
        if (loop_variable.size() == 1 && loop_variable[0]._type == ITEM_VAR) {
            variables[loop_variable[0]._value] = at_impl(steps, index);
        }
        sp = sp_begin;
        result = append(result, run_impl(sp, program, program_size, variables, functions, debug, depth+1));
        index += 1;
    }
    if (loop_variable.size() == 1 && loop_variable[0]._type == ITEM_VAR) {
        variables[loop_variable[0]._value] = old_value;
    }
    return result;
}


static List run(Item* program, int program_size, vector<List>& variables, vector<Function>& functions, bool debug) {
    List result;
    try {
        int sp = 0;
        result = run_impl(sp, program, program_size, variables, functions, debug, 0);
        assert(sp == program_size);
    }
    catch (const exception& e) {
        if (strncmp(e.what(), "warning", 7) != 0) {
            printf("exception %s\n", e.what());
        }
    }
    return result;
}


extern "C"
#if defined(_MSC_VER)
__declspec(dllexport)
#endif
int run_non_recursive_level1_function(
        int n_params, int* param_sizes, Item* params, // actual params, param[i] = params[sum(param_sizes[:i]):sum(param_sizes[:i+1])
        int n_local_variables,
        Item* function_body, int function_body_size, // 
        int output_bufsize, Item* output_buf, int* n_output
) {
    vector<List> variables;
    variables.resize(n_params + n_local_variables);
    for (int i = 0; i < n_params; ++i) {
        variables[i].resize(param_sizes[i]);
        for (int j = 0; j < param_sizes[i]; ++j) {
            variables[i][j] = *params++;
        }
    }
    for (int i = 0; i < n_local_variables; ++i) {
        variables[n_params + i] = {{ITEM_INT, 0, 0}};
    }
    vector<Function> functions;
    List result = run(function_body, function_body_size, variables, functions, false);
    *n_output = int(result.size());
    if (*n_output > output_bufsize) {
        *n_output = 0;
    }
    for (int i = 0; i < *n_output; ++i) {
        output_buf[i] = result[i];
    }
    return 0;
}
