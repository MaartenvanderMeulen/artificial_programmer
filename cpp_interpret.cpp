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
static const int ITEM_FCALL = 2; // call build-in function
static const int ITEM_VAR = 3;
static const int ITEM_LIST = 4; // {ITEM_LIST, *, arity} == {ITEM_FCALL, F_LIST, arity}
static const int ITEM_FUSERCALL = 5; // call user defined function

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

static const char* g_fname[] = {
    "",
    "lt", "le", "ge", "gt", "add", "sub", "mul", "div",
    "eq", "ne", "and", "or", "not",
    "first", "rest", "extend", "append", "cons", "len",
    "at", "list", "last",
    "var", "assign", "function", "if", "for",
    "print", "assert", "exit",
    "sum"
};


// =================================== types


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
    int _params_count;
    int _locals_count;
    List _code;
};


// =========================================== globals

static int g_count_runs_calls = 0;


// =========================================== Functions

void Assert(bool cond, const string& msg) {
    if (!cond) {
        throw runtime_error(msg);
    }
}


int len(const List& aa) {    
    return aa[0]._arity;
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
        result.push_back(aa[0]);
        result[0]._arity -= 1;
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
        for (int i = 1; i < int(bb.size()); ++i) {
            result.push_back(bb[i]);
        }
    }
    return result;
}


void print_indent(int depth) {
    for (int i = 0; i < depth; ++i) {
        printf("    ");
    }
}
    
    
void print_code_impl(const Item*& item) {
    if (item->_type == ITEM_LIST || item->_type == ITEM_FCALL || item->_type == ITEM_FUSERCALL) {
        printf("(");
        if (item->_type == ITEM_FCALL) {
            printf("%s", g_fname[item->_value]);
        } else if (item->_type == ITEM_FUSERCALL) {
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
    if (program.size() > 0) {
        const Item* sp = &program[0];
        print_code_impl(sp);
    } else {
        printf("None");
    }
    printf("\n");
}


void print_code(const Item* program) {
    const Item* sp = program;
    print_code_impl(sp);
    printf("\n");
}


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


void append_inplace(List& result, const List& bb) {
    if (result[0]._type == ITEM_LIST) {
        for (int i = 0; i < int(bb.size()); ++i) {
            result.push_back(bb[i]);
        }
        result[0]._arity += 1;
    }
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


void get_subtree(List& result, const List& data, int& sp) {
    result.push_back(data[sp]);
    int n = data[sp]._arity;
    sp++;
    while (n > 0) {
        get_subtree(result, data, sp);
        n--;
    }
}


void get_subtree(List& result, const Item* data, int& sp) {
    result.push_back(data[sp]);
    int n = data[sp]._arity;
    sp++;
    while (n > 0) {
        get_subtree(result, data, sp);
        n--;
    }
}


void skip_subtree(const List& tree, int& sp) {
    int n = tree[sp]._arity;
    sp++;
    while (n > 0) {
        skip_subtree(tree, sp);
        n--;
    }
}


void skip_subtree(const Item* tree, int& sp) {
    int n = tree[sp]._arity;
    sp++;
    while (n > 0) {
        skip_subtree(tree, sp);
        n--;
    }
}


List at_impl(const List& data, int at_index) {
    List result;
    if (at_index >= 0 && data.size() > 0 && at_index < data[0]._arity) {
        int sp = 1;
        for (int i = 0; i < at_index; ++i) {
            skip_subtree(data, sp);
        }
        get_subtree(result, data, sp);
    }
    return result;    
}


List at(const vector<List>& params) {
    List result = params[0];
    for (int dim = 1; dim < int(params.size()); ++dim) {
        if (params[dim].size() > 0 && params[dim][0]._type == ITEM_INT) {
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


void add_function(const vector<List>& params, vector<Function>& functions) {
    // 4 params : func_id, n_params, n_locals, code
    Assert(params.size() == 4, "expected: function func_id n_params n_locals code");
    Assert(params[0].size() == 1 && params[0][0]._type == ITEM_INT, "expected: func_id integer");
    int func_id = params[0][0]._value;
    Assert(params[1].size() == 1 && params[1][0]._type == ITEM_INT, "expected: n_params integer");
    int n_params = params[1][0]._value;
    Assert(params[2].size() == 1 && params[2][0]._type == ITEM_INT, "expected: n_locals integer");
    int n_locals = params[2][0]._value;
    while (func_id >= int(functions.size())) {
        functions.push_back({0, 0, {{ITEM_LIST, 0, 0}}});
    }
    functions[func_id] = {n_params, n_locals, params[3]};
}


List assign(int& sp, const Item* program, int program_size,
         vector<List>& variables, vector<Function>& functions, bool debug, int depth);
List var(int& sp, const Item* program, int program_size,
         vector<List>& variables, vector<Function>& functions, bool debug, int depth);
List for_loop(int& sp, const Item* program, int program_size,
         vector<List>& variables, vector<Function>& functions, bool debug, int depth);


List
run_impl(int& sp, const Item* program, int program_size,
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
        print_indent(depth);
        print_code(&program[sp]);        
        for (int i = 0; i < int(variables.size()); ++i) {
            print_indent(depth);
            printf("v%d=", i);
            print_code(variables[i]);
        }
        for (int i = 0; i < int(functions.size()); ++i) {
            print_indent(depth);
            printf("f%d(%d,%d)=", i, functions[i]._params_count, functions[i]._locals_count);
            print_code(functions[i]._code);
        }
    }
    Assert(sp < program_size, "Stack pointer outside the program");
    int _type = program[sp]._type;
    switch (_type) {
        case ITEM_INT : {
            Assert(program[sp]._arity == 0, "Int must have arity 0");
            result = {{ITEM_INT, program[sp]._value, 0}};
            sp += 1;
            break;
        }
        case ITEM_FCALL : {
            int orig_sp = sp;
            int func_index = program[sp]._value, arity = program[sp]._arity;
            sp += 1;
            vector<List> params;
            if (// a function of which all parameters has to be evaluated ALWAYS                
                    // this exclude : AND, OR and IF because they use Lazy evaluation
                    func_index != F_AND && func_index != F_OR && func_index != F_IF
                    // and this exclude : VAR and ASSIGN because of the special variable semantics
                    && func_index != F_VAR && func_index != F_ASSIGN
                    // and this exclude : FOR because of the special loop body semantics
                    && func_index != F_FOR
                    // and this exclude : FUNCTION because of the special function body semantics
                    && func_index != F_FUNCTION
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
                    Assert(arity == 2, "le, lt, ge, gt, add, sub, mil, div: arity must be 2");
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
                            skip_subtree(program, sp);
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
                            skip_subtree(program, sp);
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
                case F_ASSIGN : result = assign(sp, program, program_size, variables, functions, debug, depth); break;
                case F_FUNCTION : {
                    params.resize(arity);
                    for (int i = 0; i < arity; ++i) {
                        get_subtree(params[i], program, sp);
                    }
                    add_function(params, functions);
                    break;
                }
                case F_IF : {
                    bool cond = is_true(run_impl(sp, program, program_size, variables, functions, debug, depth+1));
                    if (cond) {
                        result = run_impl(sp, program, program_size, variables, functions, debug, depth+1);
                        if (arity > 2) {
                            skip_subtree(program, sp); // skip else
                        }
                    } else {
                        skip_subtree(program, sp); // skip then
                        if (arity > 2) {
                            result = run_impl(sp, program, program_size, variables, functions, debug, depth+1);
                        } else {
                            result = {{ITEM_INT, 0, 0}};
                        }
                    }
                    break;
                }
                case F_FOR : result = for_loop(sp, program, program_size, variables, functions, debug, depth); break;
                case F_PRINT : result = params[0]; print_code(params[0]); break;
                case F_ASSERT : {
                    result = params[0];
                    if (!is_true(params[0])) {
                        printf("Assertion failed: ");
                        print_code(&program[orig_sp]);
                    }
                    break;
                }
                case F_EXIT : exit(0); break;
                case F_SUM : {
                    int sum = compute_sum(params[0]);
                    result = {{ITEM_INT, sum, 0}};
                    break;
                }
                default : {
                    throw runtime_error("Call to unknown build-in function");        
                }
            }
			break;
        }
        case ITEM_VAR : {
            Assert(program[sp]._arity == 0, "Var must have arity 0");
            Assert(0 <= program[sp]._value && program[sp]._value < int(variables.size()), "Unknown var id");
            result = variables[program[sp]._value];
            sp += 1;
            break;
        }
        case ITEM_LIST : {
            int arity = program[sp]._arity;
            sp += 1;
            vector<List> params;
            for (int i = 0; i < arity; ++i) {
                params.push_back(run_impl(sp, program, program_size, variables, functions, debug, depth+1));
            }
            result = list(params);
            break;
        }
        case ITEM_FUSERCALL : {
            Assert(0 <= program[sp]._value && program[sp]._value < int(functions.size()), "Unknown function id");
            int func_index = program[sp]._value, arity = program[sp]._arity;
            const Function& f = functions[func_index];
            Assert(arity == f._params_count, "Function gets wrong number of parameters");
            sp += 1;
            vector<List> new_variables;
            for (int i = 0; i < arity; ++i) {
                new_variables.push_back(run_impl(sp, program, program_size, variables, functions, debug, depth+1));
            }            
            for (int i = 0; i < f._locals_count; ++i) {
                new_variables.push_back({{ITEM_INT, 0, 0}});
            }
            int new_sp = 0;            
            result = run_impl(new_sp, &f._code[0], int(f._code.size()), new_variables, functions, debug, depth+1);
            const char* msg = (new_sp < int(f._code.size()) ? "Garbage after end of function code" : "Unexpected end of function code");
            Assert(new_sp == int(f._code.size()), msg);
            break;
        }
        default : {
            throw runtime_error("Unknown code snippet ITEM type");        
        }
    }
    if (debug) {
        print_indent(depth);
        printf("result: ");
        print_code(result);
    }
    return result;
}


List assign(int& sp, const Item* program, int program_size,
         vector<List>& variables, vector<Function>& functions, bool debug, int depth
) {
    List result;
    List variable;
    get_subtree(variable, program, sp);
    result = run_impl(sp, program, program_size, variables, functions, debug, depth+1);
    if (variable.size() == 1 && variable[0]._type == ITEM_VAR) {
        variables[variable[0]._value] = result;
    }
    return result;
}


List var(int& sp, const Item* program, int program_size,
         vector<List>& variables, vector<Function>& functions, bool debug, int depth
) {
    List result;
    List old_value;
    List variable;
    get_subtree(variable, program, sp);
    if (variable.size() == 1 && variable[0]._type == ITEM_VAR) {
        old_value = variables[variable[0]._value];
    }
    List value = run_impl(sp, program, program_size, variables, functions, debug, depth+1);
    if (variable.size() == 1 && variable[0]._type == ITEM_VAR) {
        variables[variable[0]._value] = value;
    }
    result = run_impl(sp, program, program_size, variables, functions, debug, depth+1);
    if (variable.size() == 1 && variable[0]._type == ITEM_VAR) {
        variables[variable[0]._value] = old_value;
    }
    return result;
}


List for_loop(int& sp, const Item* program, int program_size,
         vector<List>& variables, vector<Function>& functions, bool debug, int depth
) {
    List result = {{ITEM_LIST, 0, 0}};
    List old_value;
    List loop_variable;
    get_subtree(loop_variable, program, sp); // don't evaluate, we need the identifyer id
    if (loop_variable.size() == 1 && loop_variable[0]._type == ITEM_VAR) {
        old_value = variables[loop_variable[0]._value];
        variables[loop_variable[0]._value] = {{ITEM_INT, 0, 0}}; // make sure the old value cannot be accessed anymore
    }
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
    Assert(steps[0]._type == ITEM_LIST, "For loop steps must be of List type");
    int steps_sp = 1;
    int for_iteration = 0;
    int sp_begin_for_body = sp;
    while (for_iteration < steps[0]._arity) {
        List loop_variable_value;
        loop_variable_value.clear();
        get_subtree(loop_variable_value, steps, steps_sp);
        if (loop_variable.size() == 1 && loop_variable[0]._type == ITEM_VAR) {
            variables[loop_variable[0]._value] = loop_variable_value;
        }
        sp = sp_begin_for_body;
        List iter_result = run_impl(sp, program, program_size, variables, functions, debug, depth+1);
        append_inplace(result, iter_result);
        for_iteration += 1;
    }
    if (loop_variable.size() == 1 && loop_variable[0]._type == ITEM_VAR) {
        variables[loop_variable[0]._value] = old_value;
    }
    return result;
}


static List run(const Item* program, int program_size, vector<List>& variables, vector<Function>& functions, bool debug) {
    List result;
    try {
        int sp = 0;
        result = run_impl(sp, program, program_size, variables, functions, debug, 0);
        Assert(sp == program_size, (sp < program_size ? "Garbage after end of program" : "Unexpected end of program"));
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
    bool debug = true;
    if (debug) {
        printf("C++\n");
        printf("    n_params %d\n", n_params);        
        for (int i = 0; i < n_params; ++i) {
            printf("    param size %d = %d ", i, param_sizes[i]);        
            print_code(params);
        }
    }
    vector<List> variables;
    variables.resize(n_params + n_local_variables);
    for (int i = 0; i < n_params; ++i) {
        variables[i].resize(param_sizes[i]);
        for (int j = 0; j < param_sizes[i]; ++j) {
            variables[i][j] = *params++;
        }
    }
    if (debug) {
        for (int i = 0; i < n_params; ++i) {
            printf("    param ");        
            print_code(variables[i]);
        }
        printf("    body ");        
        print_code(function_body);
        printf("    body size %d\n", function_body_size);        
    }
    for (int i = 0; i < n_local_variables; ++i) {
        variables[n_params + i] = {{ITEM_INT, 0, 0}};
    }
    vector<Function> functions;
    List result = run(function_body, function_body_size, variables, functions, false);
    if (debug) {
        printf("    output buf size %d\n", output_bufsize);        
        printf("    output ");        
        print_code(result);
    }
    *n_output = int(result.size());
    if (*n_output > output_bufsize) {
        *n_output = 0;
    }
    for (int i = 0; i < *n_output; ++i) {
        output_buf[i] = result[i];
    }
    return 0;
}
