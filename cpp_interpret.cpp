/* compile on Windows with:
call "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build\vcvars64.bat"
Cl /nologo /EHsc /F 8000000 /D NDEBUG /O2 /Ob1 /MD /LD cpp_interpret.cpp
// Uitleg : /LD create DLL; /MD use multithreaded runtime for DLL's
*/
/* compile on Linux with:
g++ -shared -fPIC -O2 -o cpp_interpret.so cpp_interpret.cpp
*/

#include <vector>
#include <string>
#include <map>
#include <set>
#include <exception>
#include <string.h> // strncmp
#include <cassert>
#include <cmath> // pow
#include <algorithm> // sort

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
    } else {
        result = {{ITEM_INT, 0, 0}};
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
    for (int i = 0; i < depth+2; ++i) {
        printf("    ");
    }
}
    
    
void print_vcode_impl(const List& code, int& i, int len) {
    if (i >= len) {
        printf("{error:i>=n}");
    } else {
        if (code[i]._type == ITEM_LIST || code[i]._type == ITEM_FCALL || code[i]._type == ITEM_FUSERCALL) {
            printf("(");
            if (code[i]._type == ITEM_FCALL) {
                printf("%s", g_fname[code[i]._value]);
            } else if (code[i]._type == ITEM_FUSERCALL) {
                printf("f%d", code[i]._value);
            }           
            int n = code[i]._arity;
            i++;
            while (n > 0) {
                printf(" ");
                print_vcode_impl(code, i, len);
                n--;
            }
            printf(")");
        } else if (code[i]._type == ITEM_INT) {
            printf("%d", code[i]._value);
            i++;
        } else if (code[i]._type == ITEM_VAR) {
            printf("v%d", code[i]._value);
            i++;
        } else {
            printf("?%d", code[i]._type);
            i++;
        }
    }
}


void print_vcode(const List& program) {
    if (program.size() > 0) {
        int i = 0;
        print_vcode_impl(program, i, int(program.size()));
    } else {        
        printf("None // List.size()==0");
    }
    printf("\n");
}


void print_code_impl2(const Item* code, int& i, int len) {
    if (i >= len) {
        printf("{error:i>=len}");
    } else {
        while (i < len) {
            switch (code[i]._type) {
                case ITEM_INT : printf(" i%d", code[i]._value); break;
                case ITEM_FCALL : printf(" %s", g_fname[code[i]._value]); break;
                case ITEM_VAR : printf(" v%d", code[i]._value); break;
                case ITEM_LIST : printf(" L%d", code[i]._arity); break;
                case ITEM_FUSERCALL : printf(" f%d", code[i]._value); break;
                default : printf(" ?%d", code[i]._type); break;
            }
            i++;
        }
    }
}


void print_code(const Item* program, int i, int len) {
    print_code_impl2(program, i, len);
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
    if (aa.size() > 0 && aa[0]._type == ITEM_LIST && bb.size() > 0 && bb[0]._type == ITEM_LIST) {
        result.reserve(int(aa.size()) + int(bb.size()) - 1);
        result = aa;
        for (int i = 1; i < int(bb.size()); ++i) {
            result.push_back(bb[i]);
        }
        result[0]._arity += bb[0]._arity;
    } else {
        result = {{ITEM_INT, 0, 0}};
    }
    return result;
}


List append(const List& aa, const List& bb) {
    List result;
    if (aa.size() > 0 && aa[0]._type == ITEM_LIST) {
        result.reserve(aa.size() + bb.size());
        result = aa;
        for (int i = 0; i < int(bb.size()); ++i) {
            result.push_back(bb[i]);
        }
        result[0]._arity += 1;
    } else {
        result = {{ITEM_INT, 0, 0}};
    }
    return result;
}


void append_inplace(List& result, const List& bb) {
    if (result.size() > 0 && result[0]._type == ITEM_LIST) {
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


void check_depth(const List& data, int current_depth) {
    if (int(data.size()) > 1000) {
        throw runtime_error("warning: data size exceeded");
    }
}


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
        printf("run_impl depth %d on code:", depth);
        print_code(program, sp, program_size);        
        if (false) {
            for (int i = 0; i < int(variables.size()); ++i) {
                print_indent(depth);
                printf("v%d=", i);
                print_vcode(variables[i]);
            }
        }
        for (int i = 0; i < int(functions.size()); ++i) {
            print_indent(depth);
            printf("f%d(%d,%d)=", i, functions[i]._params_count, functions[i]._locals_count);
            print_vcode(functions[i]._code);
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
                        if (debug) {
                            print_indent(depth);
                            printf("DEBUG %d, if true then result", __LINE__);
                            print_vcode(result);
                        }
                        if (arity > 2) {
                            skip_subtree(program, sp); // skip else
                        }
                    } else {
                        if (debug) {
                            print_indent(depth);
                            printf("DEBUG %d, sp before skip 'then'", __LINE__);
                            print_code(program, sp, program_size);
                        }
                        skip_subtree(program, sp); // skip then
                        if (debug) {
                            print_indent(depth);
                            printf("DEBUG %d, sp after skip 'then'", __LINE__);
                            print_code(program, sp, program_size);
                        }
                        if (arity > 2) {
                            result = run_impl(sp, program, program_size, variables, functions, debug, depth+1);
                            if (debug) {
                                print_indent(depth);
                                printf("DEBUG %d, else result ", __LINE__);
                                print_vcode(result);
                            }
                        } else {
                            result = {{ITEM_INT, 0, 0}};
                        }
                    }
                    break;
                }
                case F_FOR : result = for_loop(sp, program, program_size, variables, functions, debug, depth); break;
                case F_PRINT : result = params[0]; print_vcode(params[0]); break;
                case F_ASSERT : {
                    result = params[0];
                    if (!is_true(params[0])) {
                        printf("Assertion failed: ");
                        print_code(program, orig_sp, program_size);
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
        print_vcode(result);
    }
    check_depth(result, 0);
    return result;
}


List assign(int& sp, const Item* program, int program_size,
         vector<List>& variables, vector<Function>& functions, bool debug, int depth
) {
    List result;
    List variable;
    get_subtree(variable, program, sp);
    result = run_impl(sp, program, program_size, variables, functions, debug, depth+1);
    check_depth(result, 0);
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
    //printf("DEBUG cpp 736 code"); print_code(program, sp, program_size);
    List result = {{ITEM_LIST, 0, 0}};
    List old_value;
    List loop_variable;
    get_subtree(loop_variable, program, sp); // don't evaluate, we need the identifyer id
    List steps = run_impl(sp, program, program_size, variables, functions, debug, depth+1);
    //printf("DEBUG cpp 741 loop variable"); print_vcode(loop_variable);
    if (loop_variable.size() == 1 && loop_variable[0]._type == ITEM_VAR) {
        //printf("DEBUG cpp 743 real loop variable : yes\n");
        old_value = variables[loop_variable[0]._value];
    }
    //printf("DEBUG cpp 748 steps"); print_vcode(steps);
    if (steps.size() == 1 && steps[0]._type == ITEM_INT) {
        int n = steps[0]._value;
        //printf("DEBUG cpp 750 int steps %d\n", n);
        if (n > 1000) {
            throw runtime_error("warning: for loop max iterations exceeded");
        }
        steps = {{ITEM_LIST, 0, n}};
        for (int i = 0; i < n; ++i) {
            steps.push_back({ITEM_INT, i, 0});
        }
    }
    //printf("DEBUG cpp 755 steps[0]._arity = %d\n", steps[0]._arity);
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
    if (sp == sp_begin_for_body) {
        // body not executed, sp needs to be advanced
        skip_subtree(program, sp);        
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
        g_count_runs_calls = 0;
        result = run_impl(sp, program, program_size, variables, functions, debug, 0);        
        const char* msg = "Unexpected end of program";
        if (sp < program_size) {
            List garbage;
            int i = sp;
            while (i < program_size) {
                garbage.push_back(program[sp]);
                i += 1;
            }
            print_vcode(garbage);
            msg = "Garbage at end of program";
        }
        Assert(sp == program_size, msg);
    }
    catch (const exception& e) {
        if (debug || strncmp(e.what(), "warning", 7) != 0) {
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
        int output_bufsize, Item* output_buf, int* n_output,
        int debug
) {
    if (debug) {
        printf("C++ start\n");
    }
    vector<List> variables;
    variables.resize(n_params + n_local_variables);
    for (int i = 0; i < n_params; ++i) {
        variables[i].resize(param_sizes[i]);
        for (int j = 0; j < param_sizes[i]; ++j) {
            variables[i][j] = *params;
            params++;
        }
    }
    for (int i = 0; i < n_local_variables; ++i) {
        variables[n_params + i] = {{ITEM_INT, 0, 0}};
    }
    if (debug) {
        for (int i = 0; i < int(variables.size()); ++i) {
            printf("    v%d=", i);        
            print_vcode(variables[i]);
        }
        printf("    body ");        
        List body;
        body.resize(function_body_size);
        for (int i = 0; i < function_body_size; ++i) {
            body[i] = function_body[i];
        }
        print_vcode(body);
    }
    vector<Function> functions;
    List result = run(function_body, function_body_size, variables, functions, debug > 1);
    if (result.size() == 0) {
        result = {{ITEM_INT, 0, 0}};
    }
    if (debug) {
        printf("    output ");        
        print_vcode(result);
        printf("C++ ends (%d run calls)\n", g_count_runs_calls);
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



void extract_numbers_list(int actual_output_size, Item* actual_output, int& sp, vector<int>& result) {
    Assert(sp < actual_output_size, "buffer size error in model output");
    if (actual_output[sp]._type == ITEM_INT) {
        result.push_back(actual_output[sp]._value);
        sp++;
    } else {
        Assert(actual_output[sp]._type == ITEM_LIST, "expected a tree of integers");
        int n = actual_output[sp]._arity;
        sp++;
        if (n == 0) {
            result.push_back(0); // [] is extracted as '0' 
        } else {
            for (int i = 0; i < n; ++i) {
                extract_numbers_list(actual_output_size, actual_output, sp, result);
            }
        }
    }
}


int count_empty_sublists(int actual_output_size, Item* actual_output, int& sp) {
    int result = 0;
    if (actual_output[sp]._type == ITEM_INT) {
        sp++;
    } else {
        Assert(actual_output[sp]._type == ITEM_LIST, "expected a tree of integers");
        int n = actual_output[sp]._arity;
        sp++;
        if (n == 0) {
            result += 1;
        } else {
            for (int i = 0; i < n; ++i) {
                result += count_empty_sublists(actual_output_size, actual_output, sp);
            }
        }
    }
    return result;
}


int _distance_with_closest_numbers(int x, const set<int>& values) {
    int result;
    if (x > 1000000) {
        x = 1000000;
    }
    if (values.size() > 0) {
        result = 1000000;
        for (auto value : values) {
            if (value > 1000000) {
                value = 1000000;
            }
            if (result > abs(x - value)) {
               result = abs(x - value);
               if (result == 0) {
                   break;
               }
            }
        }
    } else {
        result = abs(x - 0);
    }
    return result;
}


int _distance_with_closest_sorted_numbers(int x, const vector<int>& values) {
    int result;
    if (values.size() > 0) {
        result = abs(x - values[0]);
        for (int i = 1; i < int(values.size()); ++i) {
            const int value = values[i];
            const int distance = abs(x - value);
            if (result > distance) {
               result = distance;
            }
            if (value >= x) {
                break;
            }
        }
    } else {
        result = abs(x - 0);
    }
    return result;
}


const double g_w1 = 0.3;
const double g_w2a = 1.5;
const double g_w2b = 1.1;
const double g_w3 = 1.6;
const double g_w4 = 1.5;
const double g_w5 = 1.5;
const double g_w6 = 0.1;
const double g_w7 = 0.1;
const double g_w8 = 0.4;

void compute_error_vector_impl(
    int expected_output_size, int* expected_output,
    int actual_output_size, Item* actual_output,
    int error_vector_size, double* error_vector, 
    int debug
) {
    if (debug) {
        printf("    C++, actual output:");
        int sp = 0;
        print_code(actual_output, sp, actual_output_size);
        printf("    C++, expected output: [");
        for (int i = 0; i < expected_output_size; ++i) {
            printf("%d, ", expected_output[i]);
        }
        printf("]\n");
    }


    // error1 : type difference
    double error = 0.0;
    if (actual_output[0]._type != ITEM_LIST) {
        Assert(actual_output[0]._type == ITEM_INT, "unexpected model output");
        error = 1.0 + expected_output_size;
        int value = actual_output[0]._value;
        actual_output[0]._type = ITEM_LIST;
        actual_output[0]._value = 0;
        actual_output[0]._arity = 1;
        Assert(actual_output_size == 1, "actual_output_size must be 1, otherwise its a syntax error");
        actual_output_size = 2; // IMPORTANT. Caller has to make sure there is room to do this
        actual_output[1]._type = ITEM_INT;
        actual_output[1]._value = value;
        actual_output[1]._arity = 0;
    } else {
        int sp = 1;
        for (int i = 0; i < expected_output_size; ++i) {
            if (i < actual_output[0]._arity) {
                if (actual_output[sp]._type != ITEM_INT) {
                    error += 1;
                }
                skip_subtree(actual_output, sp);
            } else {
                error += 1;
            }
        }
    }
    if (error > 0) {
        error = pow(error, g_w1);
    }
    error_vector[0] = error;

    // error2 : length difference
    vector<int> actual_list;
    int sp = 0;
    if (actual_output_size > 0) {
        extract_numbers_list(actual_output_size, actual_output, sp, actual_list);
    }
    Assert(sp == actual_output_size, "model output syntax error");
    if (debug) {
        printf("    C++, expected_output_size %d, actual output size %d\n", expected_output_size, int(actual_list.size()));
    }
    if (int(actual_list.size()) < expected_output_size) {
        error_vector[1] = pow(double(expected_output_size - int(actual_list.size())), g_w2a);
    } else if (int(actual_list.size()) > expected_output_size) {
        error_vector[1] = pow(double(int(actual_list.size()) - expected_output_size), g_w2b);
    } else {
        error_vector[1] = 0.0;
    }

    // error3 : set getallen vergelijken
    set<int> actual_set;
    for (int actual : actual_list) {        
        actual_set.insert(actual);
    }
    if (actual_set.size() == 0) {
        actual_set.insert(0);
    }
    std::vector<int> actual_sorted;
    for (int actual_unique : actual_set) {
        actual_sorted.push_back(actual_unique);
    }
    sort(actual_sorted.begin(), actual_sorted.end());
    error = 0.0;
    for (int i = 0; i < expected_output_size; ++i) {
        error += pow(double(_distance_with_closest_sorted_numbers(expected_output[i], actual_sorted)), g_w3);
    }
    error_vector[2] = error;
     
    // error4 : set getallen vergelijken
    std::vector<int> expect_sorted;
    for ( int i = 0; i < expected_output_size; ++i) {        
        expect_sorted.push_back(expected_output[i]);
    }
    sort(expect_sorted.begin(), expect_sorted.end());
    error = 0.0;
    for (int actual : actual_set) {
        error += pow(double(_distance_with_closest_sorted_numbers(actual, expect_sorted)), g_w4);
    }
    error_vector[3] = error;
     
    // error5 : staan ze op de juiste plaats
    error = 0.0;
    sp = 1;
    for (int i = 0; i < expected_output_size; ++i) {
        if (i < actual_output[0]._arity) {
            if (actual_output[sp]._type == ITEM_INT) {
                error += pow(abs(actual_output[sp]._value - expected_output[i]), g_w5);
            } else {
                error += pow(abs(expected_output[i]), g_w5);
            }
            skip_subtree(actual_output, sp);
        }
    }
    error_vector[4] = error;
     
    // error6 : staan ze op de juiste volgorde voor naar achter
    error = 0.0;
    int j = 0 ;
    for (int i = 0; i < expected_output_size; ++i) {
        while (j < int(actual_list.size()) && expected_output[i] != actual_list[j]) {
            j += 1;
        }
        if (j >= int(actual_list.size())) {
            error += 1;
        }
    }
    if (error > 0) {
        error = pow(error, g_w6);
    }
    error_vector[5] = error;

    // error7 : staan ze op de juiste volgorde achter naar voor
    error = 0.0;
    j = int(actual_list.size()) - 1;
    for (int i = expected_output_size-1; i >= 0; --i) {
        while (j >= 0 && expected_output[i] != actual_list[j]) {
            j -= 1;
        }
        if (j < 0) {
            error += 1;
        }
    }
    if (error > 0) {
        error = pow(error, g_w7);
    }
    error_vector[6] = error;

    // error8 : aantal nul subtrees
    sp = 0;
    error = count_empty_sublists(actual_output_size, actual_output, sp);
    Assert(sp == actual_output_size, "model output syntax error");
    if (error > 0) {
        error = pow(error, g_w8);
    }
    error_vector[7] = error;
}

extern "C"
#if defined(_MSC_VER)
__declspec(dllexport)
#endif
int compute_error_vector(
        int expected_output_size, int* expected_output,
        int actual_output_size, Item* actual_output,
        int error_vector_size, double* error_vector, 
        int debug
) {
    if (debug) {
        printf("C++ compute_error_vector start\n");
    }
    Assert(error_vector_size == 8, "Expected error buffer of size 8");
    compute_error_vector_impl(expected_output_size, expected_output, actual_output_size, actual_output,
        error_vector_size, error_vector, debug);
    if (debug) {
        printf("    output ");
        for (int i = 0; i < error_vector_size; ++i) {
            printf(" %.1f", error_vector[i]);
        }
        printf("\nC++ compute_error_vector ends\n");
    }
    return 0;
}
