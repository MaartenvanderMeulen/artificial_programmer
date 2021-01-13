// compile on Windows with:
// call "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build\vcvars64.bat"
// Cl /nologo /EHsc /F 8000000 /D NDEBUG /O2 /Ob1 /MD /LD cpp_interpret.cpp
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

static const int CODEITEM_INT = 1;
static const int CODEITEM_FCALL = 2;
static const int CODEITEM_VAR = 3;

static const int F_LT = 1;
static const int F_LE = 2;
static const int F_GE = 3;
static const int F_GT = 4;
static const int F_ADD = 5;
static const int F_SUB = 6;
static const int F_MUL = 7;
static const int F_DIV = 8;



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


struct CodeItem {
public:
    int _type; // 0 integer value, 1 function call
    int _value;
    int _arity;
};

struct Function {
public:
    string _name;
    vector<string> _params;
    vector<CodeItem> _code;
};


// =========================================== globals

static int g_count_runs_calls = 0;
static SaveFp g_fp_log;


// =========================================== functions


//void log_code(const vector<CodeItem>& program);
//void log_variables(const vector<int>& variables);
//void log_functions(const map<string, Function>& functions);


vector<CodeItem> run_impl(int& sp, vector<CodeItem>& program, vector<int>& variables, map<string, Function>& functions, bool debug, int depth) {
    vector<CodeItem> result;
    if (depth > 100) {
        throw runtime_error("warning: code depth exceeded");
    }
    g_count_runs_calls += 1;
    if (g_count_runs_calls > 10000) {
        throw runtime_error("warning: code run calls exceeded");
    }
    if (debug) {
        fprintf(g_fp_log._fp, "depth %d\n", depth);
        //log_code(program);
        //log_variables(variables);
        //log_functions(functions);
        fprintf(g_fp_log._fp, "\n");
    }
    assert(sp < program.size());
    if (program[sp]._type == CODEITEM_FCALL) {
        switch (program[sp]._value) {
            case F_LT : 
            case F_LE :
            case F_GE :
            case F_GT :
            case F_ADD :
            case F_SUB :
            case F_MUL :
            case F_DIV : {
                assert(program[sp]._arity == 2);
                int func_index = program[sp]._value;
                sp += 1;
                vector<CodeItem> aa = run_impl(sp, program, variables, functions, debug, depth+1);
                vector<CodeItem> bb = run_impl(sp, program, variables, functions, debug, depth+1);
                if (aa.size() != 1 || aa[0]._type != CODEITEM_INT || bb.size() != 1 || bb[0]._type != CODEITEM_INT) {
                    result.push_back({CODEITEM_INT, 0, 0});
                } else {
                    int a = aa[0]._value, b = bb[0]._value;
                    switch (func_index) {
                        case F_LT : result = {{CODEITEM_INT, (a < b ? 1 : 0), 0}}; break;
                        case F_LE : result = {{CODEITEM_INT, (a <= b ? 1 : 0), 0}}; break;
                        case F_GE : result = {{CODEITEM_INT, (a >= b ? 1 : 0), 0}}; break;
                        case F_GT : result = {{CODEITEM_INT, (a > b ? 1 : 0), 0}}; break;
                        case F_ADD : result = {{CODEITEM_INT, a + b, 0}}; break;
                        case F_SUB : result = {{CODEITEM_INT, a - b, 0}}; break;
                        case F_MUL : {
                            int c = (((long long)(a) * (long long)(b) <= 1000000000) ? a * b : 0);
                            result = {{CODEITEM_INT, c, 0}};
                            break;
                        }
                        case F_DIV : result = {{CODEITEM_INT, (b ? a / b : 0), 0}}; break;
                    }
                }
                break;
            }
        }
    } else if (program[sp]._type == CODEITEM_INT) {
        assert(program[sp]._arity == 0);
        result = {{CODEITEM_INT, program[sp]._value, 0}};
        sp += 1;
    } else if (program[sp]._type == CODEITEM_VAR) {
        assert(program[sp]._arity == 0);
        if (0 <= program[sp]._value && program[sp]._value < variables.size()) {
            result = {{CODEITEM_INT, variables[program[sp]._value], 0}};
        } else {
            throw runtime_error("error: unknown variable");        
        }
        sp += 1;
    } else {
        throw runtime_error("error: cannot parse code snippet");        
    }
    return result;
}


static vector<CodeItem> run(vector<CodeItem>& program, vector<int>& variables, map<string, Function>& functions, bool debug) {
    vector<CodeItem> result;
    try {
        int sp = 0;
        result = run_impl(sp, program, variables, functions, debug, 0);
        assert(sp == program.size());
    }
    catch (const exception& e) {
        if (strncmp(e.what(), "warning", 7) != 0) {
            fprintf(g_fp_log._fp, "exception %s\n", e.what());
        }
    }
    return result;
}


int self_test() {
    int err_count = 0;
    printf("start selftest\n");
    vector<CodeItem> program;
    vector<int> variables;
    map<string, Function> functions;
    vector<CodeItem> result;
    g_fp_log._fp = stdout; // fopen("tmp/log.txt", "w");
    g_fp_log._manage = false;
    bool debug = false;
    try {
        int a, b, f, expected;
        
        a = 3; b = 17; f = F_ADD; expected = a + b;
        program = {{CODEITEM_FCALL, f, 2}, {CODEITEM_INT, a, 0}, {CODEITEM_INT, b, 0}};
        result = run(program, variables, functions, debug);        
        if (result[0]._type != CODEITEM_INT || result[0]._value != expected) {
            printf("expected %d instead of %d\n", expected, result[0]._value);
            err_count += 1;
        }

        a = 3; b = 17; f = F_SUB; expected = a - b;
        program = {{CODEITEM_FCALL, f, 2}, {CODEITEM_INT, a, 0}, {CODEITEM_INT, b, 0}};
        result = run(program, variables, functions, debug);        
        if (result[0]._type != CODEITEM_INT || result[0]._value != expected) {
            printf("expected %d instead of %d\n", expected, result[0]._value);
            err_count += 1;
        }

        a = 3; b = 17; f = F_MUL; expected = a * b;
        program = {{CODEITEM_FCALL, f, 2}, {CODEITEM_INT, a, 0}, {CODEITEM_INT, b, 0}};
        result = run(program, variables, functions, debug);        
        if (result[0]._type != CODEITEM_INT || result[0]._value != expected) {
            printf("expected %d instead of %d\n", expected, result[0]._value);
            err_count += 1;
        }
        
        a = 17; b = 3; f = F_DIV; expected = a / b;
        program = {{CODEITEM_FCALL, f, 2}, {CODEITEM_INT, a, 0}, {CODEITEM_INT, b, 0}};
        result = run(program, variables, functions, debug);        
        if (result[0]._type != CODEITEM_INT || result[0]._value != expected) {
            printf("expected %d instead of %d\n", expected, result[0]._value);
            err_count += 1;
        }
              
        a = 17; b = 3; f = F_LT; expected = a < b ? 1 : 0;
        program = {{CODEITEM_FCALL, f, 2}, {CODEITEM_INT, a, 0}, {CODEITEM_INT, b, 0}};
        result = run(program, variables, functions, debug);        
        if (result[0]._type != CODEITEM_INT || result[0]._value != expected) {
            printf("expected %d instead of %d\n", expected, result[0]._value);
            err_count += 1;
        }

        a = 3; b = 17; f = F_LT; expected = a < b ? 1 : 0;
        program = {{CODEITEM_FCALL, f, 2}, {CODEITEM_INT, a, 0}, {CODEITEM_INT, b, 0}};
        result = run(program, variables, functions, debug);        
        if (result[0]._type != CODEITEM_INT || result[0]._value != expected) {
            printf("expected %d instead of %d\n", expected, result[0]._value);
            err_count += 1;
        }
        
        a = 17; b = 3; f = F_LE; expected = a <= b ? 1 : 0;
        program = {{CODEITEM_FCALL, f, 2}, {CODEITEM_INT, a, 0}, {CODEITEM_INT, b, 0}};
        result = run(program, variables, functions, debug);        
        if (result[0]._type != CODEITEM_INT || result[0]._value != expected) {
            printf("expected %d instead of %d\n", expected, result[0]._value);
            err_count += 1;
        }

        a = 3; b = 17; f = F_LE; expected = a <= b ? 1 : 0;
        program = {{CODEITEM_FCALL, f, 2}, {CODEITEM_INT, a, 0}, {CODEITEM_INT, b, 0}};
        result = run(program, variables, functions, debug);        
        if (result[0]._type != CODEITEM_INT || result[0]._value != expected) {
            printf("expected %d instead of %d\n", expected, result[0]._value);
            err_count += 1;
        }
        
        a = 17; b = 3; f = F_GE; expected = a >= b ? 1 : 0;
        program = {{CODEITEM_FCALL, f, 2}, {CODEITEM_INT, a, 0}, {CODEITEM_INT, b, 0}};
        result = run(program, variables, functions, debug);        
        if (result[0]._type != CODEITEM_INT || result[0]._value != expected) {
            printf("expected %d instead of %d\n", expected, result[0]._value);
            err_count += 1;
        }

        a = 3; b = 17; f = F_GE; expected = a >= b ? 1 : 0;
        program = {{CODEITEM_FCALL, f, 2}, {CODEITEM_INT, a, 0}, {CODEITEM_INT, b, 0}};
        result = run(program, variables, functions, debug);        
        if (result[0]._type != CODEITEM_INT || result[0]._value != expected) {
            printf("expected %d instead of %d\n", expected, result[0]._value);
            err_count += 1;
        }
        
        a = 17; b = 3; f = F_GT; expected = a > b ? 1 : 0;
        program = {{CODEITEM_FCALL, f, 2}, {CODEITEM_INT, a, 0}, {CODEITEM_INT, b, 0}};
        result = run(program, variables, functions, debug);        
        if (result[0]._type != CODEITEM_INT || result[0]._value != expected) {
            printf("expected %d instead of %d\n", expected, result[0]._value);
            err_count += 1;
        }

        a = 3; b = 17; f = F_GT; expected = a > b ? 1 : 0;
        program = {{CODEITEM_FCALL, f, 2}, {CODEITEM_INT, a, 0}, {CODEITEM_INT, b, 0}};
        result = run(program, variables, functions, debug);        
        if (result[0]._type != CODEITEM_INT || result[0]._value != expected) {
            printf("expected %d instead of %d\n", expected, result[0]._value);
            err_count += 1;
        }

        int c;
        a = 3; b = 17; c = 4; expected = a + b * c;
        program = {{CODEITEM_FCALL, F_ADD, 2}, {CODEITEM_INT, a, 0}, {CODEITEM_FCALL, F_MUL, 2}, {CODEITEM_INT, b, 0}, {CODEITEM_INT, c, 0}};
        result = run(program, variables, functions, debug);        
        if (result[0]._type != CODEITEM_INT || result[0]._value != expected) {
            printf("expected %d instead of %d\n", expected, result[0]._value);
            err_count += 1;
        }
        a = 3; b = 17; c = 4; expected = a * b + c;
        program = {{CODEITEM_FCALL, F_ADD, 2}, {CODEITEM_FCALL, F_MUL, 2}, {CODEITEM_INT, a, 0}, {CODEITEM_INT, b, 0}, {CODEITEM_INT, c, 0}};
        result = run(program, variables, functions, debug);        
        if (result[0]._type != CODEITEM_INT || result[0]._value != expected) {
            printf("expected %d instead of %d\n", expected, result[0]._value);
            err_count += 1;
        }

        a = 3; b = 17; c = 4; expected = a * b + c;
        variables = {a, b, c};
        program = {{CODEITEM_FCALL, F_ADD, 2}, {CODEITEM_FCALL, F_MUL, 2}, {CODEITEM_VAR, 0, 0}, {CODEITEM_VAR, 1, 0}, {CODEITEM_VAR, 2, 0}};
        result = run(program, variables, functions, debug);        
        if (result[0]._type != CODEITEM_INT || result[0]._value != expected) {
            printf("expected %d instead of %d\n", expected, result[0]._value);
            err_count += 1;
        }
    }
    catch (const exception& e) {
        printf("exception %s\n", e.what());
    }
    printf("%d errors encountered in selftest\n", err_count);
    return 0;
}


extern "C"
#if defined(_MSC_VER)
__declspec(dllexport)
#endif
void run_code(int n, void*  pv_types, void* pv_values, void* pv_arities) {
    int* pi_types = (int*) pv_types;
    int* pi_values = (int*) pv_values;
    int* pi_arities = (int*) pv_arities;
    vector<CodeItem> program;
    program.resize(n);
    printf("run_code result:\n");
    for (int i = 0; i < n; ++i) {
        program[i]._type = pi_types[i];
        program[i]._value = pi_values[i];
        program[i]._arity = pi_arities[i];
        printf("    %d (%d,%d,%d)\n", i, program[i]._type, program[i]._value, program[i]._arity);
    }
    vector<int> variables;
    map<string, Function> functions;
    vector<CodeItem> result = run(program, variables, functions, false);
    printf("run_code result:\n");
    for (int i = 0; i < int(result.size()); ++i) {
        printf("    %d (%d,%d,%d)\n", i, result[i]._type, result[i]._value, result[i]._arity);
    }
}


int main(int argc, char* argv[]) {
    if (!self_test()) {
        return 1;
    }
    return 0;
}