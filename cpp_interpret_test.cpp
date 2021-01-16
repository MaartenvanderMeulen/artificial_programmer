// compile on Windows with:
// call "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build\vcvars64.bat"
// Cl /nologo /EHsc /F 8000000 cpp_interpret_test.cpp
// compile on Linux with:
// g++ -shared -fPIC -O2 -o upstream_area.so upstream_area.cpp


#include "cpp_interpret.cpp"


// static const int F_LT = 1;
// static const int F_LE = 2;
// static const int F_GE = 3;
// static const int F_GT = 4;
// static const int F_ADD = 5;
// static const int F_SUB = 6;
// static const int F_MUL = 7;
// static const int F_DIV = 8;
void test1() {
    int err_count = 0;
    printf("start test1\n");
    List program;
    vector<List> variables;
    vector<Function> functions;
    List result;
    bool debug = false;
    try {
        int a, b, f, expected;
        
        a = 3; b = 17; f = F_ADD; expected = a + b;
        program = {{ITEM_FCALL, f, 2}, {ITEM_INT, a, 0}, {ITEM_INT, b, 0}};
        result = run(&program[0], int(program.size()), variables, functions, debug);        
        if (result[0]._type != ITEM_INT || result[0]._value != expected) {
            printf("expected %d instead of %d\n", expected, result[0]._value);
            err_count += 1;
        }

        a = 3; b = 17; f = F_SUB; expected = a - b;
        program = {{ITEM_FCALL, f, 2}, {ITEM_INT, a, 0}, {ITEM_INT, b, 0}};
        result = run(&program[0], int(program.size()), variables, functions, debug);        
        if (result[0]._type != ITEM_INT || result[0]._value != expected) {
            printf("expected %d instead of %d\n", expected, result[0]._value);
            err_count += 1;
        }

        a = 3; b = 17; f = F_MUL; expected = a * b;
        program = {{ITEM_FCALL, f, 2}, {ITEM_INT, a, 0}, {ITEM_INT, b, 0}};
        result = run(&program[0], int(program.size()), variables, functions, debug);        
        if (result[0]._type != ITEM_INT || result[0]._value != expected) {
            printf("expected %d instead of %d\n", expected, result[0]._value);
            err_count += 1;
        }
        
        a = 17; b = 3; f = F_DIV; expected = a / b;
        program = {{ITEM_FCALL, f, 2}, {ITEM_INT, a, 0}, {ITEM_INT, b, 0}};
        result = run(&program[0], int(program.size()), variables, functions, debug);        
        if (result[0]._type != ITEM_INT || result[0]._value != expected) {
            printf("expected %d instead of %d\n", expected, result[0]._value);
            err_count += 1;
        }
              
        a = 17; b = 3; f = F_LT; expected = a < b ? 1 : 0;
        program = {{ITEM_FCALL, f, 2}, {ITEM_INT, a, 0}, {ITEM_INT, b, 0}};
        result = run(&program[0], int(program.size()), variables, functions, debug);        
        if (result[0]._type != ITEM_INT || result[0]._value != expected) {
            printf("expected %d instead of %d\n", expected, result[0]._value);
            err_count += 1;
        }

        a = 3; b = 17; f = F_LT; expected = a < b ? 1 : 0;
        program = {{ITEM_FCALL, f, 2}, {ITEM_INT, a, 0}, {ITEM_INT, b, 0}};
        result = run(&program[0], int(program.size()), variables, functions, debug);        
        if (result[0]._type != ITEM_INT || result[0]._value != expected) {
            printf("expected %d instead of %d\n", expected, result[0]._value);
            err_count += 1;
        }
        
        a = 17; b = 3; f = F_LE; expected = a <= b ? 1 : 0;
        program = {{ITEM_FCALL, f, 2}, {ITEM_INT, a, 0}, {ITEM_INT, b, 0}};
        result = run(&program[0], int(program.size()), variables, functions, debug);        
        if (result[0]._type != ITEM_INT || result[0]._value != expected) {
            printf("expected %d instead of %d\n", expected, result[0]._value);
            err_count += 1;
        }

        a = 3; b = 17; f = F_LE; expected = a <= b ? 1 : 0;
        program = {{ITEM_FCALL, f, 2}, {ITEM_INT, a, 0}, {ITEM_INT, b, 0}};
        result = run(&program[0], int(program.size()), variables, functions, debug);        
        if (result[0]._type != ITEM_INT || result[0]._value != expected) {
            printf("expected %d instead of %d\n", expected, result[0]._value);
            err_count += 1;
        }
        
        a = 17; b = 3; f = F_GE; expected = a >= b ? 1 : 0;
        program = {{ITEM_FCALL, f, 2}, {ITEM_INT, a, 0}, {ITEM_INT, b, 0}};
        result = run(&program[0], int(program.size()), variables, functions, debug);        
        if (result[0]._type != ITEM_INT || result[0]._value != expected) {
            printf("expected %d instead of %d\n", expected, result[0]._value);
            err_count += 1;
        }

        a = 3; b = 17; f = F_GE; expected = a >= b ? 1 : 0;
        program = {{ITEM_FCALL, f, 2}, {ITEM_INT, a, 0}, {ITEM_INT, b, 0}};
        result = run(&program[0], int(program.size()), variables, functions, debug);        
        if (result[0]._type != ITEM_INT || result[0]._value != expected) {
            printf("expected %d instead of %d\n", expected, result[0]._value);
            err_count += 1;
        }
        
        a = 17; b = 3; f = F_GT; expected = a > b ? 1 : 0;
        program = {{ITEM_FCALL, f, 2}, {ITEM_INT, a, 0}, {ITEM_INT, b, 0}};
        result = run(&program[0], int(program.size()), variables, functions, debug);        
        if (result[0]._type != ITEM_INT || result[0]._value != expected) {
            printf("expected %d instead of %d\n", expected, result[0]._value);
            err_count += 1;
        }

        a = 3; b = 17; f = F_GT; expected = a > b ? 1 : 0;
        program = {{ITEM_FCALL, f, 2}, {ITEM_INT, a, 0}, {ITEM_INT, b, 0}};
        result = run(&program[0], int(program.size()), variables, functions, debug);        
        if (result[0]._type != ITEM_INT || result[0]._value != expected) {
            printf("expected %d instead of %d\n", expected, result[0]._value);
            err_count += 1;
        }

        int c;
        a = 3; b = 17; c = 4; expected = a + b * c;
        program = {{ITEM_FCALL, F_ADD, 2}, {ITEM_INT, a, 0}, {ITEM_FCALL, F_MUL, 2}, {ITEM_INT, b, 0}, {ITEM_INT, c, 0}};
        result = run(&program[0], int(program.size()), variables, functions, debug);        
        if (result[0]._type != ITEM_INT || result[0]._value != expected) {
            printf("expected %d instead of %d\n", expected, result[0]._value);
            err_count += 1;
        }
        a = 3; b = 17; c = 4; expected = a * b + c;
        program = {{ITEM_FCALL, F_ADD, 2}, {ITEM_FCALL, F_MUL, 2}, {ITEM_INT, a, 0}, {ITEM_INT, b, 0}, {ITEM_INT, c, 0}};
        result = run(&program[0], int(program.size()), variables, functions, debug);        
        if (result[0]._type != ITEM_INT || result[0]._value != expected) {
            printf("expected %d instead of %d\n", expected, result[0]._value);
            err_count += 1;
        }

        a = 3; b = 17; c = 4; expected = a * b + c;
        variables = {{{ITEM_INT, a, 0}}, {{ITEM_INT, b, 0}}, {{ITEM_INT, c, 0}}};
        program = {{ITEM_FCALL, F_ADD, 2}, {ITEM_FCALL, F_MUL, 2}, {ITEM_VAR, 0, 0}, {ITEM_VAR, 1, 0}, {ITEM_VAR, 2, 0}};
        result = run(&program[0], int(program.size()), variables, functions, debug);        
        if (result[0]._type != ITEM_INT || result[0]._value != expected) {
            printf("expected %d instead of %d\n", expected, result[0]._value);
            err_count += 1;
        }
    }
    catch (const exception& e) {
        printf("exception %s\n", e.what());
    }
    printf("%d errors encountered in test1\n", err_count);
}


// static const int F_EQ = 9;
// static const int F_NE = 10;
// static const int F_AND = 11;
// static const int F_OR = 12;
// static const int F_NOT = 13;
void test2() {
    int err_count = 0;
    printf("start test1\n");
    List program;
    vector<List> variables;
    vector<Function> functions;
    List result;
    int a, b, f, expected;    
    bool debug = false;
    
    a = 17; b = 3;
    expected = 1;
    program = {{ITEM_FCALL, F_EQ, 2}, {ITEM_LIST, 0, 2}, {ITEM_INT, a, 0}, {ITEM_INT, b, 0}, {ITEM_LIST, 0, 2}, {ITEM_INT, a, 0}, {ITEM_INT, b, 0}};
    result = run(&program[0], int(program.size()), variables, functions, debug);        
    if (result[0]._type != ITEM_INT || result[0]._value != expected) {
        printf("expected %d instead of %d\n", expected, result[0]._value);
        err_count += 1;
    }
    
    expected = 0;
    program = {{ITEM_FCALL, F_EQ, 2}, {ITEM_LIST, 0, 2}, {ITEM_INT, a, 0}, {ITEM_INT, b, 0}, {ITEM_LIST, 0, 2}, {ITEM_INT, b, 0}, {ITEM_INT, b, 0}};
    result = run(&program[0], int(program.size()), variables, functions, debug);        
    if (result[0]._type != ITEM_INT || result[0]._value != expected) {
        printf("expected %d instead of %d\n", expected, result[0]._value);
        err_count += 1;
    }

    expected = 0;
    program = {{ITEM_FCALL, F_NE, 2}, {ITEM_LIST, 0, 2}, {ITEM_INT, a, 0}, {ITEM_INT, b, 0}, {ITEM_LIST, 0, 2}, {ITEM_INT, a, 0}, {ITEM_INT, b, 0}};
    result = run(&program[0], int(program.size()), variables, functions, debug);        
    if (result[0]._type != ITEM_INT || result[0]._value != expected) {
        printf("expected %d instead of %d\n", expected, result[0]._value);
        err_count += 1;
    }
    
    expected = 1;
    program = {{ITEM_FCALL, F_NE, 2}, {ITEM_LIST, 0, 2}, {ITEM_INT, a, 0}, {ITEM_INT, b, 0}, {ITEM_LIST, 0, 2}, {ITEM_INT, b, 0}, {ITEM_INT, b, 0}};
    result = run(&program[0], int(program.size()), variables, functions, debug);        
    if (result[0]._type != ITEM_INT || result[0]._value != expected) {
        printf("expected %d instead of %d\n", expected, result[0]._value);
        err_count += 1;
    }

    expected = 1;
    program = {{ITEM_FCALL, F_AND, 2}, {ITEM_LIST, 0, 2}, {ITEM_INT, a, 0}, {ITEM_INT, b, 0}, {ITEM_LIST, 0, 2}, {ITEM_INT, b, 0}, {ITEM_INT, b, 0}};
    result = run(&program[0], int(program.size()), variables, functions, debug);        
    if (result[0]._type != ITEM_INT || result[0]._value != expected) {
        printf("expected %d instead of %d\n", expected, result[0]._value);
        err_count += 1;
    }

    expected = 0;
    program = {{ITEM_FCALL, F_AND, 2}, {ITEM_LIST, 0, 0}, {ITEM_LIST, 0, 2}, {ITEM_INT, b, 0}, {ITEM_INT, b, 0}};
    result = run(&program[0], int(program.size()), variables, functions, debug);        
    if (result[0]._type != ITEM_INT || result[0]._value != expected) {
        printf("expected %d instead of %d\n", expected, result[0]._value);
        err_count += 1;
    }


    expected = 1;
    program = {{ITEM_FCALL, F_OR, 2}, {ITEM_LIST, 0, 0}, {ITEM_LIST, 0, 2}, {ITEM_INT, b, 0}, {ITEM_INT, b, 0}};
    result = run(&program[0], int(program.size()), variables, functions, debug);        
    if (result[0]._type != ITEM_INT || result[0]._value != expected) {
        printf("expected %d instead of %d\n", expected, result[0]._value);
        err_count += 1;
    }

    expected = 0;
    program = {{ITEM_FCALL, F_OR, 2}, {ITEM_LIST, 0, 0}, {ITEM_INT, 0, 0}};
    result = run(&program[0], int(program.size()), variables, functions, debug);        
    if (result[0]._type != ITEM_INT || result[0]._value != expected) {
        printf("expected %d instead of %d\n", expected, result[0]._value);
        err_count += 1;
    }

    expected = 1;
    program = {{ITEM_FCALL, F_NOT, 1}, {ITEM_LIST, 0, 0},};
    result = run(&program[0], int(program.size()), variables, functions, debug);        
    if (result[0]._type != ITEM_INT || result[0]._value != expected) {
        printf("expected %d instead of %d\n", expected, result[0]._value);
        err_count += 1;
    }

    expected = 0;
    program = {{ITEM_FCALL, F_NOT, 1}, {ITEM_LIST, 0, 1}, {ITEM_INT, b, 0}};
    result = run(&program[0], int(program.size()), variables, functions, debug);        
    if (result[0]._type != ITEM_INT || result[0]._value != expected) {
        printf("expected %d instead of %d\n", expected, result[0]._value);
        err_count += 1;
    }

    printf("%d errors encountered in test2\n", err_count);
}

// static const int F_FIRST = 14;
// static const int F_REST = 15;
// static const int F_EXTEND = 16;
// static const int F_APPEND = 17;
// static const int F_CONS = 18;
// static const int F_LEN = 19;
// static const int F_AT = 20;
// static const int F_LIST = 21;
// static const int F_LAST = 22;
void test3() {
}


// static const int F_VAR = 23;
// static const int F_ASSIGN = 24;
// static const int F_FUNCTION = 25;
void test4() {
}


// static const int F_IF = 26;
// static const int F_FOR = 27;
void test5() {
}

// static const int F_PRINT = 28;
// static const int F_ASSERT = 29;
// static const int F_EXIT = 30;
void test6() {
}

// static const int F_SUM = 31;
void test7() {
    int err_count = 0;
    printf("start test7\n");
    List program;
    vector<List> variables;
    vector<Function> functions;
    List result;
    int a, b, f, expected;    
    bool debug = false;
    
    a = 17; b = 3;
    expected = a + b;
    program = {{ITEM_FCALL, F_SUM, 1}, {ITEM_LIST, 0, 2}, {ITEM_INT, a, 0}, {ITEM_INT, b, 0}};
    result = run(&program[0], int(program.size()), variables, functions, debug);        
    if (result[0]._type != ITEM_INT || result[0]._value != expected) {
        printf("expected %d instead of %d\n", expected, result[0]._value);
        err_count += 1;
    }
    
    printf("%d errors encountered in test7\n", err_count);
}


int main(int argc, char* argv[]) {
    test1();
    test2();
    test3();
    test4();
    test5();
    test6();
    test7();
    return 0;
}