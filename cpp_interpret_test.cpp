/* compile on Windows with:
call "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build\vcvars64.bat"
Cl /nologo /EHsc /F 8000000 cpp_interpret_test.cpp
*/
/* compile on Linux with:
g++ cpp_interpret_test.cpp
*/


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
    List program;
    vector<List> variables;
    vector<Function> functions;
    List result;
    bool debug = false;
    int a, b, f, expected;
    
    a = 3; b = 17; f = F_ADD; expected = a + b;
    program = {{ITEM_FCALL, f, 2}, {ITEM_INT, a, 0}, {ITEM_INT, b, 0}};
    result = run(&program[0], int(program.size()), variables, functions, debug);        
    if (result.size() != 1 || result[0]._type != ITEM_INT || result[0]._value != expected) {
        printf("%d: expected %d instead of ", __LINE__, expected);
        print_code(result);
        err_count += 1;
    }

    a = 3; b = 17; f = F_SUB; expected = a - b;
    program = {{ITEM_FCALL, f, 2}, {ITEM_INT, a, 0}, {ITEM_INT, b, 0}};
    result = run(&program[0], int(program.size()), variables, functions, debug);        
    if (result.size() != 1 || result[0]._type != ITEM_INT || result[0]._value != expected) {
        printf("%d: expected %d instead of ", __LINE__, expected);
        print_code(result);
        err_count += 1;
    }

    a = 3; b = 17; f = F_MUL; expected = a * b;
    program = {{ITEM_FCALL, f, 2}, {ITEM_INT, a, 0}, {ITEM_INT, b, 0}};
    result = run(&program[0], int(program.size()), variables, functions, debug);        
    if (result.size() != 1 || result[0]._type != ITEM_INT || result[0]._value != expected) {
        printf("%d: expected %d instead of ", __LINE__, expected);
        print_code(result);
        err_count += 1;
    }
    
    a = 17; b = 3; f = F_DIV; expected = a / b;
    program = {{ITEM_FCALL, f, 2}, {ITEM_INT, a, 0}, {ITEM_INT, b, 0}};
    result = run(&program[0], int(program.size()), variables, functions, debug);        
    if (result.size() != 1 || result[0]._type != ITEM_INT || result[0]._value != expected) {
        printf("%d: expected %d instead of ", __LINE__, expected);
        print_code(result);
        err_count += 1;
    }
          
    a = 17; b = 3; f = F_LT; expected = a < b ? 1 : 0;
    program = {{ITEM_FCALL, f, 2}, {ITEM_INT, a, 0}, {ITEM_INT, b, 0}};
    result = run(&program[0], int(program.size()), variables, functions, debug);        
    if (result.size() != 1 || result[0]._type != ITEM_INT || result[0]._value != expected) {
        printf("%d: expected %d instead of ", __LINE__, expected);
        print_code(result);
        err_count += 1;
    }

    a = 3; b = 17; f = F_LT; expected = a < b ? 1 : 0;
    program = {{ITEM_FCALL, f, 2}, {ITEM_INT, a, 0}, {ITEM_INT, b, 0}};
    result = run(&program[0], int(program.size()), variables, functions, debug);        
    if (result.size() != 1 || result[0]._type != ITEM_INT || result[0]._value != expected) {
        printf("%d: expected %d instead of ", __LINE__, expected);
        print_code(result);
        err_count += 1;
    }
    
    a = 17; b = 3; f = F_LE; expected = a <= b ? 1 : 0;
    program = {{ITEM_FCALL, f, 2}, {ITEM_INT, a, 0}, {ITEM_INT, b, 0}};
    result = run(&program[0], int(program.size()), variables, functions, debug);        
    if (result.size() != 1 || result[0]._type != ITEM_INT || result[0]._value != expected) {
        printf("%d: expected %d instead of ", __LINE__, expected);
        print_code(result);
        err_count += 1;
    }

    a = 3; b = 17; f = F_LE; expected = a <= b ? 1 : 0;
    program = {{ITEM_FCALL, f, 2}, {ITEM_INT, a, 0}, {ITEM_INT, b, 0}};
    result = run(&program[0], int(program.size()), variables, functions, debug);        
    if (result.size() != 1 || result[0]._type != ITEM_INT || result[0]._value != expected) {
        printf("%d: expected %d instead of ", __LINE__, expected);
        print_code(result);
        err_count += 1;
    }
    
    a = 17; b = 3; f = F_GE; expected = a >= b ? 1 : 0;
    program = {{ITEM_FCALL, f, 2}, {ITEM_INT, a, 0}, {ITEM_INT, b, 0}};
    result = run(&program[0], int(program.size()), variables, functions, debug);        
    if (result.size() != 1 || result[0]._type != ITEM_INT || result[0]._value != expected) {
        printf("%d: expected %d instead of ", __LINE__, expected);
        print_code(result);
        err_count += 1;
    }

    a = 3; b = 17; f = F_GE; expected = a >= b ? 1 : 0;
    program = {{ITEM_FCALL, f, 2}, {ITEM_INT, a, 0}, {ITEM_INT, b, 0}};
    result = run(&program[0], int(program.size()), variables, functions, debug);        
    if (result.size() != 1 || result[0]._type != ITEM_INT || result[0]._value != expected) {
        printf("%d: expected %d instead of ", __LINE__, expected);
        print_code(result);
        err_count += 1;
    }
    
    a = 17; b = 3; f = F_GT; expected = a > b ? 1 : 0;
    program = {{ITEM_FCALL, f, 2}, {ITEM_INT, a, 0}, {ITEM_INT, b, 0}};
    result = run(&program[0], int(program.size()), variables, functions, debug);        
    if (result.size() != 1 || result[0]._type != ITEM_INT || result[0]._value != expected) {
        printf("%d: expected %d instead of ", __LINE__, expected);
        print_code(result);
        err_count += 1;
    }

    a = 3; b = 17; f = F_GT; expected = a > b ? 1 : 0;
    program = {{ITEM_FCALL, f, 2}, {ITEM_INT, a, 0}, {ITEM_INT, b, 0}};
    result = run(&program[0], int(program.size()), variables, functions, debug);        
    if (result.size() != 1 || result[0]._type != ITEM_INT || result[0]._value != expected) {
        printf("%d: expected %d instead of ", __LINE__, expected);
        print_code(result);
        err_count += 1;
    }

    int c;
    a = 3; b = 17; c = 4; expected = a + b * c;
    program = {{ITEM_FCALL, F_ADD, 2}, {ITEM_INT, a, 0}, {ITEM_FCALL, F_MUL, 2}, {ITEM_INT, b, 0}, {ITEM_INT, c, 0}};
    result = run(&program[0], int(program.size()), variables, functions, debug);        
    if (result.size() != 1 || result[0]._type != ITEM_INT || result[0]._value != expected) {
        printf("%d: expected %d instead of ", __LINE__, expected);
        print_code(result);
        err_count += 1;
    }
    a = 3; b = 17; c = 4; expected = a * b + c;
    program = {{ITEM_FCALL, F_ADD, 2}, {ITEM_FCALL, F_MUL, 2}, {ITEM_INT, a, 0}, {ITEM_INT, b, 0}, {ITEM_INT, c, 0}};
    result = run(&program[0], int(program.size()), variables, functions, debug);        
    if (result.size() != 1 || result[0]._type != ITEM_INT || result[0]._value != expected) {
        printf("%d: expected %d instead of ", __LINE__, expected);
        print_code(result);
        err_count += 1;
    }

    a = 3; b = 17; c = 4; expected = a * b + c;
    variables = {{{ITEM_INT, a, 0}}, {{ITEM_INT, b, 0}}, {{ITEM_INT, c, 0}}};
    program = {{ITEM_FCALL, F_ADD, 2}, {ITEM_FCALL, F_MUL, 2}, {ITEM_VAR, 0, 0}, {ITEM_VAR, 1, 0}, {ITEM_VAR, 2, 0}};
    result = run(&program[0], int(program.size()), variables, functions, debug);        
    if (result.size() != 1 || result[0]._type != ITEM_INT || result[0]._value != expected) {
        printf("%d: expected %d instead of ", __LINE__, expected);
        print_code(result);
        err_count += 1;
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
    List program;
    vector<List> variables;
    vector<Function> functions;
    List result;
    int a, b, expected;    
    bool debug = false;
    
    a = 17; b = 3;
    expected = 1;
    program = {{ITEM_FCALL, F_EQ, 2}, {ITEM_LIST, 0, 2}, {ITEM_INT, a, 0}, {ITEM_INT, b, 0}, {ITEM_LIST, 0, 2}, {ITEM_INT, a, 0}, {ITEM_INT, b, 0}};
    result = run(&program[0], int(program.size()), variables, functions, debug);        
    if (result.size() != 1 || result[0]._type != ITEM_INT || result[0]._value != expected) {
			printf("%d: expected %d instead of ", __LINE__, expected);
			print_code(result);
        err_count += 1;
    }
    
    expected = 0;
    program = {{ITEM_FCALL, F_EQ, 2}, {ITEM_LIST, 0, 2}, {ITEM_INT, a, 0}, {ITEM_INT, b, 0}, {ITEM_LIST, 0, 2}, {ITEM_INT, b, 0}, {ITEM_INT, b, 0}};
    result = run(&program[0], int(program.size()), variables, functions, debug);        
    if (result.size() != 1 || result[0]._type != ITEM_INT || result[0]._value != expected) {
		printf("%d: expected %d instead of ", __LINE__, expected);
		print_code(result);
        err_count += 1;
    }

    expected = 0;
    program = {{ITEM_FCALL, F_NE, 2}, {ITEM_LIST, 0, 2}, {ITEM_INT, a, 0}, {ITEM_INT, b, 0}, {ITEM_LIST, 0, 2}, {ITEM_INT, a, 0}, {ITEM_INT, b, 0}};
    result = run(&program[0], int(program.size()), variables, functions, debug);        
    if (result.size() != 1 || result[0]._type != ITEM_INT || result[0]._value != expected) {
		printf("%d: expected %d instead of ", __LINE__, expected);
		print_code(result);
        err_count += 1;
    }
    
    expected = 1;
    program = {{ITEM_FCALL, F_NE, 2}, {ITEM_LIST, 0, 2}, {ITEM_INT, a, 0}, {ITEM_INT, b, 0}, {ITEM_LIST, 0, 2}, {ITEM_INT, b, 0}, {ITEM_INT, b, 0}};
    result = run(&program[0], int(program.size()), variables, functions, debug);        
    if (result.size() != 1 || result[0]._type != ITEM_INT || result[0]._value != expected) {
		printf("%d: expected %d instead of ", __LINE__, expected);
		print_code(result);
        err_count += 1;
    }

    expected = 1;
    program = {{ITEM_FCALL, F_AND, 2}, {ITEM_LIST, 0, 2}, {ITEM_INT, a, 0}, {ITEM_INT, b, 0}, {ITEM_LIST, 0, 2}, {ITEM_INT, b, 0}, {ITEM_INT, b, 0}};
    result = run(&program[0], int(program.size()), variables, functions, debug);        
    if (result.size() != 1 || result[0]._type != ITEM_INT || result[0]._value != expected) {
		printf("%d: expected %d instead of ", __LINE__, expected);
		print_code(result);
        err_count += 1;
    }

    expected = 0;
    program = {{ITEM_FCALL, F_AND, 2}, {ITEM_LIST, 0, 0}, {ITEM_LIST, 0, 2}, {ITEM_INT, b, 0}, {ITEM_INT, b, 0}};
    result = run(&program[0], int(program.size()), variables, functions, debug);        
    if (result.size() != 1 || result[0]._type != ITEM_INT || result[0]._value != expected) {
		printf("%d: expected %d instead of ", __LINE__, expected);
		print_code(result);
        err_count += 1;
    }


    expected = 1;
    program = {{ITEM_FCALL, F_OR, 2}, {ITEM_LIST, 0, 0}, {ITEM_LIST, 0, 2}, {ITEM_INT, b, 0}, {ITEM_INT, b, 0}};
    result = run(&program[0], int(program.size()), variables, functions, debug);        
    if (result.size() != 1 || result[0]._type != ITEM_INT || result[0]._value != expected) {
		printf("%d: expected %d instead of ", __LINE__, expected);
		print_code(result);
        err_count += 1;
    }

    expected = 0;
    program = {{ITEM_FCALL, F_OR, 2}, {ITEM_LIST, 0, 0}, {ITEM_INT, 0, 0}};
    result = run(&program[0], int(program.size()), variables, functions, debug);        
    if (result.size() != 1 || result[0]._type != ITEM_INT || result[0]._value != expected) {
		printf("%d: expected %d instead of ", __LINE__, expected);
		print_code(result);
        err_count += 1;
    }

    expected = 1;
    program = {{ITEM_FCALL, F_NOT, 1}, {ITEM_LIST, 0, 0},};
    result = run(&program[0], int(program.size()), variables, functions, debug);        
    if (result.size() != 1 || result[0]._type != ITEM_INT || result[0]._value != expected) {
		printf("%d: expected %d instead of ", __LINE__, expected);
		print_code(result);
        err_count += 1;
    }

    expected = 0;
    program = {{ITEM_FCALL, F_NOT, 1}, {ITEM_LIST, 0, 1}, {ITEM_INT, b, 0}};
    result = run(&program[0], int(program.size()), variables, functions, debug);        
    if (result.size() != 1 || result[0]._type != ITEM_INT || result[0]._value != expected) {
		printf("%d: expected %d instead of ", __LINE__, expected);
		print_code(result);
        err_count += 1;
    }

    printf("%d errors encountered in test2\n", err_count);
}

// static const int F_FIRST = 14;
// static const int F_REST = 15;
// static const int F_EXTEND = 16;
// static const int F_APPEND = 17;
// static const int F_CONS = 18;
void test3a() {
    int err_count = 0;
    List program;
    vector<List> variables;
    vector<Function> functions;
    List result;
    int a, b, expected;    
    bool debug = false;

    a = 17; b = 3;
    expected = 1;
    program = {{ITEM_FCALL, F_EQ, 2}, {ITEM_FCALL, F_FIRST, 1}, {ITEM_LIST, 0, 2}, {ITEM_INT, a, 0}, {ITEM_INT, b, 0}, {ITEM_INT, a, 0}};
    result = run(&program[0], int(program.size()), variables, functions, debug);        
    if (result.size() != 1 || result[0]._type != ITEM_INT || result[0]._value != expected) {
		printf("%d: expected %d instead of ", __LINE__, expected);
		print_code(result);
        err_count += 1;
    }

    expected = 1;
    program = {{ITEM_FCALL, F_EQ, 2},
        {ITEM_FCALL, F_REST, 1}, {ITEM_LIST, 0, 2}, {ITEM_INT, a, 0}, {ITEM_INT, b, 0},
        {ITEM_LIST, 0, 1}, {ITEM_INT, b, 0}};
    result = run(&program[0], int(program.size()), variables, functions, debug);        
    if (result.size() != 1 || result[0]._type != ITEM_INT || result[0]._value != expected) {
		printf("%d: expected %d instead of ", __LINE__, expected);
		print_code(result);
        err_count += 1;
    }
    
    expected = 1;
    program = {{ITEM_FCALL, F_EQ, 2},
        {ITEM_FCALL, F_EXTEND, 2}, {ITEM_LIST, 0, 2}, {ITEM_INT, a, 0}, {ITEM_INT, b, 0}, {ITEM_LIST, 0, 1}, {ITEM_INT, b, 0},
        {ITEM_LIST, 0, 3}, {ITEM_INT, a, 0}, {ITEM_INT, b, 0}, {ITEM_INT, b, 0},
        };
    result = run(&program[0], int(program.size()), variables, functions, debug);        
    if (result.size() != 1 || result[0]._type != ITEM_INT || result[0]._value != expected) {
		printf("%d: expected %d instead of ", __LINE__, expected);
		print_code(result);
        err_count += 1;
    }
    
    expected = 1;
    program = {{ITEM_FCALL, F_EQ, 2},
        {ITEM_FCALL, F_APPEND, 2}, {ITEM_LIST, 0, 2}, {ITEM_INT, a, 0}, {ITEM_INT, b, 0}, {ITEM_LIST, 0, 1}, {ITEM_INT, b, 0},
        {ITEM_LIST, 0, 3}, {ITEM_INT, a, 0}, {ITEM_INT, b, 0}, {ITEM_LIST, 0, 1}, {ITEM_INT, b, 0},
        };
    result = run(&program[0], int(program.size()), variables, functions, debug);        
    if (result.size() != 1 || result[0]._type != ITEM_INT || result[0]._value != expected) {
		printf("%d: expected %d instead of ", __LINE__, expected);
		print_code(result);
        err_count += 1;
    }
    
    expected = 1;
    program = {{ITEM_FCALL, F_EQ, 2},
        {ITEM_FCALL, F_CONS, 2}, {ITEM_INT, a, 0}, {ITEM_LIST, 0, 1}, {ITEM_INT, b, 0}, 
        {ITEM_LIST, 0, 2}, {ITEM_INT, a, 0}, {ITEM_INT, b, 0},
        };
    result = run(&program[0], int(program.size()), variables, functions, debug);        
    if (result.size() != 1 || result[0]._type != ITEM_INT || result[0]._value != expected) {
		printf("%d: expected %d instead of ", __LINE__, expected);
		print_code(result);
        print_code(result);
        err_count += 1;
    }
    
    printf("%d errors encountered in test3a\n", err_count);
}


// static const int F_LEN = 19;
// static const int F_AT = 20;
// static const int F_LIST = 21;
// static const int F_LAST = 22;
void test3b() {
    int err_count = 0;
    List program;
    vector<List> variables;
    vector<Function> functions;
    List result;
    int a, b, expected;    
    bool debug = false;

    a = 17; b = 31;
    expected = 1;
    program = {{ITEM_FCALL, F_EQ, 2},
        {ITEM_FCALL, F_LEN, 1}, {ITEM_LIST, 0, 3}, {ITEM_INT, a, 0}, {ITEM_INT, b, 0}, {ITEM_INT, a+b, 0},
        {ITEM_INT, 3, 0}};
    result = run(&program[0], int(program.size()), variables, functions, debug);        
    if (result.size() == 0 || result[0]._type != ITEM_INT || result[0]._value != expected) {
        printf("%d: expected %d instead of ", __LINE__, expected);
        print_code(result);
        err_count += 1;
    }

    a = 17; b = 31;
    expected = 1;
    program = {{ITEM_FCALL, F_EQ, 2},
        {ITEM_FCALL, F_AT, 2}, {ITEM_LIST, 0, 2}, {ITEM_INT, a, 0}, {ITEM_INT, b, 0}, {ITEM_INT, 1, 0},
        {ITEM_INT, b, 0}};
    result = run(&program[0], int(program.size()), variables, functions, debug);        
    if (result.size() != 1 || result[0]._type != ITEM_INT || result[0]._value != expected) {
        printf("%d: expected %d instead of ", __LINE__, expected);
        print_code(result);
        err_count += 1;
    }

    expected = 1;
    program = {{ITEM_FCALL, F_EQ, 2},
        {ITEM_FCALL, F_AT, 2}, {ITEM_LIST, 0, 2},
            {ITEM_LIST, 0, 2}, {ITEM_INT, 100, 0}, {ITEM_INT, 101, 0},
            {ITEM_LIST, 0, 2}, {ITEM_INT, 110, 0}, {ITEM_INT, 111, 0},
            {ITEM_INT, 1, 0},
        {ITEM_LIST, 0, 2}, {ITEM_INT, 110, 0}, {ITEM_INT, 111, 0}};
    result = run(&program[0], int(program.size()), variables, functions, debug);        
    if (result.size() != 1 || result[0]._type != ITEM_INT || result[0]._value != expected) {
        printf("%d: expected %d instead of ", __LINE__, expected);
        print_code(result);
        err_count += 1;
    }
    
    expected = 1;
    program = {{ITEM_FCALL, F_EQ, 2},
        {ITEM_FCALL, F_AT, 3}, {ITEM_LIST, 0, 2},
            {ITEM_LIST, 0, 2}, {ITEM_INT, 100, 0}, {ITEM_INT, 101, 0},
            {ITEM_LIST, 0, 2}, {ITEM_INT, 110, 0}, {ITEM_INT, 111, 0},
            {ITEM_INT, 1, 0},
            {ITEM_INT, 1, 0},
        {ITEM_INT, 111, 0}};
    result = run(&program[0], int(program.size()), variables, functions, debug);        
    if (result.size() != 1 || result[0]._type != ITEM_INT || result[0]._value != expected) {
        printf("%d: expected %d instead of ", __LINE__, expected);
        print_code(result);
        err_count += 1;
    }

    expected = 1;
    program = {{ITEM_FCALL, F_EQ, 2},
        {ITEM_FCALL, F_LIST, 3}, {ITEM_INT, 100, 0}, {ITEM_INT, 101, 0}, {ITEM_LIST, 0, 1}, {ITEM_INT, 110, 0},
        {ITEM_LIST, 0, 3}, {ITEM_INT, 100, 0}, {ITEM_INT, 101, 0}, {ITEM_LIST, 0, 1}, {ITEM_INT, 110, 0}};
    result = run(&program[0], int(program.size()), variables, functions, debug);        
    if (result.size() != 1 || result[0]._type != ITEM_INT || result[0]._value != expected) {
        printf("%d: expected %d instead of ", __LINE__, expected);
        print_code(result);
        err_count += 1;
    }

    expected = 1;
    program = {{ITEM_FCALL, F_EQ, 2},
        {ITEM_FCALL, F_LAST, 3},
            {ITEM_LIST, 0, 2},
                {ITEM_LIST, 0, 2}, {ITEM_INT, 100, 0}, {ITEM_INT, 101, 0},
                {ITEM_LIST, 0, 2}, {ITEM_INT, 110, 0}, {ITEM_INT, 111, 0},
            {ITEM_INT, 201, 0},
            {ITEM_INT, 301, 0},
        {ITEM_INT, 301, 0}};
    result = run(&program[0], int(program.size()), variables, functions, debug);        
    if (result.size() != 1 || result[0]._type != ITEM_INT || result[0]._value != expected) {
        printf("%d: expected %d instead of ", __LINE__, expected);
        print_code(result);
        err_count += 1;
    }

    printf("%d errors encountered in test3b\n", err_count);
}


// static const int F_VAR = 23;
// static const int F_ASSIGN = 24;
void test4a() {
    int err_count = 0;
    List program;
    vector<List> variables;
    vector<Function> functions;
    List result;
    int a, b, c, expected;    
    bool debug = false;

    variables.resize(1);
    a = 17; b = 3; c = 42;
    expected = a*a + c;
    program = {{ITEM_FCALL, F_SUM, 1},
        {ITEM_LIST, 0, 2},
            {ITEM_FCALL, F_VAR, 3}, {ITEM_VAR, 0, 0}, {ITEM_INT, a, 0}, {ITEM_FCALL, F_MUL, 2}, {ITEM_VAR, 0, 0}, {ITEM_VAR, 0, 0},
            {ITEM_INT, c, 0}};
    if (debug) {
        printf("%d run ...\n", __LINE__);
    }
    result = run(&program[0], int(program.size()), variables, functions, debug);        
    if (result.size() != 1 || result[0]._type != ITEM_INT || result[0]._value != expected) {
        printf("%d: expected %d instead of ", __LINE__, expected);
        print_code(result);
        err_count += 1;
    } else if (debug) {
        printf("%d ok\n", __LINE__);
    }
    
    a = 17; b = 3; c = 42;
    expected = b*b + c;
    program = {
        {ITEM_FCALL, F_SUM, 1},
            {ITEM_FCALL, F_LAST, 2},
                {ITEM_FCALL, F_ASSIGN, 2}, {ITEM_VAR, 0, 0}, {ITEM_INT, b, 0},
                {ITEM_LIST, 0, 1},
                    {ITEM_FCALL, F_ADD, 2}, {ITEM_FCALL, F_MUL, 2}, {ITEM_VAR, 0, 0}, {ITEM_VAR, 0, 0}, {ITEM_INT, c, 0}};
    if (debug) {
        printf("%d run ...\n", __LINE__);
    }
    result = run(&program[0], int(program.size()), variables, functions, debug);        
    if (result.size() != 1 || result[0]._type != ITEM_INT || result[0]._value != expected) {
        printf("%d: expected %d instead of ", __LINE__, expected);
        print_code(result);
        err_count += 1;
    } else if (debug) {
        printf("%d ok\n", __LINE__);
    }
    
    
    printf("%d errors encountered in test4a\n", err_count);
}

// static const int F_FUNCTION = 25;
void test4b() {
    int err_count = 0;
    List program;
    vector<List> variables;
    vector<Function> functions;
    List result;
    int expected;    
    bool debug = false;

    functions = {{1, 0, {
        {ITEM_FCALL, F_IF, 3},
            {ITEM_FCALL, F_LE, 2}, {ITEM_VAR, 0, 0}, {ITEM_INT, 1, 0},
            {ITEM_INT, 1, 0},
            {ITEM_FCALL, F_MUL, 2},
                {ITEM_VAR, 0, 0},
                {ITEM_FUSERCALL, 0, 1}, {ITEM_FCALL, F_SUB, 2}, {ITEM_VAR, 0, 0}, {ITEM_INT, 1, 0}}}};    
    expected = 1*2*3;
    program = {{ITEM_FUSERCALL, 0, 1}, {ITEM_INT, 3, 0}};
    if (debug) {
        printf("%d run ...\n", __LINE__);
    }
    result = run(&program[0], int(program.size()), variables, functions, debug);        
    if (result.size() != 1 || result[0]._type != ITEM_INT || result[0]._value != expected) {
        printf("%d: expected %d instead of ", __LINE__, expected);
        print_code(result);
        err_count += 1;
    } else if (debug) {
        printf("%d ok\n", __LINE__);
    }
    
    functions.clear();
    expected = 1*2*3;
    program = {
        {ITEM_FCALL, F_LAST, 2},
            {ITEM_FCALL, F_FUNCTION, 4},
                {ITEM_INT, 0, 0},
                {ITEM_INT, 1, 0},
                {ITEM_INT, 0, 0},
                {ITEM_FCALL, F_IF, 3},
                    {ITEM_FCALL, F_LE, 2}, {ITEM_VAR, 0, 0}, {ITEM_INT, 1, 0},
                    {ITEM_INT, 1, 0},
                    {ITEM_FCALL, F_MUL, 2},
                        {ITEM_VAR, 0, 0},
                        {ITEM_FUSERCALL, 0, 1}, {ITEM_FCALL, F_SUB, 2}, {ITEM_VAR, 0, 0}, {ITEM_INT, 1, 0},
            {ITEM_FUSERCALL, 0, 1}, {ITEM_INT, 3, 0}};
    if (debug) {
        printf("%d run ...\n", __LINE__);
    }
    result = run(&program[0], int(program.size()), variables, functions, debug);        
    if (result.size() != 1 || result[0]._type != ITEM_INT || result[0]._value != expected) {
        printf("%d: expected %d instead of ", __LINE__, expected);
        print_code(result);
        err_count += 1;
    } else if (debug) {
        printf("%d ok\n", __LINE__);
    }
        
    printf("%d errors encountered in test4b\n", err_count);
}


// static const int F_IF = 26;
// static const int F_FOR = 27;
void test5() {
    int err_count = 0;
    List program;
    vector<List> variables;
    vector<Function> functions;
    List result;
    int a, b, expected;    
    bool debug = false;

    a = 17; b = 3;
    int c = 42;
    expected = a + c;
    program = {{ITEM_FCALL, F_SUM, 1}, {ITEM_LIST, 0, 2},
        {ITEM_FCALL, F_IF, 3}, {ITEM_INT, 1, 0}, {ITEM_INT, a, 0}, {ITEM_INT, b, 0},
        {ITEM_INT, c, 0}};
    if (debug) {
        printf("%d run ...\n", __LINE__);
    }
    result = run(&program[0], int(program.size()), variables, functions, debug);        
    if (result.size() != 1 || result[0]._type != ITEM_INT || result[0]._value != expected) {
        printf("%d: expected %d instead of ", __LINE__, expected);
        print_code(result);
        err_count += 1;
    } else if (debug) {
        printf("%d ok\n", __LINE__);
    }
    
    expected = b + c;
    program = {{ITEM_FCALL, F_SUM, 1}, {ITEM_LIST, 0, 2},
        {ITEM_FCALL, F_IF, 3}, {ITEM_INT, 0, 0}, {ITEM_INT, a, 0}, {ITEM_INT, b, 0},
        {ITEM_INT, c, 0}};
    if (debug) {
        printf("%d run ...\n", __LINE__);
    }
    result = run(&program[0], int(program.size()), variables, functions, debug);        
    if (result.size() != 1 || result[0]._type != ITEM_INT || result[0]._value != expected) {
        printf("%d: expected %d instead of ", __LINE__, expected);
        print_code(result);
        err_count += 1;
    } else if (debug) {
        printf("%d ok\n", __LINE__);
    }
    
    variables.resize(1);
    expected = a*b + c;
    program = {
        {ITEM_FCALL, F_SUM, 1},
            {ITEM_FCALL, F_APPEND, 2},
                {ITEM_FCALL, F_FOR, 3}, {ITEM_VAR, 0, 0}, {ITEM_INT, a, 0}, {ITEM_INT, b, 0},
                {ITEM_INT, c, 0}};
    if (debug) {
        printf("%d run ...\n", __LINE__);
    }
    result = run(&program[0], int(program.size()), variables, functions, debug);        
    if (result.size() != 1 || result[0]._type != ITEM_INT || result[0]._value != expected) {
        printf("%d: expected %d instead of ", __LINE__, expected);
        print_code(result);
        err_count += 1;
    } else if (debug) {
        printf("%d ok\n", __LINE__);
    }
    
    a = 5;
    expected = (a+1)*a/2 + c;
    program = {
        {ITEM_FCALL, F_SUM, 1},
            {ITEM_FCALL, F_APPEND, 2},
                {ITEM_FCALL, F_FOR, 3}, {ITEM_VAR, 0, 0}, {ITEM_FCALL, F_ADD, 2}, {ITEM_INT, a, 0}, {ITEM_INT, 1, 0}, {ITEM_VAR, 0, 0},
                {ITEM_INT, c, 0}};
    if (debug) {
        printf("%d run ...\n", __LINE__);
    }
    result = run(&program[0], int(program.size()), variables, functions, debug);        
    if (result.size() != 1 || result[0]._type != ITEM_INT || result[0]._value != expected) {
        printf("%d: expected %d instead of ", __LINE__, expected);
        print_code(result);
        err_count += 1;
    } else if (debug) {
        printf("%d ok\n", __LINE__);
    }

    expected = a*a + b*b + c;
    program = {
        {ITEM_FCALL, F_SUM, 1},
            {ITEM_FCALL, F_APPEND, 2},
                {ITEM_FCALL, F_FOR, 3}, {ITEM_VAR, 0, 0}, {ITEM_LIST, 0, 2}, {ITEM_INT, a, 0}, {ITEM_INT, b, 0}, {ITEM_FCALL, F_MUL, 2}, {ITEM_VAR, 0, 0}, {ITEM_VAR, 0, 0},
                {ITEM_INT, c, 0}};
    if (debug) {
        printf("%d run ...\n", __LINE__);
    }
    result = run(&program[0], int(program.size()), variables, functions, debug);        
    if (result.size() != 1 || result[0]._type != ITEM_INT || result[0]._value != expected) {
        printf("%d: expected %d instead of ", __LINE__, expected);
        print_code(result);
        err_count += 1;
    } else if (debug) {
        printf("%d ok\n", __LINE__);
    }

    printf("%d errors encountered in test5\n", err_count);
}

// static const int F_PRINT = 28;
// static const int F_ASSERT = 29;
// static const int F_EXIT = 30;
void test6() {
    int err_count = 0;
    List program;
    vector<List> variables;
    vector<Function> functions;
    List result;
    int a, b, expected;    
    bool debug = false;

    if (false) { // test print
        a = 17; b = 3;
        expected = a + b;
        printf("test6 will print '20' as side effect of testing the print command\n");
        program = {{ITEM_FCALL, F_PRINT, 1}, {ITEM_FCALL, F_SUM, 1}, {ITEM_LIST, 0, 2}, {ITEM_INT, a, 0}, {ITEM_INT, b, 0}};
        result = run(&program[0], int(program.size()), variables, functions, debug);        
        if (result.size() != 1 || result[0]._type != ITEM_INT || result[0]._value != expected) {
			printf("%d: expected %d instead of ", __LINE__, expected);
			print_code(result);
            err_count += 1;
        }
    }
    
    a = 17; b = 3;
    expected = 1;
    program = {{ITEM_FCALL, F_ASSERT, 1}, {ITEM_FCALL, F_EQ, 2}, {ITEM_FCALL, F_SUM, 1}, {ITEM_LIST, 0, 2}, {ITEM_INT, a, 0}, {ITEM_INT, b, 0}, {ITEM_INT, a+b, 0}};
    result = run(&program[0], int(program.size()), variables, functions, debug);        
    if (result.size() != 1 || result[0]._type != ITEM_INT || result[0]._value != expected) {
		printf("%d: expected %d instead of ", __LINE__, expected);
		print_code(result);
        err_count += 1;
    }
    
    if (false) { // test assertion failure
        a = 17; b = 3;
        expected = 0;
        printf("test6 will give an assertion failure as side effect of testing the assert command\n");
        program = {{ITEM_FCALL, F_ASSERT, 1}, {ITEM_FCALL, F_EQ, 2}, {ITEM_FCALL, F_SUM, 1}, {ITEM_LIST, 0, 2}, {ITEM_INT, a, 0}, {ITEM_INT, b, 0}, {ITEM_INT, a+b+1, 0}};
        result = run(&program[0], int(program.size()), variables, functions, debug);        
        if (result.size() != 1 || result[0]._type != ITEM_INT || result[0]._value != expected) {
			printf("%d: expected %d instead of ", __LINE__, expected);
			print_code(result);
            err_count += 1;
        }
    }

    if (false) { // test exit
        program = {{ITEM_FCALL, F_EXIT, 0}};
        result = run(&program[0], int(program.size()), variables, functions, debug);        
        printf("expected exit\n");
        err_count += 1;
    }

    printf("%d errors encountered in test6\n", err_count);
}

// static const int F_SUM = 31;
void test7() {
    int err_count = 0;
    List program;
    vector<List> variables;
    vector<Function> functions;
    List result;
    int a, b, expected;    
    bool debug = false;
    
    a = 17; b = 3;
    expected = a + b;
    program = {{ITEM_FCALL, F_SUM, 1}, {ITEM_LIST, 0, 2}, {ITEM_INT, a, 0}, {ITEM_INT, b, 0}};
    result = run(&program[0], int(program.size()), variables, functions, debug);        
    if (result.size() != 1 || result[0]._type != ITEM_INT || result[0]._value != expected) {
		printf("%d: expected %d instead of ", __LINE__, expected);
		print_code(result);
        err_count += 1;
    }
    
    printf("%d errors encountered in test7\n", err_count);
}


int main(int argc, char* argv[]) {
    try {
        test1();
        test2();
        test3a();
        test3b();
        test4a();
        test4b();
        test5();
        test6();
        test7();
    }
    catch (const exception& e) {
        printf("exception %s\n", e.what());
    }
    catch (...) {
        printf("unknown exception\n");
    }
    printf("all tests completed\n");
    return 0;
}