call "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build\vcvars64.bat"
Cl /nologo /EHsc /F 8000000 cpp_interpret_test.cpp || ( pause && exit /B )
cpp_interpret_test
pause