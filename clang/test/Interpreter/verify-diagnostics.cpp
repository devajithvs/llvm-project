// RUN: cat %s | %clang-repl -Xcc -Xclang -Xcc -verify

auto x = unknown_val;
// expected-error@input_line_3:1 {{use of undeclared identifier 'unknown_val'}}

int get_int() { return 42; }
char* ptr = get_int();
// expected-error@input_line_7:1 {{cannot initialize a variable of type 'char *' with an rvalue of type 'int'}}

// Verify without input_line_*
int y = ;
// expected-error {{expected expression}}
