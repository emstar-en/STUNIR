%% STUNIR Generated YAP Code with Tabling
%% DO-178C Level A Compliant
%% Example: Efficient Fibonacci computation

:- module(fibonacci_tabled, [fibonacci/2, fibonacci_list/2]).

:- use_module(library(clpfd)).

%% Tabled Fibonacci (efficient for repeated queries)
:- table fibonacci/2.

fibonacci(0, 0) :- !.
fibonacci(1, 1) :- !.
fibonacci(N, F) :-
    N > 1,
    N1 is N - 1,
    N2 is N - 2,
    fibonacci(N1, F1),
    fibonacci(N2, F2),
    F is F1 + F2.

%% Generate Fibonacci list
fibonacci_list(Max, List) :-
    findall(F, (between(0, Max, N), fibonacci(N, F)), List).

%% Fibonacci with CLP(FD) constraints
fibonacci_constrained(N, F) :-
    N #>= 0,
    F #>= 0,
    fibonacci(N, F).

%% Example usage:
%% ?- fibonacci(10, F).
%%    F = 55.
%% ?- fibonacci_list(10, List).
%%    List = [0,1,1,2,3,5,8,13,21,34,55].
