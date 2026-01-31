%% STUNIR Generated SWI_PROLOG Code
%% DO-178C Level A Compliant
%% Example: Mathematical predicates

:- module(math_predicates, [add/3, multiply/3, factorial/2]).

%% Add two integers
add(X, Y, Result) :-
    Result is X + Y.

%% Multiply two integers
multiply(X, Y, Result) :-
    Result is X * Y.

%% Calculate factorial of N
factorial(0, 1) :- !.
factorial(N, Result) :-
    N > 0,
    N1 is N - 1,
    factorial(N1, R1),
    Result is N * R1.

%% Example usage:
%% ?- add(5, 3, R).
%%    R = 8.
%% ?- factorial(5, R).
%%    R = 120.
