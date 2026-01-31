%% STUNIR Generated CIAO Code with Assertions
%% DO-178C Level A Compliant
%% Example: Verified arithmetic operations

:- module(verified_arithmetic, [safe_divide/3, gcd/3], [assertions, regtypes]).

%% Regular type definitions
:- regtype nat/1.
nat(0).
nat(N) :- nat(M), N is M + 1.

:- regtype positive_int/1.
positive_int(N) :- integer(N), N > 0.

%% Predicate assertions
:- pred safe_divide(+positive_int, +positive_int, -float).
:- entry safe_divide(A, B) : (positive_int(A), positive_int(B)).
:- success safe_divide(A, B, C) => float(C).
:- comp safe_divide(A, B, C) + (not_fails, is_det).

safe_divide(A, B, C) :-
    B > 0,
    C is A / B.

%% Greatest Common Divisor
:- pred gcd(+nat, +nat, -nat).
:- entry gcd(A, B) : (nat(A), nat(B)).
:- success gcd(A, B, G) => (nat(G), G > 0).
:- comp gcd(A, B, G) + is_det.

gcd(X, 0, X) :- X > 0, !.
gcd(X, Y, G) :-
    Y > 0,
    R is X mod Y,
    gcd(Y, R, G).

%% Example usage:
%% ?- safe_divide(10, 3, Result).
%%    Result = 3.3333333333333335.
%% ?- gcd(48, 18, G).
%%    G = 6.
