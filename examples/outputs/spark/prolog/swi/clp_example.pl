%% STUNIR Generated SWI_PROLOG Code with CLP(FD)
%% DO-178C Level A Compliant
%% Example: Constraint Logic Programming

:- module(clp_example, [send_more_money/1, nqueens/2]).

:- use_module(library(clpfd)).

%% SEND + MORE = MONEY puzzle
send_more_money([S,E,N,D,M,O,R,Y]) :-
    Vars = [S,E,N,D,M,O,R,Y],
    Vars ins 0..9,
    all_different(Vars),
    S #\= 0,
    M #\= 0,
             1000*S + 100*E + 10*N + D
           + 1000*M + 100*O + 10*R + E
    #= 10000*M + 1000*O + 100*N + 10*E + Y,
    labeling([], Vars).

%% N-Queens problem
nqueens(N, Qs) :-
    length(Qs, N),
    Qs ins 1..N,
    safe_queens(Qs),
    labeling([], Qs).

safe_queens([]).
safe_queens([Q|Qs]) :-
    safe_queens(Qs, Q, 1),
    safe_queens(Qs).

safe_queens([], _, _).
safe_queens([Q|Qs], Q0, D0) :-
    Q #\= Q0,
    abs(Q - Q0) #\= D0,
    D1 #= D0 + 1,
    safe_queens(Qs, Q0, D1).

%% Example usage:
%% ?- send_more_money(Solution).
%%    Solution = [9,5,6,7,1,0,8,2].
%% ?- nqueens(8, Qs).
%%    Qs = [1,5,8,6,3,7,2,4] ;
