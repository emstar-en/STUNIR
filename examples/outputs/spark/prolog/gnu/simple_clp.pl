% STUNIR Generated GNU_PROLOG Code
% DO-178C Level A Compliant
% Example: Simple constraint logic programming

% Module: simple_clp

:- include('clpfd.pl').

% Sudoku cell constraint
sudoku_cell(X) :-
    X #>= 1,
    X #=< 9.

% Simple constraint problem
solve_constraint(X, Y, Z) :-
    X #> 0,
    Y #> 0,
    Z #> 0,
    X #+ Y #= 10,
    Y #+ Z #= 15,
    X #=< Z,
    fd_labeling([X, Y, Z]).

% Magic square 3x3
magic_square([A,B,C,D,E,F,G,H,I]) :-
    Vars = [A,B,C,D,E,F,G,H,I],
    fd_domain(Vars, 1, 9),
    fd_all_different(Vars),
    Sum #= 15,
    A #+ B #+ C #= Sum,
    D #+ E #+ F #= Sum,
    G #+ H #+ I #= Sum,
    A #+ D #+ G #= Sum,
    B #+ E #+ H #= Sum,
    C #+ F #+ I #= Sum,
    A #+ E #+ I #= Sum,
    C #+ E #+ G #= Sum,
    fd_labeling(Vars).

% Example usage:
% ?- solve_constraint(X, Y, Z).
%    X = 1, Y = 9, Z = 6 ;
