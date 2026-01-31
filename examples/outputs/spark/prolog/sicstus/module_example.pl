%% STUNIR Generated SICSTUS Code
%% DO-178C Level A Compliant
%% Example: Module system and CLP

:- module(module_example, [list_sum/2, list_product/2, range_check/3]).

:- use_module(library(clpfd)).

%% Mode declarations
:- mode list_sum(+list, -integer).
:- mode list_product(+list, -integer).
:- mode range_check(+integer, +integer, +integer).

%% Sum of list elements
list_sum([], 0).
list_sum([H|T], Sum) :-
    list_sum(T, Rest),
    Sum #= H + Rest.

%% Product of list elements
list_product([], 1).
list_product([H|T], Product) :-
    list_product(T, Rest),
    Product #= H * Rest.

%% Check if value is in range
range_check(Value, Min, Max) :-
    Value #>= Min,
    Value #=< Max.

%% Constrained list generation
generate_constrained_list(N, Min, Max, List) :-
    length(List, N),
    domain(List, Min, Max),
    labeling([], List).

%% Example usage:
%% ?- list_sum([1,2,3,4,5], Sum).
%%    Sum = 15.
%% ?- list_product([2,3,4], Prod).
%%    Prod = 24.
