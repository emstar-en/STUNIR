%% STUNIR Generated ECLIPSE Code
%% DO-178C Level A Compliant
%% Example: Constraint optimization

:- module(optimization).

:- lib(ic).
:- lib(branch_and_bound).

%% Knapsack problem
%% Items: [(weight, value), ...]
%% Capacity: maximum weight
knapsack(Items, Capacity, Selection, TotalValue) :-
    length(Items, N),
    length(Selection, N),
    Selection :: 0..1,  % Binary: item included or not
    
    % Extract weights and values
    ( foreach((W, V), Items),
      foreach(Weight, Weights),
      foreach(Value, Values)
    do
      Weight = W,
      Value = V
    ),
    
    % Weight constraint
    ( foreach(S, Selection),
      foreach(W, Weights),
      fromto(0, In, Out, TotalWeight)
    do
      Out #= In + S * W
    ),
    TotalWeight #=< Capacity,
    
    % Calculate total value
    ( foreach(S, Selection),
      foreach(V, Values),
      fromto(0, In, Out, TotalValue)
    do
      Out #= In + S * V
    ),
    
    % Maximize total value
    Cost #= -TotalValue,
    minimize(labeling(Selection), Cost).

%% Example: Resource allocation
allocate_resources(Workers, Tasks, Assignment, Cost) :-
    dim(Assignment, [Workers, Tasks]),
    Assignment :: 0..1,  % Binary assignment matrix
    
    % Each task assigned to exactly one worker
    ( for(T, 1, Tasks), param(Assignment, Workers)
    do
      ( for(W, 1, Workers),
        foreach(A, TaskAssignments),
        param(Assignment, T)
      do
        A is Assignment[W,T]
      ),
      sum(TaskAssignments) #= 1
    ),
    
    % Calculate total cost
    flatten_array(Assignment, Flat),
    sum(Flat) #= Cost,
    
    minimize(labeling(Flat), Cost).

%% Example usage:
%% ?- knapsack([(2,3), (3,4), (4,5), (5,6)], 9, Sel, Val).
%%    Sel = [1,1,1,0], Val = 12.
