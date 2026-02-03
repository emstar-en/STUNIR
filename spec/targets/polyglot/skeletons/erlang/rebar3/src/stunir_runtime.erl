-module(stunir_runtime).
-export([println/1, program/0]).

println(S) when is_list(S) ->
    io:format("~s~n", [S]);
println(S) ->
    io:format("~p~n", [S]).

program() ->
    %% STUB: codegen should replace this body.
    println("STUNIR Erlang skeleton: program stub"),
    0.
