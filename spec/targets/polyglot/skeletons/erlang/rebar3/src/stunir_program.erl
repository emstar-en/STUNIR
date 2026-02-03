-module(stunir_program).
-export([main/0]).

main() ->
    stunir_runtime:program().
