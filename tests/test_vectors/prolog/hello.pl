:- initialization(main).

main :-
    open('out.txt', write, S),
    write(S, 'hello
'),
    close(S),
    halt(0).
