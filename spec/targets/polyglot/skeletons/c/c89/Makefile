CC ?= cc
CFLAGS ?= -O2 -Wall -Wextra -pedantic

all: stunir_program

stunir_program: main.c stunir_runtime.c stunir_runtime.h
	$(CC) $(CFLAGS) -o $@ main.c stunir_runtime.c

clean:
	rm -f stunir_program
