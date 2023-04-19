# Makefile for BST

CC = g++ -std=c++17
CFLAGS = -Wall -O3 -g -pthread

all: solution tutorial

solution: seq hoh opt 
tutorial: seq_tutorial hoh_tutorial opt_tutorial

seq:
	$(CC) $(CFLAGS) -o seq -DSEQUENTIAL -I common main.cpp bst_solution/seq_bst/seq_bst.cpp

hoh:
	$(CC) $(CFLAGS) -o hoh -DHOH_LOCKING -I common main.cpp bst_solution/hoh_bst/hoh_bst.cpp

opt:
	$(CC) $(CFLAGS) -o opt -DOPTIMISTIC_LOCKING -I common main.cpp bst_solution/optimistic_locking/opt_bst.cpp

seq_tutorial:
	$(CC) $(CFLAGS) -o seq_tutorial -DSEQUENTIAL_TUTORIAL -I common main.cpp bst_tutorial/seq_bst/seq_bst.cpp

hoh_tutorial:
	$(CC) $(CFLAGS) -o hoh_tutorial -DHOH_LOCKING_TUTORIAL -I common main.cpp bst_tutorial/hoh_bst/hoh_bst.cpp

opt_tutorial:
	$(CC) $(CFLAGS) -o opt_tutorial -DOPTIMISTIC_LOCKING_TUTORIAL -I common main.cpp bst_tutorial/optimistic_locking/opt_bst.cpp

clean:
	if [ -f hoh ]; then rm hoh; fi
	if [ -f opt ]; then rm opt; fi
	if [ -f seq ]; then rm seq; fi
	if [ -f seq_tutorial ]; then rm seq_tutorial; fi
	if [ -f hoh_tutorial ]; then rm hoh_tutorial; fi
	if [ -f opt_tutorial ]; then rm opt_tutorial; fi
