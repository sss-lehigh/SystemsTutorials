# Makefile for BST

CC = g++
CFLAGS = -Wall -O3 -g -pthread

all: hoh opt

hoh:
	$(CC) $(CFLAGS) -o main_hoh -DHOH_LOCKING -I common -I hoh_bst main.cpp bst_variations/hoh_bst/bst.cpp

opt:
	$(CC) $(CFLAGS) -o main_opt -DOPTIMISTIC_LOCKING -I common -I optimistic_locking main.cpp bst_variations/optimistic_locking/bst.cpp

seq:
	$(CC) $(CFLAGS) -o main_seq -DSEQUENTIAL -I common -I seq_bst main.cpp bst_variations/seq_bst/bst.cpp

clean:
	rm -r *.out