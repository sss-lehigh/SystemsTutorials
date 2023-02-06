# Makefile for BST

CC = g++
CFLAGS = -Wall -g

default: main

main: main.o bst.o
	$(CC) $(CFLAGS) -o main main.o bst.o

main.o: main.cpp bst.h node.h
	$(CC) $(CFLAGS) -c main.cpp

bst.o: bst.h node.h
	$(CC) $(CFLAGS) -c bst.cpp

clean:
	rm -r *.o