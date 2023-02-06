/* 
 * File:   bst.h
 * Author: Olivia Grimes
 *
 * Created on February 2, 2023
 */
#include <iostream>
#include "node.h"

#ifndef BST_H
#define BST_H

class BST {
private:
    nodeptr root;

public:
    BST() : root(NULL) {}

    void printLevelOrder();
    void printInOrder();
    nodeptr insert(int key);
    nodeptr remove(int key);
    bool search(int key);
};

#endif