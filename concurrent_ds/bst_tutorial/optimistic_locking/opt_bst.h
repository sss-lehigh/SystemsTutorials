/* 
 * File:   opt_bst.h
 * Author: Olivia Grimes
 *
 * Created on February 2, 2023
 */
#include <iostream>
#include "opt_node.h"

#ifndef BST_H
#define BST_H

class BST {
private:
    nodeptr root;

public:
    BST() : root(NULL) {}

    // BST API
    nodeptr insert(int key);
    nodeptr remove(int key);
    bool contains(int key);

    // printing methods
    void printLevelOrder();
    void printInOrder();
};

#endif