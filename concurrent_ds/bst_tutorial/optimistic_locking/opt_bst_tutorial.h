/* 
 * File:   bst.h
 * Author: Olivia Grimes
 *
 * Created on February 2, 2023
 */
#include <iostream>
#include "opt_node_tutorial.h"

#define SENTINEL -1
#define SENTINEL_BEG -2

#ifndef BST_H
#define BST_H

class BST {
private:
    nodeptr root;

public:
    BST();
    
    bool insert(int key);
    bool remove(int key);
    bool contains(int key);

    // printing methods - single threaded
    void printLevelOrder();
    void printInOrder();
};

#endif