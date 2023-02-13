/* 
 * File:   bst.h
 * Author: Olivia Grimes
 *
 * Created on February 2, 2023
 */
#include <iostream>
#include "node.h"

// NOTE: alternatively could add a "type" field to the node to allow for negative keys
#define SENTINEL -1

#ifndef BST_H
#define BST_H

class BST {
private:
    nodeptr root;
    bool insertRoot(int key);

public:
    //BST() : root(NULL) {}
    BST();
    void printLevelOrder();
    void printInOrder();
    nodeptr insert(int key);
    nodeptr remove(int key);
    bool contains(int key);
};

#endif