/* 
 * File:   node.h
 * Author: Olivia Grimes
 *
 * Created on February 2, 2023
 */
#include <cstddef>
#include <mutex>
#include <thread>
#include <iostream>

using namespace std;

#ifndef NODE_H
#define NODE_H

#define nodeptr Node* volatile

class Node {
public:
    int key;
    nodeptr left;
    nodeptr right;
    mutex mtx;

    Node(int key)
        : key(key)
        , left(NULL)
        , right(NULL) {}
};

#endif