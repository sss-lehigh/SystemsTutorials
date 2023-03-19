/* 
 * File:   node.h
 *
 * Created on February 2, 2023
 */
#include <cstddef>

using namespace std;

#ifndef NODE_H
#define NODE_H

#define nodeptr Node* volatile

class Node {
public:
    int key;
    nodeptr left;
    nodeptr right;

    Node(int key)
        : key(key)
        , left(NULL)
        , right(NULL) {}
};

#endif