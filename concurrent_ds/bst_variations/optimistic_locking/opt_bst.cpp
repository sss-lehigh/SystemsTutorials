/* 
 * File:   bst_impl.h
 * Author: Olivia Grimes
 *
 * Created on February 2, 2023
 */

#include <mutex>
#include <unistd.h>
#include <iostream>
#include <cstdlib>
#include <vector>

#include <iostream>
#include <fstream>

#include "opt_bst.h"

using namespace std;

void addSentinels(Node* node) {
    Node* left_sentinel = new Node(SENTINEL);
    Node* right_sentinel = new Node(SENTINEL);
    node->left = left_sentinel;
    node->right = right_sentinel;
}

BST::BST() {
    Node* sentinel_beg = new Node(SENTINEL_BEG);
    root = sentinel_beg;
    addSentinels(sentinel_beg);
}

bool BST::verifyTraversal(nodeptr prev_check, nodeptr curr_check, int key) {
    nodeptr prev = root;
    nodeptr curr = prev->right;

    // verify that we can still get to prev
    while (curr->key != SENTINEL) {
        if (curr->key > key) {
            prev = curr;
            curr = curr->left;
        } else if (curr->key < key) {
            prev = curr;
            curr = curr->right;
        } else
            break;
    }
    
    if (prev != prev_check) {
        return false;
    }

    // due to traversal above, checking if curr = curr_check checks that prev_check still points to curr_check
    if (curr != curr_check) {
        return false;
    }
    return true;
}

/**
 * @brief insert a node into the binary search tree
 * 
 * @param key to be inserted
 * @return nodeptr - NULL if present else root
 */
bool BST::insert(int key) {
    nodeptr parent = root;
    nodeptr curr = parent->right; // the "true root"

    // find the key, and it's parent
    while (curr->key != SENTINEL) {
        if (curr->key > key) {
            parent = curr;
            curr = curr->left;
        } else if (curr->key < key) {
            parent = curr;
            curr = curr->right;
        }
        else
            return false;
    }

    // NOTE: curr is a sentinel node
    parent->mtx.lock();
    curr->mtx.lock();
    if (!verifyTraversal(parent, curr, key)) {
        parent->mtx.unlock();
        curr->mtx.unlock();
        return false;
    }

    // add sentinels to the new node
    addSentinels(curr);
    // TODO: add memory barrier?
    // curr is the sentinel that we want to replace with the new node
    curr->key = key;

    // unlock
    curr->mtx.unlock();
    parent->mtx.unlock();
    
    return true;
}

// return the root if success, NULL if root is null / key not present
bool BST::remove(int key) {
    nodeptr prev = root;
    nodeptr curr = prev->right; // "true root"
    
    // if tree is empty
    if (curr->key == SENTINEL) {
        return false;
    }

    // traverse to the node
    while (curr->key != SENTINEL) {
        if (curr->key > key) {
            prev = curr;
            curr = curr->left;
        }
        else if (curr->key < key) {
            prev = curr;
            curr = curr->right;
        } else
            break;
    }

    // didn't find the node
    if (curr->key == SENTINEL) {
        return false;
    }

    // lock & verify traversal
    prev->mtx.lock();
    curr->mtx.lock();
    if (!verifyTraversal(prev, curr, key)) {
        prev->mtx.unlock();
        curr->mtx.unlock();
        return false;
    }

    // at most one child (0-1 children)
    if (curr->left->key == SENTINEL || curr->right->key == SENTINEL) {
        // replaces node to be deleted (with l/r child or NULL)
        nodeptr newCurr;

        // check if the left child exists, set newCurr accordingly
        if (curr->left->key == SENTINEL)
            newCurr = curr->right;
        else
            newCurr = curr->left;

        // check if we are deleting the root
        if (prev->key == SENTINEL_BEG) {
            root->right = newCurr;
            prev->mtx.unlock();
            curr->mtx.unlock();
            return true;
        }

        // reset prev accordingly
        if (curr == prev->left)
            prev->left = newCurr;
        else
            prev->right = newCurr;
    }
    // two children
    else {
        nodeptr p = NULL;
        nodeptr temp;

        // TODO: how to adapt this for optimstic locking ??
        // locking here --> at most, one (additional) lock held at a time 
        // compute in-order successor
        temp = curr->right;
        temp->mtx.lock();
        while (temp->left->key != SENTINEL) {
            temp->mtx.unlock();
            p = temp;
            temp = temp->left;
            temp->mtx.lock();
        }

        // p's left child is the in-order successor
        // if temp has a right subtree, set it as p's left-subtree
        if (p != NULL)
            p->left = temp->right;
            
        // in-order successor is curr's right ptr
        // if temp has a right subtree, set it as curr's right-subtree
        else
            curr->right = temp->right;

        // change the data in which curr points to
        curr->key = temp->key;
        temp->mtx.unlock();
    }
    prev->mtx.unlock();
    curr->mtx.unlock();
    return true;
}

bool BST::contains(int key) {
    if (root->right->key == SENTINEL)
        return false;

    nodeptr prev = root;
    nodeptr curr = prev->right;
    
    while (curr->key != SENTINEL) {
        if (curr->key > key) {
            prev = curr;
            curr = curr->left;
        } else if (curr->key < key) {
            prev = curr;
            curr = curr->right;
        } else
            break;
    }

    // didn't find the key, return false;
    if (curr->key == SENTINEL) {
        return false;
    }

    // lock prev and curr
    prev->mtx.lock();
    curr->mtx.lock();
    bool ret = verifyTraversal(prev, curr, key);

    // unlock nodes
    prev->mtx.unlock();
    curr->mtx.unlock();

    return ret;
}


/**
 * 
 * NOTE:
 * The following methods are helper methods which print the 
 * tree in two different ways (in-order and by level).
 * 
 * They do not support concurrency.
 * 
 */

/**
 * @brief Compute the "height" of a tree -- the number of nodes along
 * the longest path from the root node down to the farthest leaf node
 * 
 * @param node root of tree to find height of
 * @return int height of tree
 */
int height(Node* node) {
    if (node == NULL)
        return 0;
    else {
        /* compute the height of each subtree */
        int lheight = height(node->left);
        int rheight = height(node->right);
 
        /* use the larger one */
        if (lheight > rheight)
            return (lheight + 1);
        else
            return (rheight + 1);
    }
}

void printGivenLevel(Node* root, int level) {
    if (root == NULL)
        return;
    if (level == 1)
        printf("%d ", root->key);
    else if (level > 1) {
        printGivenLevel(root->left, level - 1);
        printGivenLevel(root->right, level - 1);
    }
}

void BST::printLevelOrder() {
    Node* temp = root;
    int h = height(temp);
    int i;
    for (i = 1; i <= h; i++) {
        printGivenLevel(temp, i);
        printf("\n");
    }
}

void printInOrderHelper(nodeptr curr) {
    if (curr != NULL) {
        printInOrderHelper(curr->left);
        cout << curr->key << ' ';
        printInOrderHelper(curr->right);
    }
}

void BST::printInOrder() {
    nodeptr curr = root;
    printInOrderHelper(curr);
    cout << std::endl;
}