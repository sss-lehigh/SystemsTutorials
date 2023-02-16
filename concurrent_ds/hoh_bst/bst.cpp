/* 
 * File:   bst_impl.h
 * Author: Olivia Grimes
 *
 * Created on February 2, 2023
 */

#include <mutex>
#include "bst.h"

using namespace std;

std::mutex rootMtx;

BST::BST() {
    Node* sentinel_beg = new Node(SENTINEL);
    root = sentinel_beg;
}

void addSentinels(Node* node) {
    Node* left_sentinel = new Node(SENTINEL);
    Node* right_sentinel = new Node(SENTINEL);
    node->left = left_sentinel;
    node->right = right_sentinel;
}

bool BST::insertRoot(int key) {
    // block threads that don't acquire rootMtx first to avoid issues w inserting until the root is set
    rootMtx.lock();

    // someone else already initialized the root
    if (root->right) {
        rootMtx.unlock();
        return false;
    }
    // create the root and add it to the tree (NOTE: beg sentinel's left ptr is NULL, right is to "true root")
    Node* trueRoot = new Node(key);
    addSentinels(trueRoot);
    root->right = trueRoot;

    rootMtx.unlock();
    return true;
}

/**
 * @brief insert a node into the binary search tree
 * 
 * @param key to be inserted
 * @return nodeptr - NULL if present else root
 */
bool BST::insert(int key) {
    // if there is no true root, insert the new node as the root

    // if (!root->right) {
    //     if (insertRoot(key)) {
    //         return root;
    //     }
    // }

    nodeptr parent = root;
    parent->mtx.lock();

    if (!parent->right) {
        Node* trueRoot = new Node(key);
        addSentinels(trueRoot);
        root->right = trueRoot;
        return root;
    }

    nodeptr curr = parent->right; // the "true root"
    curr->mtx.lock();

    // find the key, and it's parent
    while (curr->key != SENTINEL) {
        if (curr->key > key) {
            parent->mtx.unlock();
            parent = curr;
            curr = curr->left;
            curr->mtx.lock();
        } else if (curr->key < key) {
            parent->mtx.unlock();
            parent = curr;
            curr = curr->right;
            curr->mtx.lock();
        }
        // key is already present
        else {
            parent->mtx.unlock();
            curr->mtx.unlock();
            return NULL;
        }
    }

    // curr is now the sentinel that we want to replace with the new node
    curr->key = key;
    // add sentinels to the new node
    addSentinels(curr);

    // unlock
    curr->mtx.unlock();
    parent->mtx.unlock();
    
    return root;
}

// return the root if success, NULL if root is null / key not present
bool BST::remove(int key) {
    nodeptr prev = root;
    prev->mtx.lock();
    if (!prev->right) {
        prev->mtx.unlock();
        return false;
    }
    nodeptr curr = prev->right;
    curr->mtx.lock();

    // traverse to the node
    while (curr->key != SENTINEL) {
        if (curr->key > key) {
            prev->mtx.unlock();
            prev = curr;
            curr = curr->left;
            curr->mtx.lock();
        }
        else if (curr->key < key) {
            prev->mtx.unlock();
            prev = curr;
            curr = curr->right;
            curr->mtx.lock();
        } else
            break;
    }

    // didn't find the node
    // TODO: could it be SENTINEL_BEG possibly ?
    if (curr->key == SENTINEL)
        return false;

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
    return true;;
}

bool BST::contains(int key) {
    if (!root->right)
        return false;

    nodeptr prev = root;
    prev->mtx.lock();
    nodeptr curr = prev->right;
    curr->mtx.lock();
    while (curr->key != SENTINEL) {
        if (curr->key > key) {
            prev->mtx.unlock();
            prev = curr;
            curr = curr->left;
            curr->mtx.lock();
        } else if (curr->key < key) {
            prev->mtx.unlock();
            prev = curr;
            curr = curr->right;
            curr->mtx.lock();
        } else {
            prev->mtx.unlock();
            curr->mtx.unlock();
            return true;
        }     
    }
    prev->mtx.unlock();
    curr->mtx.unlock();
    return false;
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