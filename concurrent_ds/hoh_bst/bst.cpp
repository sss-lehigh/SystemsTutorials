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
nodeptr BST::insert(int key) {
    // if there is no true root, insert the new node as the root
    if (!root->right) {
        if (insertRoot(key)) {
            return root;
        }
    }

    cout << "locking sentinel and root" << endl;
    nodeptr parent = root;
    parent->mtx.lock();
    nodeptr curr = parent->right; // the "true root"
    curr->mtx.lock();

    cout << "curr->key is " << curr->key << endl;
    cout << "key is " << key << endl;

    // find the key, and it's parent
    while (curr->key != SENTINEL) {
        cout << "HoH" << endl;
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
            cout << "somehow I am here ..." << endl;
            parent->mtx.unlock();
            curr->mtx.unlock();
            return NULL;
        }
            
    }
    cout << "here lolz" << endl;

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
nodeptr BST::remove(int key) {
    nodeptr prev = NULL;
    nodeptr curr = root;

    if (root == NULL)
        return root;

    while (curr) {
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

    // didn't find the node -> return null
    if (curr == NULL)
        return NULL;

    // at most one child (0-1 children)
    if (curr->left == NULL || curr->right == NULL) {
        // replaces node to be deleted
        nodeptr newCurr;

        // check if the left child exists, set newCurr accordingly
        if (curr->left == NULL)
            newCurr = curr->right;
        else
            newCurr = curr->left;

        // check if we are deleting the root
        if (prev == NULL) {
            root = newCurr;
            return root;
        }

        // reset prev accordingly
        if (curr == prev->left)
            prev->left = newCurr;
        else
            prev->right = newCurr;

        // free memory of curr
        delete curr;

    }
    // two children
    else {
        nodeptr p = NULL;
        nodeptr temp;

        // compute in-order successor
        temp = curr->right;
        while (temp->left) {
            p = temp;
            temp = temp->left;
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
        // delete temp which points to the node whose data we moved to the "true" node to be deleted
        delete temp;
    }
    return root;
}

bool BST::contains(int key) {
    if (root == NULL)
        return false;

    nodeptr curr = root;
    while (curr) {
        if (curr->key > key)
            curr = curr->left;
        else if (curr->key < key)
            curr = curr->right;
        else
            return true;
    }
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