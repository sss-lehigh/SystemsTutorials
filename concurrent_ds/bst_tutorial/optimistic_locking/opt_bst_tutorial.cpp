/* 
 * File:   bst_impl.h
 * Author: Olivia Grimes
 *
 * Created on February 2, 2023
 */
#include <mutex>
#include <shared_mutex>
#include "hoh_bst.h"

using namespace std;

void addSentinels(Node* node) {
    node->left = new Node(SENTINEL);;
    node->right = new Node(SENTINEL);;
}

BST::BST() {
    root = new Node(SENTINEL_BEG);
    addSentinels(root);
}

/**
 * @brief insert a node into the binary search tree
 * 
 * @param key to be inserted
 * @return nodeptr - NULL if present else root
 */
bool BST::insert(int key) {
    // lock the sentinel node
    nodeptr parent = root;
    parent->mtx.lock_shared();

    // add root to the tree
    if (parent->right->key == SENTINEL) {
        parent->mtx.unlock_shared();
        parent->mtx.lock(); // lock in exclusive mode, then determine if we still need to add first element
        if (parent->right->key == SENTINEL) {
            // sentinel_beg's right sentinel becomes the "true root"
            parent->right->key = key;
            // add sentinels to the true root
            addSentinels(parent->right);
            parent->mtx.unlock();
            return true;
        }
        // if another thread beat us to inserting the root, unlock in exclusive mode, and re-lock in shared mode
        parent->mtx.unlock();
        parent->mtx.lock_shared(); // if we don't return, re-aquire the lock
    }

    nodeptr curr = parent->right; // the "true root"
    curr->mtx.lock_shared();

    // find the key, and it's parent
    while (curr->key != SENTINEL) {
        if (curr->key > key) {
            parent->mtx.unlock_shared();
            parent = curr;
            curr = curr->left;
            curr->mtx.lock_shared();
        } else if (curr->key < key) {
            parent->mtx.unlock_shared();
            parent = curr;
            curr = curr->right;
            curr->mtx.lock_shared();
        }
        // key is already present
        else {
            parent->mtx.unlock_shared();
            curr->mtx.unlock_shared();
            return false;
        }
    }
    // unlock in shared mode, and re-lock in exclusive
    parent->mtx.unlock_shared();
    parent->mtx.lock();
    curr->mtx.unlock_shared();
    curr->mtx.lock();

    // curr is now the sentinel that we want to replace with the new node
    curr->key = key;
    // add sentinels to the new node
    addSentinels(curr);

    // unlock
    curr->mtx.unlock();
    parent->mtx.unlock();
    
    return true;
}

// return the root if success, NULL if root is null / key not present
bool BST::remove(int key) {
    nodeptr prev = root;
    prev->mtx.lock_shared();

    // if tree is empty
    if (prev->right->key == SENTINEL) {
        prev->mtx.unlock_shared();
        return false;
    }
    // get the "true root"
    nodeptr curr = prev->right;
    curr->mtx.lock_shared();

    // traverse to the node
    while (curr->key != SENTINEL) {
        if (curr->key > key) {
            prev->mtx.unlock_shared();
            prev = curr;
            curr = curr->left;
            curr->mtx.lock_shared();
        }
        else if (curr->key < key) {
            prev->mtx.unlock_shared();
            prev = curr;
            curr = curr->right;
            curr->mtx.lock_shared();
        } else
            break;
    }

    // didn't find the node
    if (curr->key == SENTINEL) {
        prev->mtx.unlock_shared();
        curr->mtx.unlock_shared();
        return false;
    }
    
    // unlock in shared mode, lock in exclusive mode
    prev->mtx.unlock_shared();
    curr->mtx.unlock_shared();
    prev->mtx.lock();
    curr->mtx.lock();

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
    if (root->right->key == SENTINEL)
        return false;

    nodeptr prev = root;
    prev->mtx.lock_shared();
    nodeptr curr = prev->right;
    curr->mtx.lock_shared();
    while (curr->key != SENTINEL) {
        if (curr->key > key) {
            prev->mtx.unlock_shared();
            prev = curr;
            curr = curr->left;
            curr->mtx.lock_shared();
        } else if (curr->key < key) {
            prev->mtx.unlock_shared();
            prev = curr;
            curr = curr->right;
            curr->mtx.lock_shared();
        } else {
            prev->mtx.unlock_shared();
            curr->mtx.unlock_shared();
            return true;
        }     
    }
    prev->mtx.unlock_shared();
    curr->mtx.unlock_shared();
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