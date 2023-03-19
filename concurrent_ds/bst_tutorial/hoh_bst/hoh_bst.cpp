/* 
 * File:   hoh_bst.h
 *
 * Created on February 2, 2023
 */

#include "hoh_bst.h"

BST::BST() {
    root = NULL;
}

bool BST::contains(int key) {
    if (root == NULL)
        return false;

    nodeptr curr = root;
    // iterate through the tree until we find the key we are searching for, or reach a null sentinel
    while (curr) {
        if (curr->key > key)
            curr = curr->left;
        else if (curr->key < key)
            curr = curr->right;
        else
            return true; // found key
    }
    return false; // reached a null sentinel
}

bool BST::insert(int key) {
    // if there is no root, insert the new node as the root
    if (!root) {
        root = new Node(key);
        return true;
    }

    nodeptr prev = NULL;
    nodeptr curr = root;

    // search for the key to insert
    while (curr) {
        if (curr->key > key) {
            prev = curr;
            curr = curr->left;
        } else if (curr->key < key) {
            prev = curr;
            curr = curr->right;
        }
        // key is already present
        else
            return false;
    }
    // key not found -> prev points to the leaf node that will be the parent of the new node

    // create the new node, and add it to the tree
    Node* node = new Node(key);
    // determine if the new node should be the L or R child of its parent (prev)
    if (prev->key > key)
        prev->left = node;
    else
        prev->right = node;
    
    return true;
}

// return the root if success, NULL if root is null / key not present
bool BST::remove(int key) {
    nodeptr prev = NULL;
    nodeptr curr = root;

    if (root == NULL)
        return false;

    // find node to remove
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
        return false;

    // node to remove has at most one child (0-1 children)
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
            return true;
        }

        // reset prev accordingly
        if (curr == prev->left)
            prev->left = newCurr;
        else
            prev->right = newCurr;

        // free memory of curr
        delete curr;

    }
    // node to remove has two children
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
        // set p's left subtree as temp's right subtree
        if (p != NULL)
            p->left = temp->right;
        // p is null, i.e., in-order successor IS curr's right ptr
        // set curr's right-subtree as temp's right subtree
        else
            curr->right = temp->right;

        // change the data in which curr points to
        curr->key = temp->key;
        // delete temp which points to the node whose data we moved to the "true" node to be deleted
        delete temp;
    }
    return true;
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
 
/* Compute the "height" of a tree -- the number of
 nodes along the longest path from the root node
 down to the farthest leaf node.*/
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