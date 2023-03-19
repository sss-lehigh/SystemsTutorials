/* 
 * File:   bst_impl.h
 * Author: Olivia Grimes
 *
 * Created on February 2, 2023
 */

#include "bst.h"

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

nodeptr BST::insert(int key) {
    // if there is no root, insert the new node as the root
    if (!root) {
        root = new Node(key);
        return root;
    }

    nodeptr prev = NULL;
    nodeptr curr = root;

    // find the key, and it's parent
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
            return NULL;
    }

    // create the new node, and add it to the tree
    Node* node = new Node(key);
    if (prev->key > key)
        prev->left = node;
    else
        prev->right = node;
    
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