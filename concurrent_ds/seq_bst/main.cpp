/* 
 * File:   main.cpp
 * Author: Olivia Grimes
 *
 * Created on February 6, 2023
 */

#include "bst.h"

void insert(BST* tree, int key) {
    printf("Inserting: %i\n", key);
    Node* ret = tree->insert(key);
    if (ret == NULL) {
        printf("   Key %i already inserted.\n", key);
    } else {
        tree->printInOrder();
    }
    printf("\n");
}

void remove(BST* tree, int key) {
    printf("\nRemoving: %i\n", key);
    Node* ret = tree->remove(key);
    if (ret == NULL) {
        printf("   Key %i not present in structure.\n", key);
    } else {
        tree->printInOrder();
    }
    printf("\n");
}

void search(BST* tree, int key) {
    printf("\nSearching for: %i\n", key);
    bool present = tree->search(key);
    if (present)
        printf("  Key %i is present", key);
    else
        printf("  Key %i is NOT present", key);
}

int main() {
    BST* tree = new BST();
    
    insert(tree, 50);
    insert(tree, 40);
    insert(tree, 60);
    insert(tree, 60);
    insert(tree, 70);
    insert(tree, 10);
    insert(tree, 80);
    insert(tree, 20);
    insert(tree, 55);
    
    printf("\nLevel Order:\n");
    tree->printLevelOrder();

    search(tree, 60);
    search(tree, 57);

    remove(tree, 60);
    search(tree, 60);
    remove(tree, 45);
    
    printf("\nLevel Order:\n");
    tree->printLevelOrder();
}